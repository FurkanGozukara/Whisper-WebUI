import os
import time
import huggingface_hub
import numpy as np
import torch
from typing import BinaryIO, Union, Tuple, List, Callable, Optional
import faster_whisper
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions
import ast
import ctranslate2
import whisper
import gradio as gr
from argparse import Namespace

from modules.utils.paths import (FASTER_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, UVR_MODELS_DIR, OUTPUT_DIR)
from modules.whisper.data_classes import *
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.utils.logger import get_logger

logger = get_logger()


class FasterWhisperInference(BaseTranscriptionPipeline):
    def __init__(self,
                 model_dir: str = FASTER_WHISPER_MODELS_DIR,
                 diarization_model_dir: str = DIARIZATION_MODELS_DIR,
                 uvr_model_dir: str = UVR_MODELS_DIR,
                 output_dir: str = OUTPUT_DIR,
                 ):
        super().__init__(
            model_dir=model_dir,
            diarization_model_dir=diarization_model_dir,
            uvr_model_dir=uvr_model_dir,
            output_dir=output_dir
        )
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.model_paths = self.get_model_paths()
        self.device = self.get_device()
        self.available_models = self.model_paths.keys()

    def transcribe(self,
                   audio: Union[str, BinaryIO, np.ndarray],
                   progress: gr.Progress = gr.Progress(),
                   progress_callback: Optional[Callable] = None,
                   *whisper_params,
                   ) -> Tuple[List[Segment], float]:
        """
        transcribe method for faster-whisper.

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio path or file binary or Audio numpy array
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        progress_callback: Optional[Callable]
            callback function to show progress. Can be used to update progress in the backend.
        *whisper_params: tuple
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

        Returns
        ----------
        segments_result: List[Segment]
            list of Segment that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for transcription
        """
        start_time = time.time()

        params = WhisperParams.from_list(list(whisper_params))

        if params.model_size != self.current_model_size or self.model is None or self.current_compute_type != params.compute_type:
            self.update_model(params.model_size, params.compute_type, progress)

        progress(0, desc="Loading audio..")
        segments, info = self._transcribe_with_batching(audio=audio, params=params)

        segments_result = []
        for idx, segment in enumerate(segments):
            progress_n = 0.0 if not info.duration else min(segment.end / info.duration, 0.99)
            seg_obj = Segment.from_faster_whisper(segment)
            segments_result.append(seg_obj)

            # Live transcription display in terminal
            logger.info(f"[{self.format_timestamp(seg_obj.start)} -> {self.format_timestamp(seg_obj.end)}] {seg_obj.text}")

            # Update progress with current segment info
            progress(progress_n, desc=f"Transcribing.. [{idx+1} segments] {seg_obj.text[:50]}...")

            self.emit_progress_callback(progress_callback, progress_n, seg_obj)

        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    def _transcribe_with_batching(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        params: WhisperParams,
    ):
        batch_pipeline_cls = getattr(faster_whisper, "BatchedInferencePipeline", None)
        if batch_pipeline_cls is None:
            logger.warning("Installed faster-whisper build does not support BatchedInferencePipeline. Falling back to standard transcription.")
            return self._transcribe_with_standard_pipeline(audio=audio, params=params)

        sampling_rate = self.model.feature_extractor.sampling_rate
        audio_array = self.prepare_audio_array(audio=audio, sampling_rate=sampling_rate)
        clip_timestamps = self.build_clip_timestamps(
            audio=audio_array,
            chunk_length=params.chunk_length,
            sampling_rate=sampling_rate,
        )

        batch_pipeline = batch_pipeline_cls(model=self.model)
        return batch_pipeline.transcribe(
            audio=audio_array,
            language=params.lang,
            task="translate" if params.is_translate else "transcribe",
            beam_size=params.beam_size,
            log_prob_threshold=params.log_prob_threshold,
            no_speech_threshold=params.no_speech_threshold,
            best_of=params.best_of,
            patience=params.patience,
            temperature=params.temperature,
            initial_prompt=params.initial_prompt,
            compression_ratio_threshold=params.compression_ratio_threshold,
            length_penalty=params.length_penalty,
            repetition_penalty=params.repetition_penalty,
            no_repeat_ngram_size=params.no_repeat_ngram_size,
            prefix=params.prefix,
            suppress_blank=params.suppress_blank,
            suppress_tokens=params.suppress_tokens,
            without_timestamps=False,
            max_initial_timestamp=params.max_initial_timestamp,
            word_timestamps=params.word_timestamps,
            prepend_punctuations=params.prepend_punctuations,
            append_punctuations=params.append_punctuations,
            max_new_tokens=params.max_new_tokens,
            chunk_length=params.chunk_length,
            clip_timestamps=clip_timestamps,
            hallucination_silence_threshold=params.hallucination_silence_threshold,
            batch_size=max(1, int(params.batch_size)),
            hotwords=params.hotwords,
            language_detection_threshold=params.language_detection_threshold,
            language_detection_segments=params.language_detection_segments,
            condition_on_previous_text=params.condition_on_previous_text,
            prompt_reset_on_temperature=params.prompt_reset_on_temperature,
        )

    def _transcribe_with_standard_pipeline(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        params: WhisperParams,
    ):
        return self.model.transcribe(
            audio=audio,
            language=params.lang,
            task="translate" if params.is_translate else "transcribe",
            beam_size=params.beam_size,
            log_prob_threshold=params.log_prob_threshold,
            no_speech_threshold=params.no_speech_threshold,
            best_of=params.best_of,
            patience=params.patience,
            temperature=params.temperature,
            initial_prompt=params.initial_prompt,
            compression_ratio_threshold=params.compression_ratio_threshold,
            length_penalty=params.length_penalty,
            repetition_penalty=params.repetition_penalty,
            no_repeat_ngram_size=params.no_repeat_ngram_size,
            prefix=params.prefix,
            suppress_blank=params.suppress_blank,
            suppress_tokens=params.suppress_tokens,
            max_initial_timestamp=params.max_initial_timestamp,
            word_timestamps=params.word_timestamps,
            prepend_punctuations=params.prepend_punctuations,
            append_punctuations=params.append_punctuations,
            max_new_tokens=params.max_new_tokens,
            chunk_length=params.chunk_length,
            hallucination_silence_threshold=params.hallucination_silence_threshold,
            hotwords=params.hotwords,
            language_detection_threshold=params.language_detection_threshold,
            language_detection_segments=params.language_detection_segments,
            condition_on_previous_text=params.condition_on_previous_text,
            prompt_reset_on_temperature=params.prompt_reset_on_temperature,
        )

    @staticmethod
    def prepare_audio_array(audio: Union[str, BinaryIO, np.ndarray], sampling_rate: int) -> np.ndarray:
        if not isinstance(audio, np.ndarray):
            audio_array = decode_audio(audio, sampling_rate=sampling_rate)
        else:
            audio_array = np.asarray(audio, dtype=np.float32)

        if audio_array.ndim > 1:
            channel_axis = 0 if audio_array.shape[0] <= audio_array.shape[-1] else -1
            audio_array = audio_array.mean(axis=channel_axis)

        return np.ascontiguousarray(audio_array.squeeze(), dtype=np.float32)

    @staticmethod
    def build_clip_timestamps(
        audio: np.ndarray,
        chunk_length: Optional[int],
        sampling_rate: int,
    ) -> List[dict]:
        total_samples = int(audio.shape[-1]) if audio.size else 0
        if total_samples <= 0:
            return []

        if chunk_length is None or chunk_length <= 0:
            return [{"start": 0, "end": total_samples}]

        chunk_samples = max(1, int(chunk_length * sampling_rate))
        return [
            {
                "start": start,
                "end": min(start + chunk_samples, total_samples),
            }
            for start in range(0, total_samples, chunk_samples)
        ]

    @staticmethod
    def emit_progress_callback(
        progress_callback: Optional[Callable],
        progress_value: float,
        segment: Optional[Segment] = None,
    ):
        if progress_callback is None:
            return

        try:
            progress_callback(progress_value, segment)
        except TypeError:
            progress_callback(progress_value)
    
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Format seconds to HH:MM:SS.mmm"""
        if seconds is None:
            return "00:00:00.000"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress = gr.Progress()
                     ):
        """
        Update current model setting

        Parameters
        ----------
        model_size: str
            Size of whisper model. If you enter the huggingface repo id, it will try to download the model
            automatically from huggingface.
        compute_type: str
            Compute type for transcription.
            see more info : https://opennmt.net/CTranslate2/quantization.html
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        """
        progress(0, desc="Initializing Model..")

        model_size_dirname = model_size.replace("/", "--") if "/" in model_size else model_size
        if model_size not in self.model_paths and model_size_dirname not in self.model_paths:
            print(f"Model is not detected. Trying to download \"{model_size}\" from huggingface to "
                  f"\"{os.path.join(self.model_dir, model_size_dirname)} ...")
            huggingface_hub.snapshot_download(
                model_size,
                local_dir=os.path.join(self.model_dir, model_size_dirname),
            )
            self.model_paths = self.get_model_paths()
            gr.Info(f"Model is downloaded with the name \"{model_size_dirname}\"")

        self.current_model_size = self.model_paths[model_size_dirname]

        local_files_only = False
        hf_prefix = "models--Systran--faster-whisper-"
        official_model_path = os.path.join(self.model_dir, hf_prefix+model_size)
        if ((os.path.isdir(self.current_model_size) and os.path.exists(self.current_model_size)) or
            (model_size in faster_whisper.available_models() and os.path.exists(official_model_path))):
            local_files_only = True

        self.current_compute_type = compute_type
        self.model = faster_whisper.WhisperModel(
            device=self.device,
            model_size_or_path=self.current_model_size,
            download_root=self.model_dir,
            compute_type=self.current_compute_type,
            local_files_only=local_files_only
        )

    def get_model_paths(self):
        """
        Get available models from models path including fine-tuned model.

        Returns
        ----------
        Name list of models
        """
        model_paths = {model:model for model in faster_whisper.available_models()}
        faster_whisper_prefix = "models--Systran--faster-whisper-"

        existing_models = os.listdir(self.model_dir)
        wrong_dirs = [".locks", "faster_whisper_models_will_be_saved_here"]
        existing_models = list(set(existing_models) - set(wrong_dirs))

        for model_name in existing_models:
            if faster_whisper_prefix in model_name:
                model_name = model_name[len(faster_whisper_prefix):]

            if model_name not in whisper.available_models():
                model_paths[model_name] = os.path.join(self.model_dir, model_name)
        return model_paths

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "auto"

    @staticmethod
    def format_suppress_tokens_str(suppress_tokens_str: str) -> List[int]:
        try:
            suppress_tokens = ast.literal_eval(suppress_tokens_str)
            if not isinstance(suppress_tokens, list) or not all(isinstance(item, int) for item in suppress_tokens):
                raise ValueError("Invalid Suppress Tokens. The value must be type of List[int]")
            return suppress_tokens
        except Exception as e:
            raise ValueError("Invalid Suppress Tokens. The value must be type of List[int]")
