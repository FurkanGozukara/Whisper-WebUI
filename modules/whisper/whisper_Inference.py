import whisper
import gradio as gr
import time
from typing import BinaryIO, Union, Tuple, List, Callable, Optional
import numpy as np
import torch
import os
from argparse import Namespace

from modules.utils.paths import (WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR, UVR_MODELS_DIR)
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.whisper.data_classes import *
from modules.whisper.data_classes import Segment, Word
from modules.utils.logger import get_logger
from modules.utils.torch_compat import torch_load_safe_globals

logger = get_logger()


class WhisperInference(BaseTranscriptionPipeline):
    def __init__(self,
                 model_dir: str = WHISPER_MODELS_DIR,
                 diarization_model_dir: str = DIARIZATION_MODELS_DIR,
                 uvr_model_dir: str = UVR_MODELS_DIR,
                 output_dir: str = OUTPUT_DIR,
                 ):
        super().__init__(
            model_dir=model_dir,
            output_dir=output_dir,
            diarization_model_dir=diarization_model_dir,
            uvr_model_dir=uvr_model_dir
        )

    def transcribe(self,
                   audio: Union[str, np.ndarray, torch.Tensor],
                   progress: gr.Progress = gr.Progress(),
                   progress_callback: Optional[Callable] = None,
                   *whisper_params,
                   log_console: bool = True,
                   log_model_banner: bool = True,
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
        audio = self.prepare_audio_input(audio)
        if isinstance(audio, np.ndarray) and audio.size <= 0:
            logger.info("Whisper skipped empty audio input.")
            return [], time.time() - start_time

        if params.model_size != self.current_model_size or self.model is None or self.current_compute_type != params.compute_type:
            self.update_model(params.model_size, params.compute_type, progress)

        def progress_hook(progress_value):
            progress(progress_value, desc="Transcribing..")
            if progress_callback is None:
                return
            try:
                progress_callback(progress_value, None)
            except TypeError:
                progress_callback(progress_value)

        result = self.model.transcribe(audio=audio,
                                       language=params.lang,
                                       verbose=False,
                                       beam_size=params.beam_size,
                                       logprob_threshold=params.log_prob_threshold,
                                       no_speech_threshold=params.no_speech_threshold,
                                       task="translate" if params.is_translate else "transcribe",
                                       fp16=True if params.compute_type == "float16" else False,
                                       best_of=params.best_of,
                                       patience=params.patience,
                                       temperature=params.temperature,
                                       compression_ratio_threshold=params.compression_ratio_threshold,
                                       word_timestamps=params.word_timestamps,
                                       initial_prompt=params.initial_prompt,
                                       condition_on_previous_text=params.condition_on_previous_text,
                                       prepend_punctuations=params.prepend_punctuations,
                                       append_punctuations=params.append_punctuations,
                                       hallucination_silence_threshold=params.hallucination_silence_threshold,
                                       progress_callback=progress_hook,)["segments"]
        segments_result = []
        for segment in result:
            # Extract word-level timestamps if available
            words = None
            if "words" in segment and segment["words"]:
                words = [
                    Word(
                        start=w.get("start"),
                        end=w.get("end"),
                        word=w.get("word"),
                        probability=w.get("probability")
                    ) for w in segment["words"]
                ]
            
            segments_result.append(Segment(
                start=segment["start"],
                end=segment["end"],
                text=segment["text"],
                words=words
            ))

        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    @staticmethod
    def prepare_audio_input(audio: Union[str, np.ndarray, torch.Tensor]):
        if isinstance(audio, torch.Tensor):
            audio_array = audio.detach().cpu().float().numpy()
        elif isinstance(audio, np.ndarray):
            audio_array = np.asarray(audio, dtype=np.float32)
        else:
            return audio

        if audio_array.ndim > 1:
            channel_axis = 0 if audio_array.shape[0] <= audio_array.shape[-1] else -1
            audio_array = audio_array.mean(axis=channel_axis)

        audio_array = np.ascontiguousarray(audio_array.reshape(-1), dtype=np.float32)
        if audio_array.size and not np.isfinite(audio_array).all():
            invalid_samples = int(audio_array.size - np.isfinite(audio_array).sum())
            logger.warning(
                "Whisper audio contains %d non-finite sample(s); replacing them with silence.",
                invalid_samples,
            )
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return audio_array

    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress = gr.Progress(),
                     ):
        """
        Update current model setting

        Parameters
        ----------
        model_size: str
            Size of whisper model
        compute_type: str
            Compute type for transcription.
            see more info : https://opennmt.net/CTranslate2/quantization.html
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        """
        progress(0, desc="Initializing Model..")
        self.current_compute_type = compute_type
        self.current_model_size = model_size
        # PyTorch 2.6 changed torch.load default to weights_only=True. Some Whisper checkpoints
        # include TorchVersion metadata which must be allowlisted for weights-only loading.
        with torch_load_safe_globals():
            self.model = whisper.load_model(
                name=model_size,
                device=self.device,
                download_root=self.model_dir
            )
