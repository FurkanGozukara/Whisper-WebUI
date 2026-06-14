import os
import re
import time
import numpy as np
import inspect
from pathlib import Path
from typing import BinaryIO, Union, Tuple, List, Callable, Optional, Dict
import torch
from faster_whisper.audio import decode_audio
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import gradio as gr
from huggingface_hub import hf_hub_download
import whisper
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
from argparse import Namespace

from modules.utils.paths import (INSANELY_FAST_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, UVR_MODELS_DIR, OUTPUT_DIR)
from modules.whisper.data_classes import *
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.utils.logger import get_logger

logger = get_logger()


class InsanelyFastWhisperInference(BaseTranscriptionPipeline):
    WHISPER_TOKEN_LIMIT_FALLBACK = 448
    PROMPT_TOKEN_RESERVE = 16
    MIN_RETRY_MAX_NEW_TOKENS = 16
    MAX_PROMPT_LENGTH_RETRIES = 5
    DEFAULT_LIVE_CHUNK_SECONDS = 5
    MAX_LIVE_CHUNK_SECONDS = 5
    REQUIRED_MODEL_FILES = (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
    )
    WEIGHT_FILE_PATTERNS = (
        "model.safetensors",
        "pytorch_model.bin",
        "model-*.safetensors",
        "pytorch_model-*.bin",
        "*.safetensors",
    )
    TOKENIZER_FILE_CANDIDATES = (
        "tokenizer.json",
        "vocab.json",
        "tokenizer.model",
    )
    REQUIRED_DOWNLOAD_FILES = (
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    OPTIONAL_DOWNLOAD_FILES = (
        "added_tokens.json",
        "special_tokens_map.json",
        "vocab.json",
    )

    def __init__(self,
                 model_dir: str = INSANELY_FAST_WHISPER_MODELS_DIR,
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
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.available_models = self.get_model_paths()

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
        progress = self.ensure_progress_callable(progress)
        params = WhisperParams.from_list(list(whisper_params))

        if self.should_load_model_for_selection(params.model_size, params.compute_type):
            self.update_model(params.model_size, params.compute_type, progress)

        progress(0, desc="Starting Insanely Fast Whisper transcription..")
        self.emit_status_callback(progress_callback, "Starting Insanely Fast Whisper transcription..")
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(style="yellow1", pulse_style="white"),
                TimeElapsedColumn(),
        ) as rich_progress:
            rich_progress.add_task("[yellow]Transcribing...", total=None)

            generate_kwargs = self.build_generate_kwargs(params)
            pipeline_input = self.prepare_pipeline_input(audio)
            audio_duration = self.audio_duration_seconds(pipeline_input)
            if progress_callback is not None:
                segments_result = self.run_pipeline_in_live_chunks(
                    audio_array=pipeline_input,
                    audio_duration=audio_duration,
                    params=params,
                    generate_kwargs=generate_kwargs,
                    progress=progress,
                    progress_callback=progress_callback,
                    log_console=log_console,
                )
            else:
                segments = self.run_pipeline_with_safe_generation(
                    inputs=pipeline_input,
                    return_timestamps=True,
                    chunk_length_s=self.resolve_transformers_chunk_length(params.chunk_length),
                    batch_size=params.batch_size,
                    generate_kwargs=generate_kwargs,
                )
                segments_result = self.pipeline_output_to_segments(segments, audio_duration)
                self.emit_segments_to_progress(
                    segments_result=segments_result,
                    total_duration=audio_duration,
                    progress=progress,
                    progress_callback=progress_callback,
                    log_console=log_console,
                )

        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    def run_pipeline_in_live_chunks(
        self,
        audio_array: np.ndarray,
        audio_duration: float,
        params: WhisperParams,
        generate_kwargs: Dict,
        progress: Callable,
        progress_callback: Optional[Callable],
        log_console: bool,
    ) -> List[Segment]:
        sampling_rate = self.get_feature_extractor_sampling_rate()
        if audio_array is None or audio_array.size == 0 or sampling_rate <= 0:
            return []

        chunk_seconds = self.resolve_live_chunk_length_seconds(params.chunk_length)
        chunk_samples = max(1, int(chunk_seconds * sampling_rate))
        total_samples = int(audio_array.shape[0])
        total_chunks = max(1, int(np.ceil(total_samples / chunk_samples)))
        segments_result: List[Segment] = []

        for chunk_index, start_sample in enumerate(range(0, total_samples, chunk_samples), start=1):
            end_sample = min(total_samples, start_sample + chunk_samples)
            chunk_audio = np.ascontiguousarray(audio_array[start_sample:end_sample], dtype=np.float32)
            chunk_start = start_sample / sampling_rate
            chunk_end = end_sample / sampling_rate
            chunk_duration = max(0.0, chunk_end - chunk_start)
            raw_chunk_progress = min(chunk_start / audio_duration, 0.99) if audio_duration else 0.0
            status = (
                f"Transcribing Insanely Fast Whisper chunk {chunk_index}/{total_chunks} "
                f"({self.format_timestamp(chunk_start)} -> {self.format_timestamp(chunk_end)}).."
            )
            progress(self.map_transcription_progress(raw_chunk_progress), desc=status)

            output = self.run_pipeline_with_safe_generation(
                inputs=chunk_audio,
                return_timestamps=True,
                chunk_length_s=None,
                batch_size=params.batch_size,
                generate_kwargs=generate_kwargs,
            )
            chunk_segments = self.pipeline_output_to_segments(output, chunk_duration)
            offset_segments = self.offset_segments(chunk_segments, chunk_start, audio_duration)
            segments_result.extend(offset_segments)

            if offset_segments:
                self.emit_segments_to_progress(
                    segments_result=offset_segments,
                    total_duration=audio_duration,
                    progress=progress,
                    progress_callback=progress_callback,
                    log_console=log_console,
                    start_index=len(segments_result) - len(offset_segments) + 1,
                )
            else:
                raw_progress = min(chunk_end / audio_duration, 0.99) if audio_duration else 0.99
                status = f"No speech detected in IFW chunk {chunk_index}/{total_chunks}."
                progress(self.map_transcription_progress(raw_progress), desc=status)
                self.emit_status_callback(progress_callback, status)

        return segments_result

    def emit_segments_to_progress(
        self,
        segments_result: List[Segment],
        total_duration: float,
        progress: Callable,
        progress_callback: Optional[Callable],
        log_console: bool,
        start_index: int = 1,
    ) -> None:
        total_duration = total_duration or (segments_result[-1].end if segments_result else 0.0) or 0.0
        for idx, segment in enumerate(segments_result, start=start_index):
            if not segment.text:
                continue

            raw_progress = min((segment.end or 0.0) / total_duration, 0.99) if total_duration else 0.99
            ui_progress = self.map_transcription_progress(raw_progress)
            progress(ui_progress, desc=f"Transcribing.. [{idx} segments] {segment.text[:50]}...")

            if log_console:
                logger.info(
                    "[%s -> %s] %s",
                    self.format_timestamp(segment.start),
                    self.format_timestamp(segment.end),
                    segment.text,
                )

            self.emit_progress_callback(progress_callback, raw_progress, segment)

    @classmethod
    def resolve_live_chunk_length_seconds(cls, chunk_length: Optional[int]) -> int:
        shared_default = WhisperParams.model_fields["chunk_length"].default
        chunk_length = cls.coerce_positive_int(chunk_length, default=None)
        if chunk_length is None or chunk_length == shared_default:
            chunk_length = cls.DEFAULT_LIVE_CHUNK_SECONDS
        return max(1, min(int(chunk_length), cls.MAX_LIVE_CHUNK_SECONDS))

    @staticmethod
    def offset_segments(segments: List[Segment], offset_seconds: float, audio_duration: float) -> List[Segment]:
        offset_segments = []
        for segment in segments:
            text = segment.text
            if not text:
                continue
            start = offset_seconds + float(segment.start or 0.0)
            end = offset_seconds + float(segment.end or 0.0)
            if audio_duration:
                start = min(start, audio_duration)
                end = min(end, audio_duration)
            if end <= start:
                end = min(audio_duration or (start + 0.01), start + 0.01)
            offset_segments.append(Segment(text=text, start=start, end=end))
        return offset_segments

    def prepare_pipeline_input(self, audio: Union[str, BinaryIO, np.ndarray, torch.Tensor]) -> np.ndarray:
        sampling_rate = self.get_feature_extractor_sampling_rate()
        if isinstance(audio, torch.Tensor):
            audio_array = audio.detach().cpu().float().numpy()
        elif isinstance(audio, np.ndarray):
            audio_array = np.asarray(audio, dtype=np.float32)
        else:
            audio_array = decode_audio(audio, sampling_rate=sampling_rate)

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

    def get_feature_extractor_sampling_rate(self) -> int:
        feature_extractor = getattr(self.model, "feature_extractor", None)
        sampling_rate = self.coerce_positive_int(getattr(feature_extractor, "sampling_rate", None), default=16000)
        return int(sampling_rate or 16000)

    def build_generate_kwargs(self, params: WhisperParams) -> Dict:
        kwargs = {
            "no_speech_threshold": params.no_speech_threshold,
            "temperature": params.temperature,
            "compression_ratio_threshold": params.compression_ratio_threshold,
            "logprob_threshold": params.log_prob_threshold,
            "condition_on_prev_tokens": bool(params.condition_on_previous_text),
        }

        target_token_limit = self.get_target_token_limit()
        max_new_tokens = self.resolve_max_new_tokens(
            requested_max_new_tokens=params.max_new_tokens,
            target_token_limit=target_token_limit,
        )
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens

        if not self.current_model_size.endswith(".en"):
            kwargs["language"] = params.lang
            kwargs["task"] = "translate" if params.is_translate else "transcribe"

        return kwargs

    def run_pipeline_with_safe_generation(
        self,
        inputs,
        return_timestamps: bool,
        chunk_length_s: Optional[int],
        batch_size: Optional[int],
        generate_kwargs: Dict,
    ):
        self.disable_transformers_torchcodec_probe()
        call_kwargs = {
            "inputs": inputs,
            "return_timestamps": return_timestamps,
            "batch_size": max(1, int(batch_size or 1)),
            "generate_kwargs": dict(generate_kwargs),
        }
        if chunk_length_s is not None:
            chunk_length_s = self.coerce_positive_int(chunk_length_s, default=None)
            if chunk_length_s:
                call_kwargs["chunk_length_s"] = chunk_length_s

        retries = 0
        while True:
            try:
                return self.model(**call_kwargs)
            except ValueError as exc:
                retry_generate_kwargs = self.build_prompt_length_retry_kwargs(
                    generate_kwargs=call_kwargs["generate_kwargs"],
                    error=exc,
                )
                if retry_generate_kwargs is None or retries >= self.MAX_PROMPT_LENGTH_RETRIES:
                    raise

                if retry_generate_kwargs == call_kwargs["generate_kwargs"]:
                    current_max_new_tokens = self.coerce_positive_int(
                        retry_generate_kwargs.get("max_new_tokens"),
                        default=None,
                    )
                    if not current_max_new_tokens or current_max_new_tokens <= 1:
                        raise
                    retry_generate_kwargs = dict(retry_generate_kwargs)
                    retry_generate_kwargs["max_new_tokens"] = current_max_new_tokens - 1

                retries += 1
                logger.warning(
                    "Whisper generation prompt was too long for the requested max_new_tokens. "
                    "Retrying with max_new_tokens=%s.",
                    retry_generate_kwargs.get("max_new_tokens"),
                )
                call_kwargs["generate_kwargs"] = retry_generate_kwargs

    def build_prompt_length_retry_kwargs(self, generate_kwargs: Dict, error: Exception) -> Optional[Dict]:
        parsed_error = self.parse_prompt_length_error(str(error))
        if parsed_error is None:
            return None

        prompt_length, _reported_max_new_tokens, target_token_limit = parsed_error
        safe_max_new_tokens = max(1, target_token_limit - prompt_length - 1)
        retry_kwargs = dict(generate_kwargs)
        current_max_new_tokens = self.coerce_positive_int(
            retry_kwargs.get("max_new_tokens"),
            default=self.get_configured_max_new_tokens(),
        )

        if (
            safe_max_new_tokens < self.MIN_RETRY_MAX_NEW_TOKENS
            and retry_kwargs.get("condition_on_prev_tokens")
        ):
            retry_kwargs["condition_on_prev_tokens"] = False
            retry_kwargs["max_new_tokens"] = self.resolve_max_new_tokens(
                requested_max_new_tokens=current_max_new_tokens,
                target_token_limit=target_token_limit,
            )
            return retry_kwargs

        retry_kwargs["max_new_tokens"] = min(current_max_new_tokens, safe_max_new_tokens)
        return retry_kwargs

    @classmethod
    def parse_prompt_length_error(cls, message: str) -> Optional[Tuple[int, int, int]]:
        prompt_patterns = (
            r"length of the prompt is\s*(\d+)",
            r"length of `decoder_input_ids`.*?is\s*(\d+)",
        )
        prompt_length = cls.first_regex_int(prompt_patterns, message)
        max_new_tokens = cls.first_regex_int((r"`max_new_tokens`(?:\s+is)?\s*(\d+)",), message)
        target_token_limit = cls.first_regex_int(
            (
                r"`?(?:max_length|max_target_positions)`?\s+of the Whisper model:\s*(\d+)",
                r"`max_length`(?:\s+is set to)?\s*(\d+)",
            ),
            message,
        )

        if prompt_length is None or max_new_tokens is None:
            return None

        return (
            prompt_length,
            max_new_tokens,
            target_token_limit or cls.WHISPER_TOKEN_LIMIT_FALLBACK,
        )

    @staticmethod
    def first_regex_int(patterns: Tuple[str, ...], text: str) -> Optional[int]:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return int(match.group(1))
        return None

    def get_target_token_limit(self) -> int:
        candidates = [
            getattr(getattr(self.model, "model", None), "config", None),
            getattr(self.model, "generation_config", None),
            getattr(getattr(self.model, "model", None), "generation_config", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            for attribute in ("max_target_positions", "max_length"):
                value = getattr(candidate, attribute, None)
                value = self.coerce_positive_int(value, default=None)
                if value:
                    return value
        return self.WHISPER_TOKEN_LIMIT_FALLBACK

    def get_configured_max_new_tokens(self) -> int:
        candidates = [
            getattr(self.model, "generation_config", None),
            getattr(getattr(self.model, "model", None), "generation_config", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            value = self.coerce_positive_int(getattr(candidate, "max_new_tokens", None), default=None)
            if value:
                return value
        return 256

    def resolve_max_new_tokens(
        self,
        requested_max_new_tokens: Optional[int],
        target_token_limit: Optional[int] = None,
    ) -> int:
        target_token_limit = self.coerce_positive_int(
            target_token_limit,
            default=self.WHISPER_TOKEN_LIMIT_FALLBACK,
        )
        requested_max_new_tokens = self.coerce_positive_int(
            requested_max_new_tokens,
            default=None,
        )
        if requested_max_new_tokens is None:
            return None
        max_without_prompt_reserve = max(1, target_token_limit - self.PROMPT_TOKEN_RESERVE)
        return min(requested_max_new_tokens, max_without_prompt_reserve)

    def pipeline_output_to_segments(self, output: Dict, audio_duration: float = 0.0) -> List[Segment]:
        chunks = output.get("chunks") if isinstance(output, dict) else None
        segments = []
        if isinstance(chunks, list):
            for item in chunks:
                if not isinstance(item, dict):
                    continue
                text = item.get("text") or ""
                start, end = self.normalize_timestamp_pair(item.get("timestamp"), audio_duration)
                if text.strip():
                    segments.append(Segment(text=text, start=start, end=end))

        if segments:
            return segments

        text = output.get("text") if isinstance(output, dict) else None
        if isinstance(text, str) and text.strip():
            return [Segment(text=text, start=0.0, end=audio_duration)]

        return []

    @staticmethod
    def normalize_timestamp_pair(timestamp, audio_duration: float = 0.0) -> Tuple[float, float]:
        start = 0.0
        end = audio_duration or 0.0
        if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
            start = 0.0 if timestamp[0] is None else float(timestamp[0])
            end = start if timestamp[1] is None else float(timestamp[1])
        return start, end

    def audio_duration_seconds(self, audio_array: np.ndarray) -> float:
        sampling_rate = self.get_feature_extractor_sampling_rate()
        if sampling_rate <= 0 or audio_array is None:
            return 0.0
        return float(len(audio_array) / sampling_rate)

    @classmethod
    def map_transcription_progress(cls, raw_progress: float) -> float:
        try:
            raw_progress = float(raw_progress)
        except (TypeError, ValueError):
            raw_progress = 0.0
        return min(max(raw_progress, 0.0), 0.99)

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
    def emit_status_callback(
        progress_callback: Optional[Callable],
        status: str,
    ):
        if progress_callback is None:
            return

        try:
            progress_callback(None, None, status)
        except TypeError:
            progress_callback(None)

    @staticmethod
    def resolve_transformers_chunk_length(chunk_length: Optional[int]) -> Optional[int]:
        try:
            chunk_length = int(chunk_length)
        except (TypeError, ValueError):
            return None

        shared_default = WhisperParams.model_fields["chunk_length"].default
        if chunk_length <= 0 or chunk_length == shared_default:
            return None
        return chunk_length

    @staticmethod
    def coerce_positive_int(value, default: Optional[int]) -> Optional[int]:
        try:
            value = int(value)
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    @staticmethod
    def disable_transformers_torchcodec_probe() -> None:
        try:
            import transformers.pipelines.automatic_speech_recognition as asr_pipeline

            asr_pipeline.is_torchcodec_available = lambda: False
        except Exception:
            pass

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
        progress = self.ensure_progress_callable(progress)
        progress(0, desc="Initializing Model..")
        model_path = self.resolve_model_target(model_size=model_size, progress=progress)

        self.current_compute_type = compute_type
        self.current_model_size = model_size
        torch_dtype = self.torch_dtype_for_compute_type(compute_type)
        self.log_model_load_start(
            implementation=self.implementation_label(WhisperImpl.INSANELY_FAST_WHISPER.value),
            selected_model=model_size,
            resolved_model=model_path,
            compute_type=compute_type,
        )
        self.model = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            **self.pipeline_dtype_kwargs(torch_dtype),
            device=self.device,
            model_kwargs=self.model_kwargs_for_torch_dtype(torch_dtype),
        )
        self.disable_bpe_tokenizer_cleanup_warning(self.model)
        self.log_model_load_complete(
            implementation=self.implementation_label(WhisperImpl.INSANELY_FAST_WHISPER.value),
            selected_model=model_size,
            active_model=self.current_model_size,
            compute_type=self.current_compute_type,
        )

    def resolve_model_target(self, model_size: str, progress: gr.Progress) -> str:
        local_model_path = os.path.join(self.model_dir, model_size)
        if self.has_transformers_model_files(local_model_path):
            logger.info('Using existing Transformers Whisper model "%s" from "%s".', model_size, local_model_path)
            return local_model_path

        if os.path.isdir(local_model_path) and os.listdir(local_model_path):
            logger.warning(
                'Insanely Fast Whisper model folder "%s" exists but is incomplete for Transformers. '
                "Trying Hugging Face cache before downloading missing files.",
                local_model_path,
            )

        cached_model_path = self.find_cached_transformers_model(model_size=model_size)
        if cached_model_path:
            logger.info(
                'Using cached Transformers Whisper model "%s" from "%s".',
                model_size,
                cached_model_path,
            )
            return cached_model_path

        return self.download_model(
            model_size=model_size,
            download_root=local_model_path,
            progress=progress,
        )

    @staticmethod
    def torch_dtype_for_compute_type(compute_type: str):
        normalized = str(compute_type or "").strip().lower()
        if normalized == "float16":
            return torch.float16
        if normalized == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def get_compute_type(self):
        if "float16" in self.available_compute_types:
            return "float16"
        if "bfloat16" in self.available_compute_types:
            return "bfloat16"
        if "float32" in self.available_compute_types:
            return "float32"
        return self.available_compute_types[0]

    @staticmethod
    def model_kwargs_for_torch_dtype(torch_dtype) -> Dict:
        supports_flash_dtype = torch_dtype in (torch.float16, torch.bfloat16)
        if supports_flash_dtype and is_flash_attn_2_available():
            return {"attn_implementation": "flash_attention_2"}
        return {"attn_implementation": "sdpa"}

    @staticmethod
    def pipeline_dtype_kwargs(torch_dtype) -> Dict:
        if "dtype" in inspect.signature(pipeline).parameters:
            return {"dtype": torch_dtype}
        return {"torch_dtype": torch_dtype}

    @staticmethod
    def disable_bpe_tokenizer_cleanup_warning(model_pipeline) -> None:
        tokenizer = getattr(model_pipeline, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "clean_up_tokenization_spaces"):
            tokenizer.clean_up_tokenization_spaces = False

    def get_model_paths(self):
        """
        Get available models from models path including fine-tuned model.

        Returns
        ----------
        Name set of models
        """
        openai_models = whisper.available_models()
        distil_models = ["distil-large-v2", "distil-large-v3", "distil-medium.en", "distil-small.en"]
        default_models = openai_models + distil_models

        existing_models = os.listdir(self.model_dir)
        wrong_dirs = [".locks", "insanely_fast_whisper_models_will_be_saved_here"]

        available_models = default_models + existing_models
        available_models = [model for model in available_models if model not in wrong_dirs]
        available_models = sorted(set(available_models), key=available_models.index)

        return available_models

    @classmethod
    def download_model(
        cls,
        model_size: str,
        download_root: str,
        progress: gr.Progress
    ) -> str:
        progress = cls.ensure_progress_callable(progress)
        progress(0, "Preparing model..")
        repo_id = cls.repo_id_for_model_size(model_size)
        logger.info(
            'Downloading Transformers Whisper model "%s" (%s) to "%s".',
            model_size,
            repo_id,
            download_root,
        )

        os.makedirs(download_root, exist_ok=True)
        for item in cls.REQUIRED_DOWNLOAD_FILES + cls.OPTIONAL_DOWNLOAD_FILES:
            try:
                hf_hub_download(repo_id=repo_id, filename=item, local_dir=download_root)
            except Exception as exc:
                if item in cls.OPTIONAL_DOWNLOAD_FILES and cls.is_hf_entry_not_found(exc):
                    logger.debug('Optional Hugging Face file "%s" is not present in %s.', item, repo_id)
                    continue
                raise

        if not cls.has_transformers_model_files(download_root):
            raise RuntimeError(
                f'Downloaded model "{model_size}" is incomplete at "{download_root}". '
                "Remove the folder and try again, or place a complete Transformers-format Whisper model there."
            )
        return download_root

    @classmethod
    def has_transformers_model_files(cls, model_path: Union[str, os.PathLike]) -> bool:
        path = Path(model_path)
        if not path.is_dir():
            return False

        for filename in cls.REQUIRED_MODEL_FILES:
            if not cls.is_nonempty_file(path / filename):
                return False

        if not any(cls.is_nonempty_file(path / filename) for filename in cls.TOKENIZER_FILE_CANDIDATES):
            return False

        return any(
            cls.is_nonempty_file(candidate)
            for pattern in cls.WEIGHT_FILE_PATTERNS
            for candidate in path.glob(pattern)
        )

    @staticmethod
    def ensure_progress_callable(progress):
        if callable(progress):
            return progress

        def noop_progress(*_args, **_kwargs):
            return None

        return noop_progress

    @staticmethod
    def is_nonempty_file(path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    @classmethod
    def find_cached_transformers_model(cls, model_size: str) -> Optional[str]:
        repo_cache_name = f"models--{cls.repo_id_for_model_size(model_size).replace('/', '--')}"
        for cache_dir in cls.candidate_hf_cache_dirs():
            repo_dir = Path(cache_dir) / repo_cache_name
            snapshots_dir = repo_dir / "snapshots"
            if not snapshots_dir.is_dir():
                continue
            snapshots = sorted(
                (snapshot for snapshot in snapshots_dir.iterdir() if snapshot.is_dir()),
                key=lambda snapshot: cls.safe_mtime(snapshot),
                reverse=True,
            )
            for snapshot in snapshots:
                if cls.has_transformers_model_files(snapshot):
                    return str(snapshot)
        return None

    @classmethod
    def candidate_hf_cache_dirs(cls) -> List[str]:
        candidates = [
            os.environ.get("HF_HUB_CACHE"),
            os.environ.get("TRANSFORMERS_CACHE"),
        ]
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            candidates.append(os.path.join(hf_home, "hub"))

        try:
            from huggingface_hub import constants as hf_constants

            candidates.extend([
                getattr(hf_constants, "HF_HUB_CACHE", None),
                os.path.join(getattr(hf_constants, "HF_HOME", ""), "hub"),
            ])
        except Exception:
            pass

        user_profile = os.environ.get("USERPROFILE")
        if user_profile:
            candidates.append(os.path.join(user_profile, ".cache", "huggingface", "hub"))

        local_models_dir = Path(INSANELY_FAST_WHISPER_MODELS_DIR).resolve().parents[1] / "hub"
        candidates.append(str(local_models_dir))

        unique_candidates = []
        seen = set()
        for candidate in candidates:
            if not candidate:
                continue
            normalized = os.path.normcase(os.path.abspath(os.path.expanduser(str(candidate))))
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_candidates.append(normalized)
        return unique_candidates

    @staticmethod
    def safe_mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    @staticmethod
    def repo_id_for_model_size(model_size: str) -> str:
        return f"distil-whisper/{model_size}" if model_size.startswith("distil") else f"openai/whisper-{model_size}"

    @staticmethod
    def is_hf_entry_not_found(exc: Exception) -> bool:
        return exc.__class__.__name__ in {
            "EntryNotFoundError",
            "RemoteEntryNotFoundError",
            "LocalEntryNotFoundError",
        }
