import importlib
import json
import os
import re
import sys
import time
from typing import BinaryIO, Callable, List, Optional, Tuple, Union

import gradio as gr
import numpy as np
import torch
from faster_whisper.audio import decode_audio

from modules.utils.constants import GRADIO_NONE_STR
from modules.utils.logger import get_logger
from modules.utils.paths import CANARY_QWEN_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR, UVR_MODELS_DIR
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.whisper.data_classes import Segment, WhisperParams


logger = get_logger()


class CanaryQwenInference(BaseTranscriptionPipeline):
    DEFAULT_MODEL_ID = "nvidia/canary-qwen-2.5b"
    SAMPLE_RATE = 16000
    MAX_CHUNK_SECONDS = 40.0
    DEFAULT_MAX_NEW_TOKENS = 256
    TRANSCRIPTION_PROGRESS_START = 0.15
    TRANSCRIPTION_PROGRESS_END = 0.98

    def __init__(
        self,
        model_dir: str = CANARY_QWEN_MODELS_DIR,
        diarization_model_dir: str = DIARIZATION_MODELS_DIR,
        uvr_model_dir: str = UVR_MODELS_DIR,
        output_dir: str = OUTPUT_DIR,
    ):
        super().__init__(
            model_dir=model_dir,
            output_dir=output_dir,
            diarization_model_dir=diarization_model_dir,
            uvr_model_dir=uvr_model_dir,
        )
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.available_models = self.get_model_paths()
        self.available_langs = ["english"]
        self.device = self.get_device()
        self.available_compute_types = self.get_available_compute_type()
        self.current_compute_type = self.get_compute_type()

    @staticmethod
    def supports_word_timestamps() -> bool:
        return False

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        progress: gr.Progress = gr.Progress(),
        progress_callback: Optional[Callable] = None,
        *whisper_params,
        log_console: bool = True,
        log_model_banner: bool = True,
    ) -> Tuple[List[Segment], float]:
        start_time = time.time()
        params = WhisperParams.from_list(list(whisper_params))
        self.validate_supported_params(params)

        if (
            params.model_size != self.current_model_size
            or self.model is None
            or self.current_compute_type != params.compute_type
        ):
            self.update_model(params.model_size, params.compute_type, progress)

        if log_model_banner:
            logger.info(
                "Using NVIDIA Canary-Qwen through NeMo SALM. "
                "The model is English ASR only and returns chunk-level timestamps."
            )

        progress(0.05, desc="Loading audio..")
        audio_array = self.prepare_audio_array(audio)
        chunks = self.build_audio_chunks(audio_array, params.chunk_length)
        if not chunks:
            return [], time.time() - start_time

        batch_size = max(1, int(params.batch_size or 1))
        generation_kwargs = self.build_generation_kwargs(params)
        segments: List[Segment] = []
        total_duration = len(audio_array) / float(self.SAMPLE_RATE)
        previous_text = ""

        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start : batch_start + batch_size]
            audios, audio_lens = self.collate_audio_batch(batch_chunks)
            prompts = [
                self.build_asr_prompt(params=params, previous_text=previous_text)
                for _ in batch_chunks
            ]

            progress_value = self.map_transcription_progress(batch_start / float(len(chunks)))
            progress(progress_value, desc=f"Transcribing Canary-Qwen chunks {batch_start + 1}-{batch_start + len(batch_chunks)}..")

            with torch.inference_mode():
                output_ids = self.model.generate(
                    prompts=prompts,
                    audios=audios.to(self.device, non_blocking=True),
                    audio_lens=audio_lens.to(self.device, non_blocking=True),
                    **generation_kwargs,
                )

            for offset, (chunk, token_ids) in enumerate(zip(batch_chunks, output_ids)):
                text = self.decode_output(token_ids)
                segment_id = len(segments) + 1
                segment = Segment(
                    id=segment_id,
                    seek=int(round(chunk["start_seconds"] * 100.0)),
                    start=chunk["start_seconds"],
                    end=chunk["end_seconds"],
                    text=text,
                    temperature=params.temperature,
                    words=None,
                )
                if text:
                    segments.append(segment)
                    previous_text = self.update_previous_text(previous_text, text)

                    if log_console:
                        logger.info(
                            "[%s -> %s] %s",
                            self.format_timestamp(segment.start),
                            self.format_timestamp(segment.end),
                            segment.text,
                        )

                    raw_progress = min((segment.end or 0.0) / total_duration, 0.99) if total_duration else 0.99
                    progress(
                        self.map_transcription_progress(raw_progress),
                        desc=f"Transcribing.. [{segment_id} segments] {segment.text[:50]}...",
                    )
                    self.emit_progress_callback(progress_callback, raw_progress, segment)
                else:
                    raw_progress = (batch_start + offset + 1) / float(len(chunks))
                    self.emit_progress_callback(progress_callback, raw_progress, None)

        elapsed_time = time.time() - start_time
        return segments, elapsed_time

    def update_model(
        self,
        model_size: str,
        compute_type: str,
        progress: gr.Progress = gr.Progress(),
    ):
        progress(0.02, desc="Initializing Canary-Qwen model..")
        self.configure_hf_cache()
        salm_cls = self.import_salm()

        dtype = self.torch_dtype_for_compute_type(compute_type)
        model_target = self.resolve_model_target(model_size)
        logger.info("Loading Canary-Qwen model '%s' into %s with %s.", model_target, self.device, compute_type)

        model = salm_cls.from_pretrained(
            model_target,
            cache_dir=self.get_hf_hub_cache_dir(),
            torch_dtype=dtype,
            token=os.environ.get("HF_TOKEN") or None,
        )
        model.eval()
        model.to(self.device)
        if dtype != torch.float32:
            model.to(dtype=dtype)

        self.model = model
        self.current_model_size = model_size
        self.current_compute_type = compute_type
        progress(0.1, desc="Canary-Qwen model loaded.")

    def validate_supported_params(self, params: WhisperParams) -> None:
        if params.is_translate:
            raise ValueError("Canary-Qwen does not support Whisper-style speech translation. Use English ASR output and translate it in the translation tab.")

        normalized_lang = params.lang.lower() if isinstance(params.lang, str) else params.lang
        if normalized_lang not in (None, "en", "english"):
            raise ValueError("Canary-Qwen is English-only. Set Language to English or Automatic Detection.")

    def build_asr_prompt(self, params: WhisperParams, previous_text: str = "") -> List[dict]:
        del previous_text

        prompt = f"Transcribe the following: {self.model.audio_locator_tag}"
        return [{"role": "user", "content": prompt}]

    def build_generation_kwargs(self, params: WhisperParams) -> dict:
        kwargs = {
            "max_new_tokens": int(params.max_new_tokens or self.DEFAULT_MAX_NEW_TOKENS),
            "num_beams": max(1, int(params.beam_size or 1)),
        }

        if params.temperature and params.temperature > 0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = float(params.temperature)
        else:
            kwargs["do_sample"] = False

        if params.length_penalty and params.length_penalty != 1.0:
            kwargs["length_penalty"] = float(params.length_penalty)
        if params.repetition_penalty and params.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = float(params.repetition_penalty)
        if params.no_repeat_ngram_size and params.no_repeat_ngram_size > 0:
            kwargs["no_repeat_ngram_size"] = int(params.no_repeat_ngram_size)

        kwargs["enable_thinking"] = bool(params.canary_enable_thinking)
        kwargs.update(self.parse_canary_generation_kwargs(params.canary_generation_kwargs))
        return kwargs

    @staticmethod
    def parse_canary_generation_kwargs(raw_kwargs) -> dict:
        if raw_kwargs in (None, "", GRADIO_NONE_STR):
            return {}

        if isinstance(raw_kwargs, dict):
            parsed = dict(raw_kwargs)
        elif isinstance(raw_kwargs, str):
            stripped = raw_kwargs.strip()
            if stripped in ("", "None", "null", GRADIO_NONE_STR):
                return {}
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid Canary Generation Kwargs JSON: {exc}") from exc
        else:
            raise ValueError("Canary Generation Kwargs must be a JSON object.")

        if not isinstance(parsed, dict):
            raise ValueError("Canary Generation Kwargs must be a JSON object.")

        reserved_keys = {"prompts", "audios", "audio_lens"}
        conflicting_keys = sorted(reserved_keys.intersection(parsed))
        if conflicting_keys:
            raise ValueError(
                "Canary Generation Kwargs cannot override internal inputs: "
                + ", ".join(conflicting_keys)
            )

        generation_config = parsed.get("generation_config")
        if isinstance(generation_config, dict):
            from transformers import GenerationConfig

            parsed["generation_config"] = GenerationConfig(**generation_config)

        return parsed

    def prepare_audio_array(self, audio: Union[str, BinaryIO, np.ndarray]) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            audio_array = np.asarray(audio, dtype=np.float32)
        else:
            audio_array = decode_audio(audio, sampling_rate=self.SAMPLE_RATE)

        if audio_array.ndim > 1:
            channel_axis = 0 if audio_array.shape[0] <= audio_array.shape[-1] else -1
            audio_array = audio_array.mean(axis=channel_axis)

        return np.ascontiguousarray(audio_array.squeeze(), dtype=np.float32)

    def build_audio_chunks(self, audio: np.ndarray, chunk_length: Optional[int]) -> List[dict]:
        total_samples = int(audio.shape[-1]) if audio.size else 0
        if total_samples <= 0:
            return []

        requested_chunk_seconds = float(chunk_length or self.MAX_CHUNK_SECONDS)
        if requested_chunk_seconds <= 0:
            requested_chunk_seconds = self.MAX_CHUNK_SECONDS
        chunk_seconds = min(requested_chunk_seconds, self.MAX_CHUNK_SECONDS)
        if requested_chunk_seconds > self.MAX_CHUNK_SECONDS:
            logger.info(
                "Canary-Qwen was trained with audio windows up to %.0fs; capping requested chunk length %.1fs to %.0fs.",
                self.MAX_CHUNK_SECONDS,
                requested_chunk_seconds,
                self.MAX_CHUNK_SECONDS,
            )

        chunk_samples = max(1, int(round(chunk_seconds * self.SAMPLE_RATE)))
        chunks = []
        for start_sample in range(0, total_samples, chunk_samples):
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunks.append(
                {
                    "audio": audio[start_sample:end_sample],
                    "start_seconds": start_sample / float(self.SAMPLE_RATE),
                    "end_seconds": end_sample / float(self.SAMPLE_RATE),
                }
            )
        return chunks

    @staticmethod
    def collate_audio_batch(chunks: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = [int(chunk["audio"].shape[-1]) for chunk in chunks]
        max_length = max(lengths) if lengths else 0
        batch = np.zeros((len(chunks), max_length), dtype=np.float32)
        for index, chunk in enumerate(chunks):
            chunk_audio = np.asarray(chunk["audio"], dtype=np.float32)
            batch[index, : chunk_audio.shape[-1]] = chunk_audio

        return torch.from_numpy(batch), torch.as_tensor(lengths, dtype=torch.long)

    def decode_output(self, token_ids) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu()
        text = self.model.tokenizer.ids_to_text(token_ids)
        return self.clean_generated_text(text)

    @staticmethod
    def clean_generated_text(text: str) -> str:
        if text is None:
            return ""
        text = str(text)
        text = re.sub(r"<\|im_start\|>\s*assistant", "", text)
        text = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def update_previous_text(previous_text: str, text: str, max_chars: int = 1000) -> str:
        combined = f"{previous_text} {text}".strip()
        if len(combined) <= max_chars:
            return combined
        return combined[-max_chars:]

    def resolve_model_target(self, model_size: str) -> str:
        model_size = model_size or self.DEFAULT_MODEL_ID
        if os.path.isabs(model_size) and os.path.exists(model_size):
            return model_size

        candidate = os.path.join(self.model_dir, model_size)
        if os.path.isdir(candidate):
            return candidate

        safe_name = model_size.replace("/", "--")
        candidate = os.path.join(self.model_dir, safe_name)
        if os.path.isdir(candidate):
            return candidate

        return model_size

    def get_model_paths(self) -> List[str]:
        ignored = {
            ".locks",
            "hub",
            "xet",
            "transformers",
            "canary_qwen_models_will_be_saved_here",
        }
        models = [self.DEFAULT_MODEL_ID]
        if os.path.isdir(self.model_dir):
            for item in os.listdir(self.model_dir):
                if item in ignored:
                    continue
                if os.path.isdir(os.path.join(self.model_dir, item)):
                    models.append(item)
        return sorted(dict.fromkeys(models), key=models.index)

    def configure_hf_cache(self) -> None:
        hub_cache = self.get_hf_hub_cache_dir()
        transformers_cache = os.path.join(self.model_dir, "transformers")
        os.makedirs(hub_cache, exist_ok=True)
        os.makedirs(transformers_cache, exist_ok=True)

        os.environ["HF_HOME"] = self.model_dir
        os.environ["HF_HUB_CACHE"] = hub_cache
        os.environ["HUGGINGFACE_HUB_CACHE"] = hub_cache
        os.environ["TRANSFORMERS_CACHE"] = transformers_cache

        try:
            import huggingface_hub.constants as hf_constants

            hf_constants.HF_HOME = self.model_dir
            hf_constants.HF_HUB_CACHE = hub_cache
            if hasattr(hf_constants, "HUGGINGFACE_HUB_CACHE"):
                hf_constants.HUGGINGFACE_HUB_CACHE = hub_cache
        except Exception:
            pass

        try:
            import transformers.utils.hub as transformers_hub

            if hasattr(transformers_hub, "TRANSFORMERS_CACHE"):
                transformers_hub.TRANSFORMERS_CACHE = transformers_cache
        except Exception:
            pass

    def get_hf_hub_cache_dir(self) -> str:
        return os.path.join(self.model_dir, "hub")

    @classmethod
    def import_salm(cls):
        cls.patch_nemo_import_compat()
        from nemo.collections.speechlm2.models import SALM

        return SALM

    @staticmethod
    def patch_nemo_import_compat() -> None:
        try:
            import overrides

            overrides_module = importlib.import_module("overrides.overrides")

            def relaxed_override(method=None, *args, **kwargs):
                del args, kwargs

                def decorate(func):
                    try:
                        setattr(func, "__override__", True)
                    except Exception:
                        pass
                    return func

                if method is None:
                    return decorate
                return decorate(method)

            overrides.override = relaxed_override
            overrides_module.override = relaxed_override
        except Exception:
            pass

        try:
            import webdataset

            sys.modules.setdefault("nemo.utils.webdataset", webdataset)
        except Exception:
            pass

        try:
            from torch.distributed import fsdp

            if not hasattr(fsdp, "fully_shard"):
                fsdp.fully_shard = lambda *a, **k: a[0] if len(a) == 1 and callable(a[0]) else (lambda f: f)
        except Exception:
            pass

    @staticmethod
    def torch_dtype_for_compute_type(compute_type: str):
        compute_type = (compute_type or "float32").lower()
        if compute_type == "bfloat16" and torch.cuda.is_available():
            return torch.bfloat16
        if compute_type == "float16" and torch.cuda.is_available():
            return torch.float16
        return torch.float32

    def get_compute_type(self):
        if "bfloat16" in self.available_compute_types:
            return "bfloat16"
        if "float16" in self.available_compute_types:
            return "float16"
        return "float32"

    def get_available_compute_type(self):
        if torch.cuda.is_available():
            compute_types = ["float16", "float32"]
            if torch.cuda.is_bf16_supported():
                compute_types.insert(0, "bfloat16")
            return compute_types
        return ["float32"]

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @classmethod
    def map_transcription_progress(cls, raw_progress: float) -> float:
        bounded = min(max(raw_progress, 0.0), 0.99)
        span = cls.TRANSCRIPTION_PROGRESS_END - cls.TRANSCRIPTION_PROGRESS_START
        return cls.TRANSCRIPTION_PROGRESS_START + (bounded * span)

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
