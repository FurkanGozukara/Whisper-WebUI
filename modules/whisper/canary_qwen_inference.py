import gc
import importlib
import itertools
import json
import os
import re
import sys
import time
import zlib
from contextlib import contextmanager
from typing import BinaryIO, Callable, Iterator, List, Optional, Tuple, Union

import gradio as gr
import numpy as np
import torch

from modules.utils.audio_manager import decode_audio
from modules.utils.constants import GRADIO_NONE_STR
from modules.utils.logger import get_logger
from modules.utils.paths import CANARY_QWEN_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR, UVR_MODELS_DIR
from modules.whisper.convrot import triton_status
from modules.whisper.convrot.registry import (CANARY_CONVROT_FALLBACK_MODELS, HOSTED_CANARY_CONVROT_MODELS,
                                             convrot_runtime_supported, download_convrot_model,
                                             is_canary_convrot_model_dir)
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.whisper.data_classes import Segment, WhisperImpl, WhisperParams


logger = get_logger()


class CanaryQwenInference(BaseTranscriptionPipeline):
    DEFAULT_MODEL_ID = "nvidia/canary-qwen-2.5b"
    SAMPLE_RATE = 16000
    MAX_CHUNK_SECONDS = 40.0
    # Automatic chunk length (0): a recording up to AUTO_SINGLE_CHUNK_SECONDS is transcribed in one piece, as in
    # the model's training (windows up to 40 s) and its published results; cutting a 20 s sentence into 10 s
    # pieces lost words and most of the punctuation at the cut. Longer recordings are cut at the clearest pause
    # between AUTO_LONG_MIN_CHUNK_SECONDS and AUTO_LONG_MAX_CHUNK_SECONDS.
    AUTO_SINGLE_CHUNK_SECONDS = 40.0
    AUTO_LONG_MIN_CHUNK_SECONDS = 15.0
    AUTO_LONG_MAX_CHUNK_SECONDS = 30.0
    MIN_CHUNK_SECONDS = 0.1
    MIN_CHUNK_SAMPLES = int(SAMPLE_RATE * MIN_CHUNK_SECONDS)
    # A chunk ends at a pause instead of exactly at the chunk length, which cut through words that were then lost
    # or garbled on both sides of the cut. "vad": lowest Silero speech probability (robust to background noise and
    # music); "energy": quietest moment. A chunk of a set length (chunk_length > 0) searches its last third.
    CUT_STRATEGY = "vad"
    VAD_FRAME_SAMPLES = 512
    VAD_SMOOTHING_FRAMES = 5
    PAUSE_FRAME_SECONDS = 0.02
    PAUSE_SMOOTHING_FRAMES = 5
    _vad_model = None
    DEFAULT_MAX_NEW_TOKENS = 256
    MAX_SAFE_MAX_NEW_TOKENS = 512
    MAX_SAFE_NUM_BEAMS = 8
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
        # set when a batch ran out of GPU memory; later batches use at most this size until a model loads
        self.oom_batch_limit: Optional[int] = None

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
            self.should_load_model_for_selection(params.model_size, params.compute_type)
        ):
            self.update_model(params.model_size, params.compute_type, progress, progress_callback=progress_callback)

        if log_model_banner:
            if self.is_convrot_engine():
                logger.info(
                    "Using NVIDIA Canary-Qwen INT8 ConvRot (INT8 weights, Triton kernels, CUDA graphs). "
                    "The model is English ASR only and returns chunk-level timestamps."
                )
            else:
                logger.info(
                    "Using NVIDIA Canary-Qwen through NeMo SALM. "
                    "The model is English ASR only and returns chunk-level timestamps."
                )

        self.emit_status_callback(progress_callback, "Preparing audio for Canary-Qwen transcription..")
        progress(0.05, desc="Loading audio..")
        audio_array = self.prepare_audio_array(audio)
        batch_size = max(1, int(params.batch_size or 1))
        generation_kwargs = self.build_generation_kwargs(params)
        segments: List[Segment] = []
        total_samples = int(audio_array.shape[-1])
        total_duration = total_samples / float(self.SAMPLE_RATE)
        previous_text = ""
        triton_counts = triton_status.counts()

        # Every batch of a long recording is padded to the longest chunk length, so the encoder runs one shape (one
        # CUDA graph) per batch size. Chunks cut at pauses have many different lengths, and each length kept its own
        # graph and activation memory (33 GB at batch 16 on an hour-long recording in testing).
        _, max_chunk_samples = self.chunk_limits(total_samples, params.chunk_length)
        pad_samples = max_chunk_samples if total_samples > max_chunk_samples else None

        chunk_iter = self.iter_audio_chunks(audio_array, params.chunk_length)
        # The generative decoder otherwise invents words such as "Okay" for
        # an exactly silent recording. Only drop digital zero; quiet speech
        # and real background noise remain available to the model/VAD.
        chunks = (chunk for chunk in chunk_iter if np.any(chunk["audio"]))
        batch_start = 0
        try:
            while True:
                batch_chunks = list(itertools.islice(chunks, min(batch_size, self.oom_batch_limit or batch_size)))
                if not batch_chunks:
                    break
                position = batch_chunks[0]["start_seconds"] / total_duration if total_duration else 0.0
                progress(self.map_transcription_progress(position),
                         desc=f"Transcribing Canary-Qwen chunks {batch_start + 1}-{batch_start + len(batch_chunks)}..")

                results = self.transcribe_chunks(batch_chunks, params, generation_kwargs, pad_samples,
                                                 progress_callback, previous_text=previous_text)
                for batch_chunk, pieces in zip(batch_chunks, results):
                    emitted = False
                    for chunk, text in pieces:
                        if not text:
                            continue
                        emitted = True
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
                    if not emitted:
                        raw_progress = min(batch_chunk["end_seconds"] / total_duration, 0.99) if total_duration else 0.99
                        self.emit_progress_callback(progress_callback, raw_progress, None)
                batch_start += len(batch_chunks)
        finally:
            chunk_iter.close()  # stops the voice detection of a cancelled or failed transcription

        triton_summary = triton_status.summary_since(triton_counts)
        if triton_summary:
            logger.info(triton_summary)
            self.emit_status_callback(progress_callback, triton_summary)

        elapsed_time = time.time() - start_time
        return segments, elapsed_time

    # A transcript that fills max_new_tokens without ending, or that compresses like a loop ("Kwame Kwame Kwame
    # ..." for a chant, as Whisper's compression ratio check), is decoded again as two halves cut at a pause.
    DEGENERATE_COMPRESSION_RATIO = 2.4
    MAX_RETRY_SPLITS = 2
    MIN_RETRY_SPLIT_SECONDS = 4.0
    MAX_REPEATED_NGRAMS = 3

    def transcribe_chunks(self, chunks: List[dict], params: WhisperParams, generation_kwargs: dict,
                          pad_samples: Optional[int], progress_callback: Optional[Callable] = None,
                          previous_text: str = "", depth: int = 0) -> List[List[Tuple[dict, str]]]:
        """(chunk, text) pieces for every chunk; a chunk whose transcript degenerated becomes several pieces."""
        output_ids = self.generate_chunk_ids(chunks, params, generation_kwargs, pad_samples, progress_callback,
                                             previous_text=previous_text)
        results = []
        for chunk, token_ids in zip(chunks, output_ids):
            text = self.decode_output(token_ids)
            if not self.is_degenerate_output(token_ids, text, generation_kwargs):
                results.append([(chunk, text)])
                continue
            duration = chunk["end_seconds"] - chunk["start_seconds"]
            if depth < self.MAX_RETRY_SPLITS and duration >= self.MIN_RETRY_SPLIT_SECONDS:
                logger.info(
                    "Canary-Qwen output for %s -> %s repeated itself; transcribing that chunk again in two parts.",
                    self.format_timestamp(chunk["start_seconds"]), self.format_timestamp(chunk["end_seconds"]),
                )
                halves = self.split_chunk(chunk)
                # the same padded length as the batch, so the retry reuses the encoder's graph memory
                sub_results = self.transcribe_chunks(halves, params, generation_kwargs, pad_samples, progress_callback,
                                                     previous_text=previous_text, depth=depth + 1)
                results.append([piece for pieces in sub_results for piece in pieces])
            else:
                results.append([(chunk, self.collapse_repetitions(text))])
        return results

    def generate_chunk_ids(self, chunks: List[dict], params: WhisperParams, generation_kwargs: dict,
                           pad_samples: Optional[int], progress_callback: Optional[Callable] = None,
                           previous_text: str = "") -> list:
        """Output token ids for every chunk.

        Out of GPU memory, the engine's cached CUDA graphs and KV caches are released and the batch runs again:
        the graphs keep memory nothing else can use, so after one failure every later file failed too, even a
        2 second clip (8 GB card at batch 8 in testing). A batch that still does not fit runs in halves, and
        later batches keep the smaller size until a model loads.
        """
        for attempt in range(2):
            try:
                return self.generate_batch_ids(chunks, params, generation_kwargs, pad_samples, progress_callback,
                                               previous_text)
            except RuntimeError as exc:
                if not self.is_out_of_memory(exc):
                    raise
                if attempt and len(chunks) == 1:
                    self.release_gpu_memory()
                    raise
            # outside the except block, whose traceback still holds the failed batch's tensors
            self.release_gpu_memory()
            if not attempt:
                logger.warning("Canary-Qwen ran out of GPU memory; released its cached CUDA graphs, trying again.")
        half = (len(chunks) + 1) // 2
        self.oom_batch_limit = min(half, self.oom_batch_limit or half)
        message = (f"Canary-Qwen ran out of GPU memory at batch size {len(chunks)}; continuing with batch size "
                   f"{self.oom_batch_limit}. Close other programs that use the GPU or lower Batch Size.")
        logger.warning(message)
        self.emit_status_callback(progress_callback, message)
        return (self.generate_chunk_ids(chunks[:half], params, generation_kwargs, pad_samples, progress_callback,
                                        previous_text)
                + self.generate_chunk_ids(chunks[half:], params, generation_kwargs, pad_samples, progress_callback,
                                          previous_text))

    def generate_batch_ids(self, chunks: List[dict], params: WhisperParams, generation_kwargs: dict,
                           pad_samples: Optional[int], progress_callback: Optional[Callable] = None,
                           previous_text: str = "") -> list:
        audios, audio_lens = self.collate_audio_batch(chunks, pad_samples)
        prompts = [self.build_asr_prompt(params=params, previous_text=previous_text) for _ in chunks]
        # The ConvRot engine's first-run Triton tuning is reported in the Live Transcription box.
        with torch.inference_mode(), triton_status.report_to(
            lambda message: self.emit_status_callback(progress_callback, message)
        ):
            output_ids = self.model.generate(
                prompts=prompts,
                audios=audios.to(self.device, non_blocking=True),
                audio_lens=audio_lens.to(self.device, non_blocking=True),
                **generation_kwargs,
            )

        # Hugging Face returns a structured output when the advanced JSON
        # requests return_dict_in_generate. Transcription still needs its
        # token sequences, one row per input chunk.
        if hasattr(output_ids, "sequences"):
            output_ids = output_ids.sequences
        if len(output_ids) != len(chunks):
            raise ValueError(
                "Canary-Qwen must return one transcript per audio chunk. "
                "Set num_return_sequences to 1 in Canary Generation Kwargs."
            )
        return list(output_ids)

    @staticmethod
    def is_out_of_memory(exc: BaseException) -> bool:
        # PyTorch's allocator raises OutOfMemoryError; the CUDA driver (graph capture, libraries) a RuntimeError
        return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()

    def release_gpu_memory(self) -> None:
        release = getattr(self.model, "release_cuda_graphs", None)
        if callable(release):
            release()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_degenerate_output(self, token_ids, text: str, generation_kwargs: dict) -> bool:
        if not text:
            return False
        max_new_tokens = int(generation_kwargs.get("max_new_tokens") or self.DEFAULT_MAX_NEW_TOKENS)
        eos_id = getattr(self.model, "text_eos_id", None)
        pad_id = getattr(self.model, "text_pad_id", None)
        if eos_id is not None:
            ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
            if eos_id not in ids and len([i for i in ids if i != pad_id]) >= max_new_tokens:
                return True
        encoded = text.encode("utf-8")
        return len(encoded) >= 64 and len(encoded) / len(zlib.compress(encoded)) > self.DEGENERATE_COMPRESSION_RATIO

    def split_chunk(self, chunk: dict) -> List[dict]:
        audio = chunk["audio"]
        total = int(audio.shape[-1])
        probs = self.speech_probabilities(audio) if self.CUT_STRATEGY == "vad" else None
        cut = self.find_pause_cut(audio, total // 3, (2 * total) // 3, speech_probs=probs)
        cut = int(min(max(cut, self.MIN_CHUNK_SAMPLES), total - self.MIN_CHUNK_SAMPLES))
        start = chunk["start_seconds"]
        middle = start + cut / float(self.SAMPLE_RATE)
        return [
            {"audio": audio[:cut], "start_seconds": start, "end_seconds": middle},
            {"audio": audio[cut:], "start_seconds": middle, "end_seconds": chunk["end_seconds"]},
        ]

    @classmethod
    def collapse_repetitions(cls, text: str) -> str:
        """Keep at most MAX_REPEATED_NGRAMS consecutive copies of a repeated one to four word phrase."""
        words = text.split()
        limit = cls.MAX_REPEATED_NGRAMS

        def key(span):
            return [w.strip(".,!?;:").lower() for w in span]

        for n in range(1, 5):
            out = []
            i = 0
            while i < len(words):
                gram = key(words[i:i + n])
                if len(gram) < n:
                    out.extend(words[i:])
                    break
                repeats = 1
                while key(words[i + repeats * n:i + (repeats + 1) * n]) == gram:
                    repeats += 1
                if repeats > limit:
                    out.extend(words[i:i + limit * n])
                    i += repeats * n
                else:
                    out.append(words[i])
                    i += 1
            words = out
        return " ".join(words)

    def model_to_device(self, device: str) -> None:
        self.model.to(device)

    def update_model(
        self,
        model_size: str,
        compute_type: str,
        progress: gr.Progress = gr.Progress(),
        progress_callback: Optional[Callable] = None,
    ):
        progress(0.02, desc="Initializing Canary-Qwen model..")
        self.emit_status_callback(progress_callback, "Initializing Canary-Qwen model..")
        self.oom_batch_limit = None
        selected_model = model_size
        fallback_model = CANARY_CONVROT_FALLBACK_MODELS.get(model_size)
        if fallback_model is not None:
            supported, reason = convrot_runtime_supported()
            if not supported:
                message = (
                    f"INT8 ConvRot model '{model_size}' cannot run on this system ({reason}); "
                    f"loading '{fallback_model}' instead."
                )
                logger.warning(message)
                self.emit_status_callback(progress_callback, message)
                model_size = fallback_model
        convrot_path = self.resolve_convrot_model(model_size, progress=progress, progress_callback=progress_callback)
        if convrot_path is not None:
            self.load_convrot_model(selected_model, convrot_path, compute_type, progress, progress_callback)
            return
        dtype = self.torch_dtype_for_compute_type(compute_type)
        with self.hf_cache_scope():
            model_target = self.resolve_model_target(model_size, progress=progress, progress_callback=progress_callback)
            self.emit_status_callback(
                progress_callback,
                f"Loading Canary-Qwen model from {model_target}. This can take a while..",
            )
            salm_cls = self.import_salm()
            logger.info("Loading Canary-Qwen model '%s' into %s with %s.", model_target, self.device, compute_type)
            self.log_model_load_start(
                implementation=self.implementation_label(WhisperImpl.CANARY_QWEN.value),
                selected_model=model_size,
                resolved_model=model_target,
                compute_type=compute_type,
            )

            self.release_model_before_load()
            model = salm_cls.from_pretrained(
                model_target,
                cache_dir=self.get_hf_hub_cache_dir(),
                torch_dtype=dtype,
                token=os.environ.get("HF_TOKEN") or None,
            )
        model.eval()
        # One move that also casts: the float32 weights (about 10 GB) were moved to the GPU first and cast
        # there, which ran out of memory on 8-12 GB cards; PyTorch casts each weight before copying it.
        if dtype != torch.float32:
            model.to(device=self.device, dtype=dtype)
        else:
            model.to(self.device)

        self.model = model
        # the selection, which may be an INT8 ConvRot model this system replaced with its NeMo fallback
        self.current_model_size = selected_model
        self.current_compute_type = compute_type
        self.log_model_load_complete(
            implementation=self.implementation_label(WhisperImpl.CANARY_QWEN.value),
            selected_model=model_size,
            active_model=self.current_model_size,
            compute_type=self.current_compute_type,
        )
        self.emit_status_callback(progress_callback, "Canary-Qwen model loaded. Starting transcription..")
        progress(0.1, desc="Canary-Qwen model loaded.")

    def is_convrot_engine(self) -> bool:
        return type(self.model).__name__ == "CanaryConvRot"

    def resolve_convrot_model(
        self,
        model_size: str,
        progress: gr.Progress = None,
        progress_callback: Optional[Callable] = None,
    ) -> Optional[str]:
        """Folder of the INT8 ConvRot model selected by ``model_size`` (a hosted model is downloaded on first
        use), or None for NeMo models."""
        name = str(model_size or "")
        if name in HOSTED_CANARY_CONVROT_MODELS:
            target = os.path.join(self.model_dir, name)
            if not is_canary_convrot_model_dir(target):
                self.download_convrot_snapshot(name, target, progress=progress, progress_callback=progress_callback)
            return target
        candidate = name if os.path.isabs(name) else os.path.join(self.model_dir, name)
        if not is_canary_convrot_model_dir(candidate):
            return None
        supported, reason = convrot_runtime_supported()
        if not supported:
            raise RuntimeError(
                "INT8 ConvRot Canary-Qwen models need an NVIDIA Ampere (RTX 30 series) or newer GPU with Triton "
                f"and flash-attn ({reason}). Select {self.DEFAULT_MODEL_ID} instead."
            )
        return candidate

    def download_convrot_snapshot(
        self,
        name: str,
        target_dir: str,
        progress: gr.Progress = None,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        repo_id, subfolder = HOSTED_CANARY_CONVROT_MODELS[name]
        if progress is not None:
            progress(0.02, desc=f"Downloading Canary-Qwen INT8 ConvRot model to {target_dir}..")
        self.live_phase = self.LIVE_PHASE_DOWNLOADING
        self.emit_status_callback(progress_callback, f"Downloading Canary-Qwen INT8 ConvRot model to {target_dir}..")
        logger.info("Downloading INT8 ConvRot model '%s' from '%s/%s' to '%s'.", name, repo_id, subfolder, target_dir)
        download_convrot_model(
            name,
            target_dir,
            tqdm_class=self.make_download_tqdm_class(progress_callback),
            token=os.environ.get("HF_TOKEN") or None,
        )
        self.available_models = self.get_model_paths()
        self.emit_status_callback(progress_callback, "Canary-Qwen model download finished.")

    def load_convrot_model(
        self,
        selected_model: str,
        model_path: str,
        compute_type: str,
        progress: gr.Progress = gr.Progress(),
        progress_callback: Optional[Callable] = None,
    ) -> None:
        from modules.whisper.convrot.canary.engine import CanaryConvRot

        self.emit_status_callback(progress_callback, f"Loading Canary-Qwen INT8 ConvRot model from {model_path}..")
        logger.info(
            "INT8 ConvRot Canary-Qwen model: running the ConvRot engine (INT8 weights, Triton kernels, CUDA "
            "graphs); the compute type does not apply to it."
        )
        self.log_model_load_start(
            implementation=self.implementation_label(WhisperImpl.CANARY_QWEN.value),
            selected_model=selected_model,
            resolved_model=model_path,
            compute_type="int8 ConvRot",
        )
        self.release_model_before_load()
        self.model = CanaryConvRot.from_folder(model_path, device_index=torch.cuda.current_device())
        self.current_model_size = selected_model
        self.current_compute_type = compute_type
        self.log_model_load_complete(
            implementation=self.implementation_label(WhisperImpl.CANARY_QWEN.value),
            selected_model=selected_model,
            active_model=model_path,
            compute_type="int8 ConvRot",
        )
        self.emit_status_callback(progress_callback, "Canary-Qwen model loaded. Starting transcription..")
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
        return self.sanitize_generation_kwargs(kwargs)

    @classmethod
    def sanitize_generation_kwargs(cls, kwargs: dict) -> dict:
        sanitized = dict(kwargs)
        generation_config = sanitized.get("generation_config")
        num_return_sequences = sanitized.get(
            "num_return_sequences", getattr(generation_config, "num_return_sequences", 1)
        )
        if num_return_sequences not in (None, 1):
            raise ValueError(
                "Canary-Qwen transcription requires num_return_sequences=1 so each transcript "
                "matches its audio chunk."
            )
        sanitized["max_new_tokens"] = cls.clamp_int(
            sanitized.get("max_new_tokens"),
            default=cls.DEFAULT_MAX_NEW_TOKENS,
            minimum=1,
            maximum=cls.MAX_SAFE_MAX_NEW_TOKENS,
        )
        sanitized["num_beams"] = cls.clamp_int(
            sanitized.get("num_beams"),
            default=1,
            minimum=1,
            maximum=cls.MAX_SAFE_NUM_BEAMS,
        )
        if sanitized.get("temperature") is not None:
            sanitized["temperature"] = cls.clamp_float(
                sanitized.get("temperature"),
                default=1.0,
                minimum=0.01,
                maximum=5.0,
            )
        if sanitized.get("top_p") is not None:
            sanitized["top_p"] = cls.clamp_float(
                sanitized.get("top_p"),
                default=1.0,
                minimum=0.01,
                maximum=1.0,
            )
        if sanitized.get("top_k") is not None:
            sanitized["top_k"] = cls.clamp_int(
                sanitized.get("top_k"),
                default=0,
                minimum=0,
                maximum=1000,
            )
        if sanitized.get("repetition_penalty") is not None:
            sanitized["repetition_penalty"] = cls.clamp_float(
                sanitized.get("repetition_penalty"),
                default=1.0,
                minimum=0.01,
                maximum=10.0,
            )
        if sanitized.get("length_penalty") is not None:
            sanitized["length_penalty"] = cls.clamp_float(
                sanitized.get("length_penalty"),
                default=1.0,
                minimum=0.01,
                maximum=10.0,
            )
        if sanitized.get("no_repeat_ngram_size") is not None:
            sanitized["no_repeat_ngram_size"] = cls.clamp_int(
                sanitized.get("no_repeat_ngram_size"),
                default=0,
                minimum=0,
                maximum=20,
            )
        generation_config = sanitized.get("generation_config")
        if generation_config is not None:
            cls.clamp_generation_config(generation_config)
        return sanitized

    @classmethod
    def clamp_generation_config(cls, generation_config) -> None:
        attribute_specs = {
            "max_new_tokens": (cls.DEFAULT_MAX_NEW_TOKENS, 1, cls.MAX_SAFE_MAX_NEW_TOKENS, cls.clamp_int),
            "num_beams": (1, 1, cls.MAX_SAFE_NUM_BEAMS, cls.clamp_int),
            "temperature": (1.0, 0.01, 5.0, cls.clamp_float),
            "top_p": (1.0, 0.01, 1.0, cls.clamp_float),
            "top_k": (0, 0, 1000, cls.clamp_int),
            "repetition_penalty": (1.0, 0.01, 10.0, cls.clamp_float),
            "length_penalty": (1.0, 0.01, 10.0, cls.clamp_float),
            "no_repeat_ngram_size": (0, 0, 20, cls.clamp_int),
        }
        for attribute, (default, minimum, maximum, clamp_fn) in attribute_specs.items():
            value = getattr(generation_config, attribute, None)
            if value is None:
                continue
            setattr(generation_config, attribute, clamp_fn(value, default, minimum, maximum))

    @staticmethod
    def clamp_int(value, default: int, minimum: int, maximum: int) -> int:
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            coerced = default
        return min(max(coerced, minimum), maximum)

    @staticmethod
    def clamp_float(value, default: float, minimum: float, maximum: float) -> float:
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            coerced = default
        return min(max(coerced, minimum), maximum)

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

        audio_array = np.ascontiguousarray(audio_array.reshape(-1), dtype=np.float32)
        if audio_array.size and not np.isfinite(audio_array).all():
            invalid_samples = int(audio_array.size - np.isfinite(audio_array).sum())
            logger.warning(
                "Canary-Qwen audio contains %d non-finite sample(s); replacing them with silence.",
                invalid_samples,
            )
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        return audio_array

    def build_audio_chunks(self, audio: np.ndarray, chunk_length: Optional[int]) -> List[dict]:
        return list(self.iter_audio_chunks(audio, chunk_length))

    def chunk_limits(self, total_samples: int, chunk_length: Optional[int], log: bool = False) -> Tuple[int, int]:
        """(shortest, longest) chunk in samples for a recording of total_samples."""
        duration_seconds = total_samples / float(self.SAMPLE_RATE)
        if chunk_length is not None and float(chunk_length) == 0:
            if duration_seconds <= self.AUTO_SINGLE_CHUNK_SECONDS:
                if log:
                    logger.info("Canary-Qwen automatic chunking: %.3fs of audio in one piece.", duration_seconds)
                max_seconds = min_seconds = duration_seconds
            else:
                max_seconds = self.AUTO_LONG_MAX_CHUNK_SECONDS
                min_seconds = min(self.AUTO_LONG_MIN_CHUNK_SECONDS, max_seconds)
                if log:
                    logger.info(
                        "Canary-Qwen automatic chunking: %.0f-%.0fs pieces cut at pauses for %.3fs of audio.",
                        min_seconds, max_seconds, duration_seconds,
                    )
        else:
            requested_chunk_seconds = float(self.MAX_CHUNK_SECONDS if chunk_length is None else chunk_length)
            if requested_chunk_seconds <= 0:
                requested_chunk_seconds = self.MAX_CHUNK_SECONDS
            if log and requested_chunk_seconds < self.MIN_CHUNK_SECONDS:
                logger.info(
                    "Canary-Qwen chunk length %.3fs is too short; using %.3fs to avoid invalid audio features.",
                    requested_chunk_seconds,
                    self.MIN_CHUNK_SECONDS,
                )
            if log and requested_chunk_seconds > self.MAX_CHUNK_SECONDS:
                logger.info(
                    "Canary-Qwen was trained with audio windows up to %.0fs; capping requested chunk length %.1fs to %.0fs.",
                    self.MAX_CHUNK_SECONDS,
                    requested_chunk_seconds,
                    self.MAX_CHUNK_SECONDS,
                )
            max_seconds = min(max(requested_chunk_seconds, self.MIN_CHUNK_SECONDS), self.MAX_CHUNK_SECONDS)
            min_seconds = max_seconds * 2.0 / 3.0

        max_chunk_samples = max(self.MIN_CHUNK_SAMPLES, int(round(max_seconds * self.SAMPLE_RATE)))
        min_chunk_samples = max(self.MIN_CHUNK_SAMPLES, min(max_chunk_samples, int(round(min_seconds * self.SAMPLE_RATE))))
        return min_chunk_samples, max_chunk_samples

    def iter_audio_chunks(self, audio: np.ndarray, chunk_length: Optional[int]) -> Iterator[dict]:
        """The chunks of a recording in order. Voice detection runs in the background and each chunk is cut as
        soon as its part of the recording is analysed, so transcription starts before the whole recording is."""
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1), dtype=np.float32)
        total_samples = int(audio.shape[-1]) if audio.size else 0
        if total_samples <= 0:
            return
        if total_samples < self.MIN_CHUNK_SAMPLES:
            logger.info(
                "Canary-Qwen skipped %.3fs of audio because it is shorter than the %.3fs minimum safe window.",
                total_samples / float(self.SAMPLE_RATE),
                self.MIN_CHUNK_SECONDS,
            )
            return

        min_chunk_samples, max_chunk_samples = self.chunk_limits(total_samples, chunk_length, log=True)
        limit_samples = int(round(self.MAX_CHUNK_SECONDS * self.SAMPLE_RATE))
        stream = None
        if self.CUT_STRATEGY == "vad" and total_samples > max_chunk_samples:
            stream = self.speech_probability_stream(audio)

        try:
            # a chunk is handed out once the next one is known: a too short tail joins the chunk before it
            pending = None
            start_sample = 0
            while start_sample < total_samples:
                remaining_samples = total_samples - start_sample
                if remaining_samples < self.MIN_CHUNK_SAMPLES:
                    if pending is not None and pending["audio"].shape[-1] + remaining_samples <= limit_samples:
                        previous_start = int(round(pending["start_seconds"] * self.SAMPLE_RATE))
                        pending["audio"] = audio[previous_start:total_samples]
                        pending["end_seconds"] = total_samples / float(self.SAMPLE_RATE)
                    else:
                        logger.info(
                            "Canary-Qwen skipped %.3fs trailing audio that is too short to transcribe safely.",
                            remaining_samples / float(self.SAMPLE_RATE),
                        )
                    break

                if remaining_samples <= max_chunk_samples:
                    end_sample = total_samples
                else:
                    latest = start_sample + max_chunk_samples
                    speech_probs = None
                    if stream is not None:
                        try:
                            speech_probs = stream.wait(latest // self.VAD_FRAME_SAMPLES + self.VAD_SMOOTHING_FRAMES)
                        except Exception as exc:
                            logger.warning("Voice detection for Canary-Qwen chunking failed (%s); cutting at quiet "
                                           "moments instead.", exc)
                            stream = None
                    end_sample = self.find_pause_cut(audio, start_sample + min_chunk_samples, latest,
                                                     speech_probs=speech_probs)
                if pending is not None:
                    yield pending
                pending = {
                    "audio": audio[start_sample:end_sample],
                    "start_seconds": start_sample / float(self.SAMPLE_RATE),
                    "end_seconds": end_sample / float(self.SAMPLE_RATE),
                }
                start_sample = end_sample
            if pending is not None:
                yield pending
        finally:
            if stream is not None:
                stream.close()

    @classmethod
    def speech_probabilities(cls, audio: np.ndarray) -> Optional[np.ndarray]:
        """Silero speech probability for every 512-sample frame (32 ms), or None when the detector is missing."""
        stream = cls.speech_probability_stream(audio)
        if stream is None:
            return None
        try:
            return stream.wait()
        except Exception as exc:
            logger.warning("Voice detection for Canary-Qwen chunking failed (%s); cutting at quiet moments instead.", exc)
            return None

    @classmethod
    def speech_probability_stream(cls, audio: np.ndarray):
        """Silero speech probabilities of the audio, computed in the background (None without the detector)."""
        try:
            from modules.vad.silero_vad import SpeechProbabilityStream

            if cls._vad_model is None:
                cls._vad_model = cls.load_vad_model()
            return SpeechProbabilityStream(cls._vad_model, audio)
        except Exception as exc:
            logger.warning("Voice detection for Canary-Qwen chunking failed (%s); cutting at quiet moments instead.", exc)
            return None

    @staticmethod
    def load_vad_model():
        from modules.vad.silero_vad import load_silero_vad_model

        return load_silero_vad_model()

    @classmethod
    def find_pause_cut(cls, audio: np.ndarray, earliest: int, latest: int, speech_probs: Optional[np.ndarray] = None) -> int:
        """Where a chunk that may end between the samples earliest and latest should end: the middle of the
        clearest pause (lowest smoothed speech probability, or lowest energy without the voice detector), the
        latest of equally clear pauses so chunks stay long. Without a pause, the chunk keeps its full length."""
        earliest = int(max(0, earliest))
        latest = int(min(latest, audio.shape[-1]))
        if latest - earliest <= 0:
            return latest
        if speech_probs is not None and speech_probs.size:
            frame = cls.VAD_FRAME_SAMPLES
            smoothing = cls.VAD_SMOOTHING_FRAMES
            first = earliest // frame
            last = min(latest // frame, speech_probs.shape[0])
            if last - first >= smoothing:
                # smooth with the neighbouring frames too: zero padding at the edges of the search region
                # made its first and last frames look like pauses
                half = smoothing // 2
                lo, hi = max(0, first - half), min(speech_probs.shape[0], last + half)
                padded = np.pad(speech_probs[lo:hi].astype(np.float64), (half - (first - lo), half - (hi - last)),
                                mode="edge")
                smoothed = np.convolve(padded, np.full(smoothing, 1.0 / smoothing), mode="valid")
                best = float(smoothed.min())
                clearest = int(np.nonzero(smoothed <= best + 0.02)[0][-1])
                cut = (first + clearest) * frame + frame // 2
                return int(min(max(cut, earliest), latest))
        frame = max(1, int(round(cls.PAUSE_FRAME_SECONDS * cls.SAMPLE_RATE)))
        frame_count = (latest - earliest) // frame
        smoothing = cls.PAUSE_SMOOTHING_FRAMES
        if frame_count < smoothing * 2:
            return latest
        region = audio[latest - frame_count * frame:latest]
        energy = np.square(region.reshape(frame_count, frame), dtype=np.float64).mean(axis=1)
        smoothed = np.convolve(energy, np.full(smoothing, 1.0 / smoothing), mode="valid")
        # the latest of equally quiet windows, so silence (or steady noise) keeps the full chunk length
        quietest = len(smoothed) - 1 - int(np.argmin(smoothed[::-1]))
        if quietest == len(smoothed) - 1:
            return latest
        cut = latest - frame_count * frame + (quietest * frame) + (smoothing * frame) // 2
        return int(min(max(cut, earliest), latest))

    @staticmethod
    def collate_audio_batch(chunks: List[dict], pad_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = [int(chunk["audio"].shape[-1]) for chunk in chunks]
        too_short_lengths = [
            length
            for length in lengths
            if 0 < length < CanaryQwenInference.MIN_CHUNK_SAMPLES
        ]
        if too_short_lengths:
            raise ValueError(
                "Canary-Qwen received an audio chunk shorter than the minimum safe "
                f"window ({min(too_short_lengths)} samples). Re-run with a larger chunk length."
            )
        max_length = max(lengths) if lengths else 0
        if pad_samples:
            max_length = max(max_length, int(pad_samples))
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

    @staticmethod
    def safe_model_dir_name(model_size: str) -> str:
        return str(model_size or "").replace("/", "--")

    @staticmethod
    def has_downloaded_model_files(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        has_config = False
        has_weight = False
        try:
            for root, dirs, files in os.walk(path):
                dirs[:] = [directory for directory in dirs if directory not in {".cache", ".git"}]
                file_names = set(files)
                if "config.json" in file_names:
                    has_config = True
                if any(
                    file_name.endswith((".safetensors", ".bin", ".pt", ".ckpt", ".nemo"))
                    for file_name in file_names
                ):
                    has_weight = True
                if has_config and has_weight:
                    return True
        except OSError:
            return False
        return False

    @classmethod
    def make_download_tqdm_class(cls, progress_callback: Optional[Callable]):
        from tqdm.auto import tqdm

        class CanaryDownloadTqdm(tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._last_status_emit = 0.0

            def update(self, n=1):
                result = super().update(n)
                self._emit_status()
                return result

            def close(self):
                self._emit_status(force=True)
                return super().close()

            def _emit_status(self, force: bool = False):
                if progress_callback is None:
                    return

                now = time.time()
                if not force and now - self._last_status_emit < 1.0:
                    return
                self._last_status_emit = now

                unit = self.unit or "files"
                if self.total:
                    percent = min(100.0, max(0.0, (float(self.n) / float(self.total)) * 100.0))
                    status = f"Downloading Canary-Qwen model: {percent:.0f}% ({self.n}/{self.total} {unit})"
                else:
                    status = f"Downloading Canary-Qwen model: {self.n} {unit}"
                cls.emit_status_callback(progress_callback, status)

        return CanaryDownloadTqdm

    def download_model_snapshot(
        self,
        model_size: str,
        target_dir: str,
        progress: gr.Progress = None,
        progress_callback: Optional[Callable] = None,
    ) -> str:
        if progress is not None:
            progress(0.02, desc=f"Downloading Canary-Qwen model to {target_dir}..")
        self.live_phase = self.LIVE_PHASE_DOWNLOADING
        self.emit_status_callback(progress_callback, f"Downloading Canary-Qwen model to {target_dir}..")

        logger.info("Downloading Canary-Qwen model '%s' to '%s'.", model_size, target_dir)

        from huggingface_hub import snapshot_download

        os.makedirs(target_dir, exist_ok=True)
        snapshot_path = snapshot_download(
            repo_id=model_size,
            local_dir=target_dir,
            cache_dir=self.get_hf_hub_cache_dir(),
            token=os.environ.get("HF_TOKEN") or None,
            tqdm_class=self.make_download_tqdm_class(progress_callback),
        )
        self.emit_status_callback(progress_callback, "Canary-Qwen model download finished.")
        return snapshot_path

    def resolve_model_target(
        self,
        model_size: str,
        progress: gr.Progress = None,
        progress_callback: Optional[Callable] = None,
    ) -> str:
        model_size = model_size or self.DEFAULT_MODEL_ID
        if os.path.isabs(model_size) and os.path.exists(model_size):
            return model_size

        candidate = os.path.join(self.model_dir, model_size)
        if self.has_downloaded_model_files(candidate):
            return candidate

        safe_name = self.safe_model_dir_name(model_size)
        candidate = os.path.join(self.model_dir, safe_name)
        if self.has_downloaded_model_files(candidate):
            return candidate

        if "/" in model_size:
            self.download_model_snapshot(
                model_size,
                candidate,
                progress=progress,
                progress_callback=progress_callback,
            )
            if not self.has_downloaded_model_files(candidate):
                raise RuntimeError(
                    f"Canary-Qwen download did not create a complete model folder at {candidate}."
                )
            self.available_models = self.get_model_paths()
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
        models = [self.DEFAULT_MODEL_ID, *HOSTED_CANARY_CONVROT_MODELS]
        if os.path.isdir(self.model_dir):
            for item in os.listdir(self.model_dir):
                if item in ignored or item.startswith("."):  # .download-<name>: unfinished download
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

    @contextmanager
    def hf_cache_scope(self):
        """Use the Canary folder as the Hugging Face cache while the model downloads and loads, then restore it.

        configure_hf_cache() changes process-wide settings (NeMo fetches the Qwen3 config and tokenizer by
        name). Left in place, every later download of the same worker process looked in and wrote to the
        Canary folder, and Offload Models to RAM When Idle keeps one worker for all jobs.
        """
        env_names = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")
        saved_env = {name: os.environ.get(name) for name in env_names}
        try:
            import huggingface_hub.constants as hf_constants
        except Exception:
            hf_constants = None
        saved_constants = {
            name: getattr(hf_constants, name)
            for name in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")
            if hf_constants is not None and hasattr(hf_constants, name)
        }
        try:
            import transformers.utils.hub as transformers_hub
        except Exception:
            transformers_hub = None
        saved_transformers_cache = getattr(transformers_hub, "TRANSFORMERS_CACHE", None)

        self.configure_hf_cache()
        try:
            yield
        finally:
            for name, value in saved_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            for name, value in saved_constants.items():
                setattr(hf_constants, name, value)
            if saved_transformers_cache is not None:
                transformers_hub.TRANSFORMERS_CACHE = saved_transformers_cache

    def get_hf_hub_cache_dir(self) -> str:
        return os.path.join(self.model_dir, "hub")

    @classmethod
    def import_salm(cls):
        cls.patch_nemo_import_compat()
        from nemo.collections.speechlm2.models import SALM

        return SALM

    @staticmethod
    def patch_nemo_import_compat() -> None:
        CanaryQwenInference.patch_lightning_neptune_logger_compat()

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
    def patch_lightning_neptune_logger_compat() -> None:
        """
        Keep NeMo importable with Lightning releases that removed NeptuneLogger.

        NeMo imports NeptuneLogger at module import time even when Neptune logging
        is disabled. Canary-Qwen does not use Neptune, so a lazy placeholder is
        enough to preserve the optional dependency boundary.
        """

        class NeptuneLogger:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                raise ImportError(
                    "NeptuneLogger is not available in the installed Lightning "
                    "package. Disable Neptune logging or install a compatible "
                    "Neptune/Lightning logger package."
                )

        for module_name in ("lightning.pytorch.loggers", "pytorch_lightning.loggers"):
            try:
                loggers_module = importlib.import_module(module_name)
            except Exception:
                continue

            if hasattr(loggers_module, "NeptuneLogger"):
                continue

            loggers_module.NeptuneLogger = NeptuneLogger
            exported_names = getattr(loggers_module, "__all__", None)
            if isinstance(exported_names, list):
                if "NeptuneLogger" not in exported_names:
                    exported_names.append("NeptuneLogger")
            elif isinstance(exported_names, tuple) and "NeptuneLogger" not in exported_names:
                loggers_module.__all__ = exported_names + ("NeptuneLogger",)

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
