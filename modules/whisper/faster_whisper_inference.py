import logging
import os
import time
import math
from contextlib import contextmanager
from types import MethodType
import huggingface_hub
import numpy as np
import torch
from typing import BinaryIO, Union, Tuple, List, Callable, Optional
import faster_whisper
from faster_whisper.audio import decode_audio, pad_or_trim
from faster_whisper.transcribe import Segment as FasterWhisperSegment, Word as FasterWhisperWord
from faster_whisper.utils import format_timestamp, get_end
from faster_whisper.vad import VadOptions
import ast
import ctranslate2
import whisper
import gradio as gr
from argparse import Namespace
from tqdm import tqdm

from modules.utils.paths import (FASTER_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, UVR_MODELS_DIR, OUTPUT_DIR)
from modules.whisper.data_classes import *
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.utils.logger import get_logger

logger = get_logger()


class StandardEncoderPrefetchCache:
    def __init__(self, model, features: np.ndarray, batch_size: int):
        self.model = model
        self.features = features
        self.batch_size = max(1, int(batch_size))
        self.content_frames = features.shape[-1] - 1
        self.window_frames = int(model.feature_extractor.nb_max_frames)
        self.cache = {}
        self.cache_clip_end = None

    def get(self, seek: int, seek_clip_end: int) -> ctranslate2.StorageView:
        if self.batch_size <= 1:
            return self._encode_window(seek=seek, seek_clip_end=seek_clip_end)

        if seek not in self.cache or self.cache_clip_end != seek_clip_end:
            self.cache.clear()
            self.cache_clip_end = seek_clip_end
            self._prefetch_from_seek(seek=seek, seek_clip_end=seek_clip_end)

        return self.cache.pop(seek)

    def _encode_window(self, seek: int, seek_clip_end: int) -> ctranslate2.StorageView:
        segment = self._build_segment(seek=seek, seek_clip_end=seek_clip_end)
        return self.model.encode(segment)

    def _prefetch_from_seek(self, seek: int, seek_clip_end: int) -> None:
        windows = []
        seeks = []
        for step in range(self.batch_size):
            window_seek = seek + (step * self.window_frames)
            if window_seek >= self.content_frames or window_seek >= seek_clip_end:
                break

            windows.append(self._build_segment(seek=window_seek, seek_clip_end=seek_clip_end))
            seeks.append(window_seek)

        if not windows:
            raise ValueError(f"No encoder windows available for seek {seek}")

        encoded_batch = self.model.encode(np.stack(windows, axis=0))
        for window_seek, storage_view in zip(seeks, self._split_storage_view_batch(encoded_batch)):
            self.cache[window_seek] = storage_view

    def _build_segment(self, seek: int, seek_clip_end: int) -> np.ndarray:
        segment_size = min(
            self.window_frames,
            self.content_frames - seek,
            seek_clip_end - seek,
        )
        segment = self.features[:, seek : seek + segment_size]
        return pad_or_trim(segment)

    @staticmethod
    def _split_storage_view_batch(
        encoded_batch: ctranslate2.StorageView,
    ) -> List[ctranslate2.StorageView]:
        batch_length = int(encoded_batch.shape[0])
        batch_view = encoded_batch
        if getattr(encoded_batch, "device", "cpu") == "cuda":
            batch_view = encoded_batch.to_device(ctranslate2.Device.cpu)

        try:
            batch_array = np.asarray(batch_view)
        except RuntimeError:
            batch_array = np.asarray(batch_view.to(ctranslate2.DataType.float32))
        return [
            ctranslate2.StorageView.from_array(np.ascontiguousarray(batch_array[idx : idx + 1]))
            for idx in range(batch_length)
        ]


class FasterWhisperInference(BaseTranscriptionPipeline):
    MODEL_READY_PROGRESS = 0.08
    AUDIO_PREPARED_PROGRESS = 0.16
    CHUNKS_PREPARED_PROGRESS = 0.24
    TRANSCRIPTION_PROGRESS_START = 0.3
    TRANSCRIPTION_PROGRESS_END = 0.98
    LONG_FORM_CONDITIONING_WINDOW_THRESHOLD = 60

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

        if self.should_use_batched_inference(params):
            progress(0, desc="Loading audio..")
            progress(self.MODEL_READY_PROGRESS, desc="Loading audio..")
            segments, info = self._transcribe_with_batching(audio=audio, params=params, progress=progress)
        else:
            logger.info("Using standard faster-whisper inference for maximum subtitle quality.")
            standard_audio, standard_params = self.resolve_standard_audio_and_params(audio=audio, params=params)
            segments, info = self._transcribe_with_standard_pipeline(
                audio=standard_audio,
                params=standard_params,
                progress=progress,
            )

        segments_result = []
        for idx, segment in enumerate(segments):
            progress_n = 0.0 if not info.duration else min(segment.end / info.duration, 0.99)
            ui_progress_n = self.map_transcription_progress(progress_n)
            seg_obj = Segment.from_faster_whisper(segment)
            segments_result.append(seg_obj)

            # Live transcription display in terminal
            logger.info(f"[{self.format_timestamp(seg_obj.start)} -> {self.format_timestamp(seg_obj.end)}] {seg_obj.text}")

            # Update progress with current segment info
            progress(ui_progress_n, desc=f"Transcribing.. [{idx+1} segments] {seg_obj.text[:50]}...")

            self.emit_progress_callback(progress_callback, progress_n, seg_obj)

        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    @staticmethod
    def should_use_batched_inference(params: WhisperParams) -> bool:
        return bool(getattr(params, "use_batched_inference", False))

    def resolve_standard_audio_and_params(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        params: WhisperParams,
    ) -> Tuple[Union[str, BinaryIO, np.ndarray], WhisperParams]:
        if not params.condition_on_previous_text:
            return audio, params

        sampling_rate = self.model.feature_extractor.sampling_rate
        audio_array = self.prepare_audio_array(audio=audio, sampling_rate=sampling_rate)
        estimated_windows = self.estimate_chunk_windows(
            audio=audio_array,
            chunk_length=params.chunk_length,
            sampling_rate=sampling_rate,
        )

        if estimated_windows < self.LONG_FORM_CONDITIONING_WINDOW_THRESHOLD:
            return audio_array, params

        duration_seconds = 0.0
        if sampling_rate > 0:
            duration_seconds = float(audio_array.shape[-1]) / float(sampling_rate)

        logger.info(
            "Auto-disabling condition_on_previous_text for long-form audio "
            "(%.1f minutes across ~%d windows) to prevent repetition drift.",
            duration_seconds / 60.0 if duration_seconds else 0.0,
            estimated_windows,
        )
        return audio_array, params.model_copy(update={"condition_on_previous_text": False})

    @staticmethod
    def should_repeat_initial_prompt(params: WhisperParams) -> bool:
        if not params.repeat_initial_prompt_every_window:
            return False

        prompt = params.initial_prompt
        if prompt is None:
            return False
        if isinstance(prompt, str):
            return bool(prompt.strip())
        return True

    @staticmethod
    def encode_initial_prompt(initial_prompt: Union[str, List[int], Tuple[int, ...]], tokenizer) -> List[int]:
        if initial_prompt is None:
            return []
        if isinstance(initial_prompt, str):
            normalized = initial_prompt.strip()
            if not normalized:
                return []
            return tokenizer.encode(" " + normalized)
        return list(initial_prompt)

    @staticmethod
    @contextmanager
    def repeat_initial_prompt_context(model, initial_prompt: Optional[Union[str, List[int], Tuple[int, ...]]]):
        if initial_prompt is None:
            yield
            return

        prompt_cache = {}
        original_get_prompt = model.get_prompt

        def wrapped_get_prompt(
            model_self,
            tokenizer,
            previous_tokens,
            without_timestamps: bool = False,
            prefix: Optional[str] = None,
            hotwords: Optional[str] = None,
        ):
            cache_key = id(tokenizer)
            if cache_key not in prompt_cache:
                prompt_cache[cache_key] = FasterWhisperInference.encode_initial_prompt(initial_prompt, tokenizer)

            merged_tokens = prompt_cache[cache_key] + list(previous_tokens or [])
            return original_get_prompt(
                tokenizer,
                merged_tokens,
                without_timestamps=without_timestamps,
                prefix=prefix,
                hotwords=hotwords,
            )

        model.get_prompt = MethodType(wrapped_get_prompt, model)
        try:
            yield
        finally:
            delattr(model, "get_prompt")

    @staticmethod
    @contextmanager
    def standard_encoder_batching_context(model, batch_size: int):
        batch_size = max(1, int(batch_size))
        if batch_size <= 1:
            yield
            return

        original_generate_segments = getattr(model, "generate_segments", None)
        if original_generate_segments is None:
            yield
            return

        def wrapped_generate_segments(
            model_self,
            features: np.ndarray,
            tokenizer,
            options,
            log_progress,
            encoder_output: Optional[ctranslate2.StorageView] = None,
        ):
            return FasterWhisperInference.generate_segments_with_encoder_prefetch(
                model=model_self,
                features=features,
                tokenizer=tokenizer,
                options=options,
                log_progress=log_progress,
                batch_size=batch_size,
                encoder_output=encoder_output,
            )

        model.generate_segments = MethodType(wrapped_generate_segments, model)
        try:
            yield
        finally:
            model.generate_segments = original_generate_segments

    @staticmethod
    def generate_segments_with_encoder_prefetch(
        model,
        features: np.ndarray,
        tokenizer,
        options,
        log_progress: bool,
        batch_size: int,
        encoder_output: Optional[ctranslate2.StorageView] = None,
    ):
        content_frames = features.shape[-1] - 1
        content_duration = float(content_frames * model.feature_extractor.time_per_frame)

        if isinstance(options.clip_timestamps, str):
            options.clip_timestamps = [
                float(ts)
                for ts in (
                    options.clip_timestamps.split(",")
                    if options.clip_timestamps
                    else []
                )
            ]

        seek_points = [
            round(ts * model.frames_per_second) for ts in options.clip_timestamps
        ]
        if len(seek_points) == 0:
            seek_points.append(0)
        if len(seek_points) % 2 == 1:
            seek_points.append(content_frames)
        seek_clips = list(zip(seek_points[::2], seek_points[1::2]))

        punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"

        idx = 0
        clip_idx = 0
        seek = seek_clips[clip_idx][0]
        all_tokens = []
        prompt_reset_since = 0

        if options.initial_prompt is not None:
            if isinstance(options.initial_prompt, str):
                initial_prompt = " " + options.initial_prompt.strip()
                initial_prompt_tokens = tokenizer.encode(initial_prompt)
                all_tokens.extend(initial_prompt_tokens)
            else:
                all_tokens.extend(options.initial_prompt)

        pbar = tqdm(total=content_duration, unit="seconds", disable=not log_progress)
        last_speech_timestamp = 0.0
        encoder_cache = StandardEncoderPrefetchCache(
            model=model,
            features=features,
            batch_size=batch_size,
        )

        try:
            while clip_idx < len(seek_clips):
                seek_clip_start, seek_clip_end = seek_clips[clip_idx]
                if seek_clip_end > content_frames:
                    seek_clip_end = content_frames
                if seek < seek_clip_start:
                    seek = seek_clip_start
                if seek >= seek_clip_end:
                    clip_idx += 1
                    if clip_idx < len(seek_clips):
                        seek = seek_clips[clip_idx][0]
                    continue

                time_offset = seek * model.feature_extractor.time_per_frame
                window_end_time = float(
                    (seek + model.feature_extractor.nb_max_frames)
                    * model.feature_extractor.time_per_frame
                )
                segment_size = min(
                    model.feature_extractor.nb_max_frames,
                    content_frames - seek,
                    seek_clip_end - seek,
                )
                segment_duration = segment_size * model.feature_extractor.time_per_frame

                if model.logger.isEnabledFor(logging.DEBUG):
                    model.logger.debug(
                        "Processing segment at %s", format_timestamp(time_offset)
                    )

                previous_tokens = all_tokens[prompt_reset_since:]

                if seek == 0 and encoder_output is not None:
                    current_encoder_output = encoder_output
                else:
                    current_encoder_output = encoder_cache.get(
                        seek=seek,
                        seek_clip_end=seek_clip_end,
                    )

                if options.multilingual:
                    results = model.model.detect_language(current_encoder_output)
                    language_token, _language_probability = results[0][0]
                    language = language_token[2:-2]

                    tokenizer.language = tokenizer.tokenizer.token_to_id(language_token)
                    tokenizer.language_code = language

                prompt = model.get_prompt(
                    tokenizer,
                    previous_tokens,
                    without_timestamps=options.without_timestamps,
                    prefix=options.prefix if seek == 0 else None,
                    hotwords=options.hotwords,
                )

                (
                    result,
                    avg_logprob,
                    temperature,
                    compression_ratio,
                ) = model.generate_with_fallback(
                    current_encoder_output,
                    prompt,
                    tokenizer,
                    options,
                )

                if options.no_speech_threshold is not None:
                    should_skip = result.no_speech_prob > options.no_speech_threshold

                    if (
                        options.log_prob_threshold is not None
                        and avg_logprob > options.log_prob_threshold
                    ):
                        should_skip = False

                    if should_skip:
                        model.logger.debug(
                            "No speech threshold is met (%f > %f)",
                            result.no_speech_prob,
                            options.no_speech_threshold,
                        )
                        seek += segment_size
                        continue

                tokens = result.sequences_ids[0]
                previous_seek = seek

                def word_anomaly_score(word: dict) -> float:
                    probability = word.get("probability", 0.0)
                    duration = word["end"] - word["start"]
                    score = 0.0
                    if probability < 0.15:
                        score += 1.0
                    if duration < 0.133:
                        score += (0.133 - duration) * 15
                    if duration > 2.0:
                        score += duration - 2.0
                    return score

                def is_segment_anomaly(current_segment: Optional[dict]) -> bool:
                    if current_segment is None or not current_segment["words"]:
                        return False
                    words = [
                        word for word in current_segment["words"]
                        if word["word"] not in punctuation
                    ]
                    words = words[:8]
                    score = sum(word_anomaly_score(word) for word in words)
                    return score >= 3 or score + 0.01 >= len(words)

                def next_words_segment(segments: List[dict]) -> Optional[dict]:
                    return next((segment for segment in segments if segment["words"]), None)

                (
                    current_segments,
                    seek,
                    single_timestamp_ending,
                ) = model._split_segments_by_timestamps(
                    tokenizer=tokenizer,
                    tokens=tokens,
                    time_offset=time_offset,
                    segment_size=segment_size,
                    segment_duration=segment_duration,
                    seek=seek,
                )

                if options.word_timestamps:
                    model.add_word_timestamps(
                        [current_segments],
                        tokenizer,
                        current_encoder_output,
                        segment_size,
                        options.prepend_punctuations,
                        options.append_punctuations,
                        last_speech_timestamp=last_speech_timestamp,
                    )
                    if not single_timestamp_ending:
                        last_word_end = get_end(current_segments)
                        if last_word_end is not None and last_word_end > time_offset:
                            seek = round(last_word_end * model.frames_per_second)

                    if options.hallucination_silence_threshold is not None:
                        threshold = options.hallucination_silence_threshold

                        first_segment = next_words_segment(current_segments)
                        if first_segment is not None and is_segment_anomaly(first_segment):
                            gap = first_segment["start"] - time_offset
                            if gap > threshold:
                                seek = previous_seek + round(gap * model.frames_per_second)
                                continue

                        hal_last_end = last_speech_timestamp
                        for segment_index in range(len(current_segments)):
                            current_segment = current_segments[segment_index]
                            if not current_segment["words"]:
                                continue
                            if is_segment_anomaly(current_segment):
                                next_segment = next_words_segment(
                                    current_segments[segment_index + 1 :]
                                )
                                if next_segment is not None:
                                    hal_next_start = next_segment["words"][0]["start"]
                                else:
                                    hal_next_start = time_offset + segment_duration
                                silence_before = (
                                    current_segment["start"] - hal_last_end > threshold
                                    or current_segment["start"] < threshold
                                    or current_segment["start"] - time_offset < 2.0
                                )
                                silence_after = (
                                    hal_next_start - current_segment["end"] > threshold
                                    or is_segment_anomaly(next_segment)
                                    or window_end_time - current_segment["end"] < 2.0
                                )
                                if silence_before and silence_after:
                                    seek = round(
                                        max(time_offset + 1, current_segment["start"])
                                        * model.frames_per_second
                                    )
                                    if content_duration - current_segment["end"] < threshold:
                                        seek = content_frames
                                    current_segments[segment_index:] = []
                                    break
                            hal_last_end = current_segment["end"]

                    last_word_end = get_end(current_segments)
                    if last_word_end is not None:
                        last_speech_timestamp = last_word_end

                for current_segment in current_segments:
                    segment_tokens = current_segment["tokens"]
                    text = tokenizer.decode(segment_tokens)

                    if current_segment["start"] == current_segment["end"] or not text.strip():
                        continue

                    all_tokens.extend(segment_tokens)
                    idx += 1

                    yield FasterWhisperSegment(
                        id=idx,
                        seek=previous_seek,
                        start=current_segment["start"],
                        end=current_segment["end"],
                        text=text,
                        tokens=segment_tokens,
                        temperature=temperature,
                        avg_logprob=avg_logprob,
                        compression_ratio=compression_ratio,
                        no_speech_prob=result.no_speech_prob,
                        words=(
                            [FasterWhisperWord(**word) for word in current_segment["words"]]
                            if options.word_timestamps
                            else None
                        ),
                    )

                if (
                    not options.condition_on_previous_text
                    or temperature > options.prompt_reset_on_temperature
                ):
                    if options.condition_on_previous_text:
                        model.logger.debug(
                            "Reset prompt. prompt_reset_on_temperature threshold is met %f > %f",
                            temperature,
                            options.prompt_reset_on_temperature,
                        )

                    prompt_reset_since = len(all_tokens)

                pbar.update(
                    (min(content_frames, seek) - previous_seek)
                    * model.feature_extractor.time_per_frame,
                )
        finally:
            pbar.close()

    def _transcribe_with_batching(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        params: WhisperParams,
        progress: gr.Progress = gr.Progress(),
    ):
        batch_pipeline_cls = getattr(faster_whisper, "BatchedInferencePipeline", None)
        if batch_pipeline_cls is None:
            logger.warning("Installed faster-whisper build does not support BatchedInferencePipeline. Falling back to standard transcription.")
            return self._transcribe_with_standard_pipeline(audio=audio, params=params, progress=progress)

        sampling_rate = self.model.feature_extractor.sampling_rate
        audio_array = self.prepare_audio_array(audio=audio, sampling_rate=sampling_rate)
        progress(self.AUDIO_PREPARED_PROGRESS, desc="Audio loaded. Preparing chunks..")
        clip_timestamps = self.build_clip_timestamps(
            audio=audio_array,
            chunk_length=params.chunk_length,
            sampling_rate=sampling_rate,
        )
        progress(self.CHUNKS_PREPARED_PROGRESS, desc=f"Prepared {len(clip_timestamps) or 1} chunks. Starting transcription..")

        batch_pipeline = batch_pipeline_cls(model=self.model)
        progress(self.TRANSCRIPTION_PROGRESS_START, desc="Transcribing..")
        repeat_initial_prompt = self.should_repeat_initial_prompt(params)
        with self.repeat_initial_prompt_context(
            self.model,
            params.initial_prompt if repeat_initial_prompt else None,
        ):
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
                initial_prompt=None if repeat_initial_prompt else params.initial_prompt,
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
        progress: gr.Progress = gr.Progress(),
    ):
        progress(self.TRANSCRIPTION_PROGRESS_START, desc="Transcribing..")
        repeat_initial_prompt = self.should_repeat_initial_prompt(params)
        encoder_batch_size = max(1, int(params.batch_size))
        use_encoder_batching = encoder_batch_size > 1 and bool(params.condition_on_previous_text)
        if use_encoder_batching:
            logger.info(
                "Using standard faster-whisper inference with encoder prefetch batching "
                "(batch_size=%d) for quality-preserving acceleration.",
                encoder_batch_size,
            )
        with self.repeat_initial_prompt_context(
            self.model,
            params.initial_prompt if repeat_initial_prompt else None,
        ):
            with self.standard_encoder_batching_context(
                self.model,
                encoder_batch_size if use_encoder_batching else 1,
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
                    initial_prompt=None if repeat_initial_prompt else params.initial_prompt,
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

    @classmethod
    def map_transcription_progress(cls, raw_progress: float) -> float:
        bounded = min(max(raw_progress, 0.0), 0.99)
        span = cls.TRANSCRIPTION_PROGRESS_END - cls.TRANSCRIPTION_PROGRESS_START
        return cls.TRANSCRIPTION_PROGRESS_START + (bounded * span)

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
    def estimate_chunk_windows(
        audio: np.ndarray,
        chunk_length: Optional[int],
        sampling_rate: int,
    ) -> int:
        total_samples = int(audio.shape[-1]) if audio.size else 0
        if total_samples <= 0:
            return 1

        if chunk_length is None or chunk_length <= 0:
            return 1

        chunk_samples = max(1, int(chunk_length * sampling_rate))
        return max(1, math.ceil(total_samples / chunk_samples))

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
        progress(0.02, desc="Initializing Model..")

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
