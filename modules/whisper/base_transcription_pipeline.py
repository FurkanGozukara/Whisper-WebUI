import os
import glob
import whisper
import ctranslate2
import gradio as gr
import torch
from modules.utils.torch_compat import enable_torchaudio_2_9_compat

enable_torchaudio_2_9_compat()
import torchaudio
from abc import ABC, abstractmethod
from typing import BinaryIO, Union, Tuple, List, Callable, Optional
import numpy as np
from datetime import datetime
from faster_whisper.vad import VadOptions
import gc
from contextlib import contextmanager
from copy import deepcopy
import time
from collections import deque
from queue import Empty, Queue
from threading import Lock, Thread

from modules.uvr.music_separator import MusicSeparator
from modules.utils.paths import WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR, UVR_MODELS_DIR
from modules.utils.constants import *
from modules.utils.logger import get_logger
from modules.utils.subtitle_manager import *
from modules.utils.subtitle_manager import safe_filename
from modules.utils.youtube_manager import get_latest_channel_videos, get_ytdata, get_ytaudio, remove_ytaudio
from modules.utils.files_manager import get_media_files, format_gradio_files, normalize_folder_path, read_file
from modules.utils.audio_manager import coerce_audio_input_path, validate_audio, is_digital_silence
from modules.whisper.data_classes import *
from modules.diarize.diarizer import Diarizer
from modules.vad.silero_vad import SileroVAD


logger = get_logger()


# One transcription at a time in this process. The engines share the GPU and keep their models loaded between jobs;
# a second job (another tab or user, or a new job while a cancelled one still finishes its file in a background
# thread) replaced, offloaded or parked the model under the running one and both failed. run() holds the lock in
# the thread that uses the model, so it is released only when that work has really ended.
TRANSCRIPTION_LOCK = Lock()


class BaseTranscriptionPipeline(ABC):
    LIVE_TRANSCRIPTION_HISTORY_LINES = 30
    LIVE_TRANSCRIPTION_POLL_INTERVAL_SEC = 0.1
    LIVE_TRANSCRIPTION_HEARTBEAT_INTERVAL_SEC = 5.0
    NO_WORD_TIMESTAMPS_SUFFIX = "_noword_timestaps"
    IMPLEMENTATION_LABELS = {
        WhisperImpl.FASTER_WHISPER.value: "Whisper (faster-whisper / CTranslate2)",
        WhisperImpl.WHISPER.value: "Whisper (OpenAI)",
        WhisperImpl.INSANELY_FAST_WHISPER.value: "Insanely Fast Whisper (Transformers)",
        WhisperImpl.CANARY_QWEN.value: "Canary-Qwen (NVIDIA NeMo)",
    }
    # Set during a batch: models stay loaded until its last file (batch_offload_scope)
    defer_offload = False
    # What a job is doing before its first segment, for the Live Transcription heartbeat
    live_phase = None
    # Set by run(): the last input could not be opened, so it returned an empty placeholder result
    last_input_unreadable = False
    UNREADABLE_INPUT_MESSAGE = "the file could not be opened (corrupted or unsupported format); nothing was written"
    LIVE_PHASE_DOWNLOADING = "downloading the model (first use only; the progress is shown in CMD)"
    LIVE_PHASE_LOADING = "loading the model"

    def __init__(self,
                 model_dir: str = WHISPER_MODELS_DIR,
                 diarization_model_dir: str = DIARIZATION_MODELS_DIR,
                 uvr_model_dir: str = UVR_MODELS_DIR,
                 output_dir: str = OUTPUT_DIR,
                 ):
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.diarizer = Diarizer(
            model_dir=diarization_model_dir
        )
        self.vad = SileroVAD()
        self.music_separator = MusicSeparator(
            model_dir=uvr_model_dir,
            output_dir=os.path.join(output_dir, "UVR")
        )

        self.model = None
        self.models_in_ram = False  # model parked in system RAM between jobs (Offload Models to RAM When Idle)
        self.current_model_size = None
        self.available_models = whisper.available_models()
        self.available_langs = sorted(list(whisper.tokenizer.LANGUAGES.values()))
        self.device = self.get_device()
        self.available_compute_types = self.get_available_compute_type()
        self.current_compute_type = self.get_compute_type()

    @abstractmethod
    def transcribe(self,
                   audio: Union[str, BinaryIO, np.ndarray],
                   progress: gr.Progress = gr.Progress(),
                   progress_callback: Optional[Callable] = None,
                   *whisper_params,
                   log_console: bool = True,
                   log_model_banner: bool = True,
                   ):
        """Inference whisper model to transcribe"""
        pass

    @abstractmethod
    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress = gr.Progress()
                     ):
        """Initialize whisper model"""
        pass

    @staticmethod
    def supports_word_timestamps() -> bool:
        return True

    @classmethod
    def implementation_label(cls, whisper_type: Optional[str]) -> str:
        try:
            normalized = WhisperParams(whisper_type=whisper_type).whisper_type
        except Exception:
            normalized = str(whisper_type or "").strip().lower()
        return cls.IMPLEMENTATION_LABELS.get(normalized, normalized or "unknown")

    @staticmethod
    def ensure_progress_callable(progress):
        if callable(progress):
            return progress

        def noop_progress(*_args, **_kwargs):
            return None

        return noop_progress

    def should_load_model_for_selection(self, model_size: str, compute_type: str) -> bool:
        return (
            self.model is None
            or model_size != self.current_model_size
            or compute_type != self.current_compute_type
        )

    def log_selected_model(
        self,
        whisper_type: Optional[str],
        model_size: str,
        compute_type: str,
        will_load: Optional[bool] = None,
    ) -> None:
        logger.info("Selected Base Model: %s", self.implementation_label(whisper_type))
        logger.info("Selected Model: %s", model_size or "(default)")
        logger.info("Selected Device/Compute: device=%s, compute_type=%s", self.device, compute_type)

        if will_load is None:
            return

        if will_load:
            logger.info("Model load required: selected model is not currently active.")
        else:
            logger.info(
                "Model already loaded: active=%s, compute_type=%s, device=%s",
                self.current_model_size,
                self.current_compute_type,
                self.device,
            )

    def log_model_load_start(
        self,
        implementation: str,
        selected_model: str,
        compute_type: str,
        resolved_model: Optional[str] = None,
    ) -> None:
        resolved_model = resolved_model or selected_model
        self.live_phase = self.LIVE_PHASE_LOADING
        logger.info(
            "Loading model: Base Model=%s, selected=%s, resolved=%s, device=%s, compute_type=%s",
            implementation,
            selected_model or "(default)",
            resolved_model or "(default)",
            self.device,
            compute_type,
        )

    def log_model_load_complete(
        self,
        implementation: str,
        selected_model: str,
        compute_type: str,
        active_model: Optional[str] = None,
    ) -> None:
        self.live_phase = None
        logger.info(
            "Model loaded: Base Model=%s, selected=%s, active=%s, device=%s, compute_type=%s",
            implementation,
            selected_model or "(default)",
            active_model or self.current_model_size or "(unknown)",
            self.device,
            compute_type,
        )

    def build_live_transcription_heartbeat(self, started_at: float, segment_count: int) -> str:
        elapsed = max(0.0, time.time() - started_at)
        if segment_count > 0:
            return (
                "Still transcribing... "
                f"{segment_count} segment(s) received; waiting for the next update "
                f"({self.format_time(elapsed)} elapsed)."
            )
        if self.live_phase:
            # A download or a model load is not transcription yet
            return f"Still working... {self.live_phase} ({self.format_time(elapsed)} elapsed)."
        return (
            "Still transcribing... waiting for the first segment "
            f"({self.format_time(elapsed)} elapsed)."
        )

    def log_live_heartbeat(self, heartbeat: str) -> None:
        # During a download CMD draws its progress bar; a heartbeat line every 5 seconds broke the bar apart
        if self.live_phase != self.LIVE_PHASE_DOWNLOADING:
            logger.info(heartbeat)

    def get_writer_options(self, whisper_params: WhisperParams) -> dict:
        normalize_word_timestamps = bool(
            whisper_params.word_timestamps
            and self.supports_word_timestamps()
            and whisper_params.normalize_word_timestamps
        )
        return {
            "highlight_words": bool(
                whisper_params.word_timestamps
                and self.supports_word_timestamps()
                and not normalize_word_timestamps
            ),
            "normalize_word_timestamps": normalize_word_timestamps,
        }

    def run(self,
            audio: Union[str, BinaryIO, np.ndarray],
            progress: gr.Progress = gr.Progress(),
            file_format: Union[str, List[str]] = "SRT",
            add_timestamp: bool = True,
            progress_callback: Optional[Callable] = None,
            *pipeline_params,
            ) -> Tuple[List[Segment], float]:
        """Transcribe one input (see _run_unlocked) once no other transcription runs in this process."""
        if not TRANSCRIPTION_LOCK.acquire(blocking=False):
            message = "Waiting for the running transcription to finish.."
            logger.info(message)
            self.ensure_progress_callable(progress)(0, desc=message)
            if progress_callback is not None:
                try:
                    progress_callback(None, None, message)
                except TypeError:
                    pass
            TRANSCRIPTION_LOCK.acquire()
        try:
            return self._run_unlocked(audio, progress, file_format, add_timestamp, progress_callback,
                                      *pipeline_params)
        finally:
            TRANSCRIPTION_LOCK.release()

    def _run_unlocked(self,
                      audio: Union[str, BinaryIO, np.ndarray],
                      progress: gr.Progress = gr.Progress(),
                      file_format: Union[str, List[str]] = "SRT",
                      add_timestamp: bool = True,
                      progress_callback: Optional[Callable] = None,
                      *pipeline_params,
                      ) -> Tuple[List[Segment], float]:
        """
        Run transcription with conditional pre-processing and post-processing.
        The VAD will be performed to remove noise from the audio input in pre-processing, if enabled.
        The diarization will be performed in post-processing, if enabled.
        Due to the integration with gradio, the parameters have to be specified with a `*` wildcard.

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio input. This can be file path or binary type.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        file_format: str
            Subtitle file format between ["SRT", "WebVTT", "txt", "lrc"]
        add_timestamp: bool
            Whether to add a timestamp at the end of the filename.
        progress_callback: Optional[Callable]
            callback function to show progress. Can be used to update progress in the backend.

        *pipeline_params: tuple
            Parameters for the transcription pipeline. This will be dealt with "TranscriptionPipelineParams" data class.
            This must be provided as a List with * wildcard because of the integration with gradio.
            See more info at : https://github.com/gradio-app/gradio/issues/2471

        Returns
        ----------
        segments_result: List[Segment]
            list of Segment that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for running
        """
        progress = self.ensure_progress_callable(progress)
        start_time = time.time()
        self.live_phase = None

        # Log start of transcription with timestamp
        audio_name = audio if isinstance(audio, str) else "audio stream"
        logger.info("\n" + "="*80)
        # Plain text: the classic Windows console cannot show emoji outside the BMP and printed "�" for them
        logger.info("TRANSCRIPTION STARTED")
        logger.info(f"   File: {audio_name}")
        logger.info(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80 + "\n")

        self.last_input_unreadable = False
        if not validate_audio(audio):
            self.last_input_unreadable = True
            # Only CMD showed why; the UI reported "Done! 0 segments" for an unreadable file.
            if progress_callback is not None:
                try:
                    progress_callback(None, None, "⚠️ This file could not be opened (corrupted or unsupported "
                                                  "format), so nothing was transcribed. See CMD for the reason.")
                except TypeError:
                    pass
            return [Segment()], 0

        params = TranscriptionPipelineParams.from_list(list(pipeline_params))
        file_formats = self.normalize_file_formats(file_format)
        primary_file_format = file_formats[0]
        params = self.validate_gradio_values(params)
        bgm_params, vad_params, whisper_params, diarization_params = params.bgm_separation, params.vad, params.whisper, params.diarization
        if is_digital_silence(audio):
            message = "The audio contains only digital silence; no speech was transcribed."
            logger.info(message)
            progress(1.0, desc=message)
            if progress_callback is not None:
                try:
                    progress_callback(1.0, None, message)
                except TypeError:
                    pass
            if not whisper_params.offload_to_ram and whisper_params.enable_offload and not self.defer_offload:
                self.offload()
            return [], time.time() - start_time
        self.prepare_models_for_run(whisper_params)
        self.log_selected_model(
            whisper_type=whisper_params.whisper_type,
            model_size=whisper_params.model_size,
            compute_type=whisper_params.compute_type,
            will_load=self.should_load_model_for_selection(
                whisper_params.model_size,
                whisper_params.compute_type,
            ),
        )

        if bgm_params.is_separate_bgm:
            music, audio, _ = self.music_separator.separate(
                audio=audio,
                model_name=bgm_params.uvr_model_size,
                device=bgm_params.uvr_device,
                segment_size=bgm_params.segment_size,
                save_file=bgm_params.save_file,
                progress=progress
            )

            if audio.ndim >= 2:
                audio = audio.mean(axis=1)
                if self.music_separator.audio_info is None:
                    origin_sample_rate = 16000
                else:
                    origin_sample_rate = self.music_separator.audio_info.sample_rate
                audio = self.resample_audio(audio=audio, original_sample_rate=origin_sample_rate)

            if bgm_params.enable_offload and not self.defer_offload:
                self.music_separator.offload()
            elapsed_time_bgm_sep = time.time() - start_time

        # A reference is enough: VAD and the engines return new arrays and never change the audio in place
        # (the deep copy duplicated up to hundreds of MB after the Background Music Remover)
        origin_audio = audio

        try:
            if vad_params.vad_filter:
                progress(0, desc="Filtering silent parts from audio..")
                vad_options = VadOptions(
                    threshold=vad_params.threshold,
                    min_speech_duration_ms=vad_params.min_speech_duration_ms,
                    max_speech_duration_s=vad_params.max_speech_duration_s,
                    min_silence_duration_ms=vad_params.min_silence_duration_ms,
                    speech_pad_ms=vad_params.speech_pad_ms
                )

                vad_processed, speech_chunks = self.vad.run(
                    audio=audio,
                    vad_parameters=vad_options,
                    progress=progress
                )

                if vad_processed.size == 0:
                    message = "VAD detected no speech; no audio was transcribed."
                    logger.info(message)
                    progress(1.0, desc=message)
                    if progress_callback is not None:
                        try:
                            progress_callback(1.0, None, message)
                        except TypeError:
                            pass
                    return [], time.time() - start_time
                audio = vad_processed

            result, elapsed_time_transcription = self.transcribe(
                audio,
                progress,
                progress_callback,
                *whisper_params.to_list()
            )
        finally:
            # Also release a previously loaded model after no-speech VAD results or decode errors.
            # RAM parking and batches retain their existing job-level cleanup behavior.
            if not whisper_params.offload_to_ram and whisper_params.enable_offload and not self.defer_offload:
                self.offload()

        if vad_params.vad_filter:
            restored_result = self.vad.restore_speech_timestamps(
                segments=result,
                speech_chunks=speech_chunks,
            )
            if restored_result:
                result = restored_result
            else:
                logger.info("VAD detected no speech segments in the audio.")

        if diarization_params.is_diarize:
            progress(0.99, desc="Diarizing speakers..")
            try:
                result, elapsed_time_diarization = self.diarizer.run(
                    audio=origin_audio,
                    use_auth_token=diarization_params.hf_token if diarization_params.hf_token else os.environ.get("HF_TOKEN"),
                    transcribed_result=result,
                    device=diarization_params.diarization_device
                )
                if diarization_params.enable_offload and not whisper_params.offload_to_ram and not self.defer_offload:
                    self.diarizer.offload()
            except Exception as e:
                # Diarization is optional; don't fail the whole transcription if it can't run.
                logger.warning(f"Diarization failed and will be skipped: {type(e).__name__}: {e}")

        if not result:
            logger.info(f"Whisper did not detected any speech segments in the audio.")
            result = [Segment()]

        progress(1.0, desc="Finished.")
        total_elapsed_time = time.time() - start_time
        
        # Log end of transcription with timestamp and duration
        logger.info("\n" + "="*80)
        logger.info("TRANSCRIPTION COMPLETED")
        logger.info(f"   File: {audio_name}")
        logger.info(f"   Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   Duration: {self.format_time(total_elapsed_time)}")
        logger.info(f"   Segments: {len(result)}")
        logger.info("="*80 + "\n")
        
        return result, total_elapsed_time

    def transcribe_file_with_live_output(self,
                        files: Optional[List] = None,
                        batch_mode: bool = False,
                        input_folder_path: Optional[str] = None,
                        include_subdirectory: Optional[bool] = None,
                        overwrite_existing: bool = False,
                        output_dir: Optional[str] = None,
                        file_formats: Union[str, List[str]] = "SRT",
                        add_timestamp: bool = True,
                        progress=gr.Progress(),
                        *pipeline_params,
                        ):
        """
        Transcribe with live output - yields updates as segments are transcribed
        """
        try:
            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            file_formats = self.normalize_file_formats(file_formats)
            writer_options = self.get_writer_options(params.whisper)

            if batch_mode and not input_folder_path:
                raise ValueError("Input folder path is required when batch processing is enabled.")

            if batch_mode and input_folder_path:
                input_folder_path = normalize_folder_path(input_folder_path)
                if not os.path.isdir(input_folder_path):
                    raise ValueError(f"Input folder not found: {input_folder_path}")
                files = get_media_files(input_folder_path, include_sub_directory=include_subdirectory)
            output_dir = normalize_folder_path(output_dir)

            files = self._unique_input_files(self.format_input_files(files))
            if not files:
                raise ValueError("No input files provided for transcription.")

            live_output_lines = deque(maxlen=self.LIVE_TRANSCRIPTION_HISTORY_LINES)
            live_output_lock = Lock()
            collected_paths: List[str] = []

            def append_live_lines(*lines: str) -> str:
                normalized_lines = []
                for line in lines:
                    if line is None:
                        continue
                    normalized_lines.extend(str(line).splitlines() or [""])

                with live_output_lock:
                    live_output_lines.extend(normalized_lines)
                    return "\n".join(live_output_lines)

            used_output_names = set()
            failures: List[str] = []
            with self.batch_offload_scope(len(files), params):
                for file in files:
                    try:
                        target_output_dir = self._get_output_dir_for_file(output_dir, file, batch_mode, input_folder_path)
                        file_name = self._output_name_for_input(file, target_output_dir, used_output_names)
                        output_specs = self._build_output_specs(file_name, file_formats, writer_options)
                        existing_outputs = self._find_existing_outputs(target_output_dir, output_specs)

                        live_output = append_live_lines(f"📂 Processing: {file_name}", "=" * 60, "")
                        yield live_output, "", collected_paths

                        if batch_mode and (not overwrite_existing) and len(existing_outputs) == len(output_specs):
                            skipped_paths = [sorted(paths)[-1] for paths in existing_outputs.values()]
                            collected_paths.extend(skipped_paths)
                            live_output = append_live_lines(
                                f"⏩ Skipped (outputs already exist in {target_output_dir})",
                                "",
                            )
                            result_str = f"Skipped {file_name}: outputs already present."
                            yield live_output, result_str, collected_paths
                            continue

                        segment_count = [0]  # Use list to allow modification in nested function
                        live_update_queue: Queue = Queue()
                        worker_result = {}
                        worker_error = {}

                        def live_progress_callback(progress_value, segment=None, status=None):
                            del progress_value
                            if status:
                                live_update_queue.put(append_live_lines(str(status)))
                            elif segment:
                                segment_count[0] += 1
                                start_time = self.format_timestamp(segment.start) if hasattr(segment, 'start') else "00:00:00.000"
                                end_time = self.format_timestamp(segment.end) if hasattr(segment, 'end') else "00:00:00.000"
                                text = segment.text if hasattr(segment, 'text') else ""
                                live_update_queue.put(append_live_lines(f"[{start_time} → {end_time}] {text}"))

                        def run_with_live_callback():
                            try:
                                transcribed_segments, time_for_task = self.run(
                                    file,
                                    progress,
                                    file_formats[0],
                                    add_timestamp,
                                    live_progress_callback,
                                    *pipeline_params,
                                )
                                worker_result["segments"] = transcribed_segments
                                worker_result["time_for_task"] = time_for_task
                            except Exception as e:
                                worker_error["exception"] = e
                            finally:
                                live_update_queue.put(None)

                        worker_thread = Thread(target=run_with_live_callback, daemon=True)
                        worker_thread.start()
                        heartbeat_started_at = time.time()
                        next_heartbeat_at = heartbeat_started_at + self.LIVE_TRANSCRIPTION_HEARTBEAT_INTERVAL_SEC

                        while True:
                            try:
                                queued_update = live_update_queue.get(timeout=self.LIVE_TRANSCRIPTION_POLL_INTERVAL_SEC)
                            except Empty:
                                if worker_thread.is_alive():
                                    now = time.time()
                                    if now >= next_heartbeat_at:
                                        heartbeat = self.build_live_transcription_heartbeat(
                                            heartbeat_started_at,
                                            segment_count[0],
                                        )
                                        self.log_live_heartbeat(heartbeat)
                                        yield append_live_lines(heartbeat), "", collected_paths
                                        next_heartbeat_at = now + self.LIVE_TRANSCRIPTION_HEARTBEAT_INTERVAL_SEC
                                    continue
                                break

                            latest_update = queued_update
                            saw_sentinel = queued_update is None

                            while True:
                                try:
                                    queued_update = live_update_queue.get_nowait()
                                except Empty:
                                    break

                                if queued_update is None:
                                    saw_sentinel = True
                                    continue

                                latest_update = queued_update

                            if latest_update is not None:
                                yield latest_update, "", collected_paths
                                next_heartbeat_at = time.time() + self.LIVE_TRANSCRIPTION_HEARTBEAT_INTERVAL_SEC

                            if saw_sentinel and not worker_thread.is_alive():
                                break

                        worker_thread.join()

                        if "exception" in worker_error:
                            raise worker_error["exception"]

                        if self.last_input_unreadable:
                            # counted as a failed file (empty subtitle files were written and "Done!" reported)
                            raise ValueError(self.UNREADABLE_INPUT_MESSAGE)
                        transcribed_segments = worker_result["segments"]
                        time_for_task = worker_result["time_for_task"]
                        reported_segment_count = segment_count[0] or self.count_transcribed_segments(transcribed_segments)

                        # Calculate transcription speed
                        if transcribed_segments and len(transcribed_segments) > 0:
                            last_segment = transcribed_segments[-1]
                            audio_duration = last_segment.end if hasattr(last_segment, 'end') and last_segment.end else 0
                            if audio_duration > 0 and time_for_task > 0:
                                speed_ratio = audio_duration / time_for_task
                                append_live_lines(
                                    "",
                                    f"⚡ Speed: {speed_ratio:.2f}x realtime ({self.format_time(audio_duration)} audio in {self.format_time(time_for_task)})",
                                )

                        # Generate final output(s)
                        _, generated_paths = self._write_output_files(
                            output_specs=output_specs,
                            output_dir=target_output_dir,
                            result=transcribed_segments,
                            add_timestamp=add_timestamp,
                            existing_outputs=existing_outputs,
                            batch_mode=batch_mode,
                            overwrite_existing=overwrite_existing,
                        )

                        collected_paths.extend(generated_paths)
                        live_output = append_live_lines(
                            f"✅ Completed in {self.format_time(time_for_task)}",
                            "",
                        )
                        result_str = f"Done! {reported_segment_count} segments in {self.format_time(time_for_task)}. Saved to {target_output_dir}"

                        yield live_output, result_str, collected_paths
                    except Exception as exc:
                        if len(files) == 1:
                            raise
                        # One bad file (corrupt, out of memory) ended the whole batch and cleared the
                        # download list; the other files now go on and the failures are listed at the end
                        logger.error("File '%s' failed: %s: %s", file, type(exc).__name__, exc, exc_info=True)
                        failures.append(f"{os.path.basename(str(file))}: {type(exc).__name__}: {exc}")
                        live_output = append_live_lines(f"❌ Failed: {os.path.basename(str(file))}: {exc}", "")
                        yield live_output, "", collected_paths

            if failures:
                summary = self._batch_failure_summary(failures, len(files))
                yield append_live_lines(summary), summary, collected_paths

        except Exception as e:
            # The UI only gets the message; the traceback goes to CMD so saved console logs show the cause.
            logger.error("Transcription failed: %s: %s", type(e).__name__, e, exc_info=True)
            error_msg = f"❌ Error: {str(e)}"
            yield error_msg, error_msg, []

    def transcribe_file(self,
                        files: Optional[List] = None,
                        batch_mode: bool = False,
                        input_folder_path: Optional[str] = None,
                        include_subdirectory: Optional[bool] = None,
                        overwrite_existing: bool = False,
                        output_dir: Optional[str] = None,
                        file_formats: Union[str, List[str]] = "SRT",
                        add_timestamp: bool = True,
                        progress=gr.Progress(),
                        *pipeline_params,
                        ) -> Tuple[str, List]:
        """
        Write subtitle file from Files

        Parameters
        ----------
        files: list
            List of files to transcribe from gr.Files()
        batch_mode: bool
            Enable batch mode. Requires input_folder_path and processes every media file found.
        input_folder_path: Optional[str]
            Folder path to process. When provided in batch mode, uploaded files are ignored.
        include_subdirectory: Optional[bool]
            Whether to include files in subdirectories when batch mode is enabled.
        overwrite_existing: bool
            When False, existing outputs in the target directory are skipped.
        output_dir: Optional[str]
            Custom output directory. If omitted in batch mode, outputs are saved next to the input files.
        file_formats: Union[str, List[str]]
            One or more subtitle formats to generate.
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the subtitle filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *pipeline_params: tuple
            Parameters for the transcription pipeline. This will be dealt with "TranscriptionPipelineParams" data class

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            file_formats = self.normalize_file_formats(file_formats)
            writer_options = self.get_writer_options(params.whisper)

            if batch_mode and not input_folder_path:
                raise ValueError("Input folder path is required when batch processing is enabled.")

            if batch_mode and input_folder_path:
                input_folder_path = normalize_folder_path(input_folder_path)
                if not os.path.isdir(input_folder_path):
                    raise ValueError(f"Input folder not found: {input_folder_path}")
                files = get_media_files(input_folder_path, include_sub_directory=include_subdirectory)
            output_dir = normalize_folder_path(output_dir)

            files = self._unique_input_files(self.format_input_files(files))
            if not files:
                raise ValueError("No input files provided for transcription.")

            files_info = {}
            all_paths: List[str] = []
            total_time = 0

            used_output_names = set()
            failures: List[str] = []
            with self.batch_offload_scope(len(files), params):
                for file in files:
                    try:
                        target_output_dir = self._get_output_dir_for_file(output_dir, file, batch_mode, input_folder_path)
                        file_name = self._output_name_for_input(file, target_output_dir, used_output_names)
                        output_specs = self._build_output_specs(file_name, file_formats, writer_options)
                        existing_outputs = self._find_existing_outputs(target_output_dir, output_specs)

                        if batch_mode and (not overwrite_existing) and len(existing_outputs) == len(output_specs):
                            skipped_paths = [sorted(paths)[-1] for paths in existing_outputs.values()]
                            all_paths.extend(skipped_paths)
                            files_info[file_name] = {
                                "subtitle": read_file(skipped_paths[0]) if skipped_paths else "",
                                "time_for_task": 0,
                                "paths": skipped_paths,
                                "skipped": True
                            }
                            continue

                        transcribed_segments, time_for_task = self.run(
                            file,
                            progress,
                            file_formats[0],
                            add_timestamp,
                            None,
                            *pipeline_params,
                        )
                        if self.last_input_unreadable:
                            raise ValueError(self.UNREADABLE_INPUT_MESSAGE)

                        subtitle_preview, generated_paths = self._write_output_files(
                            output_specs=output_specs,
                            output_dir=target_output_dir,
                            result=transcribed_segments,
                            add_timestamp=add_timestamp,
                            existing_outputs=existing_outputs,
                            batch_mode=batch_mode,
                            overwrite_existing=overwrite_existing,
                        )

                        all_paths.extend(generated_paths)
                        files_info[file_name] = {
                            "subtitle": subtitle_preview,
                            "time_for_task": time_for_task,
                            "paths": generated_paths,
                            "skipped": False
                        }
                        total_time += time_for_task
                    except Exception as exc:
                        if len(files) == 1:
                            raise
                        # One bad file (corrupt, out of memory) ended the whole batch and cleared the
                        # download list; the other files now go on and the failures are listed at the end
                        logger.error("File '%s' failed: %s: %s", file, type(exc).__name__, exc, exc_info=True)
                        failures.append(f"{os.path.basename(str(file))}: {type(exc).__name__}: {exc}")

            total_result = ''
            for file_name, info in files_info.items():
                total_result += '------------------------------------\n'
                total_result += f'{file_name}\n\n'
                if info["skipped"]:
                    total_result += "Skipped (outputs already exist)\n"
                else:
                    total_result += f'{info["subtitle"]}'

            result_str = f"Done in {self.format_time(total_time)}! Subtitle files saved to selected output folders.\n\n{total_result}"
            if failures:
                result_str = f"{self._batch_failure_summary(failures, len(files))}\n\n{result_str}"
            result_file_path = all_paths

            return result_str, result_file_path

        except Exception as e:
            raise RuntimeError(f"Error transcribing file: {e}") from e

    def transcribe_mic(self,
                       mic_audio: Union[str, dict, object],
                       file_format: Union[str, List[str]] = "SRT",
                       add_timestamp: bool = True,
                       progress=gr.Progress(),
                       *pipeline_params,
                       ) -> Tuple[str, str]:
        """
        Write subtitle file from microphone

        Parameters
        ----------
        mic_audio: Union[str, dict, object]
            Audio input from gr.Microphone(). Accepts filepath-style payloads.
        file_format: str
            Subtitle File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *pipeline_params: tuple
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            mic_audio_path = coerce_audio_input_path(mic_audio)
            if mic_audio_path is None:
                raise ValueError("No microphone audio was provided.")

            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            file_formats = self.normalize_file_formats(file_format)
            writer_options = self.get_writer_options(params.whisper)

            progress(0, desc="Loading Audio..")
            transcribed_segments, time_for_task = self.run(
                mic_audio_path,
                progress,
                file_formats[0],
                add_timestamp,
                None,
                *pipeline_params,
            )
            progress(1, desc="Completed!")

            file_name = safe_filename(os.path.splitext(os.path.basename(mic_audio_path))[0]) or "Mic"
            output_specs = self._build_output_specs(file_name, file_formats, writer_options)
            subtitle_preview, file_paths = self._write_output_files(
                output_specs=output_specs,
                output_dir=self.output_dir,
                result=transcribed_segments,
                add_timestamp=add_timestamp,
            )

            result_file_path = file_paths[0] if len(file_paths) == 1 else file_paths
            result_str = f"Done in {self.format_time(time_for_task)}! Subtitle file is in the outputs folder.\n\n{subtitle_preview}"
            return result_str, result_file_path
        except Exception as e:
            raise RuntimeError(f"Error transcribing mic: {e}") from e

    def transcribe_live_preview(
                       self,
                       audio: np.ndarray,
                       *pipeline_params,
                       ) -> Optional[str]:
        """
        Generate a lightweight live preview transcript from streaming microphone audio.

        This intentionally disables diarization and background-music separation because
        those are too expensive and unstable for short streaming chunks.

        Returns None, without waiting, while another transcription runs: the preview is skipped then.
        """
        if audio is None:
            return ""
        if not TRANSCRIPTION_LOCK.acquire(blocking=False):
            return None
        try:
            return self._transcribe_live_preview_unlocked(audio, *pipeline_params)
        finally:
            TRANSCRIPTION_LOCK.release()

    def _transcribe_live_preview_unlocked(self, audio: np.ndarray, *pipeline_params) -> str:

        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return ""

        if audio.ndim > 1:
            channel_axis = 0 if audio.shape[0] <= audio.shape[-1] else -1
            audio = audio.mean(axis=channel_axis)
        audio = np.ascontiguousarray(audio.squeeze(), dtype=np.float32)

        if is_digital_silence(audio):
            return ""

        params = TranscriptionPipelineParams.from_list(list(pipeline_params))
        params = self.validate_gradio_values(params)
        params.diarization.is_diarize = False
        params.bgm_separation.is_separate_bgm = False
        params.whisper.start_as_subprocess = False
        params.whisper.enable_offload = False
        self.prepare_models_for_run(params.whisper)

        vad_params = params.vad
        audio_to_transcribe = audio
        speech_chunks = None

        if vad_params.vad_filter:
            vad_options = VadOptions(
                threshold=vad_params.threshold,
                min_speech_duration_ms=vad_params.min_speech_duration_ms,
                max_speech_duration_s=vad_params.max_speech_duration_s,
                min_silence_duration_ms=vad_params.min_silence_duration_ms,
                speech_pad_ms=vad_params.speech_pad_ms
            )
            vad_processed, speech_chunks = self.vad.run(
                audio=audio,
                vad_parameters=vad_options,
                progress=gr.Progress(),
            )
            if vad_processed.size > 0:
                audio_to_transcribe = vad_processed
            else:
                return ""

        result, _ = self.transcribe(
            audio_to_transcribe,
            gr.Progress(),
            None,
            *params.whisper.to_list(),
            log_console=True,
            log_model_banner=False,
        )

        if speech_chunks:
            restored_result = self.vad.restore_speech_timestamps(
                segments=result,
                speech_chunks=speech_chunks,
            )
            if restored_result:
                result = restored_result

        preview_lines = []
        for segment in result:
            text = (segment.text or "").strip()
            if not text:
                continue
            preview_lines.append(
                f"[{self.format_timestamp(segment.start)} -> {self.format_timestamp(segment.end)}] {text}"
            )

        return "\n".join(preview_lines)

    def transcribe_mic_with_live_output(
                       self,
                       mic_audio: Union[str, dict, object],
                       file_format: Union[str, List[str]] = "SRT",
                       add_timestamp: bool = True,
                       progress=gr.Progress(),
                       *pipeline_params,
                       ):
        """Transcribe microphone input while streaming segment updates."""
        try:
            mic_audio_path = coerce_audio_input_path(mic_audio)
            if mic_audio_path is None:
                raise ValueError("No microphone audio was provided.")

            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            file_formats = self.normalize_file_formats(file_format)
            writer_options = self.get_writer_options(params.whisper)

            live_output_lines = deque(maxlen=self.LIVE_TRANSCRIPTION_HISTORY_LINES)
            live_output_lock = Lock()
            collected_paths: List[str] = []

            def append_live_lines(*lines: str) -> str:
                normalized_lines = []
                for line in lines:
                    if line is None:
                        continue
                    normalized_lines.extend(str(line).splitlines() or [""])

                with live_output_lock:
                    live_output_lines.extend(normalized_lines)
                    return "\n".join(live_output_lines)

            live_output = append_live_lines("🎤 Processing: Mic", "=" * 60, "")
            yield live_output, "", collected_paths

            segment_count = [0]
            live_update_queue: Queue = Queue()
            worker_result = {}
            worker_error = {}

            def live_progress_callback(progress_value, segment=None, status=None):
                del progress_value
                if status:
                    live_update_queue.put(append_live_lines(str(status)))
                elif segment:
                    segment_count[0] += 1
                    start_time = self.format_timestamp(segment.start) if hasattr(segment, "start") else "00:00:00.000"
                    end_time = self.format_timestamp(segment.end) if hasattr(segment, "end") else "00:00:00.000"
                    text = segment.text if hasattr(segment, "text") else ""
                    live_update_queue.put(append_live_lines(f"[{start_time} → {end_time}] {text}"))

            def run_with_live_callback():
                try:
                    transcribed_segments, time_for_task = self.run(
                        mic_audio_path,
                        progress,
                        file_formats[0],
                        add_timestamp,
                        live_progress_callback,
                        *pipeline_params,
                    )
                    worker_result["segments"] = transcribed_segments
                    worker_result["time_for_task"] = time_for_task
                except Exception as e:
                    worker_error["exception"] = e
                finally:
                    live_update_queue.put(None)

            worker_thread = Thread(target=run_with_live_callback, daemon=True)
            worker_thread.start()
            heartbeat_started_at = time.time()
            next_heartbeat_at = heartbeat_started_at + self.LIVE_TRANSCRIPTION_HEARTBEAT_INTERVAL_SEC

            while True:
                try:
                    queued_update = live_update_queue.get(timeout=self.LIVE_TRANSCRIPTION_POLL_INTERVAL_SEC)
                except Empty:
                    if worker_thread.is_alive():
                        now = time.time()
                        if now >= next_heartbeat_at:
                            heartbeat = self.build_live_transcription_heartbeat(
                                heartbeat_started_at,
                                segment_count[0],
                            )
                            self.log_live_heartbeat(heartbeat)
                            yield append_live_lines(heartbeat), "", collected_paths
                            next_heartbeat_at = now + self.LIVE_TRANSCRIPTION_HEARTBEAT_INTERVAL_SEC
                        continue
                    break

                latest_update = queued_update
                saw_sentinel = queued_update is None

                while True:
                    try:
                        queued_update = live_update_queue.get_nowait()
                    except Empty:
                        break

                    if queued_update is None:
                        saw_sentinel = True
                        continue

                    latest_update = queued_update

                if latest_update is not None:
                    yield latest_update, "", collected_paths
                    next_heartbeat_at = time.time() + self.LIVE_TRANSCRIPTION_HEARTBEAT_INTERVAL_SEC

                if saw_sentinel and not worker_thread.is_alive():
                    break

            worker_thread.join()

            if "exception" in worker_error:
                raise worker_error["exception"]

            transcribed_segments = worker_result["segments"]
            time_for_task = worker_result["time_for_task"]
            reported_segment_count = segment_count[0] or self.count_transcribed_segments(transcribed_segments)

            if transcribed_segments:
                last_segment = transcribed_segments[-1]
                audio_duration = last_segment.end if hasattr(last_segment, "end") and last_segment.end else 0
                if audio_duration > 0 and time_for_task > 0:
                    speed_ratio = audio_duration / time_for_task
                    append_live_lines(
                        "",
                        f"⚡ Speed: {speed_ratio:.2f}x realtime ({self.format_time(audio_duration)} audio in {self.format_time(time_for_task)})",
                    )

            file_name = safe_filename(os.path.splitext(os.path.basename(mic_audio_path))[0]) or "Mic"
            output_specs = self._build_output_specs(file_name, file_formats, writer_options)
            _, generated_paths = self._write_output_files(
                output_specs=output_specs,
                output_dir=self.output_dir,
                result=transcribed_segments,
                add_timestamp=add_timestamp,
            )

            collected_paths.extend(generated_paths)
            live_output = append_live_lines(
                f"✅ Completed in {self.format_time(time_for_task)}",
                "",
            )
            result_str = (
                f"Done! {reported_segment_count} segments in {self.format_time(time_for_task)}. "
                f"Subtitle files saved to {self.output_dir}"
            )

            yield live_output, result_str, collected_paths

        except Exception as e:
            logger.error("Microphone transcription failed: %s: %s", type(e).__name__, e, exc_info=True)
            error_msg = f"❌ Error: {str(e)}"
            yield error_msg, error_msg, []

    def transcribe_youtube(self,
                           youtube_link: str,
                           file_format: Union[str, List[str]] = "SRT",
                           add_timestamp: bool = True,
                           mass_transcribe_channel: bool = False,
                           latest_video_count: int = 100,
                           progress=gr.Progress(),
                           *pipeline_params,
                           ) -> Tuple[str, Union[str, List[str]]]:
        """
        Write subtitle file from Youtube

        Parameters
        ----------
        youtube_link: str
            URL of the Youtube video to transcribe from gr.Textbox()
        file_format: str
            Subtitle File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *pipeline_params: tuple
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            file_formats = self.normalize_file_formats(file_format)
            writer_options = self.get_writer_options(params.whisper)

            if mass_transcribe_channel:
                requested_count = max(1, min(9999, int(latest_video_count or 100)))
                progress(0, desc=f"Loading latest {requested_count} channel videos..")
                videos = get_latest_channel_videos(youtube_link, requested_count)
                if not videos:
                    raise ValueError("No videos were found for the provided YouTube channel.")

                all_paths: List[str] = []
                total_time = 0.0
                successful_titles: List[str] = []
                failed_titles: List[str] = []
                used_output_names: dict[str, int] = {}
                total_videos = len(videos)

                with self.batch_offload_scope(total_videos, params):
                    for index, yt in enumerate(videos, start=1):
                        video_title = getattr(yt, "title", None) or f"Video {index}"
                        progress((index - 1) / total_videos, desc=f"Transcribing {index}/{total_videos}: {video_title}")
                        try:
                            output_file_name = self._unique_output_file_name(
                                safe_filename(video_title),
                                used_output_names,
                            )
                            _, file_paths, time_for_task = self._transcribe_single_youtube_video(
                                yt=yt,
                                file_formats=file_formats,
                                add_timestamp=add_timestamp,
                                progress=progress,
                                pipeline_params=pipeline_params,
                                writer_options=writer_options,
                                output_file_name=output_file_name,
                            )
                            total_time += time_for_task
                            all_paths.extend(file_paths)
                            successful_titles.append(video_title)
                        except Exception as exc:
                            failed_titles.append(f"{video_title}: {exc}")

                progress(1, desc="Completed!")

                result_file_path = all_paths[0] if len(all_paths) == 1 else all_paths
                return self._channel_batch_summary(total_time, total_videos, successful_titles, failed_titles), result_file_path

            progress(0, desc="Loading Audio from Youtube..")
            yt = get_ytdata(youtube_link)
            subtitle_preview, file_paths, time_for_task = self._transcribe_single_youtube_video(
                yt=yt,
                file_formats=file_formats,
                add_timestamp=add_timestamp,
                progress=progress,
                pipeline_params=pipeline_params,
                writer_options=writer_options,
                output_file_name=safe_filename(yt.title),
            )
            progress(1, desc="Completed!")

            result_str = f"Done in {self.format_time(time_for_task)}! Subtitle file is in the outputs folder.\n\n{subtitle_preview}"
            result_file_path = file_paths[0] if len(file_paths) == 1 else file_paths
            return result_str, result_file_path

        except Exception as e:
            raise RuntimeError(f"Error transcribing youtube: {e}") from e

    def _channel_batch_summary(self, total_time: float, total_videos: int,
                               successful_titles: List[str], failed_titles: List[str]) -> str:
        success_count = len(successful_titles)
        if success_count == 0:
            raise RuntimeError(
                "No channel videos were transcribed successfully.\n"
                + "\n".join(failed_titles[:20])
            )

        result_lines = [
            (
                f"Done in {self.format_time(total_time)}! Transcribed "
                f"{success_count}/{total_videos} latest channel videos. "
                "Subtitle files are in the outputs folder."
            ),
            "",
            "Completed:",
        ]
        completed_preview = successful_titles[:20]
        result_lines.extend(f"- {title}" for title in completed_preview)
        if success_count > len(completed_preview):
            result_lines.append(f"- ... and {success_count - len(completed_preview)} more")
        if failed_titles:
            failed_preview = failed_titles[:20]
            result_lines.extend(["", "Failed:"])
            result_lines.extend(f"- {item}" for item in failed_preview)
            if len(failed_titles) > len(failed_preview):
                result_lines.append(f"- ... and {len(failed_titles) - len(failed_preview)} more")
        return "\n".join(result_lines)

    def _run_with_live_updates(self, audio, progress, file_format, add_timestamp, pipeline_params,
                               append_live_lines, collected_paths):
        """Run self.run() in a thread and yield (live_output, "", collected_paths) updates meanwhile.

        Returns (transcribed_segments, time_for_task, segment_count) to the `yield from` caller.
        """
        segment_count = [0]
        live_update_queue: Queue = Queue()
        worker_result = {}
        worker_error = {}

        def live_progress_callback(progress_value, segment=None, status=None):
            del progress_value
            if status:
                live_update_queue.put(append_live_lines(str(status)))
            elif segment:
                segment_count[0] += 1
                start_time = self.format_timestamp(segment.start) if hasattr(segment, "start") else "00:00:00.000"
                end_time = self.format_timestamp(segment.end) if hasattr(segment, "end") else "00:00:00.000"
                text = segment.text if hasattr(segment, "text") else ""
                live_update_queue.put(append_live_lines(f"[{start_time} → {end_time}] {text}"))

        def run_with_live_callback():
            try:
                transcribed_segments, time_for_task = self.run(
                    audio,
                    progress,
                    file_format,
                    add_timestamp,
                    live_progress_callback,
                    *pipeline_params,
                )
                worker_result["segments"] = transcribed_segments
                worker_result["time_for_task"] = time_for_task
            except Exception as e:
                worker_error["exception"] = e
            finally:
                live_update_queue.put(None)

        worker_thread = Thread(target=run_with_live_callback, daemon=True)
        worker_thread.start()
        heartbeat_started_at = time.time()
        next_heartbeat_at = heartbeat_started_at + self.LIVE_TRANSCRIPTION_HEARTBEAT_INTERVAL_SEC

        while True:
            try:
                queued_update = live_update_queue.get(timeout=self.LIVE_TRANSCRIPTION_POLL_INTERVAL_SEC)
            except Empty:
                if worker_thread.is_alive():
                    now = time.time()
                    if now >= next_heartbeat_at:
                        heartbeat = self.build_live_transcription_heartbeat(heartbeat_started_at, segment_count[0])
                        self.log_live_heartbeat(heartbeat)
                        yield append_live_lines(heartbeat), "", collected_paths
                        next_heartbeat_at = now + self.LIVE_TRANSCRIPTION_HEARTBEAT_INTERVAL_SEC
                    continue
                break

            latest_update = queued_update
            saw_sentinel = queued_update is None
            while True:
                try:
                    queued_update = live_update_queue.get_nowait()
                except Empty:
                    break
                if queued_update is None:
                    saw_sentinel = True
                    continue
                latest_update = queued_update

            if latest_update is not None:
                yield latest_update, "", collected_paths
                next_heartbeat_at = time.time() + self.LIVE_TRANSCRIPTION_HEARTBEAT_INTERVAL_SEC

            if saw_sentinel and not worker_thread.is_alive():
                break

        worker_thread.join()
        if "exception" in worker_error:
            raise worker_error["exception"]

        transcribed_segments = worker_result["segments"]
        reported_segment_count = segment_count[0] or self.count_transcribed_segments(transcribed_segments)
        return transcribed_segments, worker_result["time_for_task"], reported_segment_count

    def transcribe_youtube_with_live_output(self,
                                            youtube_link: str,
                                            file_format: Union[str, List[str]] = "SRT",
                                            add_timestamp: bool = True,
                                            mass_transcribe_channel: bool = False,
                                            latest_video_count: int = 100,
                                            progress=gr.Progress(),
                                            *pipeline_params):
        """Transcribe a YouTube video, or the latest videos of a channel, while streaming segment updates."""
        try:
            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            file_formats = self.normalize_file_formats(file_format)
            writer_options = self.get_writer_options(params.whisper)

            live_output_lines = deque(maxlen=self.LIVE_TRANSCRIPTION_HISTORY_LINES)
            live_output_lock = Lock()
            collected_paths: List[str] = []

            def append_live_lines(*lines: str) -> str:
                normalized_lines = []
                for line in lines:
                    if line is None:
                        continue
                    normalized_lines.extend(str(line).splitlines() or [""])
                with live_output_lock:
                    live_output_lines.extend(normalized_lines)
                    return "\n".join(live_output_lines)

            if mass_transcribe_channel:
                requested_count = max(1, min(9999, int(latest_video_count or 100)))
                yield append_live_lines(f"📺 Loading the latest {requested_count} channel videos.."), "", collected_paths
                videos = get_latest_channel_videos(youtube_link, requested_count)
                if not videos:
                    raise ValueError("No videos were found for the provided YouTube channel.")
            else:
                yield append_live_lines("📺 Loading the YouTube video.."), "", collected_paths
                videos = [get_ytdata(youtube_link)]

            total_videos = len(videos)
            used_output_names: dict[str, int] = {}
            successful_titles: List[str] = []
            failed_titles: List[str] = []
            total_time = 0.0
            subtitle_preview = ""

            with self.batch_offload_scope(total_videos, params):
                for index, yt in enumerate(videos, start=1):
                    video_title = getattr(yt, "title", None) or f"Video {index}"
                    counter = f"[{index}/{total_videos}] " if mass_transcribe_channel else ""
                    yield append_live_lines(
                        f"📺 {counter}Processing: {video_title}",
                        "=" * 60,
                        "",
                        "Downloading the audio from YouTube..",
                    ), "", collected_paths
                    audio = None
                    try:
                        audio = get_ytaudio(yt)
                        if not audio:
                            raise RuntimeError("Failed to download or convert the YouTube audio stream.")
                        transcribed_segments, time_for_task, segment_count = yield from self._run_with_live_updates(
                            audio, progress, file_formats[0], add_timestamp, pipeline_params,
                            append_live_lines, collected_paths,
                        )
                        output_file_name = self._unique_output_file_name(safe_filename(video_title), used_output_names)
                        output_specs = self._build_output_specs(output_file_name, file_formats, writer_options)
                        subtitle_preview, file_paths = self._write_output_files(
                            output_specs=output_specs,
                            output_dir=self.output_dir,
                            result=transcribed_segments,
                            add_timestamp=add_timestamp,
                        )
                        collected_paths.extend(file_paths)
                        total_time += time_for_task
                        successful_titles.append(video_title)
                        yield append_live_lines(
                            f"✅ Completed in {self.format_time(time_for_task)} ({segment_count} segments)",
                            "",
                        ), "", collected_paths
                    except Exception as exc:
                        if not mass_transcribe_channel:
                            raise
                        logger.error("YouTube video '%s' failed: %s: %s", video_title, type(exc).__name__, exc)
                        failed_titles.append(f"{video_title}: {exc}")
                        yield append_live_lines(f"❌ Failed: {exc}", ""), "", collected_paths
                    finally:
                        remove_ytaudio(audio)

            if mass_transcribe_channel:
                result_str = self._channel_batch_summary(total_time, total_videos, successful_titles, failed_titles)
            else:
                result_str = f"Done in {self.format_time(total_time)}! Subtitle file is in the outputs folder.\n\n{subtitle_preview}"
            yield "\n".join(live_output_lines), result_str, collected_paths

        except Exception as e:
            logger.error("YouTube transcription failed: %s: %s", type(e).__name__, e, exc_info=True)
            error_msg = f"❌ Error: {str(e)}"
            yield error_msg, error_msg, []

    @staticmethod
    def _batch_failure_summary(failures: List[str], file_count: int) -> str:
        listed = "; ".join(failures[:5]) + (" ..." if len(failures) > 5 else "")
        return f"⚠️ {len(failures)} of {file_count} files failed (full errors in CMD): {listed}"

    @staticmethod
    def _unique_input_files(files: List[str]) -> List[str]:
        """Each input once, in the given order (a file listed twice was transcribed twice)."""
        seen = set()
        unique = []
        for file in files:
            key = os.path.normcase(os.path.abspath(str(file)))
            if key not in seen:
                seen.add(key)
                unique.append(file)
        return unique

    @staticmethod
    def _output_name_for_input(input_file: str, target_output_dir: str, used_output_names: set) -> str:
        """Output name of one input of a job: its file name without extension, plus the extension when another
        input of the same job already writes that name into the same folder (talk.mp4 and talk.wav shared
        talk.srt, and the second one was skipped as "outputs already exist" or overwrote the first)."""
        stem, extension = os.path.splitext(os.path.basename(str(input_file)))
        folder = os.path.normcase(os.path.abspath(target_output_dir))
        candidates = [stem]
        if extension:
            candidates.append(f"{stem}_{extension.lstrip('.')}")
        candidates.extend(f"{stem}_{number}" for number in range(2, 10000))
        for candidate in candidates:
            name = safe_filename(candidate)
            key = (folder, os.path.normcase(name))
            if key not in used_output_names:
                used_output_names.add(key)
                return name
        return safe_filename(stem)

    @staticmethod
    def _unique_output_file_name(base_name: str, used_output_names: dict[str, int]) -> str:
        count = used_output_names.get(base_name, 0) + 1
        used_output_names[base_name] = count
        return base_name if count == 1 else f"{base_name}_{count}"

    def _transcribe_single_youtube_video(
        self,
        yt,
        file_formats: List[str],
        add_timestamp: bool,
        progress,
        pipeline_params,
        writer_options: Optional[dict],
        output_file_name: str,
    ) -> Tuple[str, List[str], float]:
        audio = None
        try:
            audio = get_ytaudio(yt)
            if not audio:
                raise RuntimeError("Failed to download or convert the YouTube audio stream.")

            transcribed_segments, time_for_task = self.run(
                audio,
                progress,
                file_formats[0],
                add_timestamp,
                None,
                *pipeline_params,
            )

            output_specs = self._build_output_specs(output_file_name, file_formats, writer_options)
            subtitle_preview, file_paths = self._write_output_files(
                output_specs=output_specs,
                output_dir=self.output_dir,
                result=transcribed_segments,
                add_timestamp=add_timestamp,
            )
            return subtitle_preview, file_paths, time_for_task
        finally:
            remove_ytaudio(audio)

    @staticmethod
    def normalize_file_formats(file_formats: Union[str, List[str], None]) -> List[str]:
        """Normalize a file format value coming from the UI to a non-empty list."""
        if file_formats is None:
            return ["SRT"]
        if isinstance(file_formats, str):
            file_formats = [file_formats]
        cleaned = []
        for fmt in file_formats:
            if not fmt:
                continue
            cleaned.append(fmt.strip())
        return cleaned or ["SRT"]

    @staticmethod
    def _normalize_format_key(file_format: str) -> str:
        """Convert format label to a normalized key for lookups."""
        normalized = file_format.strip().lower().replace(".", "")
        return "vtt" if normalized == "webvtt" else normalized

    def _get_output_dir_for_file(self, output_dir: Optional[str], input_file: str, batch_mode: bool,
                                 input_root: Optional[str] = None) -> str:
        """Decide which output directory to use for a given file.

        A batch with an output folder mirrors the input subfolders into it: written flat, two inputs with
        the same name in different subfolders shared one output, and the second was skipped as "already exists".
        """
        if output_dir and batch_mode and input_root and input_file:
            try:
                relative_dir = os.path.relpath(os.path.dirname(os.path.abspath(input_file)), os.path.abspath(input_root))
            except ValueError:  # different drive
                relative_dir = "."
            if relative_dir != "." and not relative_dir.startswith(".."):
                output_dir = os.path.join(output_dir, relative_dir)
        target_output_dir = output_dir or (os.path.dirname(input_file) if batch_mode and input_file else None) or self.output_dir
        os.makedirs(target_output_dir, exist_ok=True)
        return target_output_dir

    def _build_output_specs(self, file_name: str, file_formats: List[str], writer_options: Optional[dict] = None) -> List[dict]:
        """Build concrete output specs, including the plain SRT companion file when word timestamps are enabled."""
        output_specs = []
        writer_options = dict(writer_options or {})
        include_no_word_srt = bool(writer_options.get("highlight_words"))

        for fmt in file_formats:
            normalized_format = self._normalize_format_key(fmt)
            output_specs.append({
                "lookup_key": normalized_format,
                "output_format": fmt,
                "output_file_name": file_name,
                "writer_options": dict(writer_options),
            })

            if include_no_word_srt and normalized_format == "srt":
                output_specs.append({
                    "lookup_key": f"{normalized_format}{self.NO_WORD_TIMESTAMPS_SUFFIX}",
                    "output_format": "srt",
                    "output_file_name": f"{file_name}{self.NO_WORD_TIMESTAMPS_SUFFIX}",
                    "writer_options": {**writer_options, "highlight_words": False},
                })

        return output_specs

    def _find_existing_outputs(self, output_dir: str, output_specs: List[dict]) -> dict:
        """Find already generated outputs for each requested output spec (supports timestamped filenames)."""
        existing = {}
        for spec in output_specs:
            normalized_extension = self._normalize_format_key(spec["output_format"])
            pattern = os.path.join(output_dir, f"*.{normalized_extension}")
            matches = []
            for match in glob.glob(pattern):
                stem = os.path.splitext(os.path.basename(match))[0]
                remainder = stem[len(spec["output_file_name"]):] if stem.startswith(spec["output_file_name"]) else None
                if remainder is None:
                    continue
                # Only a run timestamp counts ("-" + 16 digits, 10 digits in older versions): any number did, so
                # lecture-2.srt was taken for the output of lecture.mp4 and lecture.mp4 was skipped
                if remainder == "" or (remainder.startswith("-") and remainder[1:].isdigit() and len(remainder) - 1 in (10, 16)):
                    matches.append(match)
            if matches:
                existing[spec["lookup_key"]] = matches
        return existing

    def _write_output_files(self,
                            output_specs: List[dict],
                            output_dir: str,
                            result: Union[dict, List[Segment]],
                            add_timestamp: bool = True,
                            existing_outputs: Optional[dict] = None,
                            batch_mode: bool = False,
                            overwrite_existing: bool = False) -> Tuple[str, List[str]]:
        """Write or reuse all requested output files and return the preview text plus saved paths."""
        existing_outputs = existing_outputs or {}
        subtitle_preview = ""
        generated_paths = []
        # One suffix for the whole job, so every format of the same run shares a file name.
        timestamp = datetime.now().strftime("%m%d%H%M%S%f") if add_timestamp else ""

        for spec in output_specs:
            lookup_key = spec["lookup_key"]
            if batch_mode and (not overwrite_existing) and lookup_key in existing_outputs:
                file_path = sorted(existing_outputs[lookup_key])[-1]
                subtitle_preview = subtitle_preview or read_file(file_path)
            else:
                output_file_name = spec["output_file_name"]
                if timestamp:
                    output_file_name = f"{output_file_name}-{timestamp}"
                subtitle, file_path = generate_file(
                    output_dir=output_dir,
                    output_file_name=output_file_name,
                    output_format=spec["output_format"],
                    result=result,
                    add_timestamp=False,
                    **spec["writer_options"],
                )
                subtitle_preview = subtitle_preview or subtitle
            generated_paths.append(file_path)

        return subtitle_preview, generated_paths

    @staticmethod
    def format_input_files(files: Optional[Union[str, List]]) -> List[str]:
        """Normalize the files input from Gradio into a list of paths."""
        if files is None:
            return []
        if isinstance(files, str):
            return [files]
        if isinstance(files, list) and files and isinstance(files[0], gr.utils.NamedString):
            return [file.name for file in files]
        return files

    @staticmethod
    def count_transcribed_segments(segments: Optional[List[Segment]]) -> int:
        if not segments:
            return 0
        return sum(1 for segment in segments if str(getattr(segment, "text", "") or "").strip())

    def get_compute_type(self):
        if "bfloat16" in self.available_compute_types:
            return "bfloat16"
        if "float16" in self.available_compute_types:
            return "float16"
        if "float32" in self.available_compute_types:
            return "float32"
        else:
            return self.available_compute_types[0]

    def get_available_compute_type(self):
        if self.device == "cuda":
            return list(ctranslate2.get_supported_compute_types("cuda"))
        else:
            return list(ctranslate2.get_supported_compute_types("cpu"))

    def model_to_device(self, device: str) -> None:
        """Move the loaded model between the GPU and system RAM ("cpu"). Backends override this."""
        raise NotImplementedError(f"{type(self).__name__} cannot move its model between devices")

    def prepare_models_for_run(self, whisper_params) -> None:
        """Bring a RAM-parked model back to the GPU, or drop it when a different model is selected."""
        if self.model is None or not self.models_in_ram:
            return
        if self.should_load_model_for_selection(whisper_params.model_size, whisper_params.compute_type):
            logger.info("Releasing the model parked in RAM: a different model or compute type is selected.")
            self.offload()
            return
        started = time.time()
        self.model_to_device(self.device)
        self.models_in_ram = False
        logger.info("Moved the model from RAM back to %s in %.1f s (no reload from disk).", self.device,
                    time.time() - started)

    def move_models_to_ram(self) -> None:
        """Park the loaded models in system RAM and release their VRAM until the next job.

        Skipped while a transcription runs: a cancelled job's thread can still be using the model, and this can be
        called from Gradio's event loop when it closes that job."""
        if not TRANSCRIPTION_LOCK.acquire(blocking=False):
            logger.info("The models stay loaded: a transcription is still running.")
            return
        try:
            self._move_models_to_ram_unlocked()
        finally:
            TRANSCRIPTION_LOCK.release()

    def _move_models_to_ram_unlocked(self) -> None:
        started = time.time()
        free_before = self._cuda_free_mb()
        moved = []
        if self.model is not None and not self.models_in_ram:
            try:
                self.model_to_device("cpu")
                self.models_in_ram = True
                moved.append("transcription model")
            except Exception as exc:
                logger.warning("Could not move the transcription model to RAM (%s: %s); unloading it instead.",
                               type(exc).__name__, exc)
                self.offload()
        if self.diarizer.move_to_ram():
            moved.append("diarization pipeline")
        # The UVR separator is a small ONNX session that cannot change device; it reloads from disk.
        self.music_separator.offload()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        free_after = self._cuda_free_mb()
        if moved:
            freed = "" if free_before is None else f"; GPU free memory {free_before:.0f} -> {free_after:.0f} MB"
            logger.info("Offloaded %s to RAM in %.1f s%s.", " and ".join(moved), time.time() - started, freed)

    @staticmethod
    def _cuda_free_mb():
        if not torch.cuda.is_available():
            return None
        try:
            free, _total = torch.cuda.mem_get_info()
            return free / (1024 ** 2)
        except Exception:
            return None

    def release_model_before_load(self) -> None:
        """Unload the current model before another one loads. The old model stayed in VRAM until the new one was
        ready, so switching models or compute types needed room for both and could run out of VRAM."""
        if self.model is not None:
            logger.info("Unloading the current model before loading the selected one.")
            self.offload()

    def offload(self):
        """Offload the model and free up the memory"""
        self.models_in_ram = False
        if self.model is not None:
            del self.model
            self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
        if self.device == "xpu":
            torch.xpu.empty_cache()
            torch.xpu.reset_accumulated_memory_stats()
            torch.xpu.reset_peak_memory_stats()
        gc.collect()

    @contextmanager
    def batch_offload_scope(self, item_count: int, params: TranscriptionPipelineParams):
        """Keep the models loaded between the files (or channel videos) of one batch, offload once at the end.

        run() offloads after every file, so a batch reloaded the transcription model from disk for each
        file, and the UVR and diarization models too when they were enabled.
        """
        if item_count <= 1:
            yield
            return
        self.defer_offload = True
        try:
            yield
        finally:
            self.defer_offload = False
            # The offloads run() skipped, with the same conditions; not while a transcription still runs (the
            # batch was cancelled and its current file continues in a background thread)
            if TRANSCRIPTION_LOCK.acquire(blocking=False):
                try:
                    if params.bgm_separation.is_separate_bgm and params.bgm_separation.enable_offload:
                        self.music_separator.offload()
                    if not params.whisper.offload_to_ram:
                        if params.whisper.enable_offload:
                            self.offload()
                        if params.diarization.is_diarize and params.diarization.enable_offload:
                            self.diarizer.offload()
                finally:
                    TRANSCRIPTION_LOCK.release()

    @staticmethod
    def format_time(elapsed_time: float) -> str:
        """
        Get {hours} {minutes} {seconds} time format string

        Parameters
        ----------
        elapsed_time: str
            Elapsed time for transcription

        Returns
        ----------
        Time format string
        """
        hours, rem = divmod(int(round(elapsed_time)), 3600)
        minutes, seconds = divmod(rem, 60)

        def unit(value, name):
            return f"{value} {name}" if value == 1 else f"{value} {name}s"

        parts = []
        if hours:
            parts.append(unit(hours, "hour"))
        if minutes:
            parts.append(unit(minutes, "minute"))
        parts.append(unit(seconds, "second"))
        return " ".join(parts)
    
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Format seconds to HH:MM:SS.mmm timestamp"""
        if seconds is None:
            return "00:00:00.000"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        if torch.xpu.is_available():
            return "xpu"
        elif torch.backends.mps.is_available():
            if not BaseTranscriptionPipeline.is_sparse_api_supported():
                # Device `SparseMPS` is not supported for now. See : https://github.com/pytorch/pytorch/issues/87886
                return "cpu"
            return "mps"
        else:
            return "cpu"

    @staticmethod
    def is_sparse_api_supported():
        if not torch.backends.mps.is_available():
            return False

        try:
            device = torch.device("mps")
            sparse_tensor = torch.sparse_coo_tensor(
                indices=torch.tensor([[0, 1], [2, 3]]),
                values=torch.tensor([1, 2]),
                size=(4, 4),
                device=device
            )
            return True
        except RuntimeError:
            return False

    @staticmethod
    def remove_input_files(file_paths: List[str]):
        """Remove gradio cached files"""
        if not file_paths:
            return

        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)

    @staticmethod
    def validate_gradio_values(params: TranscriptionPipelineParams):
        """
        Validate gradio specific values that can't be displayed as None in the UI.
        Related issue : https://github.com/gradio-app/gradio/issues/8723
        """
        params.whisper.lang = WhisperParams.normalize_lang_value(params.whisper.lang)

        if params.whisper.initial_prompt == GRADIO_NONE_STR:
            params.whisper.initial_prompt = None
        if params.whisper.prefix == GRADIO_NONE_STR:
            params.whisper.prefix = None
        if params.whisper.hotwords == GRADIO_NONE_STR:
            params.whisper.hotwords = None
        if params.whisper.max_new_tokens == GRADIO_NONE_NUMBER_MIN:
            params.whisper.max_new_tokens = None
        if params.whisper.hallucination_silence_threshold == GRADIO_NONE_NUMBER_MIN:
            params.whisper.hallucination_silence_threshold = None
        # 0 is a valid threshold (accept the first detection); turning it into None made faster-whisper
        # compare the probability with None and fail before transcribing
        if params.whisper.language_detection_threshold is None:
            params.whisper.language_detection_threshold = 0.5
        if params.vad.max_speech_duration_s == GRADIO_NONE_NUMBER_MAX:
            params.vad.max_speech_duration_s = float('inf')
        return params

    @staticmethod
    def cache_parameters(
        params: TranscriptionPipelineParams,
        file_format: str = "SRT",
        add_timestamp: bool = True
    ):
        """Runtime parameter caching is disabled; presets are saved explicitly from the UI."""
        return None

    @staticmethod
    def resample_audio(audio: Union[str, np.ndarray],
                       new_sample_rate: int = 16000,
                       original_sample_rate: Optional[int] = None,) -> np.ndarray:
        """Resamples audio to 16k sample rate, standard on Whisper model"""
        if isinstance(audio, str):
            audio, original_sample_rate = torchaudio.load(audio)
        else:
            if original_sample_rate is None:
                raise ValueError("original_sample_rate must be provided when audio is numpy array.")
            audio = torch.from_numpy(audio)
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=new_sample_rate)
        resampled_audio = resampler(audio).numpy()
        return resampled_audio
