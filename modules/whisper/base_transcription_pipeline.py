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
from modules.utils.youtube_manager import get_ytdata, get_ytaudio
from modules.utils.files_manager import get_media_files, format_gradio_files, read_file
from modules.utils.audio_manager import validate_audio
from modules.whisper.data_classes import *
from modules.diarize.diarizer import Diarizer
from modules.vad.silero_vad import SileroVAD


logger = get_logger()


class BaseTranscriptionPipeline(ABC):
    LIVE_TRANSCRIPTION_HISTORY_LINES = 30
    LIVE_TRANSCRIPTION_POLL_INTERVAL_SEC = 0.1
    NO_WORD_TIMESTAMPS_SUFFIX = "_noword_timestaps"

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

    def run(self,
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
        start_time = time.time()
        
        # Log start of transcription with timestamp
        audio_name = audio if isinstance(audio, str) else "audio stream"
        logger.info("\n" + "="*80)
        logger.info(f"🎬 TRANSCRIPTION STARTED")
        logger.info(f"   File: {audio_name}")
        logger.info(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80 + "\n")

        if not validate_audio(audio):
            return [Segment()], 0

        params = TranscriptionPipelineParams.from_list(list(pipeline_params))
        file_formats = self.normalize_file_formats(file_format)
        primary_file_format = file_formats[0]
        params = self.validate_gradio_values(params)
        bgm_params, vad_params, whisper_params, diarization_params = params.bgm_separation, params.vad, params.whisper, params.diarization

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

            if bgm_params.enable_offload:
                self.music_separator.offload()
            elapsed_time_bgm_sep = time.time() - start_time

        origin_audio = deepcopy(audio)

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

            if vad_processed.size > 0:
                audio = vad_processed
            else:
                vad_params.vad_filter = False

        result, elapsed_time_transcription = self.transcribe(
            audio,
            progress,
            progress_callback,
            *whisper_params.to_list()
        )
        if whisper_params.enable_offload:
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
                if diarization_params.enable_offload:
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
        logger.info(f"✅ TRANSCRIPTION COMPLETED")
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
            writer_options = {"highlight_words": True if params.whisper.word_timestamps else False}

            if batch_mode and not input_folder_path:
                raise ValueError("Input folder path is required when batch processing is enabled.")

            if batch_mode and input_folder_path:
                files = get_media_files(input_folder_path, include_sub_directory=include_subdirectory)

            files = self.format_input_files(files)
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

            for file in files:
                file_name = safe_filename(os.path.splitext(os.path.basename(file))[0])
                target_output_dir = self._get_output_dir_for_file(output_dir, file, batch_mode)
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

                def live_progress_callback(progress_value, segment=None):
                    if segment:
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

                while True:
                    try:
                        queued_update = live_update_queue.get(timeout=self.LIVE_TRANSCRIPTION_POLL_INTERVAL_SEC)
                    except Empty:
                        if worker_thread.is_alive():
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

                    if saw_sentinel and not worker_thread.is_alive():
                        break

                worker_thread.join()

                if "exception" in worker_error:
                    raise worker_error["exception"]

                transcribed_segments = worker_result["segments"]
                time_for_task = worker_result["time_for_task"]

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
                result_str = f"Done! {segment_count[0]} segments in {self.format_time(time_for_task)}. Saved to {target_output_dir}"

                yield live_output, result_str, collected_paths

        except Exception as e:
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
            writer_options = {
                "highlight_words": True if params.whisper.word_timestamps else False
            }

            if batch_mode and not input_folder_path:
                raise ValueError("Input folder path is required when batch processing is enabled.")

            if batch_mode and input_folder_path:
                files = get_media_files(input_folder_path, include_sub_directory=include_subdirectory)

            files = self.format_input_files(files)
            if not files:
                raise ValueError("No input files provided for transcription.")

            files_info = {}
            all_paths: List[str] = []
            total_time = 0

            for file in files:
                file_name = safe_filename(os.path.splitext(os.path.basename(file))[0])
                target_output_dir = self._get_output_dir_for_file(output_dir, file, batch_mode)
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

            total_result = ''
            for file_name, info in files_info.items():
                total_result += '------------------------------------\n'
                total_result += f'{file_name}\n\n'
                if info["skipped"]:
                    total_result += "Skipped (outputs already exist)\n"
                else:
                    total_result += f'{info["subtitle"]}'

            result_str = f"Done in {self.format_time(total_time)}! Subtitle files saved to selected output folders.\n\n{total_result}"
            result_file_path = all_paths

            return result_str, result_file_path

        except Exception as e:
            raise RuntimeError(f"Error transcribing file: {e}") from e

    def transcribe_mic(self,
                       mic_audio: str,
                       file_format: Union[str, List[str]] = "SRT",
                       add_timestamp: bool = True,
                       progress=gr.Progress(),
                       *pipeline_params,
                       ) -> Tuple[str, str]:
        """
        Write subtitle file from microphone

        Parameters
        ----------
        mic_audio: str
            Audio file path from gr.Microphone()
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
            writer_options = {
                "highlight_words": True if params.whisper.word_timestamps else False
            }

            progress(0, desc="Loading Audio..")
            transcribed_segments, time_for_task = self.run(
                mic_audio,
                progress,
                file_formats[0],
                add_timestamp,
                None,
                *pipeline_params,
            )
            progress(1, desc="Completed!")

            file_name = "Mic"
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

    def transcribe_youtube(self,
                           youtube_link: str,
                           file_format: Union[str, List[str]] = "SRT",
                           add_timestamp: bool = True,
                           progress=gr.Progress(),
                           *pipeline_params,
                           ) -> Tuple[str, str]:
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
            writer_options = {
                "highlight_words": True if params.whisper.word_timestamps else False
            }

            progress(0, desc="Loading Audio from Youtube..")
            yt = get_ytdata(youtube_link)
            audio = get_ytaudio(yt)

            transcribed_segments, time_for_task = self.run(
                audio,
                progress,
                file_formats[0],
                add_timestamp,
                None,
                *pipeline_params,
            )

            progress(1, desc="Completed!")

            file_name = safe_filename(yt.title)
            output_specs = self._build_output_specs(file_name, file_formats, writer_options)
            subtitle_preview, file_paths = self._write_output_files(
                output_specs=output_specs,
                output_dir=self.output_dir,
                result=transcribed_segments,
                add_timestamp=add_timestamp,
            )

            result_str = f"Done in {self.format_time(time_for_task)}! Subtitle file is in the outputs folder.\n\n{subtitle_preview}"

            if os.path.exists(audio):
                os.remove(audio)

            result_file_path = file_paths[0] if len(file_paths) == 1 else file_paths
            return result_str, result_file_path

        except Exception as e:
            raise RuntimeError(f"Error transcribing youtube: {e}") from e

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

    def _get_output_dir_for_file(self, output_dir: Optional[str], input_file: str, batch_mode: bool) -> str:
        """Decide which output directory to use for a given file."""
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
                if remainder == "" or (remainder.startswith("-") and remainder[1:].isdigit()):
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

        for spec in output_specs:
            lookup_key = spec["lookup_key"]
            if batch_mode and (not overwrite_existing) and lookup_key in existing_outputs:
                file_path = sorted(existing_outputs[lookup_key])[-1]
                subtitle_preview = subtitle_preview or read_file(file_path)
            else:
                subtitle, file_path = generate_file(
                    output_dir=output_dir,
                    output_file_name=spec["output_file_name"],
                    output_format=spec["output_format"],
                    result=result,
                    add_timestamp=add_timestamp,
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

    def offload(self):
        """Offload the model and free up the memory"""
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
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        time_str = ""
        if hours:
            time_str += f"{hours} hours "
        if minutes:
            time_str += f"{minutes} minutes "
        seconds = round(seconds)
        time_str += f"{seconds} seconds"

        return time_str.strip()
    
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
        if params.whisper.language_detection_threshold == GRADIO_NONE_NUMBER_MIN:
            params.whisper.language_detection_threshold = None
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
