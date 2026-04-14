import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from modules.utils.cuda_runtime import enable_cuda_runtime_autodiscovery

enable_cuda_runtime_autodiscovery()

import gradio as gr
import numpy as np

from modules.translation.deepl_api import DeepLAPI
from modules.runtime.subprocess_client import build_runtime_proxies
from modules.ui.htmls import *
from modules.ui.presets import (
    build_default_ui_config,
    clear_last_used_ui_preset,
    delete_ui_preset,
    get_default_startup_ui_preset,
    get_last_used_ui_preset,
    get_nested_value,
    is_locked_ui_preset,
    list_ui_presets,
    load_ui_preset,
    merge_ui_config,
    save_ui_preset,
    sanitize_preset_name,
    set_nested_value,
    set_last_used_ui_preset,
)
from modules.utils.cli_manager import str2bool
from modules.utils.files_manager import MEDIA_EXTENSION, is_video, load_yaml
from modules.utils.i18n import Translate, _
from modules.utils.logger import get_logger
from modules.utils.audio_manager import coerce_audio_input_path
from modules.utils.paths import (
    DEFAULT_PARAMETERS_CONFIG_PATH,
    DIARIZATION_MODELS_DIR,
    FASTER_WHISPER_MODELS_DIR,
    I18N_YAML_PATH,
    INSANELY_FAST_WHISPER_MODELS_DIR,
    NLLB_MODELS_DIR,
    OUTPUT_DIR,
    UVR_MODELS_DIR,
    WHISPER_MODELS_DIR,
)
from modules.utils.text import repair_blocks_text, repair_component_text
from modules.utils.youtube_manager import get_ytmetas
from modules.whisper.data_classes import *


logger = get_logger()

FAVICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "favicon.svg")
APP_TITLE = "Whisper TTS Premium App by SECourses V7.1 : https://www.patreon.com/posts/whisper-webui-to-145395299"
TIMESTAMP_INFO = (
    "Adds the current date and time to the output filename. "
    "Enable this if you want each run to create a unique file and avoid overwriting older outputs. "
    "Disable it if you want shorter, stable filenames."
)
BATCH_SIZE_CALIBRATION_FILENAME = "batch_size_calibration.json"
BATCH_SIZE_CALIBRATION_MEMORY_TOLERANCE_GB = 1.0


class App:
    LIVE_MIC_SAMPLE_RATE = 16000
    LIVE_MIC_MIN_PREVIEW_SECONDS = 2.0
    LIVE_MIC_REFRESH_SECONDS = 2.0
    LIVE_MIC_MAX_BUFFER_SECONDS = 15.0

    def __init__(self, args):
        self.args = args
        self.title = APP_TITLE
        self.app = gr.Blocks(
            title=self.title,
            delete_cache=(3600, 86400),
        )
        self.whisper_inf, self.nllb_inf = build_runtime_proxies(self.args)
        self.deepl_api = DeepLAPI(
            output_dir=os.path.join(self.args.output_dir, "translations"),
        )
        self.i18n = load_yaml(I18N_YAML_PATH)
        self.default_params = self.apply_dynamic_batch_size_defaults(
            load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
        )
        self.ui_default_config = build_default_ui_config(default_params=self.default_params)

        user_allowed = []
        if self.args.allowed_paths:
            try:
                user_allowed = list(eval(self.args.allowed_paths))
            except Exception:
                user_allowed = []

        combined_paths = self.detect_allowed_paths() + user_allowed + [self.args.output_dir]
        seen = set()
        self.allowed_paths = []
        for path in combined_paths:
            if path and path not in seen:
                self.allowed_paths.append(path)
                seen.add(path)

        logger.info(
            f'Use "{self.args.whisper_type}" implementation\n'
            f'Device "{self.whisper_inf.device}" is detected'
        )

    @staticmethod
    def detect_allowed_paths():
        paths = []
        if os.name == "nt":
            for code in range(ord("A"), ord("Z") + 1):
                drive = f"{chr(code)}:\\"
                if os.path.exists(drive):
                    paths.append(drive)
        else:
            paths.append("/")
        if not paths:
            paths.append(os.getcwd())
        return paths

    def get_batch_size_calibration_path(self):
        return os.path.join(self.args.output_dir, BATCH_SIZE_CALIBRATION_FILENAME)

    def load_batch_size_calibration(self, whisper_defaults):
        calibration_path = self.get_batch_size_calibration_path()
        if not os.path.exists(calibration_path):
            return None

        try:
            calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read batch-size calibration file '%s': %s", calibration_path, exc)
            return None

        if not isinstance(calibration, dict):
            return None

        recommended_batch_size = calibration.get("recommended_batch_size")
        if not isinstance(recommended_batch_size, int) or recommended_batch_size < 1:
            return None

        expected_model_size = calibration.get("model_size")
        current_model_size = whisper_defaults.get("model_size")
        if expected_model_size and current_model_size and expected_model_size != current_model_size:
            return None

        current_gpu_name = getattr(self.whisper_inf, "gpu_name", None)
        expected_gpu_name = calibration.get("gpu_name")
        if expected_gpu_name and current_gpu_name and expected_gpu_name != current_gpu_name:
            return None

        current_gpu_memory_gb = getattr(self.whisper_inf, "gpu_total_memory_gb", None)
        expected_gpu_memory_gb = calibration.get("gpu_total_memory_gb")
        if (
            current_gpu_memory_gb is not None
            and expected_gpu_memory_gb is not None
            and abs(float(current_gpu_memory_gb) - float(expected_gpu_memory_gb)) > BATCH_SIZE_CALIBRATION_MEMORY_TOLERANCE_GB
        ):
            return None

        return calibration

    def apply_dynamic_batch_size_defaults(self, default_params):
        whisper_defaults = default_params.get("whisper", {})
        if not isinstance(whisper_defaults, dict):
            return default_params

        calibration = self.load_batch_size_calibration(whisper_defaults)
        gpu_memory_gb = getattr(self.whisper_inf, "gpu_total_memory_gb", None)
        gpu_name = getattr(self.whisper_inf, "gpu_name", None)
        gpu_label = gpu_name or "GPU"

        if calibration is not None:
            recommended_batch_size = int(calibration["recommended_batch_size"])
            whisper_defaults["batch_size"] = recommended_batch_size
            if gpu_memory_gb is not None:
                logger.info(
                    "Detected %s with %.1f GB GPU memory. Using locally benchmarked default batch size %d.",
                    gpu_label,
                    gpu_memory_gb,
                    recommended_batch_size,
                )
            else:
                logger.info(
                    "Detected %s. Using locally benchmarked default batch size %d.",
                    gpu_label,
                    recommended_batch_size,
                )
            return default_params

        gpu_memory_gb = getattr(self.whisper_inf, "gpu_total_memory_gb", None)
        recommended_batch_size = 1
        whisper_defaults["batch_size"] = recommended_batch_size
        if gpu_memory_gb is not None:
            logger.info(
                "Detected %s with %.1f GB GPU memory. Default batch size set to %d for standard mode.",
                gpu_label,
                gpu_memory_gb,
                recommended_batch_size,
            )
        else:
            logger.info(
                "Default batch size set to %d for standard mode.",
                recommended_batch_size,
            )

        return default_params

    @staticmethod
    def format_media_duration(seconds):
        if seconds is None:
            return "Unknown"

        total_seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)

        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def get_media_duration_seconds(file_path):
        absolute_path = os.path.abspath(str(file_path))

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    absolute_path,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            raw_duration = result.stdout.strip()
            if raw_duration:
                duration = float(raw_duration)
                if duration >= 0:
                    return duration
        except (FileNotFoundError, subprocess.SubprocessError, ValueError):
            pass

        try:
            import av

            with av.open(absolute_path) as container:
                if container.duration is not None:
                    return float(container.duration / av.time_base)

                for stream in container.streams:
                    if stream.duration is not None and stream.time_base is not None:
                        return float(stream.duration * stream.time_base)
        except Exception:
            return None

        return None

    def get_gradio_api_prefix(self):
        root_path = (self.args.root_path or "").strip()
        if not root_path or root_path == "/":
            return "/gradio_api"

        normalized_root = f"/{root_path.lstrip('/')}"
        return f"{normalized_root.rstrip('/')}/gradio_api"

    def get_gradio_file_url(self, file_path):
        normalized_path = Path(os.path.abspath(str(file_path))).as_posix()
        return f"{self.get_gradio_api_prefix()}/file={quote(normalized_path, safe='/')}"

    @staticmethod
    def _strip_wrapping_quotes(value):
        normalized = str(value or "").strip()
        while len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
            normalized = normalized[1:-1].strip()
        return normalized

    @staticmethod
    def _is_windows_style_path(path):
        return bool(re.match(r"^[A-Za-z]:[\\/]", path)) or path.startswith("\\\\")

    def resolve_media_input_path(self, raw_path):
        user_path = self._strip_wrapping_quotes(raw_path)
        if not user_path:
            raise ValueError("Enter a file path to load.")

        candidate = user_path
        if candidate.lower().startswith("file://"):
            parsed = urlparse(candidate)
            candidate = unquote(parsed.path or "")

            if parsed.netloc and parsed.netloc not in {"", "localhost"}:
                candidate = f"//{parsed.netloc}{candidate}"

            if os.name == "nt" and re.match(r"^/[A-Za-z]:", candidate):
                candidate = candidate[1:]

        candidate = os.path.expandvars(os.path.expanduser(candidate))

        raw_candidates = [candidate]
        if os.name == "nt":
            raw_candidates.extend([
                candidate.replace("/", "\\"),
                candidate.replace("\\", "/"),
            ])
            if candidate.startswith("/") and re.match(r"^/[A-Za-z]:", candidate):
                raw_candidates.append(candidate[1:].replace("/", "\\"))
        else:
            raw_candidates.append(candidate.replace("\\", "/"))

        seen = set()
        last_normalized_candidate = None

        for raw_candidate in raw_candidates:
            if not raw_candidate or raw_candidate in seen:
                continue
            seen.add(raw_candidate)

            normalized_candidate = os.path.normpath(raw_candidate)
            normalized_candidate = os.path.abspath(normalized_candidate)
            last_normalized_candidate = normalized_candidate

            if os.path.isfile(normalized_candidate):
                extension = os.path.splitext(normalized_candidate)[1].lower()
                if extension not in MEDIA_EXTENSION:
                    raise ValueError(
                        f"Unsupported media file type: {extension or '[no extension]'}. "
                        "Load an audio or video file."
                    )
                return normalized_candidate

        if os.name != "nt" and self._is_windows_style_path(candidate):
            raise FileNotFoundError(
                "Windows-style path is not accessible from this host. "
                "If the file is mounted in Linux, enter that Linux path instead."
            )

        if last_normalized_candidate and os.path.exists(last_normalized_candidate):
            raise ValueError(f"Path is not a file: {last_normalized_candidate}")

        raise FileNotFoundError(f"File not found: {last_normalized_candidate or user_path}")

    def load_media_from_path(self, file_path):
        try:
            normalized_path = self.resolve_media_input_path(file_path)
        except Exception as exc:
            raise gr.Error(str(exc))

        summary_update, preview_update = self.update_uploaded_media_preview([normalized_path])
        return [normalized_path], gr.update(value=normalized_path), summary_update, preview_update

    def update_uploaded_media_preview(self, files):
        hidden_markdown = gr.update(value="", visible=False)
        hidden_html = gr.update(value="", visible=False)

        if not files:
            return hidden_markdown, hidden_html

        file_paths = [str(file) for file in (files if isinstance(files, list) else [files]) if file]
        if not file_paths:
            return hidden_markdown, hidden_html

        summary_lines = ["**Uploaded Media**"]
        preview_cards = []

        for file_path in file_paths:
            absolute_path = os.path.abspath(file_path)
            if not os.path.exists(absolute_path):
                continue

            media_type = "Video" if is_video(absolute_path) else "Audio"
            duration_text = self.format_media_duration(self.get_media_duration_seconds(absolute_path))
            summary_lines.append(
                f"- `{os.path.basename(absolute_path)}` ({media_type}) | Duration: `{duration_text}`"
            )

            if media_type == "Video":
                preview_cards.append(
                    f"""
                    <div class="upload-preview-card">
                      <div class="upload-preview-meta">
                        <span class="upload-preview-name">{html.escape(os.path.basename(absolute_path))}</span>
                        <span class="upload-preview-type">Video · {html.escape(duration_text)}</span>
                      </div>
                      <video controls preload="metadata" playsinline src="{self.get_gradio_file_url(absolute_path)}"></video>
                    </div>
                    """
                )
            else:
                preview_cards.append(
                    f"""
                    <div class="upload-preview-card">
                      <div class="upload-preview-meta">
                        <span class="upload-preview-name">{html.escape(os.path.basename(absolute_path))}</span>
                        <span class="upload-preview-type">Audio · {html.escape(duration_text)}</span>
                      </div>
                      <audio controls preload="metadata" src="{self.get_gradio_file_url(absolute_path)}"></audio>
                    </div>
                    """
                )

        if len(summary_lines) == 1:
            return hidden_markdown, hidden_html

        return (
            gr.update(value="\n".join(summary_lines), visible=True),
            gr.update(
                value=f'<div class="upload-preview-grid">{"".join(preview_cards)}</div>',
                visible=bool(preview_cards),
            ),
        )

    def resolve_file_output_folder(self, output_dir, batch_input, batch_enabled):
        if output_dir and str(output_dir).strip():
            return str(output_dir).strip()
        if batch_enabled and batch_input and str(batch_input).strip():
            return str(batch_input).strip()
        return self.args.output_dir

    def open_file_output_folder(self, output_dir, batch_input, batch_enabled):
        self.open_folder(self.resolve_file_output_folder(output_dir, batch_input, batch_enabled))

    @staticmethod
    def _unique_archive_name(used_names, file_path):
        base_name = os.path.basename(file_path)
        if base_name not in used_names:
            used_names.add(base_name)
            return base_name

        stem, ext = os.path.splitext(base_name)
        counter = 2
        while True:
            candidate = f"{stem}_{counter}{ext}"
            if candidate not in used_names:
                used_names.add(candidate)
                return candidate
            counter += 1

    def prepare_download_output(self, output_paths):
        if not output_paths:
            return gr.update(value=None, visible=False)

        valid_paths = []
        for path in output_paths:
            if path and os.path.exists(path):
                normalized = os.path.abspath(str(path))
                if normalized not in valid_paths:
                    valid_paths.append(normalized)

        if not valid_paths:
            return gr.update(value=None, visible=False)

        if len(valid_paths) == 1:
            return gr.update(value=valid_paths[0], visible=True)

        bundle_dir = os.path.join(self.args.output_dir, "_download_bundles")
        os.makedirs(bundle_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            prefix="whisper_outputs_",
            suffix=".zip",
            dir=bundle_dir,
            delete=False,
        ) as temp_zip:
            zip_path = temp_zip.name

        used_names = set()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in valid_paths:
                archive.write(file_path, arcname=self._unique_archive_name(used_names, file_path))

        return gr.update(value=zip_path, visible=True)

    @staticmethod
    def prepare_files_output(output_paths):
        if not output_paths:
            return gr.update(value=None, visible=False)

        valid_paths = []
        for path in output_paths:
            if path and os.path.exists(path):
                normalized = os.path.abspath(str(path))
                if normalized not in valid_paths:
                    valid_paths.append(normalized)

        if not valid_paths:
            return gr.update(value=None, visible=False)

        return gr.update(value=valid_paths, visible=True)

    def transcribe_file_with_download(self,
                                      files=None,
                                      batch_mode=False,
                                      input_folder_path=None,
                                      include_subdirectory=None,
                                      overwrite_existing=False,
                                      output_dir=None,
                                      file_formats="SRT",
                                      add_timestamp=True,
                                      progress=gr.Progress(),
                                      *pipeline_params):
        for live_output, result_str, collected_paths in self.whisper_inf.transcribe_file_with_live_output(
            files,
            batch_mode,
            input_folder_path,
            include_subdirectory,
            overwrite_existing,
            output_dir,
            file_formats,
            add_timestamp,
            progress,
            *pipeline_params,
        ):
            yield live_output, result_str, self.prepare_download_output(collected_paths)

    def transcribe_youtube_with_progress(self,
                                         youtube_link: str,
                                         file_formats="SRT",
                                         add_timestamp=True,
                                         mass_transcribe_channel=False,
                                         latest_video_count=100,
                                         progress=gr.Progress(),
                                         *pipeline_params):
        return self.whisper_inf.transcribe_youtube(
            youtube_link,
            file_formats,
            add_timestamp,
            mass_transcribe_channel,
            latest_video_count,
            progress,
            *pipeline_params,
        )

    def transcribe_mic_with_progress(self,
                                     mic_audio: str,
                                     file_formats="SRT",
                                     add_timestamp=True,
                                     progress=gr.Progress(),
                                     *pipeline_params):
        return self.whisper_inf.transcribe_mic(
            mic_audio,
            file_formats,
            add_timestamp,
            progress,
            *pipeline_params,
        )

    def transcribe_mic_with_download(self,
                                     mic_audio: str,
                                     file_formats="SRT",
                                     add_timestamp=True,
                                     progress=gr.Progress(),
                                     *pipeline_params):
        if coerce_audio_input_path(mic_audio) is None:
            yield (
                "Recorded microphone audio is not ready yet.",
                "The microphone recording is still being prepared. Wait until the recorder shows the saved audio player, then click GENERATE SUBTITLE FILE again.",
                self.prepare_files_output([]),
            )
            return

        for live_output, result_str, collected_paths in self.whisper_inf.transcribe_mic_with_live_output(
            mic_audio,
            file_formats,
            add_timestamp,
            progress,
            *pipeline_params,
        ):
            yield live_output, result_str, self.prepare_files_output(collected_paths)

    @classmethod
    def create_live_mic_state(cls):
        return {
            "audio": np.array([], dtype=np.float32),
            "sample_rate": cls.LIVE_MIC_SAMPLE_RATE,
            "last_processed_samples": 0,
            "transcript": "",
        }

    @staticmethod
    def normalize_live_mic_chunk(mic_audio):
        if mic_audio is None:
            return None, None

        sample_rate = None
        audio = None

        if isinstance(mic_audio, (tuple, list)) and len(mic_audio) == 2:
            sample_rate, audio = mic_audio
        elif isinstance(mic_audio, dict):
            sample_rate = mic_audio.get("sample_rate") or mic_audio.get("sr")
            for key in ("data", "audio", "array"):
                if key in mic_audio and mic_audio[key] is not None:
                    audio = mic_audio[key]
                    break
        elif isinstance(mic_audio, np.ndarray):
            audio = mic_audio

        if audio is None:
            return None, None

        audio = np.asarray(audio)
        if audio.size == 0:
            return None, None

        if np.issubdtype(audio.dtype, np.integer):
            max_value = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float32) / float(max_value or 1)

        if audio.ndim > 1:
            channel_axis = 0 if audio.shape[0] <= audio.shape[-1] else -1
            audio = audio.mean(axis=channel_axis)

        audio = audio.astype(np.float32)

        return int(sample_rate) if sample_rate else None, np.ascontiguousarray(audio.squeeze(), dtype=np.float32)

    @staticmethod
    def resample_live_mic_audio(audio: np.ndarray, original_sample_rate: int, target_sample_rate: int):
        if original_sample_rate <= 0 or original_sample_rate == target_sample_rate or audio.size == 0:
            return np.ascontiguousarray(audio, dtype=np.float32)

        target_length = max(int(round(audio.shape[0] * float(target_sample_rate) / float(original_sample_rate))), 1)
        old_positions = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
        new_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
        return np.ascontiguousarray(np.interp(new_positions, old_positions, audio).astype(np.float32))

    @staticmethod
    def append_live_mic_audio(existing_audio: np.ndarray, incoming_audio: np.ndarray):
        if existing_audio.size == 0:
            return np.ascontiguousarray(incoming_audio, dtype=np.float32)

        if incoming_audio.size >= existing_audio.size:
            try:
                if np.allclose(incoming_audio[:existing_audio.size], existing_audio, atol=1e-4):
                    return np.ascontiguousarray(incoming_audio, dtype=np.float32)
            except Exception:
                pass

        return np.ascontiguousarray(np.concatenate([existing_audio, incoming_audio]), dtype=np.float32)

    @staticmethod
    def trim_live_mic_audio(audio: np.ndarray, sample_rate: int, max_seconds: float):
        if audio.size == 0 or sample_rate <= 0 or max_seconds <= 0:
            return np.ascontiguousarray(audio, dtype=np.float32), False

        max_samples = int(sample_rate * max_seconds)
        if max_samples <= 0 or audio.shape[0] <= max_samples:
            return np.ascontiguousarray(audio, dtype=np.float32), False

        return np.ascontiguousarray(audio[-max_samples:], dtype=np.float32), True

    @classmethod
    def build_live_mic_status(cls, auto_live_enabled: bool, captured_seconds: float = 0.0, suffix: str = ""):
        if not auto_live_enabled:
            return "Live auto-transcribe is off. Enable the checkbox to start previewing while recording."

        base = (
            f"Live auto-transcribe is ON. Recording buffer: {captured_seconds:.1f}s. "
            f"Preview refresh target: every {cls.LIVE_MIC_REFRESH_SECONDS:.1f}s. "
            f"Rolling preview window: {cls.LIVE_MIC_MAX_BUFFER_SECONDS:.0f}s."
        )
        if suffix:
            return f"{base} {suffix}"
        return base

    def start_live_mic_recording(self, auto_live_enabled: bool):
        state = self.create_live_mic_state()
        if auto_live_enabled:
            transcript = ""
            status = self.build_live_mic_status(True, 0.0, "Listening for speech.")
        else:
            transcript = ""
            status = self.build_live_mic_status(False)
        return state, transcript, status

    def stop_live_mic_recording(self, auto_live_enabled: bool, live_state):
        state = live_state if isinstance(live_state, dict) else self.create_live_mic_state()
        audio = state.get("audio")
        if not isinstance(audio, np.ndarray):
            audio = np.array([], dtype=np.float32)
        sample_rate = int(state.get("sample_rate") or self.LIVE_MIC_SAMPLE_RATE)
        captured_seconds = float(audio.shape[0]) / float(sample_rate) if sample_rate and audio.size else 0.0
        suffix = "Recording stopped. Preview frozen." if auto_live_enabled else "Recording stopped."
        return state, state.get("transcript", ""), self.build_live_mic_status(auto_live_enabled, captured_seconds, suffix)

    @staticmethod
    def begin_record_mic_capture():
        return (
            "Recording in progress with the subtitle-generation microphone.",
            gr.update(interactive=False),
        )

    @staticmethod
    def finish_record_mic_capture():
        return (
            "Recording stopped. Preparing the recorded audio. Wait for the inline player to appear before generating subtitles.",
            gr.update(interactive=False),
        )

    @staticmethod
    def refresh_record_mic_ready_state(mic_audio):
        if coerce_audio_input_path(mic_audio) is None:
            return (
                "The recorded microphone clip is not ready yet.",
                gr.update(interactive=False),
            )

        return (
            "Recording saved. Click GENERATE SUBTITLE FILE below to create subtitles from it.",
            gr.update(interactive=True),
        )

    def transcribe_live_mic_chunk(self, mic_audio, auto_live_enabled: bool, live_state, *pipeline_params):
        state = live_state if isinstance(live_state, dict) else self.create_live_mic_state()

        if not auto_live_enabled:
            return state, state.get("transcript", ""), self.build_live_mic_status(False)

        incoming_sample_rate, incoming_audio = self.normalize_live_mic_chunk(mic_audio)
        if incoming_audio is None:
            audio = state.get("audio")
            if not isinstance(audio, np.ndarray):
                audio = np.array([], dtype=np.float32)
            captured_seconds = float(audio.shape[0]) / float(state.get("sample_rate") or self.LIVE_MIC_SAMPLE_RATE) if audio.size else 0.0
            return state, state.get("transcript", ""), self.build_live_mic_status(True, captured_seconds, "Waiting for microphone audio.")

        sample_rate = incoming_sample_rate or self.LIVE_MIC_SAMPLE_RATE
        if sample_rate != self.LIVE_MIC_SAMPLE_RATE:
            incoming_audio = self.resample_live_mic_audio(incoming_audio, sample_rate, self.LIVE_MIC_SAMPLE_RATE)
            sample_rate = self.LIVE_MIC_SAMPLE_RATE

        existing_audio = state.get("audio")
        if not isinstance(existing_audio, np.ndarray):
            existing_audio = np.array([], dtype=np.float32)

        combined_audio = self.append_live_mic_audio(existing_audio, incoming_audio)
        combined_audio, trimmed = self.trim_live_mic_audio(
            combined_audio,
            sample_rate,
            self.LIVE_MIC_MAX_BUFFER_SECONDS,
        )
        state["audio"] = combined_audio
        state["sample_rate"] = sample_rate

        total_samples = int(combined_audio.shape[0]) if combined_audio.size else 0
        captured_seconds = float(total_samples) / float(sample_rate) if sample_rate and total_samples else 0.0

        min_preview_samples = int(sample_rate * self.LIVE_MIC_MIN_PREVIEW_SECONDS)
        refresh_samples = int(sample_rate * self.LIVE_MIC_REFRESH_SECONDS)
        last_processed_samples = int(state.get("last_processed_samples") or 0)

        if trimmed:
            last_processed_samples = min(last_processed_samples, total_samples)
            state["last_processed_samples"] = last_processed_samples

        if total_samples < min_preview_samples:
            return state, state.get("transcript", ""), self.build_live_mic_status(
                True,
                captured_seconds,
                "Collecting enough audio for the first preview.",
            )

        if state.get("transcript") and total_samples - last_processed_samples < refresh_samples:
            return state, state.get("transcript", ""), self.build_live_mic_status(
                True,
                captured_seconds,
                "Listening...",
            )

        try:
            transcript = self.whisper_inf.transcribe_live_preview(combined_audio, *pipeline_params)
            state["transcript"] = transcript
            state["last_processed_samples"] = total_samples
            suffix = "Preview updated." if transcript else "Speech not detected yet."
            return state, transcript, self.build_live_mic_status(True, captured_seconds, suffix)
        except Exception as exc:
            return state, state.get("transcript", ""), self.build_live_mic_status(
                True,
                captured_seconds,
                f"Live preview error: {type(exc).__name__}: {exc}",
            )

    def cancel_active_generation(self, confirmed: bool):
        if not confirmed:
            return

        if self.whisper_inf.cancel_active_generation():
            gr.Info("Cancellation requested. Terminating the running subprocess.")
        else:
            gr.Info("No running subprocess was found.")

    def create_whisper_inputs_3col(self, whisper_params):
        inputs = []

        with gr.Row():
            inputs.append(gr.Number(
                label="Beam Size",
                value=whisper_params["beam_size"],
                precision=0,
                info="Number of beams in beam search. Higher values are usually more accurate but slower. Range: 1-20.",
            ))
            inputs.append(gr.Number(
                label="Log Probability Threshold",
                value=whisper_params["log_prob_threshold"],
                info="Reject segments with average log probability below this value. Lower values are stricter.",
            ))
            inputs.append(gr.Number(
                label="No Speech Threshold",
                value=whisper_params["no_speech_threshold"],
                info="Probability threshold for detecting silence or no-speech. Range: 0.0-1.0.",
            ))

        with gr.Row():
            inputs.append(gr.Dropdown(
                label="Compute Type",
                choices=self.whisper_inf.available_compute_types,
                value=self.whisper_inf.current_compute_type,
                info="Precision for model computation. Use the setting that best fits your device speed, stability, and VRAM budget.",
            ))
            inputs.append(gr.Number(
                label="Best Of",
                value=whisper_params["best_of"],
                precision=0,
                info="Number of candidate sequences to generate when sampling. Higher values can improve quality but are slower.",
            ))
            inputs.append(gr.Number(
                label="Patience",
                value=whisper_params["patience"],
                info="Beam search patience controls how long to wait for better candidates. Higher values search more thoroughly.",
            ))

        with gr.Row():
            inputs.append(gr.Checkbox(
                label="Condition On Previous Text",
                value=whisper_params["condition_on_previous_text"],
                info="Use the previous transcription as context for the next segment. Disable it if the model gets stuck in repetition.",
            ))
            inputs.append(gr.Slider(
                label="Prompt Reset On Temperature",
                value=whisper_params["prompt_reset_on_temperature"],
                minimum=0,
                maximum=1,
                step=0.01,
                info="Reset the conditioning prompt if temperature exceeds this value. Range: 0.0-1.0.",
            ))
            inputs.append(gr.Textbox(
                label="Initial Prompt",
                value=whisper_params.get("initial_prompt", ""),
                info="Text that guides transcription style or vocabulary. Useful for domain-specific terms. Leave empty for general transcription.",
            ))

        with gr.Row():
            inputs.append(gr.Slider(
                label="Temperature",
                value=whisper_params["temperature"],
                minimum=0.0,
                step=0.01,
                maximum=1.0,
                info="Randomness in decoding. Lower values are more deterministic and accurate.",
            ))
            inputs.append(gr.Number(
                label="Compression Ratio Threshold",
                value=whisper_params["compression_ratio_threshold"],
                info="Detect repetitive or hallucinated text by gzip compression ratio. Lower values are stricter.",
            ))
            inputs.append(gr.Number(
                label="Length Penalty",
                value=whisper_params["length_penalty"],
                info="Penalty for longer sequences. Values above 1.0 favor longer outputs, below 1.0 favor shorter outputs, and 1.0 is neutral.",
            ))

        return [repair_component_text(component) for component in inputs]

    def create_pipeline_inputs(self,
                               defaults=None,
                               open_outputs_button=None,
                               place_condition_on_previous_text_right: bool = False):
        transcription_defaults = defaults or self.ui_default_config["file_tab"]
        whisper_params = transcription_defaults["whisper"]
        vad_params = transcription_defaults["vad"]
        diarization_params = transcription_defaults["diarization"]
        uvr_params = transcription_defaults["bgm_separation"]

        with gr.Row():
            with gr.Column(scale=2):
                lang_value = WhisperParams.normalize_lang_choice(whisper_params["lang"])
                dd_model = gr.Dropdown(
                    choices=self.whisper_inf.available_models,
                    value=whisper_params["model_size"],
                    label=_("Model"),
                    allow_custom_value=True,
                )
                dd_lang = gr.Dropdown(
                    choices=WhisperParams.get_language_choices(self.whisper_inf.available_langs),
                    value=lang_value,
                    label=_("Language"),
                )
            with gr.Column(scale=2):
                cg_file_formats = gr.CheckboxGroup(
                    choices=["SRT", "WebVTT", "txt", "LRC", "JSON", "TSV"],
                    value=transcription_defaults.get("file_formats", ["SRT"]) or ["SRT"],
                    label=_("File Formats"),
                    info=_("Select one or more output formats."),
                )
                cb_translate = gr.Checkbox(
                    value=whisper_params["is_translate"],
                    label=_("Translate to English?"),
                    info=(
                        "When enabled, Whisper outputs English text directly from the speech. "
                        "When disabled, subtitles stay in the original spoken language. "
                        "This uses Whisper's built-in translation, not DeepL or NLLB."
                    ),
                    interactive=True,
                )
            with gr.Column(scale=1):
                cb_timestamp = gr.Checkbox(
                    value=transcription_defaults.get("add_timestamp", False),
                    label=_("Add a timestamp to the end of the filename"),
                    info=TIMESTAMP_INFO,
                    interactive=True,
                )
        open_outputs_btn = open_outputs_button

        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                run_btn = gr.Button(
                    _("GENERATE SUBTITLE FILE"),
                    variant="primary",
                    elem_classes=["action-button", "generate-subtitle-button"],
                )
            with gr.Column(scale=1):
                cancel_btn = gr.Button(
                    "Cancel Generation",
                    variant="stop",
                )
            if open_outputs_btn is None:
                with gr.Column(scale=1):
                    open_outputs_btn = gr.Button(
                        "OPEN OUTPUTS FOLDER",
                        elem_classes=["action-button", "open-outputs-folder-button"],
                    )

        with gr.Row(equal_height=True):
            with gr.Column(scale=3, visible=WhisperParams.supports_batch_size(self.args.whisper_type)):
                with gr.Group():
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=4):
                            gr.Markdown("**Batch Size**")
                            gr.Markdown(
                                "When Use Batched Inference is disabled, this controls the standard faster-whisper "
                                "encoder prefetch batch size on a single model instance. Higher values can improve "
                                "throughput with much lower quality risk than the batched decoder path, but the gain "
                                "depends on your audio and GPU. When Use Batched Inference is enabled, it controls "
                                "the faster-whisper batched decoder path instead."
                            )
                        with gr.Column(scale=2, min_width=220):
                            batch_size_input = WhisperParams.to_batch_size_input(
                                defaults=whisper_params,
                                whisper_type=self.args.whisper_type,
                            )
                            batch_size_input.show_label = False
                            batch_size_input.info = None
            with gr.Column(scale=2):
                condition_on_previous_text_input = WhisperParams.to_condition_on_previous_text_input(
                    defaults=whisper_params,
                )
            with gr.Column(scale=2):
                start_as_subprocess_input = WhisperParams.to_start_as_subprocess_input(
                    defaults=whisper_params,
                )

        with gr.Accordion(_("Advanced Parameters"), open=False):
            whisper_inputs = WhisperParams.to_gradio_inputs(
                defaults=whisper_params,
                only_advanced=True,
                whisper_type=self.args.whisper_type,
                available_compute_types=self.whisper_inf.available_compute_types,
                compute_type=self.whisper_inf.current_compute_type,
            )

        whisper_advanced_fields = WhisperParams.advanced_input_field_names()
        condition_on_previous_text_index = whisper_advanced_fields.index("condition_on_previous_text")
        start_as_subprocess_index = whisper_advanced_fields.index("start_as_subprocess")
        batch_size_index = whisper_advanced_fields.index("batch_size")

        whisper_inputs[condition_on_previous_text_index].visible = False
        whisper_inputs[condition_on_previous_text_index] = condition_on_previous_text_input
        whisper_inputs[start_as_subprocess_index].visible = False
        whisper_inputs[start_as_subprocess_index] = start_as_subprocess_input
        whisper_inputs[batch_size_index].visible = False
        whisper_inputs[batch_size_index] = batch_size_input

        with gr.Accordion(_("Background Music Remover Filter"), open=False):
            uvr_inputs = BGMSeparationParams.to_gradio_input(
                defaults=uvr_params,
                available_models=self.whisper_inf.music_separator.available_models,
                available_devices=self.whisper_inf.music_separator.available_devices,
                device=self.whisper_inf.music_separator.device,
            )

        with gr.Accordion(_("Voice Detection Filter"), open=False):
            vad_inputs = VadParams.to_gradio_inputs(defaults=vad_params)

        with gr.Accordion(_("Diarization"), open=False):
            diarization_inputs = DiarizationParams.to_gradio_inputs(
                defaults=diarization_params,
                available_devices=self.whisper_inf.diarizer.available_device,
                device=self.whisper_inf.diarizer.device,
            )

        return {
            "pipeline": [dd_model, dd_lang, cb_translate] + whisper_inputs + vad_inputs + diarization_inputs + uvr_inputs,
            "file_formats": cg_file_formats,
            "add_timestamp": cb_timestamp,
            "run_button": run_btn,
            "cancel_button": cancel_btn,
            "open_outputs_button": open_outputs_btn,
        }

    def launch(self, prevent_thread_lock: bool = False, quiet: bool = False):
        remembered_preset_name = get_last_used_ui_preset()
        startup_preset_name = remembered_preset_name or get_default_startup_ui_preset()
        default_ui_config = merge_ui_config(self.ui_default_config, default_params=self.default_params)
        startup_preset_status = ""
        if startup_preset_name:
            startup_cfg = load_ui_preset(startup_preset_name, default_params=self.default_params)
            if startup_cfg:
                default_ui_config = startup_cfg
                if remembered_preset_name:
                    startup_preset_status = f"Loaded last used preset **{startup_preset_name}**"
                else:
                    startup_preset_status = f"Loaded default preset **{startup_preset_name}**"
            else:
                clear_last_used_ui_preset()
                failed_startup_preset_name = startup_preset_name
                startup_preset_name = None
                startup_preset_status = f"Preset **{failed_startup_preset_name}** could not be loaded. Loaded defaults."

        file_defaults = default_ui_config["file_tab"]
        youtube_defaults = default_ui_config["youtube_tab"]
        mic_defaults = default_ui_config["mic_tab"]
        deepl_defaults = default_ui_config["translation_deepl"]
        nllb_defaults = default_ui_config["translation_nllb"]
        bgm_defaults = default_ui_config["bgm_separation_tab"]

        with self.app:
            gr.Radio(
                choices=list(self.i18n.keys()),
                label="UI Language",
                interactive=True,
                visible=False,
            )
            cancel_confirmed = gr.State(False)
            with Translate(self.i18n):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"### {self.title}", elem_id="md_project")

                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Accordion("Config Presets", open=True):
                            with gr.Row():
                                ui_preset_dropdown = gr.Dropdown(
                                    label="Select Preset",
                                    choices=list_ui_presets(),
                                    value=startup_preset_name,
                                )
                                ui_preset_name = gr.Textbox(label="New Preset Name", placeholder="my_preset")
                            with gr.Row():
                                ui_preset_save_btn = gr.Button("Save", variant="primary")
                                ui_preset_reset_btn = gr.Button("Reset Defaults", variant="secondary")
                                ui_preset_delete_btn = gr.Button("Delete", variant="stop")
                            ui_preset_status = gr.Markdown(startup_preset_status)
                    with gr.Column(scale=2):
                        file_download_btn = gr.DownloadButton(
                                    "Download Transcription",
                            visible=False,
                            elem_id="top-download-output-button",
                            elem_classes=["action-button", "download-output-button"],
                        )
                        file_open_outputs_btn = gr.Button(
                            "OPEN OUTPUTS FOLDER",
                            elem_classes=["action-button", "open-outputs-folder-button"],
                        )
                        tb_load_file_path = gr.Textbox(
                            label=_("Load From File Path"),
                            placeholder="C:\\media\\clip.mp4, /workspace/media/clip.mp4, ./media/clip.mp4, or file:///path/to/clip.mp4",
                            info=_("Supports Windows, Linux, relative, absolute, quoted, and file:// paths."),
                        )
                        btn_load_file_path = gr.Button(_("Load"))

                with gr.Tabs():
                    with gr.TabItem(_("File")):
                        with gr.Row():
                            with gr.Column():
                                input_file = gr.Files(
                                    type="filepath",
                                    label=_("Upload File here"),
                                    file_types=MEDIA_EXTENSION,
                                )
                                upload_media_summary = gr.Markdown(visible=False)
                                upload_media_preview = gr.HTML(visible=False)
                            with gr.Column():
                                gr.Markdown(_("Batch Processing"))
                                with gr.Row():
                                    cb_batch_processing = gr.Checkbox(
                                        label=_("Enable Batch Processing"),
                                        value=file_defaults["batch_processing"],
                                    )
                                    cb_include_subdirectory = gr.Checkbox(
                                        label=_("Include Subdirectory Files"),
                                        info=_("Process files in nested folders when batch is enabled."),
                                        value=file_defaults["include_subdirectory"],
                                    )
                                    cb_overwrite_existing = gr.Checkbox(
                                        label=_("Overwrite Existing Files"),
                                        value=file_defaults["overwrite_existing"],
                                        info=_("Re-create outputs when they already exist."),
                                    )
                                with gr.Row():
                                    tb_input_folder = gr.Textbox(
                                        label=_("Input Folder Path"),
                                        placeholder="C:/path/to/folder or /workspace/audio",
                                        info="Required when batch processing is enabled.",
                                        value=file_defaults["input_folder"],
                                    )
                                    tb_output_folder = gr.Textbox(
                                        label=_("Output Folder (Optional)"),
                                        placeholder="Leave blank to save next to input files",
                                        info="When provided, outputs are saved here instead of the input folder.",
                                        value=file_defaults["output_folder"],
                                    )
                                tb_live_transcription = gr.Textbox(
                                    label=_("Live Transcription"),
                                    lines=10,
                                    max_lines=10,
                                    interactive=False,
                                    buttons=["copy"],
                                    placeholder="Transcribed segments will appear here in real-time...",
                                    elem_classes=["live-transcription-box"],
                                )
                        file_transcription_ui = self.create_pipeline_inputs(
                            file_defaults,
                            open_outputs_button=file_open_outputs_btn,
                            place_condition_on_previous_text_right=True,
                        )
                        with gr.Row():
                            file_output = gr.Textbox(label=_("Output"), interactive=False)
                        file_inputs = [
                            input_file,
                            cb_batch_processing,
                            tb_input_folder,
                            cb_include_subdirectory,
                            cb_overwrite_existing,
                            tb_output_folder,
                            file_transcription_ui["file_formats"],
                            file_transcription_ui["add_timestamp"],
                        ]
                        input_file.change(
                            fn=self.update_uploaded_media_preview,
                            inputs=[input_file],
                            outputs=[upload_media_summary, upload_media_preview],
                            queue=False,
                            show_progress="hidden",
                        )
                        btn_load_file_path.click(
                            fn=self.load_media_from_path,
                            inputs=[tb_load_file_path],
                            outputs=[input_file, tb_load_file_path, upload_media_summary, upload_media_preview],
                            queue=False,
                            show_progress="hidden",
                        )
                        tb_load_file_path.submit(
                            fn=self.load_media_from_path,
                            inputs=[tb_load_file_path],
                            outputs=[input_file, tb_load_file_path, upload_media_summary, upload_media_preview],
                            queue=False,
                            show_progress="hidden",
                        )
                        file_run_event = file_transcription_ui["run_button"].click(
                            fn=self.transcribe_file_with_download,
                            inputs=file_inputs + file_transcription_ui["pipeline"],
                            outputs=[tb_live_transcription, file_output, file_download_btn],
                        )
                        file_transcription_ui["cancel_button"].click(
                            fn=self.cancel_active_generation,
                            inputs=[cancel_confirmed],
                            outputs=None,
                            js="() => [confirm('Terminate the running subprocess?')]",
                            cancels=[file_run_event],
                            queue=False,
                            show_progress="hidden",
                        )
                        file_transcription_ui["open_outputs_button"].click(
                            fn=self.open_file_output_folder,
                            inputs=[tb_output_folder, tb_input_folder, cb_batch_processing],
                            outputs=None,
                            queue=False,
                            show_progress="hidden",
                        )

                    with gr.TabItem(_("Youtube")):
                        with gr.Row():
                            tb_youtubelink = gr.Textbox(label=_("Youtube Link"))
                        with gr.Row():
                            cb_mass_transcribe_channel = gr.Checkbox(
                                label=_("Mass Transcribe Latest Channel Videos"),
                                value=youtube_defaults.get("mass_transcribe_channel", False),
                                info=_("Use a channel link to transcribe the latest N videos from that channel."),
                            )
                            sl_latest_video_count = gr.Slider(
                                label=_("Latest Videos To Scan"),
                                value=youtube_defaults.get("latest_video_count", 100),
                                minimum=1,
                                maximum=9999,
                                step=1,
                                info=_("How many latest channel videos to process when mass transcribe is enabled."),
                            )
                        with gr.Row(equal_height=True):
                            with gr.Column():
                                img_thumbnail = gr.Image(label=_("Youtube Thumbnail"))
                            with gr.Column():
                                tb_title = gr.Label(label=_("Youtube Title"))
                                tb_description = gr.Textbox(label=_("Youtube Description"), max_lines=15)

                        youtube_transcription_ui = self.create_pipeline_inputs(youtube_defaults)
                        youtube_transcription_ui["extra_components"] = [
                            cb_mass_transcribe_channel,
                            sl_latest_video_count,
                        ]

                        with gr.Row():
                            gr.Textbox(
                                label=_("Live Transcription"),
                                lines=10,
                                max_lines=10,
                                interactive=False,
                                buttons=["copy"],
                                placeholder="Transcribed segments will appear here in real-time...",
                                elem_classes=["live-transcription-box"],
                            )
                        with gr.Row():
                            youtube_output = gr.Textbox(label=_("Output"), scale=5)
                            youtube_outputs = gr.Files(label=_("Downloadable output file"), scale=4)

                        youtube_inputs = [
                            tb_youtubelink,
                            youtube_transcription_ui["file_formats"],
                            youtube_transcription_ui["add_timestamp"],
                            cb_mass_transcribe_channel,
                            sl_latest_video_count,
                        ]
                        youtube_run_event = youtube_transcription_ui["run_button"].click(
                            fn=self.transcribe_youtube_with_progress,
                            inputs=youtube_inputs + youtube_transcription_ui["pipeline"],
                            outputs=[youtube_output, youtube_outputs],
                        )
                        youtube_transcription_ui["cancel_button"].click(
                            fn=self.cancel_active_generation,
                            inputs=[cancel_confirmed],
                            outputs=None,
                            js="() => [confirm('Terminate the running subprocess?')]",
                            cancels=[youtube_run_event],
                            queue=False,
                            show_progress="hidden",
                        )
                        tb_youtubelink.change(get_ytmetas, inputs=[tb_youtubelink], outputs=[img_thumbnail, tb_title, tb_description])
                        youtube_transcription_ui["open_outputs_button"].click(
                            fn=lambda: self.open_folder(self.args.output_dir),
                            inputs=None,
                            outputs=None,
                            queue=False,
                            show_progress="hidden",
                        )

                    with gr.TabItem(_("Mic")):
                        gr.Markdown(
                            "Use either microphone mode below. "
                            "The left recorder starts live preview automatically when the checkbox is enabled. "
                            "The right recorder is for record-then-generate subtitle files."
                        )

                        live_mic_state = gr.State(self.create_live_mic_state())

                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1):
                                gr.Markdown("### Live Mic")
                                cb_live_auto_transcribe = gr.Checkbox(
                                    label="Auto transcribe while recording",
                                    value=True,
                                    info="When enabled, starting the recorder below begins live preview automatically.",
                                )
                                live_mic_input = gr.Microphone(
                                    label="Live microphone preview",
                                    type="numpy",
                                    streaming=True,
                                    interactive=True,
                                    elem_id="live-mic-recorder",
                                    elem_classes=["mic-recorder-frame", "mic-recorder-live"],
                                    waveform_options=gr.WaveformOptions(
                                        show_recording_waveform=True,
                                    ),
                                )
                                live_mic_status = gr.Markdown(
                                    self.build_live_mic_status(
                                        True,
                                        0.0,
                                        "Press Record on the microphone to begin live preview.",
                                    )
                                )
                                live_mic_transcription = gr.Textbox(
                                    label=_("Live Transcription"),
                                    lines=10,
                                    max_lines=10,
                                    interactive=False,
                                    buttons=["copy"],
                                    placeholder="Live preview text will appear here while you record.",
                                    elem_classes=["live-transcription-box"],
                                )

                            with gr.Column(scale=1):
                                gr.Markdown("### Record Then Generate")
                                record_mic_status = gr.Markdown(
                                    "Record with the microphone below, stop recording, then click "
                                    "**GENERATE SUBTITLE FILE** to create subtitle files from that recording."
                                )
                                record_mic_input = gr.Microphone(
                                    label="Record microphone for subtitle generation",
                                    type="filepath",
                                    interactive=True,
                                    buttons=["download"],
                                    elem_id="record-mic-recorder",
                                    elem_classes=["mic-recorder-frame", "mic-recorder-generate"],
                                    waveform_options=gr.WaveformOptions(
                                        show_recording_waveform=True,
                                    ),
                                )

                        mic_transcription_ui = self.create_pipeline_inputs(mic_defaults)
                        mic_transcription_ui["run_button"].interactive = False

                        with gr.Row():
                            mic_generation_progress = gr.Textbox(
                                label="Generation Progress",
                                lines=10,
                                max_lines=10,
                                interactive=False,
                                buttons=["copy"],
                                placeholder="Subtitle generation progress will appear here after you click Generate.",
                                elem_classes=["live-transcription-box"],
                            )
                        with gr.Row():
                            mic_output = gr.Textbox(label=_("Output"), scale=5)
                            mic_outputs = gr.Files(label=_("Downloadable output file"), scale=4)

                        mic_inputs = [
                            record_mic_input,
                            mic_transcription_ui["file_formats"],
                            mic_transcription_ui["add_timestamp"],
                        ]
                        live_mic_stream_inputs = [
                            live_mic_input,
                            cb_live_auto_transcribe,
                            live_mic_state,
                        ]

                        live_mic_start_event = live_mic_input.start_recording(
                            fn=self.start_live_mic_recording,
                            inputs=[cb_live_auto_transcribe],
                            outputs=[live_mic_state, live_mic_transcription, live_mic_status],
                            queue=False,
                            show_progress="hidden",
                        )
                        live_mic_stream_event = live_mic_input.stream(
                            fn=self.transcribe_live_mic_chunk,
                            inputs=live_mic_stream_inputs + mic_transcription_ui["pipeline"],
                            outputs=[live_mic_state, live_mic_transcription, live_mic_status],
                            queue=False,
                            show_progress="hidden",
                            trigger_mode="always_last",
                            concurrency_limit=1,
                            stream_every=1.0,
                        )
                        live_mic_input.stop_recording(
                            fn=self.stop_live_mic_recording,
                            inputs=[cb_live_auto_transcribe, live_mic_state],
                            outputs=[live_mic_state, live_mic_transcription, live_mic_status],
                            queue=False,
                            show_progress="hidden",
                        )

                        record_mic_input.start_recording(
                            fn=self.begin_record_mic_capture,
                            inputs=None,
                            outputs=[record_mic_status, mic_transcription_ui["run_button"]],
                            queue=False,
                            show_progress="hidden",
                        )
                        record_mic_input.stop_recording(
                            fn=self.finish_record_mic_capture,
                            inputs=None,
                            outputs=[record_mic_status, mic_transcription_ui["run_button"]],
                            queue=False,
                            show_progress="hidden",
                        )
                        record_mic_input.change(
                            fn=self.refresh_record_mic_ready_state,
                            inputs=[record_mic_input],
                            outputs=[record_mic_status, mic_transcription_ui["run_button"]],
                            queue=False,
                            show_progress="hidden",
                        )
                        record_mic_input.upload(
                            fn=self.refresh_record_mic_ready_state,
                            inputs=[record_mic_input],
                            outputs=[record_mic_status, mic_transcription_ui["run_button"]],
                            queue=False,
                            show_progress="hidden",
                        )
                        record_mic_input.input(
                            fn=self.refresh_record_mic_ready_state,
                            inputs=[record_mic_input],
                            outputs=[record_mic_status, mic_transcription_ui["run_button"]],
                            queue=False,
                            show_progress="hidden",
                        )

                        mic_run_event = mic_transcription_ui["run_button"].click(
                            fn=self.transcribe_mic_with_download,
                            inputs=mic_inputs + mic_transcription_ui["pipeline"],
                            outputs=[mic_generation_progress, mic_output, mic_outputs],
                        )
                        mic_transcription_ui["cancel_button"].click(
                            fn=self.cancel_active_generation,
                            inputs=[cancel_confirmed],
                            outputs=None,
                            js="() => [confirm('Terminate the running subprocess?')]",
                            cancels=[mic_run_event],
                            queue=False,
                            show_progress="hidden",
                        )
                        mic_transcription_ui["open_outputs_button"].click(
                            fn=lambda: self.open_folder(self.args.output_dir),
                            inputs=None,
                            outputs=None,
                            queue=False,
                            show_progress="hidden",
                        )

                    with gr.TabItem(_("T2T Translation")):
                        with gr.Row():
                            file_subs = gr.Files(type="filepath", label=_("Upload Subtitle Files to translate here"))

                        with gr.TabItem(_("DeepL API")):
                            with gr.Row():
                                tb_api_key = gr.Textbox(
                                    label=_("Your Auth Key (API KEY)"),
                                    value=deepl_defaults["api_key"],
                                )
                            with gr.Row():
                                deepl_source_lang = gr.Dropdown(
                                    label=_("Source Language"),
                                    value=AUTOMATIC_DETECTION if deepl_defaults["source_lang"] == AUTOMATIC_DETECTION.unwrap()
                                    else deepl_defaults["source_lang"],
                                    choices=list(self.deepl_api.available_source_langs.keys()),
                                )
                                deepl_target_lang = gr.Dropdown(
                                    label=_("Target Language"),
                                    value=deepl_defaults["target_lang"],
                                    choices=list(self.deepl_api.available_target_langs.keys()),
                                )
                            with gr.Row():
                                cb_is_pro = gr.Checkbox(label=_("Pro User?"), value=deepl_defaults["is_pro"])
                            with gr.Row():
                                deepl_add_timestamp = gr.Checkbox(
                                    value=deepl_defaults["add_timestamp"],
                                    label=_("Add a timestamp to the end of the filename"),
                                    info=TIMESTAMP_INFO,
                                    interactive=True,
                                )
                            with gr.Row():
                                deepl_run_btn = gr.Button(_("TRANSLATE SUBTITLE FILE"), variant="primary")
                            with gr.Row():
                                deepl_output = gr.Textbox(label=_("Output"), scale=5)
                                deepl_outputs = gr.Files(label=_("Downloadable output file"), scale=3)
                                deepl_open_btn = gr.Button('Open', scale=1)

                        deepl_run_btn.click(
                            fn=self.deepl_api.translate_deepl,
                            inputs=[tb_api_key, file_subs, deepl_source_lang, deepl_target_lang, cb_is_pro, deepl_add_timestamp],
                            outputs=[deepl_output, deepl_outputs],
                        )
                        deepl_open_btn.click(
                            fn=lambda: self.open_folder(os.path.join(self.args.output_dir, "translations")),
                            inputs=None,
                            outputs=None,
                        )

                        with gr.TabItem(_("NLLB")):
                            with gr.Row():
                                nllb_model_size = gr.Dropdown(
                                    label=_("Model"),
                                    value=nllb_defaults["model_size"],
                                    choices=self.nllb_inf.available_models,
                                )
                                nllb_source_lang = gr.Dropdown(
                                    label=_("Source Language"),
                                    value=nllb_defaults["source_lang"],
                                    choices=self.nllb_inf.available_source_langs,
                                )
                                nllb_target_lang = gr.Dropdown(
                                    label=_("Target Language"),
                                    value=nllb_defaults["target_lang"],
                                    choices=self.nllb_inf.available_target_langs,
                                )
                            with gr.Row():
                                nb_max_length = gr.Number(
                                    label="Max Length Per Line",
                                    value=nllb_defaults["max_length"],
                                    precision=0,
                                )
                            with gr.Row():
                                nllb_add_timestamp = gr.Checkbox(
                                    value=nllb_defaults["add_timestamp"],
                                    label=_("Add a timestamp to the end of the filename"),
                                    info=TIMESTAMP_INFO,
                                    interactive=True,
                                )
                            with gr.Row():
                                nllb_run_btn = gr.Button(_("TRANSLATE SUBTITLE FILE"), variant="primary")
                            with gr.Row():
                                nllb_output = gr.Textbox(label=_("Output"), scale=5)
                                nllb_outputs = gr.Files(label=_("Downloadable output file"), scale=3)
                                nllb_open_btn = gr.Button('Open', scale=1)
                            with gr.Column():
                                gr.HTML(NLLB_VRAM_TABLE, elem_id="md_nllb_vram_table")

                        nllb_run_btn.click(
                            fn=self.nllb_inf.translate_file,
                            inputs=[file_subs, nllb_model_size, nllb_source_lang, nllb_target_lang, nb_max_length, nllb_add_timestamp],
                            outputs=[nllb_output, nllb_outputs],
                        )
                        nllb_open_btn.click(
                            fn=lambda: self.open_folder(os.path.join(self.args.output_dir, "translations")),
                            inputs=None,
                            outputs=None,
                        )

                    with gr.TabItem(_("BGM Separation")):
                        files_audio = gr.Files(type="filepath", label=_("Upload Audio Files to separate background music"))
                        dd_uvr_device = gr.Dropdown(
                            label=_("Device"),
                            value=bgm_defaults["uvr_device"] or self.whisper_inf.music_separator.device,
                            choices=self.whisper_inf.music_separator.available_devices,
                        )
                        dd_uvr_model_size = gr.Dropdown(
                            label=_("Model"),
                            value=bgm_defaults["uvr_model_size"],
                            choices=self.whisper_inf.music_separator.available_models,
                        )
                        nb_uvr_segment_size = gr.Number(
                            label="Segment Size",
                            value=bgm_defaults["segment_size"],
                            precision=0,
                        )
                        cb_uvr_save_file = gr.Checkbox(
                            label=_("Save separated files to output"),
                            value=True,
                            visible=False,
                        )
                        uvr_run_btn = gr.Button(_("SEPARATE BACKGROUND MUSIC"), variant="primary")
                        with gr.Column():
                            with gr.Row():
                                ad_instrumental = gr.Audio(label=_("Instrumental"), scale=8)
                                btn_open_instrumental_folder = gr.Button('Open', scale=1)
                            with gr.Row():
                                ad_vocals = gr.Audio(label=_("Vocals"), scale=8)
                                btn_open_vocals_folder = gr.Button('Open', scale=1)

                        uvr_run_btn.click(
                            fn=self.whisper_inf.music_separator.separate_files,
                            inputs=[files_audio, dd_uvr_model_size, dd_uvr_device, nb_uvr_segment_size, cb_uvr_save_file],
                            outputs=[ad_instrumental, ad_vocals],
                        )
                        btn_open_instrumental_folder.click(
                            inputs=None,
                            outputs=None,
                            fn=lambda: self.open_folder(os.path.join(self.args.output_dir, "UVR", "instrumental")),
                        )
                        btn_open_vocals_folder.click(
                            inputs=None,
                            outputs=None,
                            fn=lambda: self.open_folder(os.path.join(self.args.output_dir, "UVR", "vocals")),
                        )

                file_format_choices = ["SRT", "WebVTT", "txt", "LRC", "JSON", "TSV"]

                def _normalize_choice(value):
                    return value.unwrap() if hasattr(value, "unwrap") else value

                def _serialize_ui_value(value):
                    if isinstance(value, list):
                        return [_serialize_ui_value(item) for item in value]
                    return _normalize_choice(value)

                def _transcription_config_paths(section_key: str, include_file_options: bool = False, extra_paths=None):
                    paths = []
                    if include_file_options:
                        paths.extend([
                            (section_key, "batch_processing"),
                            (section_key, "include_subdirectory"),
                            (section_key, "overwrite_existing"),
                            (section_key, "input_folder"),
                            (section_key, "output_folder"),
                        ])
                    paths.extend([(section_key, "file_formats"), (section_key, "add_timestamp")])
                    if extra_paths:
                        paths.extend(extra_paths)
                    for field in WhisperParams.__annotations__.keys():
                        paths.append((section_key, "whisper", field))
                    for field in VadParams.__annotations__.keys():
                        paths.append((section_key, "vad", field))
                    for field in DiarizationParams.__annotations__.keys():
                        paths.append((section_key, "diarization", field))
                    for field in BGMSeparationParams.__annotations__.keys():
                        paths.append((section_key, "bgm_separation", field))
                    return paths

                def _transcription_config_components(transcription_ui: dict, include_file_options: bool = False):
                    components = []
                    if include_file_options:
                        components.extend([
                            cb_batch_processing,
                            cb_include_subdirectory,
                            cb_overwrite_existing,
                            tb_input_folder,
                            tb_output_folder,
                        ])
                    components.extend([transcription_ui["file_formats"], transcription_ui["add_timestamp"]])
                    components.extend(transcription_ui.get("extra_components", []))
                    components.extend(transcription_ui["pipeline"])
                    return components

                config_keys = (
                    _transcription_config_paths("file_tab", include_file_options=True)
                    + _transcription_config_paths(
                        "youtube_tab",
                        extra_paths=[
                            ("youtube_tab", "mass_transcribe_channel"),
                            ("youtube_tab", "latest_video_count"),
                        ],
                    )
                    + _transcription_config_paths("mic_tab")
                    + [
                        ("translation_deepl", "api_key"),
                        ("translation_deepl", "is_pro"),
                        ("translation_deepl", "source_lang"),
                        ("translation_deepl", "target_lang"),
                        ("translation_deepl", "add_timestamp"),
                        ("translation_nllb", "model_size"),
                        ("translation_nllb", "source_lang"),
                        ("translation_nllb", "target_lang"),
                        ("translation_nllb", "max_length"),
                        ("translation_nllb", "add_timestamp"),
                        ("bgm_separation_tab", "uvr_device"),
                        ("bgm_separation_tab", "uvr_model_size"),
                        ("bgm_separation_tab", "segment_size"),
                    ]
                )

                config_components = (
                    _transcription_config_components(file_transcription_ui, include_file_options=True)
                    + _transcription_config_components(youtube_transcription_ui)
                    + _transcription_config_components(mic_transcription_ui)
                    + [
                        tb_api_key,
                        cb_is_pro,
                        deepl_source_lang,
                        deepl_target_lang,
                        deepl_add_timestamp,
                        nllb_model_size,
                        nllb_source_lang,
                        nllb_target_lang,
                        nb_max_length,
                        nllb_add_timestamp,
                        dd_uvr_device,
                        dd_uvr_model_size,
                        nb_uvr_segment_size,
                    ]
                )

                dropdown_choice_specs = {
                    ("file_tab", "whisper", "lang"): WhisperParams.get_language_choices(self.whisper_inf.available_langs),
                    ("youtube_tab", "whisper", "lang"): WhisperParams.get_language_choices(self.whisper_inf.available_langs),
                    ("mic_tab", "whisper", "lang"): WhisperParams.get_language_choices(self.whisper_inf.available_langs),
                    ("file_tab", "whisper", "compute_type"): self.whisper_inf.available_compute_types,
                    ("youtube_tab", "whisper", "compute_type"): self.whisper_inf.available_compute_types,
                    ("mic_tab", "whisper", "compute_type"): self.whisper_inf.available_compute_types,
                    ("file_tab", "diarization", "diarization_device"): self.whisper_inf.diarizer.available_device,
                    ("youtube_tab", "diarization", "diarization_device"): self.whisper_inf.diarizer.available_device,
                    ("mic_tab", "diarization", "diarization_device"): self.whisper_inf.diarizer.available_device,
                    ("file_tab", "bgm_separation", "uvr_model_size"): self.whisper_inf.music_separator.available_models,
                    ("youtube_tab", "bgm_separation", "uvr_model_size"): self.whisper_inf.music_separator.available_models,
                    ("mic_tab", "bgm_separation", "uvr_model_size"): self.whisper_inf.music_separator.available_models,
                    ("file_tab", "bgm_separation", "uvr_device"): self.whisper_inf.music_separator.available_devices,
                    ("youtube_tab", "bgm_separation", "uvr_device"): self.whisper_inf.music_separator.available_devices,
                    ("mic_tab", "bgm_separation", "uvr_device"): self.whisper_inf.music_separator.available_devices,
                    ("translation_deepl", "source_lang"): list(self.deepl_api.available_source_langs.keys()),
                    ("translation_deepl", "target_lang"): list(self.deepl_api.available_target_langs.keys()),
                    ("translation_nllb", "model_size"): self.nllb_inf.available_models,
                    ("translation_nllb", "source_lang"): self.nllb_inf.available_source_langs,
                    ("translation_nllb", "target_lang"): self.nllb_inf.available_target_langs,
                    ("bgm_separation_tab", "uvr_device"): self.whisper_inf.music_separator.available_devices,
                    ("bgm_separation_tab", "uvr_model_size"): self.whisper_inf.music_separator.available_models,
                }

                def _match_dropdown_value(value, choices, default, path=None):
                    if value is None and default is None:
                        return None
                    if path in {
                        ("file_tab", "whisper", "lang"),
                        ("youtube_tab", "whisper", "lang"),
                        ("mic_tab", "whisper", "lang"),
                    }:
                        value = WhisperParams.normalize_lang_choice(value)
                        default = WhisperParams.normalize_lang_choice(default)
                    normalized_choices = {_normalize_choice(choice): choice for choice in choices}
                    normalized_value = _normalize_choice(value)
                    if normalized_value in normalized_choices:
                        return normalized_choices[normalized_value]
                    return default

                def _match_checkbox_group(value, choices, default):
                    if not isinstance(value, list):
                        return list(default)
                    allowed = {_normalize_choice(choice) for choice in choices}
                    matched = [item for item in value if _normalize_choice(item) in allowed]
                    return matched or list(default)

                def _values_to_ui_config(*values):
                    cfg = build_default_ui_config(default_params=self.default_params)
                    for path, value in zip(config_keys, values):
                        set_nested_value(cfg, path, _serialize_ui_value(value))
                    return cfg

                def _ui_config_to_values(cfg: dict):
                    merged = merge_ui_config(cfg, default_params=self.default_params)
                    defaults = build_default_ui_config(default_params=self.default_params)

                    for section_key in ("file_tab", "youtube_tab", "mic_tab"):
                        value = get_nested_value(merged, (section_key, "file_formats"), ["SRT"])
                        default_value = get_nested_value(defaults, (section_key, "file_formats"), ["SRT"])
                        set_nested_value(
                            merged,
                            (section_key, "file_formats"),
                            _match_checkbox_group(value, file_format_choices, default_value),
                        )

                    values = []
                    for path in config_keys:
                        value = get_nested_value(merged, path)
                        if path in dropdown_choice_specs:
                            default_value = get_nested_value(defaults, path)
                            value = _match_dropdown_value(value, dropdown_choice_specs[path], default_value, path=path)
                        values.append(value)
                    return values

                def _load_preset_values_and_status(preset_name: str):
                    if not preset_name:
                        values = _ui_config_to_values(build_default_ui_config(default_params=self.default_params))
                        return values, "No preset selected. Showing defaults."

                    safe_name = sanitize_preset_name(preset_name)
                    cfg = load_ui_preset(safe_name, default_params=self.default_params)
                    if not cfg:
                        if get_last_used_ui_preset() == safe_name:
                            clear_last_used_ui_preset()
                        values = _ui_config_to_values(build_default_ui_config(default_params=self.default_params))
                        return values, f"Preset **{preset_name}** was not found. Loaded defaults."

                    set_last_used_ui_preset(safe_name)
                    values = _ui_config_to_values(cfg)
                    return values, f"Loaded preset **{safe_name}**"

                def _save_preset_ui(preset_name: str, *values):
                    try:
                        cfg = _values_to_ui_config(*values)
                        saved = save_ui_preset(preset_name, cfg, default_params=self.default_params)
                        loaded_values, _ = _load_preset_values_and_status(saved)
                        return (
                            gr.update(choices=list_ui_presets(), value=saved),
                            *loaded_values,
                            f"Saved and loaded preset **{saved}**",
                        )
                    except Exception as exc:
                        return (gr.update(), *values, f"Save failed: {exc}")

                def _load_preset_ui(preset_name: str):
                    values, status = _load_preset_values_and_status(preset_name)
                    return (*values, status)

                def _reset_defaults_ui():
                    values = _ui_config_to_values(build_default_ui_config(default_params=self.default_params))
                    return (*values, "Reset to defaults")

                def _delete_preset_ui(preset_name: str):
                    if not preset_name:
                        return gr.update(), "No preset selected"
                    if is_locked_ui_preset(preset_name):
                        return gr.update(choices=list_ui_presets(), value=preset_name), (
                            f"Preset **{preset_name}** is built in and cannot be deleted."
                        )
                    ok = delete_ui_preset(preset_name)
                    presets = list_ui_presets()
                    if ok:
                        if get_last_used_ui_preset() == sanitize_preset_name(preset_name):
                            clear_last_used_ui_preset()
                        return gr.update(choices=presets, value=None), f"Deleted preset **{preset_name}**"
                    return gr.update(choices=presets), f"Could not delete preset **{preset_name}**"

                ui_preset_save_btn.click(
                    fn=_save_preset_ui,
                    inputs=[ui_preset_name] + config_components,
                    outputs=[ui_preset_dropdown] + config_components + [ui_preset_status],
                    queue=False,
                    show_progress="hidden",
                )
                ui_preset_dropdown.change(
                    fn=_load_preset_ui,
                    inputs=[ui_preset_dropdown],
                    outputs=config_components + [ui_preset_status],
                    queue=False,
                    show_progress="hidden",
                )
                ui_preset_reset_btn.click(
                    fn=_reset_defaults_ui,
                    inputs=[],
                    outputs=config_components + [ui_preset_status],
                    queue=False,
                    show_progress="hidden",
                )
                ui_preset_delete_btn.click(
                    fn=_delete_preset_ui,
                    inputs=[ui_preset_dropdown],
                    outputs=[ui_preset_dropdown, ui_preset_status],
                    queue=False,
                    show_progress="hidden",
                )

        repair_blocks_text(self.app)
        args = self.args
        return self.app.queue(api_open=args.api_open).launch(
            share=args.share,
            inbrowser=args.inbrowser,
            auth=(args.username, args.password) if args.username and args.password else None,
            root_path=args.root_path,
            favicon_path=FAVICON_PATH,
            ssl_verify=args.ssl_verify,
            ssl_keyfile=args.ssl_keyfile,
            ssl_keyfile_password=args.ssl_keyfile_password,
            ssl_certfile=args.ssl_certfile,
            allowed_paths=self.allowed_paths,
            server_name=args.server_name,
            server_port=args.server_port,
            theme=args.theme,
            css=CSS,
            head=HEAD,
            prevent_thread_lock=prevent_thread_lock,
            quiet=quiet,
        )

    @staticmethod
    def open_folder(folder_path: str):
        absolute_path = os.path.abspath(folder_path)
        os.makedirs(absolute_path, exist_ok=True)

        if os.name == "nt":
            os.startfile(absolute_path)
            return

        opener = "open" if sys.platform == "darwin" else "xdg-open"
        if shutil.which(opener) is None:
            logger.warning(
                f"Unable to open folder automatically because '{opener}' is not available: {absolute_path}"
            )
            return

        subprocess.Popen(
            [opener, absolute_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


parser = argparse.ArgumentParser()
parser.add_argument("--whisper_type", type=str, default=WhisperImpl.FASTER_WHISPER.value, choices=[item.value for item in WhisperImpl])
parser.add_argument("--share", type=str2bool, default=False, nargs="?", const=True)
parser.add_argument("--server_name", type=str, default=None)
parser.add_argument("--server_port", type=int, default=None)
parser.add_argument("--root_path", type=str, default=None)
parser.add_argument("--username", type=str, default=None)
parser.add_argument("--password", type=str, default=None)
parser.add_argument("--theme", type=str, default="soft")
parser.add_argument("--colab", type=str2bool, default=False, nargs="?", const=True)
parser.add_argument("--api_open", type=str2bool, default=False, nargs="?", const=True)
parser.add_argument("--allowed_paths", type=str, default=None)
parser.add_argument("--inbrowser", type=str2bool, default=True, nargs="?", const=True)
parser.add_argument("--ssl_verify", type=str2bool, default=True, nargs="?", const=True)
parser.add_argument("--ssl_keyfile", type=str, default=None)
parser.add_argument("--ssl_keyfile_password", type=str, default=None)
parser.add_argument("--ssl_certfile", type=str, default=None)
parser.add_argument("--whisper_model_dir", type=str, default=WHISPER_MODELS_DIR)
parser.add_argument("--faster_whisper_model_dir", type=str, default=FASTER_WHISPER_MODELS_DIR)
parser.add_argument("--insanely_fast_whisper_model_dir", type=str, default=INSANELY_FAST_WHISPER_MODELS_DIR)
parser.add_argument("--diarization_model_dir", type=str, default=DIARIZATION_MODELS_DIR)
parser.add_argument("--nllb_model_dir", type=str, default=NLLB_MODELS_DIR)
parser.add_argument("--uvr_model_dir", type=str, default=UVR_MODELS_DIR)
parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
_args = parser.parse_args()


if __name__ == "__main__":
    app = App(args=_args)
    app.launch()
