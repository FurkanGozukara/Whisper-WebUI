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

from modules.translation.deepl_api import DeepLAPI
from modules.runtime.subprocess_client import build_runtime_proxies
from modules.ui.htmls import *
from modules.ui.presets import (
    build_default_ui_config,
    clear_last_used_ui_preset,
    delete_ui_preset,
    get_last_used_ui_preset,
    get_nested_value,
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
from modules.utils.youtube_manager import get_ytmetas
from modules.whisper.data_classes import *


logger = get_logger()

FAVICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "favicon.svg")
APP_TITLE = "Whisper TTS Premium App by SECourses V4.0 : https://www.patreon.com/posts/whisper-webui-to-145395299"
TIMESTAMP_INFO = (
    "Adds the current date and time to the output filename. "
    "Enable this if you want each run to create a unique file and avoid overwriting older outputs. "
    "Disable it if you want shorter, stable filenames."
)
BATCH_SIZE_CALIBRATION_FILENAME = "batch_size_calibration.json"
BATCH_SIZE_CALIBRATION_MEMORY_TOLERANCE_GB = 1.0


class App:
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
                info="ðŸ” Number of beams in beam search. Higher = more accurate but slower. Range: 1-20. Examples: 5 (balanced), 10 (high accuracy), 1 (fastest/greedy). Current: optimized at 10 for maximum English accuracy.",
            ))
            inputs.append(gr.Number(
                label="Log Probability Threshold",
                value=whisper_params["log_prob_threshold"],
                info="ðŸ“Š Rejects segments with average log probability below this. Lower (more negative) = stricter quality control. Examples: -1.0 (default), -0.5 (strict - rejects uncertain outputs), -1.5 (lenient). Current: -0.5 for high quality.",
            ))
            inputs.append(gr.Number(
                label="No Speech Threshold",
                value=whisper_params["no_speech_threshold"],
                info="ðŸ”‡ Probability threshold for detecting silence/no-speech. Range: 0.0-1.0. Examples: 0.6 (balanced), 0.4 (detects more speech in noisy audio), 0.8 (strict silence detection). Lower if audio has background noise.",
            ))

        with gr.Row():
            inputs.append(gr.Dropdown(
                label="Compute Type",
                choices=self.whisper_inf.available_compute_types,
                value=self.whisper_inf.current_compute_type,
                info="âš™ï¸ Precision for model computation. float32 (most accurate, 2x VRAM), float16 (balanced - recommended), int8 (fastest, less accurate). Use float16 for GPU, float32 for CPU. Current: float16 (optimal balance).",
            ))
            inputs.append(gr.Number(
                label="Best Of",
                value=whisper_params["best_of"],
                precision=0,
                info="ðŸŽ¯ Number of candidate sequences to generate when sampling (when temperature > 0). Higher = better quality but slower. Range: 1-20. Examples: 5 (default), 10 (high quality), 1 (fastest). Current: 10 for maximum accuracy.",
            ))
            inputs.append(gr.Number(
                label="Patience",
                value=whisper_params["patience"],
                info="â³ Beam search patience: how long to wait for better candidates. Higher = more thorough search. Examples: 1.0 (default), 2.0 (very thorough - current setting), 0.5 (faster). Increase for complex audio.",
            ))

        with gr.Row():
            inputs.append(gr.Checkbox(
                label="Condition On Previous Text",
                value=whisper_params["condition_on_previous_text"],
                info="ðŸ”— Use previous transcription as context for next segment. âœ… Recommended ON for better coherence and flow. Disable if getting stuck in repetitive loops. Helps maintain context across segments.",
            ))
            inputs.append(gr.Slider(
                label="Prompt Reset On Temperature",
                value=whisper_params["prompt_reset_on_temperature"],
                minimum=0,
                maximum=1,
                step=0.01,
                info="ðŸŒ¡ï¸ Reset conditioning prompt if temperature exceeds this value. Range: 0.0-1.0. Examples: 0.5 (default - balanced), 0.3 (reset more often), 0.7 (reset less often). Prevents getting stuck in bad outputs.",
            ))
            inputs.append(gr.Textbox(
                label="Initial Prompt",
                value=whisper_params.get("initial_prompt", ""),
                info="ðŸ’¬ Text to guide transcription style/vocabulary. Examples: 'Medical terminology:', 'Interview with Dr. Smith about AI', 'Technical lecture on Python'. Helps with domain-specific terms. Leave empty for general transcription.",
            ))

        with gr.Row():
            inputs.append(gr.Slider(
                label="Temperature",
                value=whisper_params["temperature"],
                minimum=0.0,
                step=0.01,
                maximum=1.0,
                info="ðŸŽ² Randomness in decoding. 0.0 = deterministic (most accurate - recommended), 0.2-0.5 = slight variation, 0.8-1.0 = creative but less accurate. Use 0 for maximum accuracy. Current: 0 (optimal for accuracy).",
            ))
            inputs.append(gr.Number(
                label="Compression Ratio Threshold",
                value=whisper_params["compression_ratio_threshold"],
                info="ðŸ“¦ Detects repetitive/hallucinated text by gzip compression ratio. If text compresses too much (< threshold), it's likely repetitive. Examples: 2.4 (default), 2.0 (stricter), 3.0 (lenient). Lower = catches more hallucinations.",
            ))
            inputs.append(gr.Number(
                label="Length Penalty",
                value=whisper_params["length_penalty"],
                info="ðŸ“ Penalty for longer sequences. >1.0 = favors longer outputs, <1.0 = favors shorter outputs. Examples: 1.0 (neutral - default), 1.2 (encourages longer segments), 0.8 (encourages shorter segments). Use 1.0 for balanced output.",
            ))

        return inputs

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
                                "When Use Batched Inference is disabled, this loads that many separate model "
                                "instances, splits the audio into equal time ranges, transcribes each range "
                                "independently, and then merges the result with corrected timestamps. When Use "
                                "Batched Inference is enabled, it controls the faster-whisper batched decoder "
                                "path instead."
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
        startup_preset_name = get_last_used_ui_preset()
        default_ui_config = merge_ui_config(self.ui_default_config, default_params=self.default_params)
        startup_preset_status = ""
        if startup_preset_name:
            startup_cfg = load_ui_preset(startup_preset_name, default_params=self.default_params)
            if startup_cfg:
                default_ui_config = startup_cfg
                startup_preset_status = f"Loaded last used preset **{startup_preset_name}**"
            else:
                clear_last_used_ui_preset()
                startup_preset_name = None
                startup_preset_status = "Last used preset was not found. Loaded defaults."

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
                        with gr.Accordion("Config Presets (Save / Load)", open=True):
                            with gr.Row():
                                ui_preset_dropdown = gr.Dropdown(
                                    label="Select Preset",
                                    choices=list_ui_presets(),
                                    value=startup_preset_name,
                                )
                                ui_preset_name = gr.Textbox(label="New Preset Name", placeholder="my_preset")
                            with gr.Row():
                                ui_preset_save_btn = gr.Button("Save", variant="primary")
                                ui_preset_load_btn = gr.Button("Load Selected")
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
                        with gr.Row(equal_height=True):
                            with gr.Column():
                                img_thumbnail = gr.Image(label=_("Youtube Thumbnail"))
                            with gr.Column():
                                tb_title = gr.Label(label=_("Youtube Title"))
                                tb_description = gr.Textbox(label=_("Youtube Description"), max_lines=15)

                        youtube_transcription_ui = self.create_pipeline_inputs(youtube_defaults)

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
                        ]
                        youtube_run_event = youtube_transcription_ui["run_button"].click(
                            fn=self.whisper_inf.transcribe_youtube,
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
                        with gr.Row():
                            mic_input = gr.Microphone(
                                label=_("Record with Mic"),
                                type="filepath",
                                interactive=True,
                                buttons=["download"],
                            )

                        mic_transcription_ui = self.create_pipeline_inputs(mic_defaults)

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
                            mic_output = gr.Textbox(label=_("Output"), scale=5)
                            mic_outputs = gr.Files(label=_("Downloadable output file"), scale=4)

                        mic_inputs = [
                            mic_input,
                            mic_transcription_ui["file_formats"],
                            mic_transcription_ui["add_timestamp"],
                        ]
                        mic_run_event = mic_transcription_ui["run_button"].click(
                            fn=self.whisper_inf.transcribe_mic,
                            inputs=mic_inputs + mic_transcription_ui["pipeline"],
                            outputs=[mic_output, mic_outputs],
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

                def _transcription_config_paths(section_key: str, include_file_options: bool = False):
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
                    components.extend(transcription_ui["pipeline"])
                    return components

                config_keys = (
                    _transcription_config_paths("file_tab", include_file_options=True)
                    + _transcription_config_paths("youtube_tab")
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

                def _save_preset_ui(preset_name: str, *values):
                    try:
                        cfg = _values_to_ui_config(*values)
                        saved = save_ui_preset(preset_name, cfg, default_params=self.default_params)
                        set_last_used_ui_preset(saved)
                        return gr.update(choices=list_ui_presets(), value=saved), f"Saved preset **{saved}**"
                    except Exception as exc:
                        return gr.update(), f"Save failed: {exc}"

                def _load_preset_ui(preset_name: str):
                    if not preset_name:
                        values = _ui_config_to_values(build_default_ui_config(default_params=self.default_params))
                        return (*values, "No preset selected. Showing defaults.")

                    cfg = load_ui_preset(preset_name, default_params=self.default_params)
                    if not cfg:
                        if get_last_used_ui_preset() == sanitize_preset_name(preset_name):
                            clear_last_used_ui_preset()
                        values = _ui_config_to_values(build_default_ui_config(default_params=self.default_params))
                        return (*values, f"Preset **{preset_name}** was not found. Loaded defaults.")

                    set_last_used_ui_preset(preset_name)
                    values = _ui_config_to_values(cfg)
                    return (*values, f"Loaded preset **{preset_name}**")

                def _reset_defaults_ui():
                    values = _ui_config_to_values(build_default_ui_config(default_params=self.default_params))
                    return (*values, "Reset to defaults")

                def _delete_preset_ui(preset_name: str):
                    if not preset_name:
                        return gr.update(), "No preset selected"
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
                    outputs=[ui_preset_dropdown, ui_preset_status],
                    queue=False,
                    show_progress="hidden",
                )
                ui_preset_load_btn.click(
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
