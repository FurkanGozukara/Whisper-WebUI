import argparse
import os

import gradio as gr

from modules.translation.deepl_api import DeepLAPI
from modules.translation.nllb_inference import NLLBInference
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
from modules.utils.files_manager import MEDIA_EXTENSION, load_yaml
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
from modules.utils.torch_compat import enable_torch_2_6_weights_only_compat
from modules.utils.youtube_manager import get_ytmetas
from modules.whisper.data_classes import *
from modules.whisper.whisper_factory import WhisperFactory


logger = get_logger()
enable_torch_2_6_weights_only_compat()
FAVICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "favicon.svg")
APP_TITLE = "Whisper TTS Premium App by SECourses V4.0 : https://www.patreon.com/posts/whisper-webui-to-145395299"


class App:
    def __init__(self, args):
        self.args = args
        self.title = APP_TITLE
        self.app = gr.Blocks(
            title=self.title,
            delete_cache=(3600, 86400),
        )
        self.whisper_inf = WhisperFactory.create_whisper_inference(
            whisper_type=self.args.whisper_type,
            whisper_model_dir=self.args.whisper_model_dir,
            faster_whisper_model_dir=self.args.faster_whisper_model_dir,
            insanely_fast_whisper_model_dir=self.args.insanely_fast_whisper_model_dir,
            uvr_model_dir=self.args.uvr_model_dir,
            output_dir=self.args.output_dir,
        )
        self.nllb_inf = NLLBInference(
            model_dir=self.args.nllb_model_dir,
            output_dir=os.path.join(self.args.output_dir, "translations"),
        )
        self.deepl_api = DeepLAPI(
            output_dir=os.path.join(self.args.output_dir, "translations"),
        )
        self.i18n = load_yaml(I18N_YAML_PATH)
        self.default_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
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

    def create_pipeline_inputs(self, defaults=None):
        transcription_defaults = defaults or self.ui_default_config["file_tab"]
        whisper_params = transcription_defaults["whisper"]
        vad_params = transcription_defaults["vad"]
        diarization_params = transcription_defaults["diarization"]
        uvr_params = transcription_defaults["bgm_separation"]

        with gr.Row():
            dd_model = gr.Dropdown(
                choices=self.whisper_inf.available_models,
                value=whisper_params["model_size"],
                label=_("Model"),
                allow_custom_value=True,
            )
            dd_lang = gr.Dropdown(
                choices=self.whisper_inf.available_langs + [AUTOMATIC_DETECTION],
                value=AUTOMATIC_DETECTION if whisper_params["lang"] == AUTOMATIC_DETECTION.unwrap() else whisper_params["lang"],
                label=_("Language"),
            )
            cg_file_formats = gr.CheckboxGroup(
                choices=["SRT", "WebVTT", "txt", "LRC", "JSON", "TSV"],
                value=transcription_defaults.get("file_formats", ["SRT"]) or ["SRT"],
                label=_("File Formats"),
                info=_("Select one or more output formats."),
            )
        with gr.Row():
            cb_translate = gr.Checkbox(
                value=whisper_params["is_translate"],
                label=_("Translate to English?"),
                info="Whisper's End-To-End Speech-To-Text translation feature",
                interactive=True,
            )
            cb_timestamp = gr.Checkbox(
                value=transcription_defaults.get("add_timestamp", False),
                label=_("Add a timestamp to the end of the filename"),
                interactive=True,
            )
            batch_size_input = WhisperParams.to_batch_size_input(
                defaults=whisper_params,
                whisper_type=self.args.whisper_type,
            )

        with gr.Accordion(_("Advanced Parameters"), open=False):
            whisper_inputs = WhisperParams.to_gradio_inputs(
                defaults=whisper_params,
                only_advanced=True,
                whisper_type=self.args.whisper_type,
                available_compute_types=self.whisper_inf.available_compute_types,
                compute_type=self.whisper_inf.current_compute_type,
                include_batch_size=False,
            )

        whisper_advanced_fields = WhisperParams.advanced_input_field_names()
        batch_size_index = whisper_advanced_fields.index("batch_size")
        whisper_inputs.insert(batch_size_index, batch_size_input)

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
                label=_("Language"),
                interactive=True,
                visible=False,
            )
            with Translate(self.i18n):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"### {self.title}", elem_id="md_project")

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

                with gr.Tabs():
                    with gr.TabItem(_("File")):
                        with gr.Row():
                            with gr.Column():
                                input_file = gr.Files(
                                    type="filepath",
                                    label=_("Upload File here"),
                                    file_types=MEDIA_EXTENSION,
                                )
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
                        file_transcription_ui = self.create_pipeline_inputs(file_defaults)

                        with gr.Row():
                            file_run_btn = gr.Button(_("GENERATE SUBTITLE FILE"), variant="primary")
                        with gr.Row():
                            tb_live_transcription = gr.Textbox(
                                label=_("Live Transcription"),
                                lines=10,
                                max_lines=15,
                                interactive=False,
                                buttons=["copy"],
                                placeholder="Transcribed segments will appear here in real-time...",
                            )
                        with gr.Row():
                            file_output = gr.Textbox(label=_("Output"), scale=5)
                            file_outputs = gr.Files(label=_("Downloadable output file"), scale=3, interactive=False)
                            file_open_btn = gr.Button('Open', scale=1)

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
                        file_run_btn.click(
                            fn=self.whisper_inf.transcribe_file_with_live_output,
                            inputs=file_inputs + file_transcription_ui["pipeline"],
                            outputs=[tb_live_transcription, file_output, file_outputs],
                        )
                        file_open_btn.click(
                            fn=lambda output_dir, batch_input, batch_enabled: self.open_folder(
                                output_dir if output_dir else (
                                    batch_input if (batch_enabled and batch_input) else self.args.output_dir
                                )
                            ),
                            inputs=[tb_output_folder, tb_input_folder, cb_batch_processing],
                            outputs=None,
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
                            youtube_run_btn = gr.Button(_("GENERATE SUBTITLE FILE"), variant="primary")
                        with gr.Row():
                            gr.Textbox(
                                label=_("Live Transcription"),
                                lines=10,
                                max_lines=15,
                                interactive=False,
                                buttons=["copy"],
                                placeholder="Transcribed segments will appear here in real-time...",
                            )
                        with gr.Row():
                            youtube_output = gr.Textbox(label=_("Output"), scale=5)
                            youtube_outputs = gr.Files(label=_("Downloadable output file"), scale=3)
                            youtube_open_btn = gr.Button('Open', scale=1)

                        youtube_inputs = [
                            tb_youtubelink,
                            youtube_transcription_ui["file_formats"],
                            youtube_transcription_ui["add_timestamp"],
                        ]
                        youtube_run_btn.click(
                            fn=self.whisper_inf.transcribe_youtube,
                            inputs=youtube_inputs + youtube_transcription_ui["pipeline"],
                            outputs=[youtube_output, youtube_outputs],
                        )
                        tb_youtubelink.change(get_ytmetas, inputs=[tb_youtubelink], outputs=[img_thumbnail, tb_title, tb_description])
                        youtube_open_btn.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)

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
                            mic_run_btn = gr.Button(_("GENERATE SUBTITLE FILE"), variant="primary")
                        with gr.Row():
                            gr.Textbox(
                                label=_("Live Transcription"),
                                lines=10,
                                max_lines=15,
                                interactive=False,
                                buttons=["copy"],
                                placeholder="Transcribed segments will appear here in real-time...",
                            )
                        with gr.Row():
                            mic_output = gr.Textbox(label=_("Output"), scale=5)
                            mic_outputs = gr.Files(label=_("Downloadable output file"), scale=3)
                            mic_open_btn = gr.Button('Open', scale=1)

                        mic_inputs = [
                            mic_input,
                            mic_transcription_ui["file_formats"],
                            mic_transcription_ui["add_timestamp"],
                        ]
                        mic_run_btn.click(
                            fn=self.whisper_inf.transcribe_mic,
                            inputs=mic_inputs + mic_transcription_ui["pipeline"],
                            outputs=[mic_output, mic_outputs],
                        )
                        mic_open_btn.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)

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
                    ("file_tab", "whisper", "lang"): self.whisper_inf.available_langs + [AUTOMATIC_DETECTION],
                    ("youtube_tab", "whisper", "lang"): self.whisper_inf.available_langs + [AUTOMATIC_DETECTION],
                    ("mic_tab", "whisper", "lang"): self.whisper_inf.available_langs + [AUTOMATIC_DETECTION],
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

                def _match_dropdown_value(value, choices, default):
                    if value is None and default is None:
                        return None
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
                            value = _match_dropdown_value(value, dropdown_choice_specs[path], default_value)
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
            prevent_thread_lock=prevent_thread_lock,
            quiet=quiet,
        )

    @staticmethod
    def open_folder(folder_path: str):
        if os.path.exists(folder_path):
            os.system(f"start {folder_path}")
        else:
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"The directory path {folder_path} has newly created.")


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
