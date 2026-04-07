from pathlib import Path

from modules.translation.deepl_api import DeepLAPI
from modules.translation.translation_base import TranslationBase
from modules.ui import presets as ui_presets
from modules.ui.presets import (
    build_default_ui_config,
    clear_last_used_ui_preset,
    delete_ui_preset,
    get_last_used_ui_preset,
    last_used_ui_preset_path,
    list_ui_presets,
    load_ui_preset,
    merge_ui_config,
    save_ui_preset,
    set_last_used_ui_preset,
)
from modules.utils.paths import CONFIGS_DIR, DEFAULT_PARAMETERS_CONFIG_PATH
from modules.uvr.music_separator import MusicSeparator
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.whisper.data_classes import (
    BGMSeparationParams,
    DiarizationParams,
    TranscriptionPipelineParams,
    VadParams,
    WhisperParams,
)


def test_default_parameters_path_is_outside_configs():
    default_path = Path(DEFAULT_PARAMETERS_CONFIG_PATH).resolve()
    configs_dir = Path(CONFIGS_DIR).resolve()

    assert default_path.is_file()
    assert default_path.parent != configs_dir
    assert configs_dir not in default_path.parents


def test_ui_presets_round_trip_and_backfill_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_presets, "PRESETS_DIR", str(tmp_path))

    partial_cfg = {
        "file_tab": {
            "batch_processing": True,
            "input_folder": "D:/audio",
        },
        "translation_deepl": {
            "api_key": "secret",
        },
    }

    saved_name = save_ui_preset("my preset", partial_cfg)
    loaded_cfg = load_ui_preset(saved_name)

    assert saved_name == "my_preset"
    assert list_ui_presets() == ["my_preset"]
    assert loaded_cfg is not None
    assert loaded_cfg["file_tab"]["batch_processing"] is True
    assert loaded_cfg["file_tab"]["input_folder"] == "D:/audio"
    assert "whisper" in loaded_cfg["file_tab"]
    assert "youtube_tab" in loaded_cfg
    assert loaded_cfg["translation_deepl"]["api_key"] == "secret"

    assert delete_ui_preset(saved_name) is True
    assert list_ui_presets() == []


def test_merge_ui_config_restores_missing_sections():
    merged = merge_ui_config({"file_tab": {"batch_processing": True}})
    defaults = build_default_ui_config()

    assert merged["file_tab"]["batch_processing"] is True
    assert merged["mic_tab"] == defaults["mic_tab"]
    assert merged["file_tab"]["whisper"]["model_size"] == defaults["file_tab"]["whisper"]["model_size"]
    assert merged["file_tab"]["whisper"]["lang"] == "english"


def test_whisper_lang_is_normalized_for_ui_and_runtime():
    defaults = build_default_ui_config()

    assert defaults["file_tab"]["whisper"]["lang"] == "english"
    assert defaults["file_tab"]["whisper"]["word_timestamps"] is True
    assert defaults["file_tab"]["whisper"]["use_batched_inference"] is False
    assert WhisperParams(lang="English").lang == "en"
    assert WhisperParams(lang="english").lang == "en"
    assert WhisperParams(lang="en").lang == "en"
    assert WhisperParams(lang="Automatic Detection").lang is None
    assert WhisperParams.normalize_lang_choice("English") == "english"
    assert WhisperParams.normalize_lang_choice("en") == "english"


def test_last_used_preset_is_persisted_and_cleared_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_presets, "PRESETS_DIR", str(tmp_path))

    saved_name = save_ui_preset("daily setup", {"file_tab": {"batch_processing": True}})
    remembered_name = set_last_used_ui_preset(saved_name)

    assert remembered_name == "daily_setup"
    assert get_last_used_ui_preset() == "daily_setup"
    assert last_used_ui_preset_path().read_text(encoding="utf-8").strip() == "daily_setup"

    assert delete_ui_preset(saved_name) is True
    assert get_last_used_ui_preset() is None
    assert not last_used_ui_preset_path().exists()

    clear_last_used_ui_preset()


def test_runtime_parameter_caching_is_disabled():
    before = Path(DEFAULT_PARAMETERS_CONFIG_PATH).read_text(encoding="utf-8")

    BaseTranscriptionPipeline.cache_parameters(
        TranscriptionPipelineParams(
            whisper=WhisperParams(),
            vad=VadParams(),
            diarization=DiarizationParams(),
            bgm_separation=BGMSeparationParams(),
        ),
        "SRT",
        True,
    )
    TranslationBase.cache_parameters("model", "eng_Latn", "kor_Hang", 200, True)
    DeepLAPI.cache_parameters("key", False, "English", "Korean", True)
    MusicSeparator.cache_parameters("UVR-MDX-NET-Inst_HQ_4", 256)

    after = Path(DEFAULT_PARAMETERS_CONFIG_PATH).read_text(encoding="utf-8")
    assert after == before
