import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from modules.utils.constants import (
    AUTOMATIC_DETECTION,
    GRADIO_NONE_NUMBER_MAX,
    GRADIO_NONE_NUMBER_MIN,
    GRADIO_NONE_STR,
)
from modules.utils.files_manager import load_yaml
from modules.utils.paths import DEFAULT_PARAMETERS_CONFIG_PATH, PRESETS_DIR, UI_SYSTEM_PRESETS_DIR
from modules.utils.whisper_languages import normalize_lang_choice as normalize_whisper_lang_choice

UI_PRESET_VERSION = "1.0"
UI_PRESET_FORMAT = "whisper_webui_ui"
LAST_USED_UI_PRESET_FILENAME = "last_used_ui_preset.txt"
DEFAULT_STARTUP_UI_PRESET = "canary_best_quality"


def sanitize_preset_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(name))
    return safe.strip("._") or "default"


def ui_preset_path(preset_name: str) -> Path:
    return Path(PRESETS_DIR) / f"{sanitize_preset_name(preset_name)}.json"


def system_ui_preset_path(preset_name: str) -> Path:
    return Path(UI_SYSTEM_PRESETS_DIR) / f"{sanitize_preset_name(preset_name)}.json"


def find_ui_preset_path(preset_name: str) -> Optional[Path]:
    safe_name = sanitize_preset_name(preset_name)
    system_path = system_ui_preset_path(safe_name)
    if system_path.exists():
        return system_path

    user_path = ui_preset_path(safe_name)
    if user_path.exists():
        return user_path

    return None


def is_locked_ui_preset(preset_name: str) -> bool:
    if not preset_name:
        return False
    return system_ui_preset_path(preset_name).exists()


def last_used_ui_preset_path() -> Path:
    return Path(PRESETS_DIR) / LAST_USED_UI_PRESET_FILENAME


def list_ui_presets() -> list[str]:
    preset_names = set()
    for root in (Path(UI_SYSTEM_PRESETS_DIR), Path(PRESETS_DIR)):
        if not root.exists():
            continue
        preset_names.update(path.stem for path in root.glob("*.json") if path.is_file())
    return sorted(preset_names)


def clear_last_used_ui_preset() -> None:
    path = last_used_ui_preset_path()
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def set_last_used_ui_preset(preset_name: Optional[str]) -> Optional[str]:
    if not preset_name or not str(preset_name).strip():
        clear_last_used_ui_preset()
        return None

    safe_name = sanitize_preset_name(str(preset_name).strip())
    path = last_used_ui_preset_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(safe_name, encoding="utf-8")
    tmp_path.replace(path)
    return safe_name


def get_last_used_ui_preset() -> Optional[str]:
    path = last_used_ui_preset_path()
    if not path.exists():
        return None

    try:
        raw_name = path.read_text(encoding="utf-8").strip()
    except Exception:
        clear_last_used_ui_preset()
        return None

    if not raw_name:
        clear_last_used_ui_preset()
        return None

    safe_name = sanitize_preset_name(raw_name)
    if find_ui_preset_path(safe_name) is None:
        clear_last_used_ui_preset()
        return None

    return safe_name


def get_default_startup_ui_preset(preset_name: str = DEFAULT_STARTUP_UI_PRESET) -> Optional[str]:
    safe_name = sanitize_preset_name(preset_name)
    if find_ui_preset_path(safe_name) is None:
        return None
    return safe_name


def get_nested_value(data: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def set_nested_value(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = data
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _as_ui_checkbox_group(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value:
        return [value]
    return list(fallback)


def _as_ui_optional_text(value: Any) -> str:
    if value is None:
        return GRADIO_NONE_STR
    return str(value)


def _as_ui_optional_number(value: Any) -> Any:
    if value is None:
        return GRADIO_NONE_NUMBER_MIN
    return value


def _normalize_ui_lang(value: Any) -> str:
    return normalize_whisper_lang_choice(value, AUTOMATIC_DETECTION.unwrap())


def _normalize_whisper_defaults(defaults: dict[str, Any]) -> dict[str, Any]:
    whisper = deepcopy(defaults)
    whisper["lang"] = _normalize_ui_lang(whisper.get("lang"))
    for key in ("initial_prompt", "prefix", "hotwords", "canary_generation_kwargs"):
        whisper[key] = _as_ui_optional_text(whisper.get(key))
    for key in ("max_new_tokens", "hallucination_silence_threshold", "language_detection_threshold"):
        whisper[key] = _as_ui_optional_number(whisper.get(key))

    suppress_tokens = whisper.get("suppress_tokens")
    if isinstance(suppress_tokens, list):
        whisper["suppress_tokens"] = str(suppress_tokens)
    elif suppress_tokens in (None, ""):
        whisper["suppress_tokens"] = "[-1]"
    else:
        whisper["suppress_tokens"] = str(suppress_tokens)

    return whisper


def _normalize_vad_defaults(defaults: dict[str, Any]) -> dict[str, Any]:
    vad = deepcopy(defaults)
    if vad.get("max_speech_duration_s") == float("inf"):
        vad["max_speech_duration_s"] = GRADIO_NONE_NUMBER_MAX
    return vad


def _transcription_tab_defaults(default_params: dict[str, Any], include_file_options: bool) -> dict[str, Any]:
    whisper = _normalize_whisper_defaults(default_params.get("whisper", {}))
    file_formats = _as_ui_checkbox_group(whisper.pop("file_format", ["SRT"]), ["SRT"])
    add_timestamp = bool(whisper.pop("add_timestamp", False))

    section = {
        "file_formats": file_formats,
        "add_timestamp": add_timestamp,
        "whisper": whisper,
        "vad": _normalize_vad_defaults(default_params.get("vad", {})),
        "diarization": deepcopy(default_params.get("diarization", {})),
        "bgm_separation": deepcopy(default_params.get("bgm_separation", {})),
    }
    if include_file_options:
        file_defaults = deepcopy(default_params.get("file", {}))
        section.update({
            "batch_processing": bool(file_defaults.get("batch_processing", False)),
            "include_subdirectory": bool(file_defaults.get("include_subdirectory", False)),
            "overwrite_existing": bool(file_defaults.get("overwrite_existing", False)),
            "input_folder": str(file_defaults.get("input_folder", "")),
            "output_folder": str(file_defaults.get("output_folder", "")),
        })
    return section


def build_default_ui_config(default_params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    params = deepcopy(default_params) if default_params is not None else load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
    translation = deepcopy(params.get("translation", {}))
    deepl = deepcopy(translation.get("deepl", {}))
    nllb = deepcopy(translation.get("nllb", {}))
    youtube_tab = _transcription_tab_defaults(params, include_file_options=False)
    youtube_tab.update({
        "mass_transcribe_channel": False,
        "latest_video_count": 100,
    })

    return {
        "_meta": {
            "version": UI_PRESET_VERSION,
            "format": UI_PRESET_FORMAT,
        },
        "file_tab": _transcription_tab_defaults(params, include_file_options=True),
        "youtube_tab": youtube_tab,
        "mic_tab": _transcription_tab_defaults(params, include_file_options=False),
        "translation_deepl": {
            "api_key": str(deepl.get("api_key", "")),
            "is_pro": bool(deepl.get("is_pro", False)),
            "source_lang": deepl.get("source_lang") or AUTOMATIC_DETECTION.unwrap(),
            "target_lang": deepl.get("target_lang", "English"),
            "add_timestamp": bool(translation.get("add_timestamp", True)),
        },
        "translation_nllb": {
            "model_size": nllb.get("model_size"),
            "source_lang": nllb.get("source_lang"),
            "target_lang": nllb.get("target_lang"),
            "max_length": nllb.get("max_length", 200),
            "add_timestamp": bool(translation.get("add_timestamp", True)),
        },
        "bgm_separation_tab": {
            "uvr_device": params.get("bgm_separation", {}).get("uvr_device"),
            "uvr_model_size": params.get("bgm_separation", {}).get("uvr_model_size"),
            "segment_size": params.get("bgm_separation", {}).get("segment_size", 256),
        },
    }


def _coerce_like_default(value: Any, default: Any) -> Any:
    if isinstance(default, dict):
        if not isinstance(value, dict):
            return deepcopy(default)
        merged = deepcopy(default)
        for key, default_value in default.items():
            if key in value:
                merged[key] = _coerce_like_default(value[key], default_value)
        return merged

    if isinstance(default, list):
        if not isinstance(value, list):
            return list(default)
        return deepcopy(value)

    if isinstance(default, bool):
        return value if isinstance(value, bool) else default

    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    if isinstance(default, str):
        if value is None:
            return default
        return str(value)

    if default is None:
        if value is None or isinstance(value, str):
            return value
        return default

    return deepcopy(value)


def merge_ui_config(cfg: Optional[dict[str, Any]], default_params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    defaults = build_default_ui_config(default_params=default_params)
    if not isinstance(cfg, dict):
        return defaults
    return _coerce_like_default(cfg, defaults)


def save_ui_preset(preset_name: str, config: dict[str, Any], default_params: Optional[dict[str, Any]] = None) -> str:
    if not preset_name or not str(preset_name).strip():
        raise ValueError("Preset name cannot be empty.")

    safe_name = sanitize_preset_name(str(preset_name).strip())
    if is_locked_ui_preset(safe_name):
        raise ValueError(f"Preset '{safe_name}' is built in and cannot be overwritten.")

    root = Path(PRESETS_DIR)
    root.mkdir(parents=True, exist_ok=True)

    cfg = merge_ui_config(config, default_params=default_params)
    cfg.setdefault("_meta", {})
    cfg["_meta"]["version"] = UI_PRESET_VERSION
    cfg["_meta"]["format"] = UI_PRESET_FORMAT
    cfg["_meta"]["last_modified"] = datetime.now().isoformat()
    if "created_at" not in cfg["_meta"]:
        cfg["_meta"]["created_at"] = cfg["_meta"]["last_modified"]

    out_path = ui_preset_path(safe_name)
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)
    return safe_name


def load_ui_preset(preset_name: str, default_params: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    if not preset_name:
        return None
    path = find_ui_preset_path(preset_name)
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return merge_ui_config(data, default_params=default_params)


def delete_ui_preset(preset_name: str) -> bool:
    if not preset_name:
        return False
    if is_locked_ui_preset(preset_name):
        return False
    path = ui_preset_path(preset_name)
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False
