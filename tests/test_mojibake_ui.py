from pathlib import Path
import sys
import types

import gradio as gr

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if "gradio_i18n" not in sys.modules:
    fake_gradio_i18n = types.ModuleType("gradio_i18n")
    fake_gradio_i18n_i18n = types.ModuleType("gradio_i18n.i18n")

    class _Translate:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _I18nString(str):
        def __new__(cls, value):
            obj = str.__new__(cls, value)
            obj.radd_values = []
            obj.add_values = []
            return obj

    class _TranslateContext:
        dictionary = {}

        @staticmethod
        def get_current_language(_request):
            return "en"

    fake_gradio_i18n.Translate = _Translate
    fake_gradio_i18n.gettext = lambda value: value
    fake_gradio_i18n_i18n.I18nString = _I18nString
    fake_gradio_i18n_i18n.TranslateContext = _TranslateContext
    sys.modules["gradio_i18n"] = fake_gradio_i18n
    sys.modules["gradio_i18n.i18n"] = fake_gradio_i18n_i18n

from modules.utils.text import repair_blocks_text, repair_mojibake_obj, repair_mojibake_text
from modules.whisper.data_classes import (
    BGMSeparationParams,
    DiarizationParams,
    VadParams,
    WhisperImpl,
    WhisperParams,
)


def _corrupt_cp1252_once(value: str) -> str:
    return value.encode("utf-8").decode("cp1252")


def _corrupt_cp1252_twice(value: str) -> str:
    return _corrupt_cp1252_once(_corrupt_cp1252_once(value))


def _assert_component_text_clean(component) -> None:
    for attribute in ("label", "info", "placeholder"):
        value = getattr(component, attribute, None)
        assert repair_mojibake_text(value) == value, (
            f"{component.__class__.__name__}.{attribute} still contains mojibake: {value!r}"
        )

    choices = getattr(component, "choices", None)
    if choices is not None:
        assert repair_mojibake_obj(choices) == choices, (
            f"{component.__class__.__name__}.choices still contain mojibake: {choices!r}"
        )


def _dump_model(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def test_repair_mojibake_text_handles_single_and_double_cp1252_corruption():
    clean = "🎵 Remove background music"

    assert repair_mojibake_text(_corrupt_cp1252_once(clean)) == clean
    assert repair_mojibake_text(_corrupt_cp1252_twice(clean)) == clean


def test_repair_blocks_text_sanitizes_existing_components():
    clean_info = "🎵 Remove background music"
    clean_choice = "English"

    with gr.Blocks() as demo:
        textbox = gr.Textbox(info=_corrupt_cp1252_twice(clean_info))
        dropdown = gr.Dropdown(choices=[_corrupt_cp1252_once(clean_choice)])

    repair_blocks_text(demo)

    assert textbox.info == clean_info
    assert dropdown.choices == [(clean_choice, clean_choice)]


def test_gradio_param_factories_return_clean_text():
    whisper_defaults = _dump_model(WhisperParams())

    with gr.Blocks():
        components = []
        components.extend(VadParams.to_gradio_inputs(defaults={}))
        components.extend(DiarizationParams.to_gradio_inputs(defaults={}, available_devices=["cpu", "cuda"], device="cpu"))
        components.extend(
            BGMSeparationParams.to_gradio_input(
                defaults={},
                available_devices=["cpu", "cuda"],
                device="cpu",
                available_models=["UVR-MDX-NET-Inst_HQ_4"],
            )
        )
        components.extend(
            WhisperParams.to_gradio_inputs(
                defaults=whisper_defaults,
                only_advanced=True,
                whisper_type=WhisperImpl.FASTER_WHISPER.value,
                available_models=["large-v2"],
                available_langs=["en"],
                available_compute_types=["bfloat16", "float16", "float32", "int8"],
                compute_type="bfloat16",
            )
        )

    for component in components:
        _assert_component_text_clean(component)


def test_app_applies_global_gradio_text_repair():
    app_source = (ROOT_DIR / "app.py").read_text(encoding="utf-8")

    assert "return [repair_component_text(component) for component in inputs]" in app_source
    assert "repair_blocks_text(self.app)" in app_source
