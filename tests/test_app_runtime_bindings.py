import importlib
import sys

import gradio as gr
from gradio.helpers import special_args


def load_app_module(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app.py"])
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def test_file_transcription_wrapper_preserves_pipeline_order(monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)
    captured = {}

    class DummyWhisperInference:
        def transcribe_file_with_live_output(self, *args):
            captured["args"] = args
            yield "live text", "done", ["tests/test.srt"]

    app_instance.whisper_inf = DummyWhisperInference()
    app_instance.prepare_download_output = lambda paths: {"visible": True, "value": paths}

    ui_inputs = [
        ["tests/jfk.wav"],
        False,
        "",
        False,
        False,
        "",
        ["SRT"],
        False,
        "large-v3",
        "english",
        False,
        8,
        -1.0,
        0.6,
        "bfloat16",
    ]

    processed_inputs, progress_index, _, _ = special_args(
        app_instance.transcribe_file_with_download,
        list(ui_inputs),
    )

    assert progress_index == 8

    live_text, result_text, download_update = next(
        app_instance.transcribe_file_with_download(*processed_inputs)
    )

    assert live_text == "live text"
    assert result_text == "done"
    assert download_update == {"visible": True, "value": ["tests/test.srt"]}
    assert captured["args"][:8] == tuple(ui_inputs[:8])
    assert isinstance(captured["args"][8], gr.Progress)
    assert captured["args"][9:] == tuple(ui_inputs[8:])
