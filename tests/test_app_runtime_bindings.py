import importlib
import sys
from pathlib import Path

import gradio as gr
import pytest
from gradio.helpers import special_args

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


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


def test_resolve_media_input_path_accepts_relative_paths_and_file_uris(tmp_path, monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)

    media_file = tmp_path / "sample.mp3"
    media_file.write_bytes(b"")

    monkeypatch.chdir(tmp_path)

    assert app_instance.resolve_media_input_path("./sample.mp3") == str(media_file.resolve())
    assert app_instance.resolve_media_input_path(f'  "{media_file.resolve().as_uri()}"  ') == str(media_file.resolve())


def test_load_media_from_path_reuses_existing_preview_flow(tmp_path, monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)

    media_file = tmp_path / "clip.wav"
    media_file.write_bytes(b"")

    captured = {}

    def fake_preview(files):
        captured["files"] = files
        return {"summary": True}, {"preview": True}

    app_instance.update_uploaded_media_preview = fake_preview

    loaded_files, path_update, summary_update, preview_update = app_instance.load_media_from_path(str(media_file))

    assert loaded_files == [str(media_file.resolve())]
    assert captured["files"] == [str(media_file.resolve())]
    assert path_update["value"] == str(media_file.resolve())
    assert summary_update == {"summary": True}
    assert preview_update == {"preview": True}


def test_resolve_media_input_path_rejects_unsupported_extensions(tmp_path, monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)

    text_file = tmp_path / "notes.txt"
    text_file.write_text("not media", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported media file type"):
        app_instance.resolve_media_input_path(str(text_file))
