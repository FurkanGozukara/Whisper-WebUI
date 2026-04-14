import importlib
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
from gradio.helpers import special_args

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_app_module(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app.py"])
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def test_importing_app_does_not_import_torch(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app.py"])
    sys.modules.pop("app", None)
    for module_name in list(sys.modules):
        if module_name == "torch" or module_name.startswith("torch."):
            sys.modules.pop(module_name, None)

    importlib.import_module("app")

    assert "torch" not in sys.modules


def test_whisper_defaults_enable_subprocess_and_disable_conditioning(monkeypatch):
    app_module = load_app_module(monkeypatch)

    assert app_module.WhisperParams().start_as_subprocess is True
    assert app_module.WhisperParams().condition_on_previous_text is False


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


def test_mic_transcription_wrapper_preserves_pipeline_order(monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)
    captured = {}
    expected_basename = "mic_record_2026_04_15_10_11_12"

    class DummyWhisperInference:
        def transcribe_mic_with_live_output(self, *args):
            captured["args"] = args
            yield "live text", "done", ["tests/test.srt"]

    app_instance.whisper_inf = DummyWhisperInference()
    app_instance.prepare_files_output = lambda paths: {"visible": True, "value": paths}
    monkeypatch.setattr(
        app_module.App,
        "build_mic_output_basename",
        staticmethod(lambda prefix, timestamp=None: f"{prefix}_2026_04_15_10_11_12"),
    )

    ui_inputs = [
        "tests/jfk.wav",
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
        app_instance.transcribe_mic_with_download,
        list(ui_inputs),
    )

    assert progress_index == 3

    live_text, result_text, files_update = next(
        app_instance.transcribe_mic_with_download(*processed_inputs)
    )

    assert live_text == "live text"
    assert result_text == "done"
    assert files_update == {"visible": True, "value": ["tests/test.srt"]}
    assert Path(captured["args"][0]).name == f"{expected_basename}.wav"
    assert captured["args"][1:3] == tuple(ui_inputs[1:3])
    assert isinstance(captured["args"][3], gr.Progress)
    assert captured["args"][4:] == tuple(ui_inputs[3:])


def test_mic_transcription_wrapper_returns_preparing_message_for_missing_audio(monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)
    app_instance.prepare_files_output = lambda paths: {"visible": False, "value": paths}

    progress_text, result_text, files_update = next(
        app_instance.transcribe_mic_with_download(None, ["SRT"], False)
    )

    assert progress_text == "Recorded microphone audio is not ready yet."
    assert "still being prepared" in result_text
    assert files_update == {"visible": False, "value": []}


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


def test_normalize_live_mic_chunk_accepts_dict_payload_with_numpy_audio(monkeypatch):
    app_module = load_app_module(monkeypatch)

    sample_rate, audio = app_module.App.normalize_live_mic_chunk(
        {
            "sample_rate": 16000,
            "data": np.array([[0, 32767], [32767, 0]], dtype=np.int16),
        }
    )

    assert sample_rate == 16000
    assert audio.dtype == np.float32
    assert audio.shape == (2,)
    assert np.allclose(audio, np.array([0.5, 0.5], dtype=np.float32), atol=1e-4)


def test_transcribe_live_mic_chunk_updates_transcript_once_buffer_is_ready(monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)
    captured = {}

    class DummyWhisperInference:
        def transcribe_live_preview(self, audio, *pipeline_params):
            captured["audio"] = audio
            captured["pipeline_params"] = pipeline_params
            return "preview transcript"

    app_instance.whisper_inf = DummyWhisperInference()
    state = app_module.App.create_live_mic_state()

    updated_state, transcript, status = app_instance.transcribe_live_mic_chunk(
        (16000, np.ones(32000, dtype=np.float32)),
        True,
        state,
        "large-v3",
        "english",
    )

    assert transcript == "preview transcript"
    assert updated_state["transcript"] == "preview transcript"
    assert updated_state["last_processed_samples"] == 32000
    assert updated_state["stream_total_samples"] == 32000
    assert captured["audio"].shape == (32000,)
    assert captured["pipeline_params"] == ("large-v3", "english")
    assert "Preview updated." in status


def test_trim_live_mic_audio_caps_buffer_to_recent_window(monkeypatch):
    app_module = load_app_module(monkeypatch)

    trimmed_audio, trimmed = app_module.App.trim_live_mic_audio(
        np.arange(0, 320000, dtype=np.float32),
        sample_rate=16000,
        max_seconds=15.0,
    )

    assert trimmed is True
    assert trimmed_audio.shape == (240000,)
    assert np.array_equal(trimmed_audio, np.arange(80000, 320000, dtype=np.float32))


def test_transcribe_live_mic_chunk_keeps_updating_for_cumulative_stream_after_trim(monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)
    captured = {}

    class DummyWhisperInference:
        def transcribe_live_preview(self, audio, *pipeline_params):
            captured["audio"] = audio
            captured["pipeline_params"] = pipeline_params
            return "preview transcript"

    app_instance.whisper_inf = DummyWhisperInference()
    state = {
        "audio": np.arange(16000, 256000, dtype=np.float32),
        "sample_rate": 16000,
        "last_processed_samples": 256000,
        "transcript": "older preview",
        "stream_total_samples": 256000,
        "stream_mode": app_module.App.LIVE_MIC_STREAM_MODE_CUMULATIVE,
    }

    updated_state, transcript, status = app_instance.transcribe_live_mic_chunk(
        (16000, np.arange(0, 288000, dtype=np.float32)),
        True,
        state,
        "large-v3",
    )

    assert transcript == "preview transcript"
    assert updated_state["audio"].shape == (240000,)
    assert updated_state["last_processed_samples"] == 288000
    assert updated_state["stream_total_samples"] == 288000
    assert captured["audio"].shape == (240000,)
    assert captured["pipeline_params"] == ("large-v3",)
    assert "Preview updated." in status


def test_prepare_live_mic_capture_for_generation_stages_named_wav(tmp_path, monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)

    monkeypatch.setattr(app_module.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        app_module.App,
        "build_mic_output_basename",
        staticmethod(lambda prefix, timestamp=None: f"{prefix}_2026_04_15_10_11_12"),
    )

    result = app_instance.prepare_live_mic_capture_for_generation(
        True,
        app_module.App.create_live_mic_state(),
        (16000, np.ones(32000, dtype=np.float32)),
    )

    _, _, status, progress_text, _, files_update, capture_update, _, record_status, run_button_update = result

    assert "live_record_2026_04_15_10_11_12" in status
    assert "live_record_2026_04_15_10_11_12" in progress_text
    assert capture_update["path"].endswith("live_record_2026_04_15_10_11_12.wav")
    assert Path(capture_update["path"]).exists()
    assert files_update["visible"] is False
    assert record_status == app_module.App.build_record_mic_idle_status()
    assert run_button_update["interactive"] is False
    Path(capture_update["path"]).unlink(missing_ok=True)


def test_stage_live_mic_audio_prefers_longer_accumulated_stream_over_short_stop_payload(tmp_path, monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)

    monkeypatch.setattr(app_module.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        app_module.App,
        "build_mic_output_basename",
        staticmethod(lambda prefix, timestamp=None: f"{prefix}_2026_04_15_10_11_12"),
    )

    staged_path, captured_seconds, fallback_message = app_instance.stage_live_mic_audio(
        (16000, np.ones(16000, dtype=np.float32)),
        {
            "sample_rate": 16000,
            "full_audio": np.ones(160000, dtype=np.float32),
            "audio": np.ones(32000, dtype=np.float32),
        },
    )

    assert staged_path.endswith("live_record_2026_04_15_10_11_12.wav")
    assert Path(staged_path).exists()
    assert captured_seconds == pytest.approx(10.0)
    assert "shorter" in fallback_message
    Path(staged_path).unlink(missing_ok=True)


def test_persist_staged_mic_audio_output_copies_recording_into_outputs(tmp_path, monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)
    app_instance.args = type("Args", (), {"output_dir": str(tmp_path / "outputs")})()

    staged_audio = tmp_path / "live_record_2026_04_15_10_11_12.wav"
    staged_audio.write_bytes(b"wave")

    persisted_path = app_instance.persist_staged_mic_audio_output(str(staged_audio))

    assert Path(persisted_path) == (tmp_path / "outputs" / staged_audio.name)
    assert Path(persisted_path).read_bytes() == b"wave"


def test_refresh_record_mic_ready_state_reflects_attached_audio(tmp_path, monkeypatch):
    app_module = load_app_module(monkeypatch)

    audio_file = tmp_path / "mic.wav"
    audio_file.write_bytes(b"")

    waiting_status, waiting_update = app_module.App.refresh_record_mic_ready_state(None)
    ready_status, ready_update = app_module.App.refresh_record_mic_ready_state({"path": str(audio_file)})

    assert "not ready yet" in waiting_status
    assert waiting_update["interactive"] is False
    assert "Recording saved." in ready_status
    assert ready_update["interactive"] is True
