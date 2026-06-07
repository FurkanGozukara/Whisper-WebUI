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
    saved_torch_modules = {
        module_name: module
        for module_name, module in list(sys.modules.items())
        if module_name == "torch" or module_name.startswith("torch.")
    }
    for module_name in list(sys.modules):
        if module_name == "torch" or module_name.startswith("torch."):
            sys.modules.pop(module_name, None)

    try:
        importlib.import_module("app")

        assert "torch" not in sys.modules
    finally:
        for module_name in list(sys.modules):
            if module_name == "torch" or module_name.startswith("torch."):
                sys.modules.pop(module_name, None)
        sys.modules.update(saved_torch_modules)


def test_whisper_defaults_enable_subprocess_and_disable_conditioning(monkeypatch):
    app_module = load_app_module(monkeypatch)

    assert app_module.WhisperParams().start_as_subprocess is True
    assert app_module.WhisperParams().condition_on_previous_text is False


def test_primary_model_choices_include_backend_details(monkeypatch):
    app_module = load_app_module(monkeypatch)

    choices = dict(app_module.App.PRIMARY_MODEL_CHOICES)

    assert choices["Whisper (faster-whisper / CTranslate2)"] == app_module.WhisperImpl.FASTER_WHISPER.value
    assert choices["Insanely Fast Whisper (Transformers)"] == app_module.WhisperImpl.INSANELY_FAST_WHISPER.value
    assert choices["Canary-Qwen (NVIDIA NeMo)"] == app_module.WhisperImpl.CANARY_QWEN.value
    assert "CTranslate2" in app_module.App.model_type_details_for_whisper_type("faster-whisper")
    assert "Transformers Whisper pipeline" in app_module.App.model_type_details_for_whisper_type("insanely_fast_whisper")
    assert "NeMo SALM" in app_module.App.model_type_details_for_whisper_type("canary-qwen")


def test_advanced_parameter_visibility_matches_selected_backend(monkeypatch):
    app_module = load_app_module(monkeypatch)

    assert app_module.App.advanced_field_visible_for_whisper_type("hotwords", "faster-whisper")
    assert app_module.App.advanced_field_visible_for_whisper_type("normalize_word_timestamps", "faster-whisper")
    assert not app_module.App.advanced_field_visible_for_whisper_type("canary_generation_kwargs", "faster-whisper")

    assert app_module.App.advanced_field_visible_for_whisper_type("max_new_tokens", "insanely_fast_whisper")
    assert app_module.App.advanced_field_visible_for_whisper_type("batch_size", "insanely_fast_whisper")
    assert not app_module.App.advanced_field_visible_for_whisper_type("hotwords", "insanely_fast_whisper")
    assert not app_module.App.advanced_field_visible_for_whisper_type("chunk_length", "insanely_fast_whisper")
    assert not app_module.App.advanced_field_visible_for_whisper_type("beam_size", "insanely_fast_whisper")

    assert app_module.App.advanced_field_visible_for_whisper_type("canary_generation_kwargs", "canary-qwen")
    assert app_module.App.advanced_field_visible_for_whisper_type("chunk_length", "canary-qwen")
    assert not app_module.App.advanced_field_visible_for_whisper_type("normalize_word_timestamps", "canary-qwen")
    assert not app_module.App.advanced_field_visible_for_whisper_type("hotwords", "canary-qwen")


def test_insanely_fast_defaults_never_fall_back_to_canary_model(monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)
    app_instance.default_params = {
        "whisper": {
            "model_size": app_module.App.CANARY_DEFAULT_MODEL,
            "lang": "english",
            "is_translate": False,
        }
    }

    class DummyWhisperInference:
        implementation_metadata = {}
        available_models = [app_module.App.CANARY_DEFAULT_MODEL]
        available_langs = ["english"]
        available_compute_types = ["bfloat16"]
        current_compute_type = "bfloat16"

    app_instance.whisper_inf = DummyWhisperInference()

    metadata = app_instance.get_whisper_metadata(app_module.WhisperImpl.INSANELY_FAST_WHISPER.value)
    defaults = app_instance.defaults_for_primary_whisper_type(app_module.WhisperImpl.INSANELY_FAST_WHISPER.value)

    assert metadata["available_models"] == ["large-v3"]
    assert app_instance.default_model_for_whisper_type(app_module.WhisperImpl.INSANELY_FAST_WHISPER.value) == "large-v3"
    assert defaults["model_size"] == "large-v3"
    assert defaults["whisper_type"] == app_module.WhisperImpl.INSANELY_FAST_WHISPER.value
    assert defaults["chunk_length"] is None
    assert defaults["max_new_tokens"] is None


def test_cancel_active_generation_runs_without_confirmation_input(monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)
    calls = []

    class DummyWhisperInference:
        def cancel_active_generation(self):
            calls.append("cancel")
            return True

    app_instance.whisper_inf = DummyWhisperInference()

    assert app_instance.cancel_active_generation() is True
    assert calls == ["cancel"]


def test_cancel_active_generation_respects_explicit_confirmation_decline(monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)
    calls = []

    class DummyWhisperInference:
        def cancel_active_generation(self):
            calls.append("cancel")
            return True

    app_instance.whisper_inf = DummyWhisperInference()

    assert app_instance.cancel_active_generation(False) is False
    assert calls == []


def test_cancel_confirm_js_returns_frontend_confirmation_payload(monkeypatch):
    app_module = load_app_module(monkeypatch)

    assert "window.confirm" in app_module.CANCEL_CONFIRM_JS
    assert "return [window.confirm" in app_module.CANCEL_CONFIRM_JS


def test_cancel_mic_generation_runs_without_confirmation_input(monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)
    app_instance.build_record_mic_idle_status = lambda: "idle"
    calls = []

    def fake_cancel(confirmed=True):
        calls.append(confirmed)
        return True

    app_instance.cancel_active_generation = fake_cancel

    _mic_update, status, _button_update, message = app_instance.cancel_mic_generation()

    assert calls == [True]
    assert status == "idle"
    assert message == "Generation cancelled. Ready for a new recording."


def test_cancel_mic_generation_respects_confirmation_decline(monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)
    app_instance.build_record_mic_idle_status = lambda: "idle"
    calls = []

    def fake_cancel(confirmed=True):
        calls.append(confirmed)
        return True

    app_instance.cancel_active_generation = fake_cancel

    _mic_update, _status, _button_update, message = app_instance.cancel_mic_generation(False)

    assert calls == []
    assert message == "Cancellation dismissed. Generation is still running."


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


def test_file_transcription_wrapper_logs_and_returns_persistent_error(monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)
    logged = []

    class DummyWhisperInference:
        def transcribe_file_with_live_output(self, *args):
            yield "partial live", "working", []
            raise RuntimeError("boom")

    app_instance.whisper_inf = DummyWhisperInference()
    app_instance.prepare_download_output = lambda paths: {"visible": bool(paths), "value": paths}
    monkeypatch.setattr(
        app_module.App,
        "log_persistent_error",
        staticmethod(lambda context, exc: logged.append((context, type(exc).__name__, str(exc)))),
    )

    outputs = list(
        app_instance.transcribe_file_with_download(
            ["tests/jfk.wav"],
            False,
            "",
            False,
            False,
            "",
            ["SRT"],
            False,
        )
    )

    assert logged == [("File transcription", "RuntimeError", "boom")]
    assert outputs[-1][0] == "partial live"
    assert "Error: File transcription failed." in outputs[-1][1]
    assert "RuntimeError: boom" in outputs[-1][1]
    assert "CMD/terminal" in outputs[-1][1]
    assert outputs[-1][2] == {"visible": False, "value": []}


def test_youtube_transcription_wrapper_logs_and_returns_persistent_error(monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)
    logged = []

    class DummyWhisperInference:
        def transcribe_youtube(self, *args):
            raise ValueError("bad youtube")

    app_instance.whisper_inf = DummyWhisperInference()
    app_instance.prepare_files_output = lambda paths: {"visible": bool(paths), "value": paths}
    monkeypatch.setattr(
        app_module.App,
        "log_persistent_error",
        staticmethod(lambda context, exc: logged.append((context, type(exc).__name__, str(exc)))),
    )

    result_text, files_update = app_instance.transcribe_youtube_with_progress("https://example.com")

    assert logged == [("YouTube transcription", "ValueError", "bad youtube")]
    assert "Error: YouTube transcription failed." in result_text
    assert "ValueError: bad youtube" in result_text
    assert files_update == {"visible": False, "value": []}


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


def test_staged_mic_transcription_wrapper_logs_and_returns_persistent_error(tmp_path, monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)
    logged = []
    staged_audio = tmp_path / "mic.wav"
    staged_audio.write_bytes(b"wav")

    class DummyWhisperInference:
        def transcribe_mic_with_live_output(self, *args):
            yield "mic live", "working", []
            raise RuntimeError("mic boom")

    app_instance.whisper_inf = DummyWhisperInference()
    app_instance.args = type("Args", (), {"output_dir": str(tmp_path / "outputs")})()
    app_instance.prepare_files_output = lambda paths: {"visible": bool(paths), "value": paths}
    monkeypatch.setattr(
        app_module.App,
        "log_persistent_error",
        staticmethod(lambda context, exc: logged.append((context, type(exc).__name__, str(exc)))),
    )

    outputs = list(app_instance.transcribe_staged_mic_with_download(str(staged_audio), ["SRT"], False))

    assert logged == [("Microphone transcription", "RuntimeError", "mic boom")]
    assert outputs[-1][0] == "mic live"
    assert "Error: Microphone transcription failed." in outputs[-1][1]
    assert "RuntimeError: mic boom" in outputs[-1][1]
    assert outputs[-1][2]["visible"] is True


def test_live_preview_logs_error_to_terminal(monkeypatch):
    app_module = load_app_module(monkeypatch)
    app_instance = app_module.App.__new__(app_module.App)
    logged = []

    class DummyWhisperInference:
        def transcribe_live_preview(self, *args):
            raise RuntimeError("preview boom")

    app_instance.whisper_inf = DummyWhisperInference()
    state = app_module.App.create_live_mic_state()
    monkeypatch.setattr(
        app_module.App,
        "log_persistent_error",
        staticmethod(lambda context, exc: logged.append((context, type(exc).__name__, str(exc)))),
    )

    updated_state, transcript, status = app_instance.transcribe_live_mic_chunk(
        (16000, np.ones(32000, dtype=np.float32)),
        True,
        state,
        "large-v3",
    )

    assert updated_state is state
    assert transcript == ""
    assert logged == [("Live microphone preview", "RuntimeError", "preview boom")]
    assert "Live preview error: RuntimeError: preview boom" in status


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
