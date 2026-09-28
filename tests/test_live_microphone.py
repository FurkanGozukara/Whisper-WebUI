"""Complete-recording delivery must not depend on receiving every preview."""

import importlib
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["app.py"])
    module = importlib.import_module("app")
    monkeypatch.setattr(gr.processing_utils, "get_upload_folder", lambda: str(tmp_path))
    monkeypatch.setattr(module.App, "get_mic_staging_dir", staticmethod(lambda: str(tmp_path / "staged")))
    (tmp_path / "staged").mkdir()
    return module.App.__new__(module.App)


def payload(tmp_path, audio, sample_rate, *, total_samples=None, mode="preview_window", recording_id="test-recording"):
    path = tmp_path / f"{mode}.wav"
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    return {"path": str(path), "sample_rate": sample_rate, "capture_mode": mode,
            "total_samples": len(audio) if total_samples is None else total_samples,
            "recording_id": recording_id}


def test_preview_windows_do_not_duplicate_audio_or_require_every_update(app, tmp_path):
    state = app.create_live_mic_state()
    state, _, status = app.transcribe_live_mic_chunk(
        payload(tmp_path, np.ones(32000) * .1, 16000), False, state)
    assert state["stream_total_samples"] == 32000
    # Skip 28 seconds of preview delivery; the next window still describes the full duration.
    state, _, status = app.transcribe_live_mic_chunk(
        payload(tmp_path, np.ones(240000) * .2, 16000, total_samples=480000), False, state)
    assert len(state["audio"]) == 240000
    assert state["full_audio"].size == 0
    assert state["stream_total_samples"] == 480000
    assert "30.0s" in status
    assert np.allclose(state["audio"], .2, atol=1 / 32768)


def test_complete_recording_survives_missed_previews_with_exact_sample_count(app, tmp_path):
    audio = np.sin(np.arange(16000 * 37) * .05).astype(np.float32) * .25
    complete = payload(tmp_path, audio, 16000, mode="complete_recording")
    state = app.create_live_mic_state()
    state["audio"] = audio[-16000 * 15:]
    result = app.prepare_live_mic_capture_for_generation(False, state, complete)
    saved, sr = sf.read(result[6]["path"], dtype="float32")
    assert sr == 16000
    assert saved.shape == audio.shape
    assert np.max(np.abs(saved - audio)) <= 1 / 32768
    assert "37.0s" in result[2]
    assert "Generating subtitle files" in result[2]


def test_browser_sample_rate_controls_duration_and_resampling(app, tmp_path):
    state, _, status = app.transcribe_live_mic_chunk(
        payload(tmp_path, np.ones(48000 * 3) * .3, 48000, total_samples=48000 * 20),
        False, app.create_live_mic_state())
    assert len(state["audio"]) == 16000 * 3
    assert state["stream_total_samples"] == 16000 * 20
    assert "20.0s" in status


def test_new_recording_resets_previous_preview_state(app, tmp_path):
    state = app.create_live_mic_state()
    state.update(recording_id="old", transcript="old words", last_processed_samples=16000 * 100)
    state, text, _ = app.transcribe_live_mic_chunk(
        payload(tmp_path, np.zeros(32000), 16000, recording_id="new"), False, state)
    assert text == ""
    assert state["last_processed_samples"] == 0
    assert state["recording_id"] == "new"


def test_uploaded_microphone_cannot_reference_file_outside_upload_cache(app, tmp_path):
    with pytest.raises(ValueError, match="not an uploaded recording"):
        app.normalize_live_mic_chunk({"path": str(tmp_path.parent / "other.wav"),
                                      "capture_mode": "complete_recording"})


def test_component_events_use_regular_inputs_and_stop_after_full_upload():
    from modules.ui.live_microphone import live_microphone

    with gr.Blocks() as demo:
        microphone = live_microphone()
        microphone.input(lambda value: value, inputs=microphone, outputs=microphone,
                         trigger_mode="always_last", concurrency_limit=1)
        microphone.start_recording(lambda: None, queue=False)
        microphone.stop_recording(lambda value: None, inputs=microphone, queue=False)
    config = demo.get_config_file()
    assert microphone.get_config()["value"] is None
    assert all(dependency["connection"] != "stream" for dependency in config["dependencies"])
