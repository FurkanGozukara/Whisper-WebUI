from argparse import Namespace

from modules.runtime.subprocess_client import SubprocessWhisperProxy
from modules.whisper.data_classes import TranscriptionPipelineParams, WhisperParams


def build_proxy():
    return SubprocessWhisperProxy(
        Namespace(),
        {
            "device": "cpu",
            "available_models": ["large-v3"],
            "available_langs": ["english"],
            "available_compute_types": ["float32"],
            "current_compute_type": "float32",
            "music_separator": {
                "device": "cpu",
                "available_devices": ["cpu"],
                "available_models": ["UVR"],
            },
            "diarizer": {
                "device": "cpu",
                "available_device": ["cpu"],
            },
        },
    )


def test_transcribe_youtube_keeps_pipeline_params_aligned_without_explicit_progress(monkeypatch):
    proxy = build_proxy()
    pipeline_params = TranscriptionPipelineParams(
        whisper=WhisperParams(),
    ).to_list()

    monkeypatch.setattr(proxy, "_use_subprocess", lambda params: True)
    monkeypatch.setattr(
        proxy._client,
        "call",
        lambda action, payload: {"action": action, "payload": payload},
    )

    result = proxy.transcribe_youtube(
        "https://www.youtube.com/watch?v=test",
        ["SRT"],
        False,
        True,
        123,
        *pipeline_params,
    )

    assert result["action"] == "transcribe_youtube"
    assert result["payload"]["mass_transcribe_channel"] is True
    assert result["payload"]["latest_video_count"] == 123
    assert result["payload"]["pipeline_params"] == pipeline_params


def test_transcribe_mic_keeps_pipeline_params_aligned_without_explicit_progress(monkeypatch):
    proxy = build_proxy()
    pipeline_params = TranscriptionPipelineParams(
        whisper=WhisperParams(),
    ).to_list()

    monkeypatch.setattr(proxy, "_use_subprocess", lambda params: True)
    monkeypatch.setattr(
        proxy._client,
        "call",
        lambda action, payload: {"action": action, "payload": payload},
    )

    result = proxy.transcribe_mic(
        "mic.wav",
        ["SRT"],
        False,
        *pipeline_params,
    )

    assert result["action"] == "transcribe_mic"
    assert result["payload"]["pipeline_params"] == pipeline_params


def test_transcribe_mic_with_live_output_keeps_pipeline_params_aligned_without_explicit_progress(monkeypatch):
    proxy = build_proxy()
    pipeline_params = TranscriptionPipelineParams(
        whisper=WhisperParams(),
    ).to_list()

    captured = {}

    monkeypatch.setattr(proxy, "_use_subprocess", lambda params: True)

    def fake_stream(action, payload):
        captured["action"] = action
        captured["payload"] = payload
        yield "live", "done", ["Mic.srt"]

    monkeypatch.setattr(proxy, "_stream_transcription_worker", fake_stream)

    result = list(
        proxy.transcribe_mic_with_live_output(
            "mic.wav",
            ["SRT"],
            False,
            *pipeline_params,
        )
    )

    assert result == [("live", "done", ["Mic.srt"])]
    assert captured["action"] == "transcribe_mic_stream"
    assert captured["payload"]["pipeline_params"] == pipeline_params


def test_transcribe_live_preview_uses_local_inferencer(monkeypatch):
    proxy = build_proxy()
    captured = {}

    class DummyInferencer:
        def transcribe_live_preview(self, audio, *pipeline_params):
            captured["audio"] = audio
            captured["pipeline_params"] = pipeline_params
            return "preview text"

    monkeypatch.setattr(proxy, "_get_local_inferencer", lambda: DummyInferencer())

    result = proxy.transcribe_live_preview("audio-buffer", "large-v3", "english")

    assert result == "preview text"
    assert captured["audio"] == "audio-buffer"
    assert captured["pipeline_params"] == ("large-v3", "english")
