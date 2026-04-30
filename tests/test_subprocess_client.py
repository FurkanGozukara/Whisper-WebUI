from argparse import Namespace
from collections import deque
import json
import os
import subprocess
import sys
import tempfile
from threading import Thread
import time

import pytest

from modules.runtime.subprocess_client import SubprocessWhisperProxy, WorkerHandle
from modules.whisper.data_classes import TranscriptionPipelineParams, WhisperImpl, WhisperParams


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


def start_test_worker(mode: str, label: str) -> WorkerHandle:
    request_file = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="whisper_webui_test_worker_",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    )
    json.dump({"mode": mode, "label": label}, request_file)
    request_file.close()

    script = r"""
import json
import sys
import time

mode = sys.argv[1]
label = sys.argv[2]

if mode in {"sleep", "finish"}:
    sys.stdout.write(json.dumps({
        "event": "update",
        "payload": {
            "live_output": f"{label} live",
            "result_str": "",
            "paths": [],
        },
    }) + "\n")
    sys.stdout.flush()

if mode == "result":
    sys.stdout.write(json.dumps({
        "event": "result",
        "payload": {
            "label": label,
            "status": "ok",
        },
    }) + "\n")
    sys.stdout.flush()
elif mode == "sleep":
    time.sleep(60)
else:
    sys.stdout.write(json.dumps({"event": "complete"}) + "\n")
    sys.stdout.flush()
"""

    popen_kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "bufsize": 0,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["preexec_fn"] = os.setsid

    process = subprocess.Popen(
        [sys.executable, "-u", "-c", script, mode, label],
        **popen_kwargs,
    )
    stderr_lines = deque(maxlen=20)

    def drain_stderr():
        if process.stderr is None:
            return
        for raw_line in process.stderr:
            stderr_lines.append(raw_line.decode("utf-8", errors="replace").rstrip())

    stderr_thread = Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()
    return WorkerHandle(
        process=process,
        request_path=request_file.name,
        stderr_lines=stderr_lines,
        stderr_thread=stderr_thread,
    )


def install_worker_sequence(monkeypatch, proxy: SubprocessWhisperProxy, modes):
    handles = []

    def fake_start_worker(action, payload):
        index = len(handles)
        whisper_type = payload.get("whisper_type", "unknown")
        handle = start_test_worker(modes[index], f"{whisper_type}:{action}:{index}")
        handles.append(handle)
        return handle

    monkeypatch.setattr(proxy._client, "start_worker", fake_start_worker)
    return handles


def wait_for_active_handle(proxy: SubprocessWhisperProxy, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with proxy._active_lock:
            handle = proxy._active_handle
        if handle is not None:
            return handle
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for an active worker handle.")


def test_transcribe_youtube_keeps_pipeline_params_aligned_without_explicit_progress(monkeypatch):
    proxy = build_proxy()
    pipeline_params = TranscriptionPipelineParams(
        whisper=WhisperParams(),
    ).to_list()

    monkeypatch.setattr(proxy, "_use_subprocess", lambda params: True)
    monkeypatch.setattr(
        proxy,
        "_call_transcription_worker",
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
        proxy,
        "_call_transcription_worker",
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


@pytest.mark.parametrize("whisper_type", [WhisperImpl.FASTER_WHISPER.value, WhisperImpl.CANARY_QWEN.value])
def test_stream_close_terminates_real_worker_and_next_generation_starts(monkeypatch, whisper_type):
    proxy = build_proxy()
    handles = install_worker_sequence(monkeypatch, proxy, ["sleep", "finish"])

    generator = proxy._stream_transcription_worker(
        "transcribe_file",
        {"whisper_type": whisper_type},
    )

    assert next(generator) == (f"{whisper_type}:transcribe_file:0 live", "", [])

    started_at = time.monotonic()
    generator.close()
    elapsed = time.monotonic() - started_at

    assert elapsed < 8
    assert handles[0].process.poll() is not None

    restarted = list(
        proxy._stream_transcription_worker(
            "transcribe_file",
            {"whisper_type": whisper_type},
        )
    )

    assert restarted == [(f"{whisper_type}:transcribe_file:1 live", "", [])]
    assert handles[1].process.poll() == 0


@pytest.mark.parametrize("whisper_type", [WhisperImpl.FASTER_WHISPER.value, WhisperImpl.CANARY_QWEN.value])
def test_explicit_cancel_terminates_real_worker_and_next_generation_starts(monkeypatch, whisper_type):
    proxy = build_proxy()
    handles = install_worker_sequence(monkeypatch, proxy, ["sleep", "finish"])

    generator = proxy._stream_transcription_worker(
        "transcribe_file",
        {"whisper_type": whisper_type},
    )

    assert next(generator) == (f"{whisper_type}:transcribe_file:0 live", "", [])
    assert proxy.cancel_active_generation() is True

    remaining = list(generator)

    assert handles[0].process.poll() is not None
    assert remaining == [
        (
            f"{whisper_type}:transcribe_file:0 live",
            "Cancelled. Running subprocess was terminated.",
            [],
        )
    ]

    restarted = list(
        proxy._stream_transcription_worker(
            "transcribe_file",
            {"whisper_type": whisper_type},
        )
    )

    assert restarted == [(f"{whisper_type}:transcribe_file:1 live", "", [])]
    assert handles[1].process.poll() == 0


@pytest.mark.parametrize("whisper_type", [WhisperImpl.FASTER_WHISPER.value, WhisperImpl.CANARY_QWEN.value])
def test_non_stream_transcription_cancel_terminates_real_worker_and_restart_returns(monkeypatch, whisper_type):
    proxy = build_proxy()
    handles = install_worker_sequence(monkeypatch, proxy, ["sleep", "result"])
    pipeline_params = TranscriptionPipelineParams(
        whisper=WhisperParams(whisper_type=whisper_type),
    ).to_list()
    outcome = {}

    def run_first_call():
        try:
            outcome["value"] = proxy.transcribe_youtube(
                "https://www.youtube.com/watch?v=test",
                ["SRT"],
                False,
                False,
                1,
                *pipeline_params,
            )
        except Exception as exc:
            outcome["error"] = exc

    thread = Thread(target=run_first_call, daemon=True)
    thread.start()
    wait_for_active_handle(proxy)

    assert proxy.cancel_active_generation() is True
    thread.join(timeout=8)

    assert not thread.is_alive()
    assert handles[0].process.poll() is not None
    assert "Cancelled. Running subprocess was terminated." in str(outcome["error"])

    restarted = proxy.transcribe_youtube(
        "https://www.youtube.com/watch?v=test",
        ["SRT"],
        False,
        False,
        1,
        *pipeline_params,
    )

    assert restarted == {
        "label": f"{whisper_type}:transcribe_youtube:1",
        "status": "ok",
    }
    assert handles[1].process.poll() == 0


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
