from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline, TRANSCRIPTION_LOCK
from modules.whisper.data_classes import Segment, TranscriptionPipelineParams, VadParams, WhisperParams


class Pipeline(BaseTranscriptionPipeline):
    def __init__(self, output_dir):
        self.output_dir = str(output_dir)
        self.model = object()
        self.models_in_ram = False
        self.current_model_size = "large-v3"
        self.current_compute_type = "float32"
        self.device = "cpu"
        self.decode_calls = 0
        self.offload_calls = 0
        self.vad = SimpleNamespace(run=lambda **kwargs: (np.array([], dtype=np.float32), []))

    def transcribe(self, *args, **kwargs):
        self.decode_calls += 1
        return [Segment(start=0.0, end=1.0, text="Invented words on noise.")], .01

    def update_model(self, *args, **kwargs):
        raise AssertionError("A no-speech result must not load a transcription model.")

    def offload(self):
        self.offload_calls += 1
        self.model = None


def config(**whisper):
    return TranscriptionPipelineParams(
        whisper=WhisperParams(enable_offload=True, compute_type="float32", **whisper),
        vad=VadParams(vad_filter=True),
    )


def nonzero_noise():
    return np.random.default_rng(17).normal(0, .015, 16000 * 3).astype(np.float32)


def test_full_pipeline_skips_decode_and_completes_progress_after_vad_rejects_noise(tmp_path):
    pipeline = Pipeline(tmp_path)
    progress, callbacks = [], []
    result, elapsed = pipeline.run(nonzero_noise(), lambda *a, **k: progress.append((a, k)),
                                   "SRT", False, lambda *a: callbacks.append(a), *config().to_list())
    assert result == []
    assert elapsed >= 0
    assert pipeline.decode_calls == 0
    assert pipeline.offload_calls == 1
    assert progress[-1][0] == (1.0,)
    assert "no speech" in callbacks[-1][2]
    assert TRANSCRIPTION_LOCK.acquire(blocking=False)
    TRANSCRIPTION_LOCK.release()


@pytest.mark.parametrize("park", [False, True])
def test_digital_silence_skips_decode_and_respects_offload(tmp_path, park):
    pipeline = Pipeline(tmp_path)
    pipeline.vad.run = lambda **kwargs: pytest.fail("Zero samples need no VAD")
    result, _ = pipeline.run(np.zeros(16000, dtype=np.float32), None, "SRT", False, None,
                             *config(offload_to_ram=park).to_list())
    assert result == []
    assert pipeline.decode_calls == 0
    assert pipeline.offload_calls == (0 if park else 1)


@pytest.mark.parametrize("defer,park", [(True, False), (False, True)])
def test_no_speech_keeps_batch_and_ram_parking_cleanup_policy(tmp_path, defer, park):
    pipeline = Pipeline(tmp_path)
    pipeline.defer_offload = defer
    result, _ = pipeline.run(nonzero_noise(), None, "SRT", False, None, *config(offload_to_ram=park).to_list())
    assert result == []
    assert pipeline.offload_calls == 0


def test_no_speech_with_offload_disabled_keeps_loaded_model(tmp_path):
    pipeline = Pipeline(tmp_path)
    params = config()
    params.whisper.enable_offload = False
    result, _ = pipeline.run(nonzero_noise(), None, "SRT", False, None, *params.to_list())
    assert result == []
    assert pipeline.offload_calls == 0


def test_vad_failure_still_offloads_and_releases_transcription_lock(tmp_path):
    pipeline = Pipeline(tmp_path)

    def fail_vad(**kwargs):
        raise RuntimeError("VAD test failure")

    pipeline.vad.run = fail_vad
    with pytest.raises(RuntimeError, match="VAD test failure"):
        pipeline.run(nonzero_noise(), None, "SRT", False, None, *config().to_list())
    assert pipeline.offload_calls == 1
    assert TRANSCRIPTION_LOCK.acquire(blocking=False)
    TRANSCRIPTION_LOCK.release()


def test_preview_does_not_send_vad_rejected_noise_to_decoder(tmp_path):
    pipeline = Pipeline(tmp_path)
    assert pipeline.transcribe_live_preview(nonzero_noise(), *config().to_list()) == ""
    assert pipeline.decode_calls == 0
    assert pipeline.offload_calls == 0


def test_disabled_vad_still_passes_nonzero_audio_to_decoder(tmp_path):
    pipeline = Pipeline(tmp_path)
    params = config()
    params.vad.vad_filter = False
    result, _ = pipeline.run(nonzero_noise(), None, "SRT", False, None, *params.to_list())
    assert len(result) == 1
    assert pipeline.decode_calls == 1
    assert pipeline.offload_calls == 1


def test_real_silero_noise_file_produces_no_subtitle_words(tmp_path):
    from modules.vad.silero_vad import SileroVAD

    pipeline = Pipeline(tmp_path)
    pipeline.vad = SileroVAD()
    path = tmp_path / "nonzero-background-noise.wav"
    sf.write(path, nonzero_noise(), 16000, subtype="FLOAT")
    # Unlike exact digital silence this passes the zero-sample guard and exercises VAD.
    result, _ = pipeline.run(str(path), None, "SRT", False, None, *config().to_list())
    assert result == []
    assert pipeline.decode_calls == 0
    assert pipeline.offload_calls == 1
    # Exercise the same generator used by the File UI, including its subtitle writer.
    updates = list(pipeline.transcribe_file_with_live_output(
        [str(path)], False, None, False, False, str(tmp_path), ["SRT", "TXT"], False,
        lambda *args, **kwargs: None, *config().to_list()))
    _, status, outputs = updates[-1]
    assert "0 segments" in status
    assert len(outputs) == 2
    assert all(Path(output).read_text(encoding="utf-8") == "" for output in outputs)
    assert pipeline.decode_calls == 0
    assert pipeline.offload_calls == 2
