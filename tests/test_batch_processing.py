from types import SimpleNamespace

import numpy as np

from modules.whisper.data_classes import WhisperParams
from modules.whisper.faster_whisper_inference import FasterWhisperInference


class DummyProgress:
    def __init__(self):
        self.events = []

    def __call__(self, value, desc=None):
        self.events.append((value, desc))


def make_dummy_segment(segment_id, start, end, text):
    return SimpleNamespace(
        id=segment_id,
        seek=int(start * 100),
        text=text,
        start=start,
        end=end,
        tokens=[segment_id],
        temperature=0.0,
        avg_logprob=-0.1,
        compression_ratio=1.0,
        no_speech_prob=0.0,
        words=None,
    )


def test_build_clip_timestamps_splits_fixed_windows():
    audio = np.zeros(60 * 16000, dtype=np.float32)

    clip_timestamps = FasterWhisperInference.build_clip_timestamps(
        audio=audio,
        chunk_length=15,
        sampling_rate=16000,
    )

    assert clip_timestamps == [
        {"start": 0, "end": 240000},
        {"start": 240000, "end": 480000},
        {"start": 480000, "end": 720000},
        {"start": 720000, "end": 960000},
    ]


def test_emit_progress_callback_supports_single_argument_callbacks():
    callback_values = []

    FasterWhisperInference.emit_progress_callback(
        progress_callback=lambda progress_value: callback_values.append(progress_value),
        progress_value=0.5,
        segment=None,
    )

    assert callback_values == [0.5]


def test_transcribe_uses_batched_pipeline_with_requested_batch_size(monkeypatch):
    captured = {}
    callback_events = []
    dummy_segments = [
        make_dummy_segment(1, 0.0, 14.8, "first chunk"),
        make_dummy_segment(2, 15.0, 29.6, "second chunk"),
    ]

    class DummyBatchedInferencePipeline:
        def __init__(self, model):
            captured["model"] = model

        def transcribe(self, **kwargs):
            captured.update(kwargs)
            return iter(dummy_segments), SimpleNamespace(duration=60.0)

    inferencer = object.__new__(FasterWhisperInference)
    inferencer.model = SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000))
    inferencer.current_model_size = "large-v3"
    inferencer.current_compute_type = "float16"
    inferencer.update_model = lambda *args, **kwargs: None

    monkeypatch.setattr(
        "modules.whisper.faster_whisper_inference.faster_whisper.BatchedInferencePipeline",
        DummyBatchedInferencePipeline,
    )

    progress = DummyProgress()
    whisper_params = WhisperParams(
        model_size="large-v3",
        compute_type="float16",
        lang="en",
        chunk_length=15,
        batch_size=4,
        word_timestamps=True,
    ).to_list()

    segments, elapsed_time = FasterWhisperInference.transcribe(
        inferencer,
        np.zeros(60 * 16000, dtype=np.float32),
        progress,
        lambda progress_value, segment: callback_events.append((progress_value, segment.text)),
        *whisper_params,
    )

    assert len(segments) == 2
    assert elapsed_time >= 0
    assert captured["batch_size"] == 4
    assert captured["without_timestamps"] is False
    assert captured["word_timestamps"] is True
    assert captured["clip_timestamps"] == [
        {"start": 0, "end": 240000},
        {"start": 240000, "end": 480000},
        {"start": 480000, "end": 720000},
        {"start": 720000, "end": 960000},
    ]
    assert callback_events == [
        (0.24666666666666667, "first chunk"),
        (0.49333333333333335, "second chunk"),
    ]
    assert progress.events[0] == (0, "Loading audio..")
