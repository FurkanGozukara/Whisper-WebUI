from types import SimpleNamespace

import numpy as np
import torch

from modules.whisper.data_classes import WhisperParams
from modules.whisper.whisper_Inference import WhisperInference


class DummyProgress:
    def __init__(self):
        self.events = []

    def __call__(self, value, desc=None):
        self.events.append((value, desc))


def build_inferencer(dummy_model):
    inferencer = object.__new__(WhisperInference)
    inferencer.model = dummy_model
    inferencer.current_model_size = "tiny"
    inferencer.current_compute_type = "float32"
    inferencer.update_model = lambda *args, **kwargs: None
    return inferencer


def test_whisper_prepare_audio_input_replaces_non_finite_and_mixes_mono():
    audio = np.array(
        [
            [np.nan, 1.0, -1.0],
            [np.inf, -1.0, 1.0],
        ],
        dtype=np.float32,
    )

    prepared = WhisperInference.prepare_audio_input(audio)

    assert prepared.shape == (3,)
    assert prepared.dtype == np.float32
    assert np.isfinite(prepared).all()
    assert prepared.tolist() == [0.0, 0.0, 0.0]


def test_whisper_prepare_audio_input_accepts_torch_tensor():
    prepared = WhisperInference.prepare_audio_input(torch.ones((2, 4), dtype=torch.float32))

    assert isinstance(prepared, np.ndarray)
    assert prepared.shape == (4,)
    assert np.allclose(prepared, np.ones(4, dtype=np.float32))


def test_whisper_transcribe_passes_prompt_and_conditioning_settings():
    captured = {}

    class DummyModel:
        def transcribe(self, **kwargs):
            captured.update(kwargs)
            kwargs["progress_callback"](0.5)
            return {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": " ok",
                        "words": [{"start": 0.0, "end": 0.5, "word": "ok", "probability": 0.9}],
                    }
                ]
            }

    progress = DummyProgress()
    progress_events = []
    inferencer = build_inferencer(DummyModel())
    params = WhisperParams(
        model_size="tiny",
        compute_type="float32",
        lang="english",
        initial_prompt="domain prompt",
        condition_on_previous_text=False,
        word_timestamps=True,
    )

    segments, _elapsed = inferencer.transcribe(
        np.ones(16000, dtype=np.float32),
        progress,
        lambda value, segment=None: progress_events.append((value, segment)),
        *params.to_list(),
    )

    assert segments[0].text == " ok"
    assert segments[0].words[0].word == "ok"
    assert captured["initial_prompt"] == "domain prompt"
    assert captured["condition_on_previous_text"] is False
    assert captured["language"] == "en"
    assert captured["audio"].dtype == np.float32
    assert progress.events[-1] == (0.5, "Transcribing..")
    assert progress_events == [(0.5, None)]


def test_whisper_transcribe_skips_empty_numpy_audio():
    class FailingModel:
        def transcribe(self, **kwargs):
            raise AssertionError("Empty audio should not be passed to the model.")

    inferencer = build_inferencer(FailingModel())

    segments, elapsed = inferencer.transcribe(
        np.array([], dtype=np.float32),
        DummyProgress(),
        None,
        *WhisperParams(model_size="tiny", compute_type="float32").to_list(),
    )

    assert segments == []
    assert elapsed >= 0
