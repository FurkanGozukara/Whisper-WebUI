from types import SimpleNamespace

import ctranslate2
import numpy as np

from modules.whisper.data_classes import WhisperParams
from modules.whisper.faster_whisper_inference import (
    FasterWhisperInference,
    StandardEncoderPrefetchCache,
)


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


def test_standard_pipeline_repeats_initial_prompt_every_window():
    captured = {}

    class DummyTokenizer:
        def __init__(self):
            self.calls = []

        def encode(self, text):
            self.calls.append(text)
            return [101, 202]

    class DummyModel:
        def __init__(self):
            self.tokenizer = DummyTokenizer()
            self.prompt_inputs = []

        def get_prompt(self, tokenizer, previous_tokens, without_timestamps=False, prefix=None, hotwords=None):
            prompt_tokens = list(previous_tokens)
            self.prompt_inputs.append(prompt_tokens)
            return prompt_tokens

        def transcribe(self, **kwargs):
            captured.update(kwargs)
            self.get_prompt(self.tokenizer, [11], False, None, None)
            self.get_prompt(self.tokenizer, [22], False, None, None)
            return iter([]), SimpleNamespace(duration=30.0)

    inferencer = object.__new__(FasterWhisperInference)
    inferencer.model = DummyModel()

    params = WhisperParams(
        initial_prompt="Welcome to the school.",
        repeat_initial_prompt_every_window=True,
    )

    FasterWhisperInference._transcribe_with_standard_pipeline(
        inferencer,
        np.zeros(16000, dtype=np.float32),
        params,
        DummyProgress(),
    )

    assert captured["initial_prompt"] is None
    assert inferencer.model.prompt_inputs == [
        [101, 202, 11],
        [101, 202, 22],
    ]
    assert inferencer.model.tokenizer.calls == [" Welcome to the school."]


def test_standard_encoder_prefetch_cache_batches_aligned_windows():
    encode_batch_sizes = []

    class DummyModel:
        feature_extractor = SimpleNamespace(nb_max_frames=4)

        def encode(self, features):
            features = np.asarray(features, dtype=np.float32)
            if features.ndim == 2:
                features = np.expand_dims(features, 0)
            encode_batch_sizes.append(int(features.shape[0]))
            return ctranslate2.StorageView.from_array(np.ascontiguousarray(features))

    cache = StandardEncoderPrefetchCache(
        model=DummyModel(),
        features=np.zeros((80, 20), dtype=np.float32),
        batch_size=3,
    )

    first = cache.get(seek=0, seek_clip_end=20)
    second = cache.get(seek=4, seek_clip_end=20)
    third = cache.get(seek=8, seek_clip_end=20)

    assert encode_batch_sizes == [3]
    assert first.shape[0] == 1
    assert second.shape[0] == 1
    assert third.shape[0] == 1


def test_standard_encoder_prefetch_cache_rebases_when_seek_changes():
    encode_batch_sizes = []

    class DummyModel:
        feature_extractor = SimpleNamespace(nb_max_frames=4)

        def encode(self, features):
            features = np.asarray(features, dtype=np.float32)
            if features.ndim == 2:
                features = np.expand_dims(features, 0)
            encode_batch_sizes.append(int(features.shape[0]))
            return ctranslate2.StorageView.from_array(np.ascontiguousarray(features))

    cache = StandardEncoderPrefetchCache(
        model=DummyModel(),
        features=np.zeros((80, 20), dtype=np.float32),
        batch_size=2,
    )

    cache.get(seek=0, seek_clip_end=20)
    cache.get(seek=3, seek_clip_end=20)

    assert encode_batch_sizes == [2, 2]


def test_standard_pipeline_enables_encoder_batching_when_batch_size_gt_one():
    captured = {}

    class DummyModel:
        def __init__(self):
            self.feature_extractor = SimpleNamespace(sampling_rate=16000)
            self.generate_segments = self._original_generate_segments

        def _original_generate_segments(self, *args, **kwargs):
            return iter(())

        def transcribe(self, **kwargs):
            current_func = getattr(self.generate_segments, "__func__", self.generate_segments)
            original_func = getattr(self._original_generate_segments, "__func__", self._original_generate_segments)
            captured["generate_segments_patched"] = (
                current_func is not original_func
            )
            return iter([]), SimpleNamespace(duration=1.0)

    inferencer = object.__new__(FasterWhisperInference)
    inferencer.model = DummyModel()

    FasterWhisperInference._transcribe_with_standard_pipeline(
        inferencer,
        np.zeros(16000, dtype=np.float32),
        WhisperParams(batch_size=4),
        DummyProgress(),
    )

    assert captured["generate_segments_patched"] is True
    restored_func = getattr(inferencer.model.generate_segments, "__func__", inferencer.model.generate_segments)
    original_func = getattr(inferencer.model._original_generate_segments, "__func__", inferencer.model._original_generate_segments)
    assert restored_func is original_func


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
        use_batched_inference=True,
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
    assert progress.events[1] == (FasterWhisperInference.MODEL_READY_PROGRESS, "Loading audio..")
    assert progress.events[2] == (FasterWhisperInference.AUDIO_PREPARED_PROGRESS, "Audio loaded. Preparing chunks..")
    assert progress.events[3] == (FasterWhisperInference.CHUNKS_PREPARED_PROGRESS, "Prepared 4 chunks. Starting transcription..")
    assert progress.events[4] == (FasterWhisperInference.TRANSCRIPTION_PROGRESS_START, "Transcribing..")
    assert progress.events[5][0] > FasterWhisperInference.TRANSCRIPTION_PROGRESS_START


def test_batched_pipeline_repeats_initial_prompt_every_window(monkeypatch):
    captured = {}
    dummy_segments = [make_dummy_segment(1, 0.0, 14.8, "first chunk")]

    class DummyTokenizer:
        def __init__(self):
            self.calls = []

        def encode(self, text):
            self.calls.append(text)
            return [303, 404]

    class DummyModel:
        def __init__(self):
            self.feature_extractor = SimpleNamespace(sampling_rate=16000)
            self.tokenizer = DummyTokenizer()
            self.prompt_inputs = []

        def get_prompt(self, tokenizer, previous_tokens, without_timestamps=False, prefix=None, hotwords=None):
            prompt_tokens = list(previous_tokens)
            self.prompt_inputs.append(prompt_tokens)
            return prompt_tokens

    class DummyBatchedInferencePipeline:
        def __init__(self, model):
            self.model = model

        def transcribe(self, **kwargs):
            captured.update(kwargs)
            self.model.get_prompt(self.model.tokenizer, [1], False, None, None)
            self.model.get_prompt(self.model.tokenizer, [2], False, None, None)
            return iter(dummy_segments), SimpleNamespace(duration=15.0)

    inferencer = object.__new__(FasterWhisperInference)
    inferencer.model = DummyModel()
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
        use_batched_inference=True,
        batch_size=1,
        initial_prompt="Welcome to the school.",
        repeat_initial_prompt_every_window=True,
    ).to_list()

    segments, elapsed_time = FasterWhisperInference.transcribe(
        inferencer,
        np.zeros(15 * 16000, dtype=np.float32),
        progress,
        None,
        *whisper_params,
    )

    assert len(segments) == 1
    assert elapsed_time >= 0
    assert captured["initial_prompt"] is None
    assert inferencer.model.prompt_inputs == [
        [303, 404, 1],
        [303, 404, 2],
    ]
    assert inferencer.model.tokenizer.calls == [" Welcome to the school."]


def test_transcribe_uses_standard_pipeline_by_default(monkeypatch):
    captured = {"standard_called": False}

    inferencer = object.__new__(FasterWhisperInference)
    inferencer.model = SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000))
    inferencer.current_model_size = "large-v3"
    inferencer.current_compute_type = "float16"
    inferencer.update_model = lambda *args, **kwargs: None

    def fake_standard_pipeline(audio, params, progress):
        captured["standard_called"] = True
        captured["params"] = params
        return iter([make_dummy_segment(1, 0.0, 1.0, "standard path")]), SimpleNamespace(duration=1.0)

    inferencer._transcribe_with_standard_pipeline = fake_standard_pipeline

    def fail_batched_pipeline(*args, **kwargs):
        raise AssertionError("Batched pipeline should not run when use_batched_inference is disabled.")

    inferencer._transcribe_with_batching = fail_batched_pipeline

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
        None,
        *whisper_params,
    )

    assert captured["standard_called"] is True
    assert captured["params"].use_batched_inference is False
    assert len(segments) == 1
    assert elapsed_time >= 0


def test_transcribe_keeps_conditioning_for_short_standard_audio():
    captured = {}

    inferencer = object.__new__(FasterWhisperInference)
    inferencer.model = SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000))
    inferencer.current_model_size = "large-v3"
    inferencer.current_compute_type = "float16"
    inferencer.update_model = lambda *args, **kwargs: None

    def fake_standard_pipeline(audio, params, progress):
        captured["audio"] = audio
        captured["params"] = params
        return iter([make_dummy_segment(1, 0.0, 1.0, "short path")]), SimpleNamespace(duration=1.0)

    inferencer._transcribe_with_standard_pipeline = fake_standard_pipeline
    inferencer._transcribe_with_batching = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("Batched pipeline should not run when use_batched_inference is disabled.")
    )

    progress = DummyProgress()
    whisper_params = WhisperParams(
        model_size="large-v3",
        compute_type="float16",
        lang="en",
        chunk_length=1,
        condition_on_previous_text=True,
        word_timestamps=True,
    ).to_list()

    segments, elapsed_time = FasterWhisperInference.transcribe(
        inferencer,
        np.zeros(30 * 16000, dtype=np.float32),
        progress,
        None,
        *whisper_params,
    )

    assert captured["params"].condition_on_previous_text is True
    assert isinstance(captured["audio"], np.ndarray)
    assert len(segments) == 1
    assert elapsed_time >= 0


def test_transcribe_auto_disables_conditioning_for_long_standard_audio():
    captured = {}

    inferencer = object.__new__(FasterWhisperInference)
    inferencer.model = SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000))
    inferencer.current_model_size = "large-v3"
    inferencer.current_compute_type = "float16"
    inferencer.update_model = lambda *args, **kwargs: None

    def fake_standard_pipeline(audio, params, progress):
        captured["audio"] = audio
        captured["params"] = params
        return iter([make_dummy_segment(1, 0.0, 1.0, "long path")]), SimpleNamespace(duration=1.0)

    inferencer._transcribe_with_standard_pipeline = fake_standard_pipeline
    inferencer._transcribe_with_batching = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("Batched pipeline should not run when use_batched_inference is disabled.")
    )

    progress = DummyProgress()
    whisper_params = WhisperParams(
        model_size="large-v3",
        compute_type="float16",
        lang="en",
        chunk_length=1,
        condition_on_previous_text=True,
        word_timestamps=True,
    ).to_list()

    segments, elapsed_time = FasterWhisperInference.transcribe(
        inferencer,
        np.zeros(61 * 16000, dtype=np.float32),
        progress,
        None,
        *whisper_params,
    )

    assert captured["params"].condition_on_previous_text is False
    assert isinstance(captured["audio"], np.ndarray)
    assert len(segments) == 1
    assert elapsed_time >= 0
