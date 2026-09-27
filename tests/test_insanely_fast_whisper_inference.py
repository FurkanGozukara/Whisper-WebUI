from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from modules.whisper.data_classes import WhisperParams
from modules.whisper.insanely_fast_whisper_inference import InsanelyFastWhisperInference


def write_complete_transformers_model(path):
    path.mkdir(parents=True, exist_ok=True)
    for filename in InsanelyFastWhisperInference.REQUIRED_MODEL_FILES:
        (path / filename).write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


class DummyPipeline:
    def __init__(self, side_effects=None):
        self.calls = []
        self.side_effects = list(side_effects or [])
        self.generation_config = SimpleNamespace(max_new_tokens=256, max_length=448)
        self.model = SimpleNamespace(config=SimpleNamespace(max_target_positions=448))
        self.feature_extractor = SimpleNamespace(sampling_rate=16000)
        self.tokenizer = SimpleNamespace(clean_up_tokenization_spaces=True)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.side_effects:
            effect = self.side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            return effect
        return {"chunks": [{"timestamp": (0.0, 1.0), "text": " ok"}]}


class DummyProgress:
    def __call__(self, value, desc=None):
        pass


class RecordingProgress:
    def __init__(self):
        self.calls = []

    def __call__(self, value, desc=None):
        self.calls.append((value, desc))


def build_inferencer(dummy_pipeline):
    inferencer = object.__new__(InsanelyFastWhisperInference)
    inferencer.model = dummy_pipeline
    inferencer.current_model_size = "large-v3"
    inferencer.current_compute_type = "float32"
    inferencer.update_model = lambda *args, **kwargs: None
    return inferencer


def test_insanely_fast_whisper_disables_previous_token_conditioning_by_default():
    dummy_pipeline = DummyPipeline()
    inferencer = build_inferencer(dummy_pipeline)

    generate_kwargs = inferencer.build_generate_kwargs(WhisperParams(model_size="large-v3"))

    assert generate_kwargs["condition_on_prev_tokens"] is False
    assert "max_new_tokens" not in generate_kwargs


def test_insanely_fast_whisper_respects_previous_token_conditioning_when_enabled():
    dummy_pipeline = DummyPipeline()
    inferencer = build_inferencer(dummy_pipeline)

    generate_kwargs = inferencer.build_generate_kwargs(
        WhisperParams(model_size="large-v3", condition_on_previous_text=True)
    )

    assert generate_kwargs["condition_on_prev_tokens"] is True


def test_insanely_fast_whisper_retries_prompt_length_error_with_safe_max_new_tokens():
    error = ValueError(
        "The length of the prompt is 197, and the `max_new_tokens` 256. "
        "Thus, the combined length of the prompt and `max_new_tokens` is: 453. "
        "This exceeds the `max_length` of the Whisper model: 448."
    )
    dummy_pipeline = DummyPipeline(side_effects=[error])
    inferencer = build_inferencer(dummy_pipeline)

    segments, _elapsed = inferencer.transcribe(
        np.zeros(16000, dtype=np.float32),
        DummyProgress(),
        None,
        *WhisperParams(model_size="large-v3", max_new_tokens=256).to_list(),
        log_console=False,
        log_model_banner=False,
    )

    assert [segment.text for segment in segments] == [" ok"]
    assert len(dummy_pipeline.calls) == 2
    assert dummy_pipeline.calls[1]["generate_kwargs"]["max_new_tokens"] == 250


def test_insanely_fast_whisper_retries_repeated_prompt_length_errors():
    first_error = ValueError(
        "The length of the prompt is 197, and the `max_new_tokens` 256. "
        "Thus, the combined length of the prompt and `max_new_tokens` is: 453. "
        "This exceeds the `max_length` of the Whisper model: 448."
    )
    second_error = ValueError(
        "The length of the prompt is 230, and the `max_new_tokens` 250. "
        "Thus, the combined length of the prompt and `max_new_tokens` is: 480. "
        "This exceeds the `max_length` of the Whisper model: 448."
    )
    dummy_pipeline = DummyPipeline(side_effects=[first_error, second_error])
    inferencer = build_inferencer(dummy_pipeline)

    segments, _elapsed = inferencer.transcribe(
        np.zeros(16000, dtype=np.float32),
        DummyProgress(),
        None,
        *WhisperParams(model_size="large-v3", max_new_tokens=256).to_list(),
        log_console=False,
        log_model_banner=False,
    )

    assert [segment.text for segment in segments] == [" ok"]
    assert len(dummy_pipeline.calls) == 3
    assert dummy_pipeline.calls[1]["generate_kwargs"]["max_new_tokens"] == 250
    assert dummy_pipeline.calls[2]["generate_kwargs"]["max_new_tokens"] == 217


def test_insanely_fast_whisper_decodes_file_before_calling_transformers(monkeypatch):
    dummy_pipeline = DummyPipeline()
    inferencer = build_inferencer(dummy_pipeline)
    decoded_audio = np.linspace(-0.5, 0.5, num=16000, dtype=np.float32)

    def fake_decode_audio(path, sampling_rate):
        assert path == "input.mp4"
        assert sampling_rate == 16000
        return decoded_audio

    monkeypatch.setattr(
        "modules.whisper.insanely_fast_whisper_inference.decode_audio",
        fake_decode_audio,
    )

    segments, _elapsed = inferencer.transcribe(
        "input.mp4",
        DummyProgress(),
        None,
        *WhisperParams(model_size="large-v3").to_list(),
        log_console=False,
        log_model_banner=False,
    )

    assert [segment.text for segment in segments] == [" ok"]
    assert isinstance(dummy_pipeline.calls[0]["inputs"], np.ndarray)
    assert np.array_equal(dummy_pipeline.calls[0]["inputs"], decoded_audio)
    assert "chunk_length_s" not in dummy_pipeline.calls[0]


def test_insanely_fast_whisper_keeps_non_default_chunk_length_explicit():
    dummy_pipeline = DummyPipeline()
    inferencer = build_inferencer(dummy_pipeline)

    inferencer.transcribe(
        np.zeros(16000, dtype=np.float32),
        DummyProgress(),
        None,
        *WhisperParams(model_size="large-v3", chunk_length=20).to_list(),
        log_console=False,
        log_model_banner=False,
    )

    assert dummy_pipeline.calls[0]["chunk_length_s"] == 20


def test_insanely_fast_whisper_parses_new_transformers_decoder_length_error():
    message = (
        "The length of `decoder_input_ids`, including special start tokens, prompt tokens, "
        "and previous tokens, is 197, and `max_new_tokens` is 256. Thus, the combined length "
        "of `decoder_input_ids` and `max_new_tokens` is: 453. This exceeds the "
        "`max_target_positions` of the Whisper model: 448."
    )

    parsed = InsanelyFastWhisperInference.parse_prompt_length_error(message)

    assert parsed == (197, 256, 448)


def test_insanely_fast_whisper_clamps_oversized_max_new_tokens():
    dummy_pipeline = DummyPipeline()
    inferencer = build_inferencer(dummy_pipeline)

    generate_kwargs = inferencer.build_generate_kwargs(
        WhisperParams(model_size="large-v3", max_new_tokens=999)
    )

    assert generate_kwargs["max_new_tokens"] == 432


def test_insanely_fast_whisper_uses_sdpa_for_float32_model_loading():
    assert InsanelyFastWhisperInference.torch_dtype_for_compute_type("float32") is torch.float32
    assert InsanelyFastWhisperInference.model_kwargs_for_torch_dtype(torch.float32) == {
        "attn_implementation": "sdpa"
    }
    assert InsanelyFastWhisperInference.pipeline_dtype_kwargs(torch.float32) == {"dtype": torch.float32}


def test_insanely_fast_whisper_prefers_float16_default_compute_type():
    inferencer = object.__new__(InsanelyFastWhisperInference)
    inferencer.available_compute_types = ["bfloat16", "float16", "float32"]

    assert inferencer.get_compute_type() == "float16"


def test_insanely_fast_whisper_disables_bpe_tokenizer_cleanup_on_loaded_pipeline():
    dummy_pipeline = DummyPipeline()

    InsanelyFastWhisperInference.disable_bpe_tokenizer_cleanup_warning(dummy_pipeline)

    assert dummy_pipeline.tokenizer.clean_up_tokenization_spaces is False


def test_insanely_fast_whisper_disables_transformers_torchcodec_probe(monkeypatch):
    import transformers.pipelines.automatic_speech_recognition as asr_pipeline

    monkeypatch.setattr(asr_pipeline, "is_torchcodec_available", lambda: True)

    InsanelyFastWhisperInference.disable_transformers_torchcodec_probe()

    assert asr_pipeline.is_torchcodec_available() is False


def test_insanely_fast_whisper_recognizes_complete_local_transformers_model(tmp_path):
    local_model = tmp_path / "large-v3"
    write_complete_transformers_model(local_model)

    inferencer = object.__new__(InsanelyFastWhisperInference)
    inferencer.model_dir = str(tmp_path)
    inferencer.download_model = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("downloaded"))

    assert inferencer.resolve_model_target("large-v3", DummyProgress()) == str(local_model)


def test_insanely_fast_whisper_rejects_incomplete_nonempty_model_folder(tmp_path):
    incomplete_model = tmp_path / "large-v3"
    incomplete_model.mkdir()
    (incomplete_model / "config.json").write_text("{}", encoding="utf-8")

    assert not InsanelyFastWhisperInference.has_transformers_model_files(incomplete_model)


def test_insanely_fast_whisper_uses_hf_cache_snapshot_before_download(tmp_path, monkeypatch):
    local_root = tmp_path / "local"
    local_root.mkdir()
    cached_model = (
        tmp_path
        / "hub"
        / "models--openai--whisper-large-v3"
        / "snapshots"
        / "abcdef"
    )
    write_complete_transformers_model(cached_model)

    monkeypatch.setattr(
        InsanelyFastWhisperInference,
        "candidate_hf_cache_dirs",
        classmethod(lambda cls: [str(tmp_path / "hub")]),
    )
    inferencer = object.__new__(InsanelyFastWhisperInference)
    inferencer.model_dir = str(local_root)
    inferencer.download_model = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("downloaded"))

    assert inferencer.resolve_model_target("large-v3", DummyProgress()) == str(cached_model)


def test_insanely_fast_whisper_downloads_only_when_local_and_cache_are_missing(tmp_path, monkeypatch):
    local_root = tmp_path / "local"
    monkeypatch.setattr(
        InsanelyFastWhisperInference,
        "candidate_hf_cache_dirs",
        classmethod(lambda cls: [str(tmp_path / "empty-hub")]),
    )
    inferencer = object.__new__(InsanelyFastWhisperInference)
    inferencer.model_dir = str(local_root)

    def fake_download(model_size, download_root, progress):
        assert model_size == "large-v3"
        assert download_root == str(local_root / "large-v3")
        write_complete_transformers_model(local_root / "large-v3")
        return download_root

    inferencer.download_model = fake_download

    assert inferencer.resolve_model_target("large-v3", DummyProgress()) == str(local_root / "large-v3")


def test_insanely_fast_whisper_falls_back_to_top_level_text_when_chunks_are_empty():
    dummy_pipeline = DummyPipeline(side_effects=[{"text": " transcript", "chunks": []}])
    inferencer = build_inferencer(dummy_pipeline)

    segments, _elapsed = inferencer.transcribe(
        np.zeros(16000, dtype=np.float32),
        DummyProgress(),
        None,
        *WhisperParams(model_size="large-v3").to_list(),
        log_console=False,
        log_model_banner=False,
    )

    assert len(segments) == 1
    assert segments[0].text == " transcript"
    assert segments[0].start == 0.0
    assert segments[0].end == 1.0


def test_insanely_fast_whisper_logs_segment_text_when_enabled(caplog):
    caplog.set_level("INFO", logger="Whisper-WebUI")
    dummy_pipeline = DummyPipeline()
    inferencer = build_inferencer(dummy_pipeline)

    inferencer.transcribe(
        np.zeros(16000, dtype=np.float32),
        DummyProgress(),
        None,
        *WhisperParams(model_size="large-v3").to_list(),
        log_console=True,
        log_model_banner=False,
    )

    assert " ok" in caplog.text


def test_insanely_fast_whisper_emits_segments_to_live_progress_callback():
    dummy_pipeline = DummyPipeline()
    inferencer = build_inferencer(dummy_pipeline)
    progress = RecordingProgress()
    callback_events = []

    def progress_callback(progress_value, segment=None, status=None):
        callback_events.append((progress_value, segment, status))

    segments, _elapsed = inferencer.transcribe(
        np.zeros(16000, dtype=np.float32),
        progress,
        progress_callback,
        *WhisperParams(model_size="large-v3").to_list(),
        log_console=False,
        log_model_banner=False,
    )

    status_events = [event for event in callback_events if event[2]]
    segment_events = [event for event in callback_events if event[1] is not None]

    assert [segment.text for segment in segments] == [" ok"]
    assert status_events[0][2] == "Starting Insanely Fast Whisper transcription.."
    assert len(segment_events) == 1
    assert segment_events[0][1].text == " ok"
    assert any(desc and "Transcribing.." in desc for _value, desc in progress.calls)
    assert any(desc and "chunk 1/1" in desc for _value, desc in progress.calls)


def test_insanely_fast_whisper_live_callback_processes_audio_in_chunks():
    dummy_pipeline = DummyPipeline(side_effects=[
        {"chunks": [{"timestamp": (0.0, 0.5), "text": " first"}]},
        {"chunks": [{"timestamp": (0.0, 0.5), "text": " second"}]},
    ])
    inferencer = build_inferencer(dummy_pipeline)
    progress = RecordingProgress()
    callback_events = []

    def progress_callback(progress_value, segment=None, status=None):
        callback_events.append((progress_value, segment, status, len(dummy_pipeline.calls)))

    segments, _elapsed = inferencer.transcribe(
        np.zeros(2 * 16000, dtype=np.float32),
        progress,
        progress_callback,
        *WhisperParams(model_size="large-v3", chunk_length=1).to_list(),
        log_console=False,
        log_model_banner=False,
    )

    segment_events = [event for event in callback_events if event[1] is not None]
    status_events = [event for event in callback_events if event[2]]

    assert len(dummy_pipeline.calls) == 2
    assert [call["inputs"].shape[0] for call in dummy_pipeline.calls] == [16000, 16000]
    assert [segment.text for segment in segments] == [" first", " second"]
    assert [(segment.start, segment.end) for segment in segments] == [(0.0, 0.5), (1.0, 1.5)]
    assert [event[1].text for event in segment_events] == [" first", " second"]
    assert segment_events[0][3] == 1
    assert any("Starting Insanely Fast Whisper" in event[2] for event in status_events)
    assert any(desc and "chunk 1/2" in desc for _value, desc in progress.calls)
    assert any(desc and "chunk 2/2" in desc for _value, desc in progress.calls)


def test_insanely_fast_whisper_live_chunk_length_uses_whisper_window():
    assert InsanelyFastWhisperInference.resolve_live_chunk_length_seconds(None) == 30
    assert InsanelyFastWhisperInference.resolve_live_chunk_length_seconds(0) == 30
    # 10 s is the shared WhisperParams default, which means "not customized" here.
    assert InsanelyFastWhisperInference.resolve_live_chunk_length_seconds(10) == 30
    assert InsanelyFastWhisperInference.resolve_live_chunk_length_seconds(20) == 20
    assert InsanelyFastWhisperInference.resolve_live_chunk_length_seconds(45) == 30
    assert InsanelyFastWhisperInference.resolve_live_chunk_length_seconds(1) == 1


def test_insanely_fast_whisper_live_chunks_end_in_a_pause():
    sampling_rate = 16000
    audio = np.full(70 * sampling_rate, 0.5, dtype=np.float32)
    pause_start = int(26.0 * sampling_rate)
    audio[pause_start:pause_start + int(0.3 * sampling_rate)] = 0.0

    bounds = InsanelyFastWhisperInference.live_chunk_bounds(audio, sampling_rate, 30)

    assert bounds[0][0] == 0
    assert pause_start < bounds[0][1] <= pause_start + int(0.3 * sampling_rate)
    assert all(end - start <= 30 * sampling_rate for start, end in bounds)
    assert [start for start, _end in bounds[1:]] == [end for _start, end in bounds[:-1]]
    assert bounds[-1][1] == len(audio)


def test_insanely_fast_whisper_maps_every_model_name_to_a_published_repo():
    # openai/whisper-large-v1 and openai/whisper-turbo do not exist on the Hub (404 on first use).
    repo = InsanelyFastWhisperInference.repo_for_model_size

    assert repo("large-v1") == ("MonsterMMORPG/Wan_GGUF", "Whisper_Transformers/large-v1")
    assert repo("large") == ("openai/whisper-large-v3", None)
    assert repo("turbo") == ("openai/whisper-large-v3-turbo", None)
    assert repo("large-v3-turbo") == ("openai/whisper-large-v3-turbo", None)
    assert repo("medium.en") == ("openai/whisper-medium.en", None)
    assert repo("distil-large-v3") == ("distil-whisper/distil-large-v3", None)


def test_insanely_fast_whisper_alias_uses_the_canonical_model_folder(tmp_path):
    local_model = tmp_path / "large-v3-turbo"
    write_complete_transformers_model(local_model)

    inferencer = object.__new__(InsanelyFastWhisperInference)
    inferencer.model_dir = str(tmp_path)
    inferencer.download_model = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("downloaded"))

    assert inferencer.resolve_model_target("turbo", DummyProgress()) == str(local_model)
    assert inferencer.model_to_download("turbo") is None


def test_insanely_fast_whisper_downloads_hosted_model_into_its_folder(tmp_path, monkeypatch):
    import modules.whisper.insanely_fast_whisper_inference as ifw_module

    calls = []

    def fake_snapshot_download(repo_id, allow_patterns, local_dir):
        calls.append((repo_id, allow_patterns))
        write_complete_transformers_model(Path(local_dir) / "Whisper_Transformers" / "large-v1")
        return local_dir

    monkeypatch.setattr(ifw_module, "snapshot_download", fake_snapshot_download)
    target = tmp_path / "large-v1"
    target.mkdir()  # left empty by an earlier failed attempt

    assert InsanelyFastWhisperInference.download_model("large-v1", str(target), DummyProgress()) == str(target)
    assert calls == [("MonsterMMORPG/Wan_GGUF", ["Whisper_Transformers/large-v1/*"])]
    assert InsanelyFastWhisperInference.has_transformers_model_files(target)
    assert [path.name for path in tmp_path.iterdir()] == ["large-v1"]  # staging folder removed


def test_insanely_fast_whisper_finds_hosted_model_in_hf_cache_subfolder(tmp_path, monkeypatch):
    cached_model = (
        tmp_path / "hub" / "models--MonsterMMORPG--Wan_GGUF" / "snapshots" / "abcdef"
        / "Whisper_Transformers" / "large-v1"
    )
    write_complete_transformers_model(cached_model)
    monkeypatch.setattr(
        InsanelyFastWhisperInference,
        "candidate_hf_cache_dirs",
        classmethod(lambda cls: [str(tmp_path / "hub")]),
    )

    assert InsanelyFastWhisperInference.find_cached_transformers_model("large-v1") == str(cached_model)


def test_insanely_fast_whisper_model_list_skips_hidden_download_folders(tmp_path):
    (tmp_path / ".download-large-v1").mkdir()
    (tmp_path / "my-finetune").mkdir()

    inferencer = object.__new__(InsanelyFastWhisperInference)
    inferencer.model_dir = str(tmp_path)
    models = inferencer.get_model_paths()

    assert "my-finetune" in models
    assert ".download-large-v1" not in models
