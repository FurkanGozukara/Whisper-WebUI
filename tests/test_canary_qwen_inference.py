import os
import sys
import types

import gradio as gr
import numpy as np
import pytest
import torch
from pathlib import Path

from modules.whisper.canary_qwen_inference import CanaryQwenInference
from modules.whisper.data_classes import WhisperImpl, WhisperParams
from modules.whisper.whisper_factory import WhisperFactory


class DummyTokenizer:
    def ids_to_text(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return f" chunk-{int(token_ids[0])} "


class DummyCanaryModel:
    audio_locator_tag = "<|audio|>"

    def __init__(self):
        self.tokenizer = DummyTokenizer()
        self.calls = []
        self.next_id = 1

    def generate(self, prompts, audios, audio_lens, **kwargs):
        self.calls.append(
            {
                "prompts": prompts,
                "audio_shape": tuple(audios.shape),
                "audio_lens": audio_lens.detach().cpu().tolist(),
                "kwargs": kwargs,
            }
        )
        ids = torch.arange(self.next_id, self.next_id + len(prompts)).unsqueeze(1)
        self.next_id += len(prompts)
        return ids


def build_inferencer(tmp_path):
    inferencer = CanaryQwenInference(model_dir=str(tmp_path / "canary"))
    inferencer.device = "cpu"
    inferencer.model = DummyCanaryModel()
    inferencer.current_model_size = CanaryQwenInference.DEFAULT_MODEL_ID
    inferencer.current_compute_type = "float32"
    # synthetic signals are not speech to the voice detector: cut them at the quietest moment
    inferencer.CUT_STRATEGY = "energy"
    return inferencer


def test_factory_creates_canary_qwen_inference(tmp_path):
    inferencer = WhisperFactory.create_whisper_inference(
        whisper_type=WhisperImpl.CANARY_QWEN.value,
        canary_qwen_model_dir=str(tmp_path / "canary"),
    )

    assert isinstance(inferencer, CanaryQwenInference)
    assert inferencer.available_models[0] == CanaryQwenInference.DEFAULT_MODEL_ID


def test_canary_transcribe_batches_chunks_and_returns_chunk_timestamps(tmp_path):
    inferencer = build_inferencer(tmp_path)
    audio = np.ones(CanaryQwenInference.SAMPLE_RATE * 3, dtype=np.float32)
    params = WhisperParams(
        model_size=CanaryQwenInference.DEFAULT_MODEL_ID,
        lang="english",
        compute_type="float32",
        beam_size=1,
        chunk_length=1,
        batch_size=2,
        max_new_tokens=12,
        word_timestamps=True,
    )

    segments, _elapsed = inferencer.transcribe(
        audio,
        gr.Progress(),
        None,
        *params.to_list(),
        log_console=False,
        log_model_banner=False,
    )

    assert [segment.text for segment in segments] == ["chunk-1", "chunk-2", "chunk-3"]
    assert [segment.start for segment in segments] == [0.0, 1.0, 2.0]
    assert [segment.end for segment in segments] == [1.0, 2.0, 3.0]
    assert segments[0].words is None
    assert [call["audio_lens"] for call in inferencer.model.calls] == [[16000, 16000], [16000]]
    assert inferencer.model.calls[0]["kwargs"]["max_new_tokens"] == 12
    assert inferencer.model.calls[0]["prompts"][0][0]["content"] == "Transcribe the following: <|audio|>"


def test_canary_merges_raw_generation_kwargs(tmp_path):
    inferencer = build_inferencer(tmp_path)
    params = WhisperParams(
        model_size=CanaryQwenInference.DEFAULT_MODEL_ID,
        lang="english",
        compute_type="float32",
        max_new_tokens=12,
        canary_generation_kwargs='{"top_p": 0.9, "top_k": 50, "do_sample": true, "max_new_tokens": 7}',
    )

    segments, _elapsed = inferencer.transcribe(
        np.ones(CanaryQwenInference.SAMPLE_RATE, dtype=np.float32),
        gr.Progress(),
        None,
        *params.to_list(),
        log_console=False,
        log_model_banner=False,
    )

    assert segments[0].text == "chunk-1"
    assert inferencer.model.calls[0]["kwargs"]["top_p"] == 0.9
    assert inferencer.model.calls[0]["kwargs"]["top_k"] == 50
    assert inferencer.model.calls[0]["kwargs"]["do_sample"] is True
    assert inferencer.model.calls[0]["kwargs"]["max_new_tokens"] == 7


def test_canary_clamps_unsafe_generation_kwargs(tmp_path):
    inferencer = build_inferencer(tmp_path)
    params = WhisperParams(
        model_size=CanaryQwenInference.DEFAULT_MODEL_ID,
        lang="english",
        compute_type="float32",
        canary_generation_kwargs=(
            '{"max_new_tokens": 999999, "num_beams": 999, "top_p": 4, '
            '"top_k": -5, "temperature": -1, "repetition_penalty": -2, '
            '"length_penalty": 99, "no_repeat_ngram_size": 100}'
        ),
    )

    inferencer.transcribe(
        np.ones(CanaryQwenInference.SAMPLE_RATE, dtype=np.float32),
        gr.Progress(),
        None,
        *params.to_list(),
        log_console=False,
        log_model_banner=False,
    )

    kwargs = inferencer.model.calls[0]["kwargs"]
    assert kwargs["max_new_tokens"] == CanaryQwenInference.MAX_SAFE_MAX_NEW_TOKENS
    assert kwargs["num_beams"] == CanaryQwenInference.MAX_SAFE_NUM_BEAMS
    assert kwargs["top_p"] == 1.0
    assert kwargs["top_k"] == 0
    assert kwargs["temperature"] == 0.01
    assert kwargs["repetition_penalty"] == 0.01
    assert kwargs["length_penalty"] == 10.0
    assert kwargs["no_repeat_ngram_size"] == 20


def test_canary_clamps_nested_generation_config():
    kwargs = CanaryQwenInference.parse_canary_generation_kwargs(
        '{"generation_config": {"max_new_tokens": 999999, "num_beams": 999, "top_p": 2}}'
    )

    sanitized = CanaryQwenInference.sanitize_generation_kwargs(kwargs)

    generation_config = sanitized["generation_config"]
    assert generation_config.max_new_tokens == CanaryQwenInference.MAX_SAFE_MAX_NEW_TOKENS
    assert generation_config.num_beams == CanaryQwenInference.MAX_SAFE_NUM_BEAMS
    assert generation_config.top_p == 1.0


@pytest.mark.parametrize("raw", [
    '{"num_beams": 2, "num_return_sequences": 2}',
    '{"generation_config": {"num_beams": 2, "num_return_sequences": 2}}',
])
def test_canary_rejects_multiple_hypotheses_per_chunk(raw):
    kwargs = CanaryQwenInference.parse_canary_generation_kwargs(raw)
    with pytest.raises(ValueError, match="num_return_sequences=1"):
        CanaryQwenInference.sanitize_generation_kwargs(kwargs)


def test_canary_accepts_structured_generation_output(tmp_path):
    inferencer = build_inferencer(tmp_path)
    inferencer.model.generate = lambda **kwargs: types.SimpleNamespace(sequences=torch.tensor([[17]]))
    params = WhisperParams(model_size=CanaryQwenInference.DEFAULT_MODEL_ID, compute_type="float32",
                           canary_generation_kwargs='{"return_dict_in_generate": true}')
    segments, _ = inferencer.transcribe(np.ones(16000, dtype=np.float32), gr.Progress(), None,
                                       *params.to_list(), log_console=False, log_model_banner=False)
    assert [s.text for s in segments] == ["chunk-17"]


def test_canary_skips_digital_silence_and_preserves_real_timestamps(tmp_path):
    inferencer = build_inferencer(tmp_path)
    params = WhisperParams(model_size=CanaryQwenInference.DEFAULT_MODEL_ID, compute_type="float32",
                           chunk_length=1, batch_size=2)
    silence = np.zeros(16000, dtype=np.float32)
    segments, _ = inferencer.transcribe(silence, gr.Progress(), None, *params.to_list(), log_console=False)
    assert segments == []
    assert inferencer.model.calls == []
    # Even extremely quiet nonzero audio must be preserved; this is not an
    # amplitude-based speech detector that can erase distant voices.
    quiet = np.full(16000, 1e-8, dtype=np.float32)
    segments, _ = inferencer.transcribe(np.concatenate([quiet, silence, quiet]), gr.Progress(), None,
                                       *params.to_list(), log_console=False)
    assert [(s.start, s.end) for s in segments] == [(0.0, 1.0), (2.0, 3.0)]


def test_canary_merges_tiny_tail_into_previous_chunk(tmp_path):
    inferencer = build_inferencer(tmp_path)
    audio = np.zeros((CanaryQwenInference.SAMPLE_RATE * 2) + 273, dtype=np.float32)

    chunks = inferencer.build_audio_chunks(audio, chunk_length=1)

    assert [chunk["audio"].shape[-1] for chunk in chunks] == [
        CanaryQwenInference.SAMPLE_RATE,
        CanaryQwenInference.SAMPLE_RATE + 273,
    ]
    assert chunks[-1]["end_seconds"] == pytest.approx(audio.shape[-1] / CanaryQwenInference.SAMPLE_RATE)


@pytest.mark.parametrize("samples,expected_first_end", [
    (30 * 16000, 30.0),
    (40 * 16000, 40.0),
    (40 * 16000 + 1, 30.0),
    (60 * 16000, 30.0),
])
def test_canary_automatic_chunks_keep_utterances_up_to_40s_whole(tmp_path, samples, expected_first_end):
    inferencer = build_inferencer(tmp_path)
    chunks = inferencer.build_audio_chunks(np.ones(samples, dtype=np.float32), chunk_length=0)
    assert chunks[0]["end_seconds"] == expected_first_end
    assert chunks[-1]["end_seconds"] == pytest.approx(samples / 16000)
    if samples <= 40 * 16000:
        assert len(chunks) == 1


class FixedSpeechProbabilities:
    def __init__(self, probs):
        self.probs = probs
        self.waited = []
        self.closed = False

    def wait(self, frame_index=None):
        self.waited.append(frame_index)
        return self.probs

    def close(self):
        self.closed = True


def test_canary_automatic_chunks_cut_long_audio_at_the_detected_pause(tmp_path, monkeypatch):
    inferencer = build_inferencer(tmp_path)
    inferencer.CUT_STRATEGY = "vad"
    samples = 70 * 16000
    frames = -(-samples // CanaryQwenInference.VAD_FRAME_SAMPLES)
    probs = np.full(frames, 0.95, dtype=np.float32)
    pause = int(22.0 * 16000) // CanaryQwenInference.VAD_FRAME_SAMPLES
    probs[pause - 8:pause + 8] = 0.01  # half a second of silence at 22 s
    monkeypatch.setattr(CanaryQwenInference, "speech_probability_stream",
                        classmethod(lambda cls, audio: FixedSpeechProbabilities(probs)))

    chunks = inferencer.build_audio_chunks(np.ones(samples, dtype=np.float32), chunk_length=0)

    assert 21.7 <= chunks[0]["end_seconds"] <= 22.3
    lengths = [c["end_seconds"] - c["start_seconds"] for c in chunks]
    assert all(length <= CanaryQwenInference.AUTO_LONG_MAX_CHUNK_SECONDS + 1e-6 for length in lengths)
    assert all(length >= CanaryQwenInference.AUTO_LONG_MIN_CHUNK_SECONDS - 1e-6 for length in lengths[:-1])


def test_canary_cuts_each_chunk_once_its_part_is_analysed(tmp_path, monkeypatch):
    inferencer = build_inferencer(tmp_path)
    inferencer.CUT_STRATEGY = "vad"
    samples = 100 * 16000
    stream = FixedSpeechProbabilities(np.full(-(-samples // 512), 0.9, dtype=np.float32))
    monkeypatch.setattr(CanaryQwenInference, "speech_probability_stream", classmethod(lambda cls, audio: stream))

    chunks = inferencer.iter_audio_chunks(np.ones(samples, dtype=np.float32), chunk_length=0)
    first = next(chunks)

    assert 29.9 < first["end_seconds"] <= 30.0
    # only the frames up to the second chunk's latest end were needed so far
    second_latest = int(round(first["end_seconds"] * 16000)) + 30 * 16000
    assert stream.waited == [30 * 16000 // 512 + 5, second_latest // 512 + 5]
    assert not stream.closed
    chunks.close()
    assert stream.closed


def test_canary_pads_every_batch_of_a_long_recording_to_one_length(tmp_path):
    inferencer = build_inferencer(tmp_path)
    params = WhisperParams(model_size=CanaryQwenInference.DEFAULT_MODEL_ID, lang="english",
                           compute_type="float32", chunk_length=0, batch_size=2)
    audio = np.ones(int(95 * 16000), dtype=np.float32)

    inferencer.transcribe(audio, gr.Progress(), None, *params.to_list(), log_console=False,
                          log_model_banner=False)

    widths = {call["audio_shape"][1] for call in inferencer.model.calls}
    assert widths == {30 * 16000}


class LoopingCanaryModel(DummyCanaryModel):
    """Loops (no end token, max_new_tokens tokens) for any chunk longer than 8 seconds."""

    text_eos_id = 2
    text_pad_id = 0

    def generate(self, prompts, audios, audio_lens, **kwargs):
        super().generate(prompts, audios, audio_lens, **kwargs)
        rows = []
        for length in audio_lens.tolist():
            if length > 8 * 16000:
                rows.append([7] * kwargs["max_new_tokens"])
            else:
                rows.append([5, self.text_eos_id] + [self.text_pad_id] * (kwargs["max_new_tokens"] - 2))
        return torch.tensor(rows)


class LoopingTokenizer:
    def ids_to_text(self, token_ids):
        ids = [int(i) for i in (token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids)]
        if ids and ids[0] == 7:
            return "Kwame " * len(ids)
        return "Fine."


def test_canary_transcribes_a_looping_chunk_again_in_two_halves(tmp_path):
    inferencer = build_inferencer(tmp_path)
    inferencer.model = LoopingCanaryModel()
    inferencer.model.tokenizer = LoopingTokenizer()
    params = WhisperParams(model_size=CanaryQwenInference.DEFAULT_MODEL_ID, lang="english",
                           compute_type="float32", chunk_length=0, batch_size=4, max_new_tokens=32)

    segments, _ = inferencer.transcribe(np.ones(12 * 16000, dtype=np.float32), gr.Progress(), None,
                                        *params.to_list(), log_console=False, log_model_banner=False)

    assert [segment.text for segment in segments] == ["Fine.", "Fine."]
    assert segments[0].start == 0.0 and segments[-1].end == pytest.approx(12.0)
    assert segments[0].end == segments[1].start


class OutOfMemoryCanaryModel(DummyCanaryModel):
    """Runs out of GPU memory for batches over ``fits`` rows, and ``fail_first`` more times whatever the size."""

    def __init__(self, fits=16, fail_first=0):
        super().__init__()
        self.fits = fits
        self.fail_first = fail_first
        self.released = 0
        self.batches = []

    def generate(self, prompts, audios, audio_lens, **kwargs):
        self.batches.append(len(prompts))
        if self.fail_first or len(prompts) > self.fits:
            self.fail_first = max(0, self.fail_first - 1)
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB")
        return super().generate(prompts, audios, audio_lens, **kwargs)

    def release_cuda_graphs(self):
        self.released += 1


def transcribe_seconds(inferencer, seconds, batch_size):
    params = WhisperParams(model_size=CanaryQwenInference.DEFAULT_MODEL_ID, lang="english",
                           compute_type="float32", chunk_length=1, batch_size=batch_size)
    segments, _ = inferencer.transcribe(np.ones(seconds * 16000, dtype=np.float32), gr.Progress(), None,
                                        *params.to_list(), log_console=False, log_model_banner=False)
    return segments


def test_canary_releases_cached_graphs_and_retries_after_running_out_of_memory(tmp_path):
    inferencer = build_inferencer(tmp_path)
    inferencer.model = OutOfMemoryCanaryModel(fail_first=1)

    segments = transcribe_seconds(inferencer, 4, batch_size=4)

    assert [segment.text for segment in segments] == ["chunk-1", "chunk-2", "chunk-3", "chunk-4"]
    assert inferencer.model.batches == [4, 4]
    assert inferencer.model.released == 1
    assert inferencer.oom_batch_limit is None


def test_canary_halves_a_batch_that_does_not_fit_and_keeps_the_smaller_size(tmp_path):
    inferencer = build_inferencer(tmp_path)
    inferencer.model = OutOfMemoryCanaryModel(fits=2)

    segments = transcribe_seconds(inferencer, 8, batch_size=8)

    assert [segment.text for segment in segments] == [f"chunk-{i}" for i in range(1, 9)]
    assert [segment.start for segment in segments] == [float(i) for i in range(8)]
    assert inferencer.oom_batch_limit == 2
    inferencer.model.batches.clear()
    transcribe_seconds(inferencer, 5, batch_size=8)
    assert inferencer.model.batches == [2, 2, 1]


def test_canary_raises_out_of_memory_when_one_chunk_does_not_fit(tmp_path):
    inferencer = build_inferencer(tmp_path)
    inferencer.model = OutOfMemoryCanaryModel(fits=0)

    with pytest.raises(torch.cuda.OutOfMemoryError):
        transcribe_seconds(inferencer, 2, batch_size=2)
    assert inferencer.model.released >= 2


def test_canary_collapses_runaway_repetitions():
    collapse = CanaryQwenInference.collapse_repetitions
    assert collapse("Kwame Kwame Kwame Kwame Kwame Kwame so how") == "Kwame Kwame Kwame so how"
    assert collapse("on the mat on the mat on the mat on the mat") == "on the mat on the mat on the mat"
    assert collapse("Quami. Quami. Quami. I saw people") == "Quami. Quami. Quami. I saw people"
    assert collapse("a normal sentence.") == "a normal sentence."


@pytest.mark.parametrize("duration,requested,expected_first_end", [
    (20, 12, 12.0),
    (60, 10, 10.0),
    (60, 12, 12.0),
    (60, None, 40.0),
])
def test_canary_manual_chunk_lengths_and_legacy_none_remain_explicit(tmp_path, duration, requested, expected_first_end):
    inferencer = build_inferencer(tmp_path)
    chunks = inferencer.build_audio_chunks(np.ones(duration * 16000, dtype=np.float32), requested)
    assert chunks[0]["end_seconds"] == expected_first_end


def test_canary_automatic_chunks_accept_empty_audio(tmp_path):
    inferencer = build_inferencer(tmp_path)
    assert inferencer.build_audio_chunks(np.array([], dtype=np.float32), chunk_length=0) == []


def test_canary_skips_audio_shorter_than_safe_feature_window(tmp_path):
    inferencer = build_inferencer(tmp_path)
    params = WhisperParams(
        model_size=CanaryQwenInference.DEFAULT_MODEL_ID,
        lang="english",
        compute_type="float32",
    )

    segments, _elapsed = inferencer.transcribe(
        np.zeros(CanaryQwenInference.MIN_CHUNK_SAMPLES - 1, dtype=np.float32),
        gr.Progress(),
        None,
        *params.to_list(),
        log_console=False,
        log_model_banner=False,
    )

    assert segments == []
    assert inferencer.model.calls == []


def test_canary_enforces_minimum_requested_chunk_length(tmp_path):
    inferencer = build_inferencer(tmp_path)
    audio = np.zeros(CanaryQwenInference.MIN_CHUNK_SAMPLES * 3, dtype=np.float32)

    chunks = inferencer.build_audio_chunks(audio, chunk_length=0.001)

    assert chunks
    assert all(chunk["audio"].shape[-1] >= CanaryQwenInference.MIN_CHUNK_SAMPLES for chunk in chunks)


def test_canary_prepare_audio_array_replaces_non_finite_samples(tmp_path):
    inferencer = build_inferencer(tmp_path)

    audio = inferencer.prepare_audio_array(np.array([np.nan, np.inf, -np.inf, 0.5], dtype=np.float32))

    assert np.isfinite(audio).all()
    assert audio.tolist() == [0.0, 0.0, 0.0, 0.5]


def test_canary_disables_word_timestamp_writer_options(tmp_path):
    inferencer = build_inferencer(tmp_path)

    assert inferencer.supports_word_timestamps() is False
    assert inferencer.get_writer_options(WhisperParams(word_timestamps=True)) == {
        "highlight_words": False,
        "normalize_word_timestamps": False,
    }


def test_canary_downloads_remote_model_to_visible_model_dir(tmp_path, monkeypatch):
    inferencer = CanaryQwenInference(model_dir=str(tmp_path / "canary"))
    calls = []

    def fake_snapshot_download(repo_id, local_dir, cache_dir, token, tqdm_class):
        calls.append(
            {
                "repo_id": repo_id,
                "local_dir": local_dir,
                "cache_dir": cache_dir,
                "token": token,
                "has_tqdm_class": tqdm_class is not None,
            }
        )
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        Path(local_dir, "config.json").write_text("{}", encoding="utf-8")
        Path(local_dir, "model.safetensors").write_text("", encoding="utf-8")
        return local_dir

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    target = inferencer.resolve_model_target(CanaryQwenInference.DEFAULT_MODEL_ID)

    expected_dir = tmp_path / "canary" / "nvidia--canary-qwen-2.5b"
    assert target == str(expected_dir)
    assert expected_dir.is_dir()
    assert calls == [
        {
            "repo_id": CanaryQwenInference.DEFAULT_MODEL_ID,
            "local_dir": str(expected_dir),
            "cache_dir": str(tmp_path / "canary" / "hub"),
            "token": None,
            "has_tqdm_class": True,
        }
    ]


def test_canary_uses_visible_model_dir_without_redownloading(tmp_path, monkeypatch):
    inferencer = CanaryQwenInference(model_dir=str(tmp_path / "canary"))
    local_model_dir = tmp_path / "canary" / "nvidia--canary-qwen-2.5b"
    local_model_dir.mkdir(parents=True)
    (local_model_dir / "config.json").write_text("{}", encoding="utf-8")
    (local_model_dir / "model.safetensors").write_text("", encoding="utf-8")

    def fail_snapshot_download(*args, **kwargs):
        raise AssertionError("Should not download when the visible model folder exists.")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fail_snapshot_download)

    assert inferencer.resolve_model_target(CanaryQwenInference.DEFAULT_MODEL_ID) == str(local_model_dir)


def test_canary_redownloads_incomplete_visible_model_dir(tmp_path, monkeypatch):
    inferencer = CanaryQwenInference(model_dir=str(tmp_path / "canary"))
    local_model_dir = tmp_path / "canary" / "nvidia--canary-qwen-2.5b"
    local_model_dir.mkdir(parents=True)
    (local_model_dir / "config.json").write_text("{}", encoding="utf-8")
    calls = []

    def fake_snapshot_download(repo_id, local_dir, cache_dir, token, tqdm_class):
        calls.append(repo_id)
        Path(local_dir, "model.safetensors").write_text("", encoding="utf-8")
        return local_dir

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    assert inferencer.resolve_model_target(CanaryQwenInference.DEFAULT_MODEL_ID) == str(local_model_dir)
    assert calls == [CanaryQwenInference.DEFAULT_MODEL_ID]


def test_canary_raises_when_download_does_not_create_complete_model_dir(tmp_path, monkeypatch):
    inferencer = CanaryQwenInference(model_dir=str(tmp_path / "canary"))

    def fake_snapshot_download(repo_id, local_dir, cache_dir, token, tqdm_class):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        Path(local_dir, "config.json").write_text("{}", encoding="utf-8")
        return local_dir

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    try:
        inferencer.resolve_model_target(CanaryQwenInference.DEFAULT_MODEL_ID)
    except RuntimeError as exc:
        assert "complete model folder" in str(exc)
    else:
        raise AssertionError("Expected incomplete Canary-Qwen download to fail.")


def test_canary_update_model_emits_download_and_load_status(tmp_path, monkeypatch):
    inferencer = CanaryQwenInference(model_dir=str(tmp_path / "canary"))
    statuses = []

    class DummyLoadedModel:
        audio_locator_tag = "<|audio|>"

        def __init__(self):
            self.tokenizer = DummyTokenizer()

        def eval(self):
            return self

        def to(self, *args, **kwargs):
            return self

    class DummySalm:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return DummyLoadedModel()

    def fake_snapshot_download(repo_id, local_dir, cache_dir, token, tqdm_class):
        bar = tqdm_class(total=2, unit="files")
        bar.update(1)
        bar.update(1)
        bar.close()
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        Path(local_dir, "config.json").write_text("{}", encoding="utf-8")
        Path(local_dir, "model.safetensors").write_text("", encoding="utf-8")
        return local_dir

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(inferencer, "import_salm", lambda: DummySalm)
    hub_cache_before = os.environ.get("HF_HUB_CACHE")

    inferencer.update_model(
        CanaryQwenInference.DEFAULT_MODEL_ID,
        "float32",
        gr.Progress(),
        progress_callback=lambda _progress, _segment=None, status=None: statuses.append(status) if status else None,
    )

    assert any("Downloading Canary-Qwen model to" in status for status in statuses)
    assert any("Downloading Canary-Qwen model:" in status for status in statuses)
    assert any("Canary-Qwen model download finished." == status for status in statuses)
    assert any("Loading Canary-Qwen model from" in status for status in statuses)
    assert statuses[-1] == "Canary-Qwen model loaded. Starting transcription.."
    # The Canary folder is the Hugging Face cache only while the model loads, not for the rest of the worker
    assert os.environ.get("HF_HUB_CACHE") == hub_cache_before


def test_canary_patches_missing_lightning_neptune_logger(monkeypatch):
    lightning_loggers = types.ModuleType("lightning.pytorch.loggers")
    lightning_loggers.__all__ = ["TensorBoardLogger"]
    pytorch_lightning_loggers = types.ModuleType("pytorch_lightning.loggers")
    pytorch_lightning_loggers.__all__ = ("TensorBoardLogger",)

    monkeypatch.setitem(sys.modules, "lightning.pytorch.loggers", lightning_loggers)
    monkeypatch.setitem(sys.modules, "pytorch_lightning.loggers", pytorch_lightning_loggers)

    CanaryQwenInference.patch_lightning_neptune_logger_compat()

    for loggers_module in (lightning_loggers, pytorch_lightning_loggers):
        assert "NeptuneLogger" in loggers_module.__all__
        with pytest.raises(ImportError, match="NeptuneLogger"):
            loggers_module.NeptuneLogger()


def test_canary_rejects_unsupported_translation(tmp_path):
    inferencer = build_inferencer(tmp_path)
    params = WhisperParams(
        model_size=CanaryQwenInference.DEFAULT_MODEL_ID,
        lang="english",
        is_translate=True,
        compute_type="float32",
    )

    try:
        inferencer.transcribe(
            np.zeros(CanaryQwenInference.SAMPLE_RATE, dtype=np.float32),
            gr.Progress(),
            None,
            *params.to_list(),
        )
    except ValueError as exc:
        assert "does not support" in str(exc)
    else:
        raise AssertionError("Expected Canary-Qwen translation request to fail")
