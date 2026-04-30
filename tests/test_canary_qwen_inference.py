import gradio as gr
import numpy as np
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
    audio = np.zeros(CanaryQwenInference.SAMPLE_RATE * 3, dtype=np.float32)
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
        np.zeros(CanaryQwenInference.SAMPLE_RATE, dtype=np.float32),
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


def test_canary_disables_word_timestamp_writer_options(tmp_path):
    inferencer = build_inferencer(tmp_path)

    assert inferencer.supports_word_timestamps() is False
    assert inferencer.get_writer_options(WhisperParams(word_timestamps=True)) == {"highlight_words": False}


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
