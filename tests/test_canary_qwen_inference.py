import gradio as gr
import numpy as np
import torch

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
