"""INT8 ConvRot Canary-Qwen: model list, hosted download, fallback, kernels and (when the model is present) the engine.

The GPU tests need an NVIDIA Ampere or newer GPU with Triton and flash-attn; the engine tests also need the
converted model in models/Whisper/canary-qwen/canary-qwen-2.5b-int8-convrot.
"""

import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
import torch

from modules.utils.paths import CANARY_QWEN_MODELS_DIR
from modules.whisper.canary_qwen_inference import CanaryQwenInference
from modules.whisper.convrot import registry
from modules.whisper.convrot.registry import (CANARY_CONVROT_FALLBACK_MODELS, HOSTED_CANARY_CONVROT_MODELS,
                                              is_canary_convrot_model_dir, is_convrot_model_dir)

CONVROT_NAME = "canary-qwen-2.5b-int8-convrot"
MODEL_DIR = os.path.join(CANARY_QWEN_MODELS_DIR, CONVROT_NAME)


def gpu_ok() -> bool:
    return registry.convrot_runtime_supported()[0]


def make_fake_convrot_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"convrot": {"runtime": {}}, "salm": {}, "llm": {}}), encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"")
    return path


def test_hosted_canary_convrot_model_is_listed(tmp_path):
    (tmp_path / "canary" / ".download-canary-qwen-2.5b-int8-convrot").mkdir(parents=True)
    inferencer = CanaryQwenInference(model_dir=str(tmp_path / "canary"))
    assert inferencer.available_models[0] == CanaryQwenInference.DEFAULT_MODEL_ID
    assert CONVROT_NAME in inferencer.available_models
    assert not any(m.startswith(".download-") for m in inferencer.available_models)
    assert HOSTED_CANARY_CONVROT_MODELS[CONVROT_NAME][1].startswith("Canary_INT8_ConvRot/")
    assert CANARY_CONVROT_FALLBACK_MODELS[CONVROT_NAME] == CanaryQwenInference.DEFAULT_MODEL_ID


def test_canary_convrot_folder_detection(tmp_path):
    folder = make_fake_convrot_dir(tmp_path / "m")
    assert is_convrot_model_dir(str(folder))
    assert is_canary_convrot_model_dir(str(folder))
    (folder / "config.json").write_text(json.dumps({"convrot": {}, "whisper_dims": {}}), encoding="utf-8")
    assert not is_canary_convrot_model_dir(str(folder))  # a Whisper ConvRot folder
    assert not is_canary_convrot_model_dir(str(tmp_path / "missing"))


def test_canary_convrot_downloads_hosted_model(tmp_path, monkeypatch):
    inferencer = CanaryQwenInference(model_dir=str(tmp_path / "canary"))
    calls = []

    def fake_download(name, target_dir, tqdm_class=None, token=None):
        calls.append((name, target_dir, tqdm_class is not None))
        make_fake_convrot_dir(Path(target_dir))
        return target_dir

    monkeypatch.setattr("modules.whisper.canary_qwen_inference.download_convrot_model", fake_download)
    monkeypatch.setattr("modules.whisper.canary_qwen_inference.convrot_runtime_supported", lambda: (True, ""))
    target = inferencer.resolve_convrot_model(CONVROT_NAME)
    assert target == str(tmp_path / "canary" / CONVROT_NAME)
    assert calls == [(CONVROT_NAME, target, True)]
    assert inferencer.resolve_convrot_model(CONVROT_NAME) == target  # no second download
    assert len(calls) == 1
    assert inferencer.resolve_convrot_model(CanaryQwenInference.DEFAULT_MODEL_ID) is None


def test_canary_convrot_falls_back_to_nemo_when_unsupported(tmp_path, monkeypatch):
    inferencer = CanaryQwenInference(model_dir=str(tmp_path / "canary"))
    loaded = []

    class DummyModel:
        audio_locator_tag = "<|audio|>"

        def eval(self):
            return self

        def to(self, *args, **kwargs):
            return self

    class DummySalm:
        @staticmethod
        def from_pretrained(target, **kwargs):
            loaded.append(target)
            return DummyModel()

    monkeypatch.setattr("modules.whisper.canary_qwen_inference.convrot_runtime_supported",
                        lambda: (False, "GPU compute capability 7.5"))
    monkeypatch.setattr(inferencer, "import_salm", lambda: DummySalm)
    monkeypatch.setattr(inferencer, "resolve_model_target", lambda model_size, **kwargs: model_size)
    statuses = []
    inferencer.update_model(CONVROT_NAME, "float32", gr.Progress(),
                            progress_callback=lambda _p, _s=None, status=None: statuses.append(status) if status else None)
    assert loaded == [CanaryQwenInference.DEFAULT_MODEL_ID]
    assert inferencer.current_model_size == CONVROT_NAME  # the selection stays; no reload on the next job
    assert not inferencer.should_load_model_for_selection(CONVROT_NAME, "float32")
    assert any("cannot run on this system" in s for s in statuses)


@pytest.mark.skipif(not gpu_ok(), reason="needs an NVIDIA Ampere+ GPU with Triton and flash-attn")
def test_canary_convrot_kernels_match_reference():
    from modules.whisper.convrot import kernels as WK
    from modules.whisper.convrot.canary import kernels as CK

    torch.manual_seed(0)
    h = WK.hadamard_matrix(256, device="cuda", dtype=torch.float64)
    for pre in (CK.PRE_NONE, CK.PRE_LAYERNORM, CK.PRE_RMSNORM, CK.PRE_SILU, CK.PRE_SWIGLU):
        for m in (1, 3, 100):
            k = 2048
            x = torch.randn(m, 2 * k if pre == CK.PRE_SWIGLU else k, device="cuda")
            w = torch.rand(k, device="cuda") + 0.5
            b = torch.randn(k, device="cuda") * 0.1
            ref = (CK.torch_pre(x, pre, w, b, 1e-5).double().view(m, k // 256, 256) @ h).view(m, k)
            rot = CK.rotate_pre(x, pre, w, b, 1e-5, out_mode=CK.OUT_ROT)
            assert ((rot.double() - ref).abs().max() / ref.abs().max()).item() < 2e-3
            q, s = CK.rotate_pre(x, pre, w, b, 1e-5, out_mode=CK.OUT_Q_GROUP)
            deq = (q.double().view(m, k // 256, 256) * s.double()[:, :, None]).view(m, k)
            assert ((deq - ref).abs().max() / ref.abs().max()).item() < 1e-2
    inv_freq = (1.0 / (1e6 ** (torch.arange(0, 128, 2, dtype=torch.int64).float() / 128))).cuda()
    qkv = torch.randn(5, 32 * 128, device="cuda").half()
    pos = torch.tensor([0, 1, 17, 300, 900], device="cuda", dtype=torch.int32)
    qw, kw = torch.rand(128, device="cuda") + 0.5, torch.rand(128, device="cuda") + 0.5
    out = CK.qk_norm_rope(qkv, pos, inv_freq, qw, kw, 16, 8, 128, 1e-6)
    ref = CK.torch_qk_norm_rope(qkv, pos, inv_freq, qw, kw, 16, 8, 128, 1e-6)
    assert ((out.float() - ref).abs().max() / ref.abs().max()).item() < 2e-3


@pytest.mark.skipif(not gpu_ok() or not is_canary_convrot_model_dir(MODEL_DIR),
                    reason="needs an Ampere+ GPU and the converted model in the Canary model folder")
def test_canary_convrot_engine_transcribes_and_survives_ram_offload():
    from modules.whisper.convrot.canary.engine import CanaryConvRot

    eng = CanaryConvRot.from_folder(MODEL_DIR)
    rng = np.random.default_rng(0)
    t = np.arange(16000 * 4) / 16000.0
    audio = (0.1 * np.sin(2 * np.pi * 220 * t) + 0.01 * rng.standard_normal(t.shape)).astype(np.float32)
    audios = torch.from_numpy(np.stack([audio, np.pad(audio[: 16000 * 3], (0, 16000))]))
    lens = torch.tensor([16000 * 4, 16000 * 3])
    prompt = [{"role": "user", "content": f"Transcribe the following: {eng.audio_locator_tag}"}]
    out1 = eng.generate([prompt, prompt], audios=audios, audio_lens=lens, max_new_tokens=32)
    assert out1.shape[0] == 2 and out1.dtype == torch.long
    eng.to("cpu")
    assert all(not L.qkv.weight.is_cuda for L in eng.llm.layers)
    eng.to("cuda")
    out2 = eng.generate([prompt, prompt], audios=audios, audio_lens=lens, max_new_tokens=32)
    assert torch.equal(out1.cpu(), out2.cpu())
    # beam search goes through Hugging Face generate on the same INT8 weights
    out3 = eng.generate([prompt], audios=audios[:1], audio_lens=lens[:1], max_new_tokens=16, num_beams=2)
    assert out3.shape[0] == 1
    assert isinstance(eng.tokenizer.ids_to_text(out1[0]), str)
    # The HF beam/sampling adapter shares GPU embeddings. Offloading must drop
    # it too, otherwise it retains the old CUDA storage after llm.to("cpu").
    eng.to("cpu")
    assert getattr(eng, "_hf_model", None) is None
    eng.to("cuda")
    out4 = eng.generate([prompt], audios=audios[:1], audio_lens=lens[:1], max_new_tokens=16, num_beams=2)
    assert torch.equal(out3.cpu(), out4.cpu())


@pytest.mark.skipif(not gpu_ok() or not is_canary_convrot_model_dir(MODEL_DIR),
                    reason="needs an Ampere+ GPU and the converted model in the Canary model folder")
def test_canary_graphs_survive_decoder_session_eviction():
    """Switching batch sizes must not free another live graph's cuBLAS workspace."""
    import gc
    from modules.whisper.convrot.canary.engine import CanaryConvRot

    eng = CanaryConvRot.from_folder(MODEL_DIR)
    audio = torch.sin(torch.arange(16000 * 3, dtype=torch.float32) * .03).mul_(.1)
    prompt = [{"role": "user", "content": f"Transcribe the following: {eng.audio_locator_tag}"}]
    outputs = []
    for batch_size in (1, 2, 3, 1):
        gc.collect()
        torch.cuda.empty_cache()
        ids = eng.generate([prompt] * batch_size,
                           audios=audio.repeat(batch_size, 1),
                           audio_lens=torch.full((batch_size,), audio.numel()),
                           max_new_tokens=8)
        outputs.append(ids[0].cpu())
    assert torch.equal(outputs[0], outputs[-1])


@pytest.mark.skipif(not gpu_ok() or not is_canary_convrot_model_dir(MODEL_DIR),
                    reason="needs an Ampere+ GPU and the converted model in the Canary model folder")
def test_canary_released_graphs_free_memory_and_are_captured_again():
    import gc
    from modules.whisper.convrot.canary.engine import CanaryConvRot

    eng = CanaryConvRot.from_folder(MODEL_DIR)
    audio = torch.sin(torch.arange(16000 * 20, dtype=torch.float32) * .03).mul_(.1)
    prompt = [{"role": "user", "content": f"Transcribe the following: {eng.audio_locator_tag}"}]

    def run():
        return eng.generate([prompt] * 4, audios=audio.repeat(4, 1), audio_lens=torch.full((4,), audio.numel()),
                            max_new_tokens=16).cpu()

    first = run()
    gc.collect()
    torch.cuda.empty_cache()
    with_graphs = torch.cuda.memory_reserved()
    eng.release_cuda_graphs()
    gc.collect()
    torch.cuda.empty_cache()
    assert not eng._sessions and not eng.encoder._graphs
    assert torch.cuda.memory_reserved() < with_graphs
    assert torch.equal(first, run())
