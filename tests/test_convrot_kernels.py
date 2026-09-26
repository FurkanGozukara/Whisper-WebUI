"""GPU tests for the INT8 ConvRot Whisper kernels (skipped without CUDA)."""

import os

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
LARGE_V3_DIR = os.path.join(TESTS_DIR, "..", "models", "Whisper", "faster-whisper", "large-v3-int8-convrot")


def _ref_rotate_quantize(x, pre, lnw, lnb, group_scales):
    from modules.whisper.convrot import kernels as K

    h = K.hadamard_matrix(256, device=x.device, dtype=torch.float32)
    xf = x.float()
    if pre == K.PRE_LAYERNORM:
        xf = torch.nn.functional.layer_norm(xf, (xf.shape[-1],), lnw, lnb, 1e-5)
    elif pre == K.PRE_GELU:
        xf = torch.nn.functional.gelu(xf)
    m, k = xf.shape
    xr = (xf.reshape(m, k // 256, 256) @ h)
    if group_scales:
        s = (xr.abs().amax(-1) / 127).clamp_min(1e-12)
        q = torch.round(xr / s[..., None]).clamp(-127, 127).reshape(m, k)
    else:
        xr = xr.reshape(m, k)
        s = (xr.abs().amax(-1) / 127).clamp_min(1e-12)
        q = torch.round(xr / s[:, None]).clamp(-127, 127)
    return q, s


def test_hadamard_is_symmetric_orthogonal():
    from modules.whisper.convrot import kernels as K

    h = K.hadamard_matrix(256)
    assert torch.allclose(h, h.T)
    assert torch.allclose(h @ h, torch.eye(256, dtype=h.dtype), atol=1e-12)


@cuda
@pytest.mark.parametrize("k", [1280, 5120])
@pytest.mark.parametrize("pre", [0, 1, 2])
@pytest.mark.parametrize("group_scales", [False, True])
def test_rotate_quantize_matches_reference(k, pre, group_scales):
    from modules.whisper.convrot import kernels as K

    torch.manual_seed(0)
    x = torch.randn(9, k, device="cuda", dtype=torch.float16) * 2
    x[:, 3] *= 30
    lnw = torch.rand(k, device="cuda") + 0.5
    lnb = torch.randn(k, device="cuda") * 0.1
    q, s = K.rotate_quantize(x, pre=pre, ln_weight=lnw, ln_bias=lnb, group_scales=group_scales)
    qr, sr = _ref_rotate_quantize(x, pre, lnw, lnb, group_scales)
    assert torch.allclose(s, sr, rtol=1e-5)
    assert (q.float() - qr).abs().max().item() <= 1  # fp32 LayerNorm summation order may flip a rounding
    assert (q.float() != qr).float().mean().item() < 1e-3


@cuda
@pytest.mark.parametrize("m,k,n", [(5, 1280, 3840), (5, 5120, 1280), (300, 1280, 1280), (1500, 5120, 1280)])
def test_int8_linear_close_to_float(m, k, n):
    from modules.whisper.convrot import kernels as K

    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    x[:, 11] *= 25
    w = torch.randn(n, k, device="cuda") * 0.03
    b = (torch.randn(n, device="cuda") * 0.1).half()
    wq, ws = K.quantize_weight_rowwise(K.rotate_weight(w))
    ref = x.float() @ w.t() + b.float()
    for gs in (False, True):
        y = K.convrot_linear(x, wq.cuda(), ws.cuda(), bias=b, group_scales=gs)
        rel = ((y.float() - ref).norm() / ref.norm()).item()
        assert rel < 0.02, (gs, rel)


@cuda
@pytest.mark.skipif(not os.path.isfile(os.path.join(LARGE_V3_DIR, "model.safetensors")),
                    reason="needs models/Whisper/faster-whisper/large-v3-int8-convrot")
def test_engine_results_survive_parking_in_ram_and_reloading_from_disk():
    import numpy as np
    from faster_whisper.audio import decode_audio
    from faster_whisper.feature_extractor import FeatureExtractor
    from modules.whisper.convrot.engine import ConvRotWhisper

    jfk = os.path.join(TESTS_DIR, "jfk.wav")  # git-ignored sample, downloaded on demand like in the other tests
    if not os.path.isfile(jfk):
        import requests
        from test_config import TEST_FILE_DOWNLOAD_URL

        with open(jfk, "wb") as file:
            file.write(requests.get(TEST_FILE_DOWNLOAD_URL, timeout=60).content)
    audio = decode_audio(jfk, sampling_rate=16000)
    audio = np.pad(audio, (0, 16000 * 30 - len(audio)))
    features = FeatureExtractor(feature_size=128)(audio)[None, :, :3000].astype(np.float32)
    model = ConvRotWhisper(LARGE_V3_DIR, device="cuda", device_index=0)
    prompt = [[50258, 50259, 50360, 50364]]  # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>

    first = model.generate(features, prompt, beam_size=5, return_scores=True)[0]
    allocated = torch.cuda.memory_allocated()
    model.unload_model(to_cpu=True)
    assert not model.model_is_loaded
    assert torch.cuda.memory_allocated() < allocated - 1_000_000_000  # the 1.6 GB of weights left the GPU
    parked = model.generate(features, prompt, beam_size=5, return_scores=True)[0]  # moved back on demand
    model.unload_model()  # weights dropped; the next call reloads them from disk
    reloaded = model.generate(features, prompt, beam_size=5, return_scores=True)[0]

    assert len(first.sequences_ids[0]) > 10
    assert first.sequences_ids == parked.sequences_ids == reloaded.sequences_ids
    assert abs(first.scores[0] - parked.scores[0]) < 1e-4
    assert abs(first.scores[0] - reloaded.scores[0]) < 1e-4
