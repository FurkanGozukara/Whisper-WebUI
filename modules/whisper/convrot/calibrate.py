"""Calibration statistics and GPTQ rounding for INT8 ConvRot Whisper conversion.

The FP16 model runs on calibration audio; for every linear we accumulate the
input Gram matrix X^T X (inputs after LayerNorm/GELU, i.e. exactly what the
INT8 kernel rotates and quantizes). Encoder statistics come from the encoder
forward, decoder statistics from teacher-forced decoding of the FP16 model's
own greedy transcripts. GPTQ then rounds each rotated weight row-wise with the
Hessian rotated into the ConvRot basis (Hb X^T X Hb).
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from . import kernels as K
from .model import Linear, WhisperRuntime

ENC_NAMES = ("qkv", "out", "fc1", "fc2")
DEC_NAMES = ("qkv", "out", "cross_q", "cross_kv", "cross_out", "fc1", "fc2")


class _GramLinear:
    """Wraps an FP16 Linear and accumulates X^T X of its (pre-processed) input."""

    def __init__(self, base: Linear, store: dict, key: str):
        self.base = base
        self.store = store
        self.key = key
        self.quantized = False

    def __call__(self, x, ln=None, pre_gelu=False, gelu=False, residual=None, out=None):
        xf = x.float()
        if ln is not None:
            xf = F.layer_norm(xf, (xf.shape[-1],), ln.weight, ln.bias, ln.eps)
        elif pre_gelu:
            xf = F.gelu(xf)
        x2 = xf.reshape(-1, xf.shape[-1])
        entry = self.store.get(self.key)
        if entry is None:
            entry = self.store[self.key] = [torch.zeros((x2.shape[1], x2.shape[1]), device=x2.device,
                                                        dtype=torch.float32), 0]
        entry[0].addmm_(x2.t(), x2)
        entry[1] += x2.shape[0]
        return self.base(x, ln=ln, pre_gelu=pre_gelu, gelu=gelu, residual=residual, out=out)


def _wrap(rt: WhisperRuntime, store: dict):
    saved = []
    for side, layers, names in (("encoder", rt.enc_layers, ENC_NAMES), ("decoder", rt.dec_layers, DEC_NAMES)):
        for i, layer in enumerate(layers):
            for n in names:
                base = getattr(layer, n)
                saved.append((layer, n, base))
                setattr(layer, n, _GramLinear(base, store, f"{side}.{i}.{n}"))
    return saved


def _unwrap(saved):
    for layer, n, base in saved:
        setattr(layer, n, base)


@torch.inference_mode()
def collect_gram(engine, mel_batches: Iterable[torch.Tensor], prompt: list[int],
                 progress: Callable[[str], None] | None = None, previous_text: bool = False) -> dict:
    """Accumulate X^T X for every linear of ``engine.runtime`` (an FP16 ConvRotWhisper).

    With ``previous_text`` each window's decoder sequence is prefixed like
    faster-whisper's condition_on_previous_text prompt
    (``<|startofprev|>`` + the previous window's transcript), so the decoder
    statistics also cover the long-context positions used during transcription.
    """
    rt = engine.runtime
    store: dict = {}
    sot_prev = engine._vocab_index.get("<|startofprev|>")
    last_tokens: list[int] = []
    for bi, mel in enumerate(mel_batches):
        mel = mel.to(rt.device)
        # 1) FP16 greedy transcript (no statistics) to get realistic decoder inputs.
        results = engine.generate(mel, [prompt] * mel.shape[0], beam_size=1, max_length=448, suppress_tokens=[-1])
        seqs = []
        for r in results:
            text = [t for t in r.sequences_ids[0]]
            head = []
            if previous_text and sot_prev is not None and last_tokens:
                head = [sot_prev] + last_tokens[-223:]
            seqs.append(head + prompt + text + [engine.eot_id])
            last_tokens = [t for t in text if t < engine.eot_id]
        saved = _wrap(rt, store)
        try:
            enc = rt.encode(mel)
            nb = enc.shape[0]
            cross = rt.compute_cross_kv(enc, out=torch.empty((rt.dims.n_text_layer, nb, enc.shape[1], 2 * rt.d_model),
                                                             device=rt.device, dtype=rt.dtype))
            # teacher forcing: one sequence at a time so padding never enters the statistics
            for b, seq in enumerate(seqs):
                tok = torch.tensor([seq], device=rt.device, dtype=torch.long)
                rt.forward_full(tok, cross[:, b:b + 1].contiguous())
        finally:
            _unwrap(saved)
        if progress:
            progress(f"calibration batch {bi + 1}: {sum(len(s) for s in seqs)} decoder tokens")
    return store


def hessian_rotated(gram: torch.Tensor, group: int = K.CONVROT_GROUP) -> torch.Tensor:
    """Hb^T G Hb for the block-diagonal regular Hadamard (fp64)."""
    k = gram.shape[0]
    h = K.hadamard_matrix(group, device=gram.device, dtype=torch.float64)
    ng = k // group
    g4 = gram.reshape(ng, group, ng, group)
    g4 = torch.einsum("ab,iajc,cd->ibjd", h, g4, h)
    return g4.reshape(k, k)


def gptq_quantize(w_rot: torch.Tensor, hessian: torch.Tensor, block: int = 128, damp: float = 0.01,
                  act_order: bool = True):
    """GPTQ with fixed symmetric per-row INT8 scales (absmax of the rotated weight).

    Returns (int8 weight [N, K], fp32 scale [N]).
    """
    W = w_rot.to(torch.float64).clone()
    n, k = W.shape
    Hm = hessian.to(torch.float64).clone()
    scale = (W.abs().amax(dim=1).clamp_min(1e-12) / 127.0)
    dead = torch.diag(Hm) == 0
    Hm[dead, dead] = 1
    W[:, dead] = 0
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(Hm), descending=True)
        W = W[:, perm]
        Hm = Hm[perm][:, perm]
    Hm += damp * torch.mean(torch.diag(Hm)) * torch.eye(k, dtype=Hm.dtype, device=Hm.device)
    L = torch.linalg.cholesky(Hm)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    s = scale[:, None]
    for i1 in range(0, k, block):
        i2 = min(i1 + block, k)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = torch.clamp(torch.round(w / scale), -127, 127) * scale
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    if perm is not None:
        inv = torch.argsort(perm)
        Q = Q[:, inv]
    q_int = torch.clamp(torch.round(Q / s), -127, 127).to(torch.int8)
    return q_int, scale.to(torch.float32)


def output_error(w_rot: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, hessian: torch.Tensor) -> float:
    """Relative output error tr(dW H dW^T) / tr(W H W^T) under the calibration Hessian."""
    Wd = q.to(torch.float64) * scale.to(torch.float64)[:, None]
    dW = Wd - w_rot.to(torch.float64)
    num = torch.einsum("nk,kj,nj->", dW, hessian, dW)
    den = torch.einsum("nk,kj,nj->", w_rot.to(torch.float64), hessian, w_rot.to(torch.float64))
    return math.sqrt(max(num.item(), 0.0) / max(den.item(), 1e-30))
