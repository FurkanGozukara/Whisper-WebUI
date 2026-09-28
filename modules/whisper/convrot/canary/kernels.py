"""Triton kernels for the INT8 ConvRot Canary-Qwen runtime.

The GEMMs (W8A8 ``int8_linear_q`` and weight-only ``w8a16_linear``) and the regular-Hadamard
rotation come from the Whisper ConvRot kernels. This module adds the activation pre-ops the
FastConformer encoder and the Qwen3 LLM need in front of a rotated INT8 linear:

  * ``PRE_LAYERNORM`` (encoder), ``PRE_RMSNORM`` (LLM), ``PRE_SILU`` (encoder feed-forward and
    convolution module), ``PRE_SWIGLU`` (LLM MLP: ``silu(gate) * up`` of a ``[gate | up]`` row);
  * output as INT8 with per-row or per-(row, 256-group) scales (W8A8) or as the rotated fp16
    activation (weight-only INT8);

and ``qk_norm_rope``: Qwen3's per-head RMSNorm of queries and keys followed by the rotary
embedding, in one pass.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from modules.whisper.convrot.kernels import CONVROT_GROUP, _rotate_groups256

PRE_NONE = 0
PRE_LAYERNORM = 1
PRE_RMSNORM = 3
PRE_SILU = 4
PRE_SWIGLU = 5

OUT_Q_ROW = 0  # INT8, one scale per row
OUT_Q_GROUP = 1  # INT8, one scale per (row, 256-group)
OUT_ROT = 2  # rotated activation in the output dtype (weight-only INT8 path)

GROUP_KERNEL_MAX_M = 64


@triton.jit
def _silu(x):
    return x / (1.0 + tl.exp(-x))


@triton.jit
def _rq_row_kernel(
    X, stride_xm, Q, S, NW, NB, eps,
    K: tl.constexpr, NG_P2: tl.constexpr, PRE: tl.constexpr, OUT: tl.constexpr,
):
    """One program per row: pre-op on the whole row, rotation of every 256-group, output."""
    row = tl.program_id(0).to(tl.int64)
    g = tl.arange(0, NG_P2)
    c = tl.arange(0, 256)
    col = g[:, None] * 256 + c[None, :]
    mask = col < K
    x_row = X + row * stride_xm
    if PRE == 5:
        a = tl.load(x_row + col, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(x_row + K + col, mask=mask, other=0.0).to(tl.float32)
        x = _silu(a) * b
    else:
        x = tl.load(x_row + col, mask=mask, other=0.0).to(tl.float32)
    if PRE == 1:
        mean = tl.sum(tl.sum(x, axis=1), axis=0) / K
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(tl.sum(xc * xc, axis=1), axis=0) / K
        w = tl.load(NW + col, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(NB + col, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, xc * tl.math.rsqrt(var + eps) * w + b, 0.0)
    elif PRE == 3:
        ms = tl.sum(tl.sum(x * x, axis=1), axis=0) / K
        w = tl.load(NW + col, mask=mask, other=0.0).to(tl.float32)
        x = x * tl.math.rsqrt(ms + eps) * w
    elif PRE == 4:
        x = _silu(x)
    flat = tl.reshape(x, (NG_P2 * 256,))
    flat = _rotate_groups256(flat, NG_P2 * 256) * 0.0625
    x = tl.reshape(flat, (NG_P2, 256))
    if OUT == 2:
        tl.store(Q + row * K + col, x.to(Q.dtype.element_ty), mask=mask)
    elif OUT == 1:
        NG: tl.constexpr = K // 256
        gscale = tl.maximum(tl.max(tl.abs(x), axis=1) / 127.0, 1e-12)
        qv = tl.extra.cuda.libdevice.rint(x / gscale[:, None])
        qv = tl.minimum(tl.maximum(qv, -127.0), 127.0)
        tl.store(Q + row * K + col, qv.to(tl.int8), mask=mask)
        tl.store(S + row * NG + g, gscale, mask=g < NG)
    else:
        amax = tl.max(tl.max(tl.abs(x), axis=1), axis=0)
        scale = tl.maximum(amax / 127.0, 1e-12)
        qv = tl.extra.cuda.libdevice.rint(x / scale)
        qv = tl.minimum(tl.maximum(qv, -127.0), 127.0)
        tl.store(Q + row * K + col, qv.to(tl.int8), mask=mask)
        tl.store(S + row, scale)


@triton.jit
def _rq_group_kernel(
    X, stride_xm, Q, S, NW, NB, eps,
    K: tl.constexpr, NG_P2: tl.constexpr, PRE: tl.constexpr, OUT: tl.constexpr,
):
    """One program per (row, 256-group); norms recompute the row statistics (small row counts)."""
    row = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    NG: tl.constexpr = K // 256
    c = tl.arange(0, 256)
    x_row = X + row * stride_xm
    if PRE == 5:
        a = tl.load(x_row + g * 256 + c).to(tl.float32)
        b = tl.load(x_row + K + g * 256 + c).to(tl.float32)
        x = _silu(a) * b
    else:
        x = tl.load(x_row + g * 256 + c).to(tl.float32)
    if PRE == 1:
        gg = tl.arange(0, NG_P2)
        col = gg[:, None] * 256 + c[None, :]
        mask = col < K
        xa = tl.load(x_row + col, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(tl.sum(xa, axis=1), axis=0) / K
        xc = tl.where(mask, xa - mean, 0.0)
        var = tl.sum(tl.sum(xc * xc, axis=1), axis=0) / K
        w = tl.load(NW + g * 256 + c).to(tl.float32)
        b = tl.load(NB + g * 256 + c).to(tl.float32)
        x = (x - mean) * tl.math.rsqrt(var + eps) * w + b
    elif PRE == 3:
        gg = tl.arange(0, NG_P2)
        col = gg[:, None] * 256 + c[None, :]
        mask = col < K
        xa = tl.load(x_row + col, mask=mask, other=0.0).to(tl.float32)
        ms = tl.sum(tl.sum(xa * xa, axis=1), axis=0) / K
        w = tl.load(NW + g * 256 + c).to(tl.float32)
        x = x * tl.math.rsqrt(ms + eps) * w
    elif PRE == 4:
        x = _silu(x)
    x = _rotate_groups256(x, 256) * 0.0625
    if OUT == 2:
        tl.store(Q + row * K + g * 256 + c, x.to(Q.dtype.element_ty))
    else:
        scale = tl.maximum(tl.max(tl.abs(x), axis=0) / 127.0, 1e-12)
        qv = tl.extra.cuda.libdevice.rint(x / scale)
        qv = tl.minimum(tl.maximum(qv, -127.0), 127.0)
        tl.store(Q + row * K + g * 256 + c, qv.to(tl.int8))
        tl.store(S + row * NG + g, scale)


def _next_pow2(v: int) -> int:
    return 1 << (int(v) - 1).bit_length()


def rotate_pre(x: torch.Tensor, pre: int = PRE_NONE, norm_w=None, norm_b=None, eps: float = 1e-5,
               out_mode: int = OUT_ROT, out_dtype: torch.dtype = torch.float16):
    """Pre-op + per-256-group rotation of each row of ``x``.

    ``x`` is ``[..., K]`` (``[..., 2K]`` for ``PRE_SWIGLU``: gate then up). Returns the rotated
    activation ``[M, K]`` (``OUT_ROT``) or ``(int8 [M, K], fp32 scales)`` with scales ``[M]``
    (``OUT_Q_ROW``) or ``[M, K/256]`` (``OUT_Q_GROUP``).
    """
    width = x.shape[-1]
    k = width // 2 if pre == PRE_SWIGLU else width
    x2 = x.reshape(-1, width)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    if k % CONVROT_GROUP:
        raise ValueError(f"ConvRot needs K divisible by {CONVROT_GROUP}, got {k}")
    ng = k // CONVROT_GROUP
    if out_mode == OUT_ROT:
        q = torch.empty((m, k), device=x.device, dtype=out_dtype)
        s = q
    else:
        q = torch.empty((m, k), device=x.device, dtype=torch.int8)
        s = torch.empty((m, ng) if out_mode == OUT_Q_GROUP else (m,), device=x.device, dtype=torch.float32)
    if m == 0:
        return q if out_mode == OUT_ROT else (q, s)
    nw = norm_w if norm_w is not None else x2
    nb = norm_b if norm_b is not None else nw
    ng_p2 = _next_pow2(ng)
    if m <= GROUP_KERNEL_MAX_M and out_mode != OUT_Q_ROW:
        _rq_group_kernel[(m, ng)](x2, x2.stride(0), q, s, nw, nb, eps, K=k, NG_P2=ng_p2, PRE=pre, OUT=out_mode,
                                  num_warps=4)
    else:
        num_warps = 4 if ng_p2 <= 8 else 8
        _rq_row_kernel[(m,)](x2, x2.stride(0), q, s, nw, nb, eps, K=k, NG_P2=ng_p2, PRE=pre, OUT=out_mode,
                             num_warps=num_warps)
    return q if out_mode == OUT_ROT else (q, s)


def torch_pre(x: torch.Tensor, pre: int, norm_w=None, norm_b=None, eps: float = 1e-5) -> torch.Tensor:
    """Reference pre-op in fp32 (float runtime and tests)."""
    xf = x.float()
    if pre == PRE_SWIGLU:
        k = xf.shape[-1] // 2
        return F_silu(xf[..., :k]) * xf[..., k:]
    if pre == PRE_LAYERNORM:
        return torch.nn.functional.layer_norm(xf, (xf.shape[-1],), norm_w.float(), norm_b.float(), eps)
    if pre == PRE_RMSNORM:
        return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * norm_w.float()
    if pre == PRE_SILU:
        return F_silu(xf)
    return xf


def F_silu(x):
    return torch.nn.functional.silu(x)


@triton.jit
def _qk_norm_rope_kernel(
    QKV, stride_qkv, POS, INV_FREQ, QW, KW, OUT, stride_out, eps,
    NQ: tl.constexpr, D: tl.constexpr,
):
    """Program (row, head) over the q and k heads of a fused qkv row: RMSNorm over the head
    dimension with the q_norm/k_norm weight, then the rotary embedding (rotate-half layout)."""
    row = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    i = tl.arange(0, HALF)
    pos = tl.load(POS + row).to(tl.float32)
    ang = pos * tl.load(INV_FREQ + i)
    cs = tl.cos(ang)
    sn = tl.sin(ang)
    base = QKV + row * stride_qkv + h * D
    x1 = tl.load(base + i).to(tl.float32)
    x2 = tl.load(base + HALF + i).to(tl.float32)
    ms = (tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / D
    r = tl.math.rsqrt(ms + eps)
    is_q = h < NQ
    w1 = tl.where(is_q, tl.load(QW + i).to(tl.float32), tl.load(KW + i).to(tl.float32))
    w2 = tl.where(is_q, tl.load(QW + HALF + i).to(tl.float32), tl.load(KW + HALF + i).to(tl.float32))
    y1 = x1 * r * w1
    y2 = x2 * r * w2
    o = OUT + row * stride_out + h * D
    tl.store(o + i, (y1 * cs - y2 * sn).to(OUT.dtype.element_ty))
    tl.store(o + HALF + i, (y2 * cs + y1 * sn).to(OUT.dtype.element_ty))


def qk_norm_rope(qkv: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor, q_w: torch.Tensor,
                 k_w: torch.Tensor, n_q: int, n_kv: int, head_dim: int, eps: float,
                 out: torch.Tensor | None = None, out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """``qkv`` [M, (n_q + 2 n_kv) * D] -> normalized, rotated q and k heads [M, n_q + n_kv, D]."""
    m = qkv.shape[0]
    if out is None:
        out = torch.empty((m, n_q + n_kv, head_dim), device=qkv.device, dtype=out_dtype)
    if m == 0:
        return out
    _qk_norm_rope_kernel[(m, n_q + n_kv)](qkv, qkv.stride(0), pos, inv_freq, q_w, k_w, out, out.stride(0), eps,
                                          NQ=n_q, D=head_dim, num_warps=1)
    return out


def torch_qk_norm_rope(qkv: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor, q_w, k_w, n_q: int,
                       n_kv: int, head_dim: int, eps: float) -> torch.Tensor:
    """fp32 reference of ``qk_norm_rope`` (Hugging Face Qwen3 semantics)."""
    m = qkv.shape[0]
    x = qkv[:, : (n_q + n_kv) * head_dim].float().view(m, n_q + n_kv, head_dim)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    w = torch.cat([q_w.float().expand(n_q, head_dim), k_w.float().expand(n_kv, head_dim)])
    x = x * w
    ang = pos.float()[:, None] * inv_freq.float()[None, :]
    emb = torch.cat([ang, ang], dim=-1)[:, None, :]
    cos, sin = emb.cos(), emb.sin()
    x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin
