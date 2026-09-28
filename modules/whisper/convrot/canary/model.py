"""PyTorch runtime for Canary-Qwen-2.5B: FastConformer encoder + projection + Qwen3-1.7B.

The same runtime runs the original BF16 weights (float ``Linear``; FP32 for reference runs) and
INT8 ConvRot checkpoints: every attention/MLP/convolution-pointwise linear of the encoder and the
LLM holds INT8 weights in the regular-Hadamard-rotated basis (ComfyUI's native ``int8_tensorwise``
+ ``convrot`` layout: ``<p>.weight`` int8, ``<p>.weight_scale`` fp32 per output channel,
``<p>.comfy_quant``). Activations are rotated online; the encoder runs W8A8 on INT8 tensor cores
(dynamic per-(row, 256-group) activation scales), the LLM weight-only INT8 with fp16 activations.

Numerics: the residual streams are fp32, rotated GEMM inputs fp16, attention fp16 (flash-attn
for the LLM, SDPA with the relative-position bias for the encoder). Tensor names are the NeMo
checkpoint's with the LoRA adapters merged (``llm.model.layers.N...``), see ``canonical_tensors``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from modules.whisper.convrot import kernels as WK

from . import kernels as CK

try:
    from flash_attn import flash_attn_with_kvcache
except Exception:  # pragma: no cover - flash-attn is part of the app requirements
    flash_attn_with_kvcache = None

QUANT_FORMAT = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": WK.CONVROT_GROUP}
INF_VAL = 10000.0  # NeMo's masking value and positional-encoding base
# CUDA work of other app threads (Gradio, GPU memory reports) must not invalidate a capture.
CAPTURE_MODE = "thread_local"


@dataclass
class CanaryDims:
    n_mels: int = 128
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    sample_rate: int = 16000
    preemph: float = 0.97
    log_guard: float = 2 ** -24
    enc_layers: int = 32
    d_model: int = 1024
    enc_heads: int = 8
    enc_ff: int = 4096
    conv_kernel: int = 9
    sub_channels: int = 256
    llm_layers: int = 28
    hidden: int = 2048
    n_q_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    intermediate: int = 6144
    vocab: int = 151936
    rope_theta: float = 1000000.0
    rms_eps: float = 1e-6
    ln_eps: float = 1e-5
    bn_eps: float = 1e-5

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_configs(cls, salm_cfg: dict, llm_cfg: dict) -> "CanaryDims":
        enc = salm_cfg["perception"]["encoder"]
        pre = salm_cfg["perception"]["preprocessor"]
        sr = int(pre.get("sample_rate", 16000))
        return cls(
            n_mels=int(pre["features"]), n_fft=int(pre["n_fft"]), win_length=int(round(pre["window_size"] * sr)),
            hop_length=int(round(pre["window_stride"] * sr)), sample_rate=sr,
            enc_layers=int(enc["n_layers"]), d_model=int(enc["d_model"]), enc_heads=int(enc["n_heads"]),
            enc_ff=int(enc["d_model"]) * int(enc["ff_expansion_factor"]), conv_kernel=int(enc["conv_kernel_size"]),
            sub_channels=int(enc["subsampling_conv_channels"]), llm_layers=int(llm_cfg["num_hidden_layers"]),
            hidden=int(llm_cfg["hidden_size"]), n_q_heads=int(llm_cfg["num_attention_heads"]),
            n_kv_heads=int(llm_cfg["num_key_value_heads"]), head_dim=int(llm_cfg["head_dim"]),
            intermediate=int(llm_cfg["intermediate_size"]), vocab=int(llm_cfg["vocab_size"]),
            rope_theta=float(llm_cfg["rope_theta"]), rms_eps=float(llm_cfg["rms_norm_eps"]),
        )


# ----------------------------------------------------------------------------------------------
# Checkpoint handling
# ----------------------------------------------------------------------------------------------

def canonical_tensors(sd: dict, lora_scale: float) -> dict:
    """NeMo SALM state dict -> runtime names; LoRA merged in fp32 (W + scale * B @ A)."""
    out = {}
    lora = {}
    for key, value in sd.items():
        if key.endswith("num_batches_tracked"):
            continue
        if ".lora_A." in key or ".lora_B." in key:
            lora[key] = value
            continue
        name = key.replace("llm.base_model.model.model.", "llm.model.").replace(".base_layer.", ".")
        name = name.replace("llm.base_model.model.lm_head.", "llm.lm_head.")
        out[name] = value
    for key, a in lora.items():
        if ".lora_A." not in key:
            continue
        b = lora[key.replace(".lora_A.", ".lora_B.")]
        base = key.split(".lora_A.")[0].replace("llm.base_model.model.model.", "llm.model.") + ".weight"
        w = out[base].float() + lora_scale * (b.float() @ a.float())
        out[base] = w
    out.pop("llm.lm_head.weight", None)  # tied to embed_tokens
    return out


def encoder_linear_names(i: int) -> list[str]:
    p = f"perception.encoder.layers.{i}."
    return [p + n for n in ("feed_forward1.linear1", "feed_forward1.linear2", "self_attn.linear_q",
                            "self_attn.linear_k", "self_attn.linear_v", "self_attn.linear_out",
                            "conv.pointwise_conv1", "conv.pointwise_conv2", "feed_forward2.linear1",
                            "feed_forward2.linear2")]


def llm_linear_names(i: int) -> list[str]:
    p = f"llm.model.layers.{i}."
    return [p + n for n in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]


def all_linear_names(dims: CanaryDims) -> list[str]:
    names = []
    for i in range(dims.enc_layers):
        names += encoder_linear_names(i)
    for i in range(dims.llm_layers):
        names += llm_linear_names(i)
    return names


def _move(t, device):
    """Move a tensor; CPU copies are pinned so moving back to the GPU is fast."""
    if t is None:
        return None
    if device.type == "cpu":
        if not t.is_cuda:
            return t
        parked = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
        parked.copy_(t)
        return parked
    return t.to(device, non_blocking=True)


# ----------------------------------------------------------------------------------------------
# Linear layers
# ----------------------------------------------------------------------------------------------

class Linear:
    """Linear layer that is either INT8 ConvRot (``scale`` set) or float (``dtype``)."""

    __slots__ = ("weight", "scale", "bias", "quantized", "out_features", "in_features", "group_scales",
                 "weight_only", "dtype")

    def __init__(self, weight, bias=None, scale=None, dtype=torch.float16):
        self.quantized = scale is not None
        self.dtype = dtype
        if self.quantized:
            self.weight = weight.contiguous()
            self.scale = scale.reshape(-1).float().contiguous()
            self.bias = bias.float().contiguous() if bias is not None else None
        else:
            self.weight = weight.to(dtype).contiguous()
            self.scale = None
            self.bias = bias.to(dtype).contiguous() if bias is not None else None
        self.out_features, self.in_features = self.weight.shape
        self.group_scales = True
        self.weight_only = False

    def to(self, device):
        self.weight = _move(self.weight, device)
        self.bias = _move(self.bias, device)
        self.scale = _move(self.scale, device)

    def scaled(self, factor: float) -> "Linear":
        """In-place output scaling (a power of two keeps it exact)."""
        if self.quantized:
            self.scale = self.scale * factor
        else:
            self.weight = self.weight * factor
        if self.bias is not None:
            self.bias = self.bias * factor
        return self

    def __call__(self, x, pre=CK.PRE_NONE, nw=None, nb=None, eps=1e-5, residual=None, out=None):
        if self.quantized:
            if self.weight_only:
                xr = CK.rotate_pre(x, pre, nw, nb, eps, out_mode=CK.OUT_ROT)
                return WK.w8a16_linear(xr, self.weight, self.scale, bias=self.bias, residual=residual, out=out)
            xq, xs = CK.rotate_pre(x, pre, nw, nb, eps,
                                   out_mode=CK.OUT_Q_GROUP if self.group_scales else CK.OUT_Q_ROW)
            out_dtype = out.dtype if out is not None else torch.float16
            return WK.int8_linear_q(xq, xs, self.weight, self.scale, bias=self.bias, residual=residual, out=out,
                                    out_dtype=out_dtype)
        xf = CK.torch_pre(x, pre, nw, nb, eps).to(self.dtype)
        y = F.linear(xf.reshape(-1, xf.shape[-1]), self.weight, self.bias)
        if residual is not None:
            if out is not None and out.data_ptr() == residual.data_ptr():
                out.add_(y.to(out.dtype))
                return out
            y = residual + y.to(residual.dtype)
        if out is not None:
            out.copy_(y)
            return out
        return y


def _cat_linear(parts: list[Linear]) -> Linear:
    """Concatenate linears that share an input (rows of the weight)."""
    weight = torch.cat([p.weight for p in parts], dim=0)
    if all(p.bias is None for p in parts):
        bias = None
    else:
        ref = next(p.bias for p in parts if p.bias is not None)
        bias = torch.cat([p.bias if p.bias is not None else torch.zeros(p.out_features, dtype=ref.dtype,
                                                                          device=ref.device) for p in parts])
    if parts[0].quantized:
        return Linear(weight, bias, torch.cat([p.scale for p in parts]))
    lin = Linear(weight, bias, dtype=parts[0].dtype)
    return lin


def load_linear(t: dict, prefix: str, device, dtype) -> Linear:
    weight = t[prefix + ".weight"]
    bias = t.get(prefix + ".bias")
    scale = t.get(prefix + ".weight_scale")
    if weight.dim() == 3:  # pointwise Conv1d (out, in, 1)
        weight = weight.squeeze(-1)
    if scale is not None:
        if weight.dtype != torch.int8:
            raise ValueError(f"{prefix}: weight_scale present but weight dtype is {weight.dtype}")
        quant = t.get(prefix + ".comfy_quant")
        if quant is not None:
            meta = json.loads(bytes(quant.to(torch.uint8).tolist()).decode("utf-8"))
            if not meta.get("convrot") or int(meta.get("convrot_groupsize", 0)) != WK.CONVROT_GROUP:
                raise ValueError(f"{prefix}: unsupported quantization metadata {meta}")
        return Linear(weight.to(device), bias.to(device) if bias is not None else None, scale.to(device))
    return Linear(weight.to(device), bias.to(device) if bias is not None else None, dtype=dtype)


class Norm:
    __slots__ = ("weight", "bias", "eps")

    def __init__(self, weight, bias=None, eps=1e-5):
        self.weight = weight.float().contiguous()
        self.bias = bias.float().contiguous() if bias is not None else None
        self.eps = eps

    def to(self, device):
        self.weight = _move(self.weight, device)
        self.bias = _move(self.bias, device)


# ----------------------------------------------------------------------------------------------
# Encoder (preprocessor + FastConformer + projection)
# ----------------------------------------------------------------------------------------------

class EncoderLayer:
    __slots__ = ("norm_ff1", "ff1_l1", "ff1_l2", "norm_att", "qkv", "att_out", "pos_w", "bias_u", "bias_v",
                 "norm_conv", "pw1", "dw_w", "dw_b", "pw2", "norm_ff2", "ff2_l1", "ff2_l2", "norm_out")


class EncoderRuntime:
    def __init__(self, t: dict, dims: CanaryDims, device, dtype=torch.float16, w8a8: bool = True,
                 group_scales: bool = True):
        self.dims = dims
        self.device = torch.device(device)
        self.dtype = dtype  # float weights / attention / intermediate activations
        dev = self.device
        eps = dims.ln_eps
        self.fb = t["perception.preprocessor.featurizer.fb"].float().reshape(dims.n_mels, -1).to(dev)
        self.window = t["perception.preprocessor.featurizer.window"].float().to(dev)
        p = "perception.encoder.pre_encode."
        self.sub = [(t[p + f"conv.{i}.weight"].to(dev, dtype), t[p + f"conv.{i}.bias"].to(dev, dtype))
                    for i in (0, 2, 3, 5, 6)]
        self.sub_out = load_linear(t, p + "out", dev, dtype)
        self.layers = []
        for i in range(dims.enc_layers):
            p = f"perception.encoder.layers.{i}."
            L = EncoderLayer()
            L.norm_ff1 = Norm(t[p + "norm_feed_forward1.weight"].to(dev), t[p + "norm_feed_forward1.bias"].to(dev), eps)
            L.ff1_l1 = load_linear(t, p + "feed_forward1.linear1", dev, dtype)
            L.ff1_l2 = load_linear(t, p + "feed_forward1.linear2", dev, dtype).scaled(0.5)
            L.norm_att = Norm(t[p + "norm_self_att.weight"].to(dev), t[p + "norm_self_att.bias"].to(dev), eps)
            L.qkv = _cat_linear([load_linear(t, p + f"self_attn.linear_{n}", dev, dtype) for n in "qkv"])
            L.att_out = load_linear(t, p + "self_attn.linear_out", dev, dtype)
            L.pos_w = t[p + "self_attn.linear_pos.weight"].to(dev, dtype).contiguous()
            L.bias_u = t[p + "self_attn.pos_bias_u"].to(dev, dtype)
            L.bias_v = t[p + "self_attn.pos_bias_v"].to(dev, dtype)
            L.norm_conv = Norm(t[p + "norm_conv.weight"].to(dev), t[p + "norm_conv.bias"].to(dev), eps)
            L.pw1 = load_linear(t, p + "conv.pointwise_conv1", dev, dtype)
            # BatchNorm (eval) folded into the depthwise convolution
            g = t[p + "conv.batch_norm.weight"].float() / torch.sqrt(t[p + "conv.batch_norm.running_var"].float() + dims.bn_eps)
            dw_w = t[p + "conv.depthwise_conv.weight"].float() * g[:, None, None]
            dw_b = (t[p + "conv.depthwise_conv.bias"].float() - t[p + "conv.batch_norm.running_mean"].float()) * g \
                + t[p + "conv.batch_norm.bias"].float()
            L.dw_w = dw_w.to(dev, dtype).contiguous()
            L.dw_b = dw_b.to(dev, dtype).contiguous()
            L.pw2 = load_linear(t, p + "conv.pointwise_conv2", dev, dtype)
            L.norm_ff2 = Norm(t[p + "norm_feed_forward2.weight"].to(dev), t[p + "norm_feed_forward2.bias"].to(dev), eps)
            L.ff2_l1 = load_linear(t, p + "feed_forward2.linear1", dev, dtype)
            L.ff2_l2 = load_linear(t, p + "feed_forward2.linear2", dev, dtype).scaled(0.5)
            L.norm_out = Norm(t[p + "norm_out.weight"].to(dev), t[p + "norm_out.bias"].to(dev), eps)
            self.layers.append(L)
        self.proj = load_linear(t, "perception.proj", dev, dtype)
        self._graphs: dict = {}
        self.set_mode(w8a8, group_scales)

    def linears(self):
        for L in self.layers:
            for n in ("ff1_l1", "ff1_l2", "qkv", "att_out", "pw1", "pw2", "ff2_l1", "ff2_l2"):
                yield getattr(L, n)

    def set_mode(self, w8a8: bool, group_scales: bool):
        """Quantized linears: W8A8 (dynamic INT8 activations) or weight-only INT8."""
        self._graphs.clear()
        self.w8a8 = bool(w8a8)
        self.group_scales = bool(group_scales)
        for lin in self.linears():
            lin.weight_only = lin.quantized and not self.w8a8
            lin.group_scales = self.group_scales

    @torch.inference_mode()
    def to(self, device):
        device = torch.device(device)
        self._graphs.clear()  # graphs hold the old weight addresses
        self.fb = _move(self.fb, device)
        self.window = _move(self.window, device)
        self.sub = [(_move(w, device), _move(b, device)) for w, b in self.sub]
        self.sub_out.to(device)
        self.proj.to(device)
        for L in self.layers:
            for n in EncoderLayer.__slots__:
                v = getattr(L, n)
                if isinstance(v, torch.Tensor):
                    setattr(L, n, _move(v, device))
                else:
                    v.to(device)
        self.device = device

    # -- preprocessor (NeMo FilterbankFeatures, eval mode) -----------------------------------
    @torch.inference_mode()
    def features(self, audio: torch.Tensor, lengths: torch.Tensor):
        """audio [B, N] float32 (zero padded) -> log-mel [B, n_mels, T] fp32 normalized per feature."""
        d = self.dims
        x = audio.to(self.device, torch.float32)
        lengths = lengths.to(self.device)
        seq_len = torch.div(lengths, d.hop_length, rounding_mode="floor")
        timemask = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        x = torch.cat((x[:, :1], x[:, 1:] - d.preemph * x[:, :-1]), dim=1).masked_fill(~timemask, 0.0)
        spec = torch.stft(x, n_fft=d.n_fft, hop_length=d.hop_length, win_length=d.win_length, center=True,
                          window=self.window, return_complex=True, pad_mode="constant")
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1)).pow(2.0)
        mel = torch.matmul(self.fb, spec)
        mel = torch.log(mel + d.log_guard)
        # per-feature normalization over the valid frames (NeMo normalize_batch)
        b, _, t = mel.shape
        valid = torch.arange(t, device=mel.device)[None, :] < seq_len[:, None]
        n = valid.sum(1)
        ref = mel[:, :, 0]
        centered = torch.where(valid[:, None, :], mel - ref[:, :, None], 0.0)
        mean = ref + centered.sum(2) / n.clamp_min(1)[:, None]
        var = torch.where(valid[:, None, :], mel - mean[:, :, None], 0.0).pow(2).sum(2) / (n[:, None] - 1.0)
        std = torch.sqrt(var)
        std = std.masked_fill(std.isnan(), 0.0) + 1e-5
        mel = ((mel - mean[:, :, None]) / std[:, :, None]).masked_fill(~valid[:, None, :], 0.0)
        return mel, seq_len

    # -- subsampling (dw_striding x8 with NeMo's MaskedConvSequential masking) -------------------
    def _subsample(self, feats: torch.Tensor, lengths: torch.Tensor):
        x = feats.transpose(1, 2).unsqueeze(1).to(self.dtype)  # B, 1, T, F

        def mask(x, lens):
            m = torch.arange(x.shape[2], device=x.device)[None, :] < lens[:, None]
            return x * m[:, None, :, None].to(x.dtype)

        def out_len(lens):
            return torch.div(lens + 2 - 3, 2, rounding_mode="floor") + 1

        (w0, b0), (wd1, bd1), (wp1, bp1), (wd2, bd2), (wp2, bp2) = self.sub
        lens = lengths
        x = F.conv2d(mask(x, lens), w0, b0, stride=2, padding=1)
        lens = out_len(lens)
        x = torch.relu(mask(x, lens))
        x = F.conv2d(mask(x, lens), wd1, bd1, stride=2, padding=1, groups=x.shape[1])
        lens = out_len(lens)
        x = F.conv2d(mask(x, lens), wp1, bp1)
        x = torch.relu(mask(x, lens))
        x = F.conv2d(mask(x, lens), wd2, bd2, stride=2, padding=1, groups=x.shape[1])
        lens = out_len(lens)
        x = F.conv2d(mask(x, lens), wp2, bp2)
        x = mask(torch.relu(mask(x, lens)), lens)
        b, c, t, f = x.shape
        x = x.transpose(1, 2).reshape(b * t, c * f)
        return x, lens, b, t

    def _pos_emb(self, t: int) -> torch.Tensor:
        """Relative positional encodings for positions t-1 ... -(t-1) (NeMo RelPositionalEncoding)."""
        d = self.dims.d_model
        pos = torch.arange(t - 1, -t, -1, dtype=torch.float32, device=self.device)[:, None]
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32, device=self.device) * -(math.log(INF_VAL) / d))
        pe = torch.zeros(2 * t - 1, d, device=self.device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        b, h, qlen, pos_len = x.size()
        x = F.pad(x, pad=(1, 0))
        x = x.view(b, h, -1, qlen)
        return x[:, :, 1:].view(b, h, qlen, pos_len)

    @torch.inference_mode()
    def encode(self, feats: torch.Tensor, feat_lens: torch.Tensor, always_mask: bool = False):
        """log-mel [B, n_mels, T] -> projected audio embeddings [B, T/8, hidden] fp32 and lengths.

        ``always_mask`` applies the padding masks without checking for padding (no host sync; CUDA graphs).
        """
        d = self.dims
        x2, lens, b, t = self._subsample(feats.to(self.device), feat_lens.to(self.device))
        x = torch.empty((b * t, d.d_model), device=self.device, dtype=torch.float32)
        self.sub_out(x2, out=x)
        valid = torch.arange(t, device=self.device)[None, :] < lens[:, None]  # B, T
        padded = True if always_mask else bool((~valid).any())
        pad_mask = ~valid
        att_mask = ~(valid[:, :, None] & valid[:, None, :]) if padded else None  # B, T, T (True = masked)
        pe = self._pos_emb(t).to(self.dtype)
        h, dk = d.enc_heads, d.d_model // d.enc_heads
        scale = 1.0 / math.sqrt(dk)
        for li, L in enumerate(self.layers):
            # feed-forward 1 (half-step residual folded into linear2)
            hid = L.ff1_l1(x, CK.PRE_LAYERNORM, L.norm_ff1.weight, L.norm_ff1.bias, L.norm_ff1.eps)
            L.ff1_l2(hid, CK.PRE_SILU, residual=x, out=x)
            # relative-position multi-head self-attention
            qkv = L.qkv(x, CK.PRE_LAYERNORM, L.norm_att.weight, L.norm_att.bias, L.norm_att.eps)
            qkv = qkv.to(self.dtype).view(b, t, 3, h, dk)
            q = qkv[:, :, 0]
            k = qkv[:, :, 1].transpose(1, 2)
            v = qkv[:, :, 2].transpose(1, 2)
            q_u = (q + L.bias_u).transpose(1, 2)
            q_v = (q + L.bias_v).transpose(1, 2)
            p = F.linear(pe, L.pos_w).view(2 * t - 1, h, dk).transpose(0, 1)  # h, 2t-1, dk
            bd = torch.matmul(q_v, p.transpose(-2, -1).unsqueeze(0))
            bd = self._rel_shift(bd)[:, :, :, :t] * scale
            if att_mask is not None:
                bd = bd.masked_fill(att_mask[:, None], -INF_VAL)
            att = F.scaled_dot_product_attention(q_u, k, v, attn_mask=bd)
            if att_mask is not None:
                att = att.masked_fill(att_mask.all(-1)[:, None, :, None], 0.0)
            att = att.transpose(1, 2).reshape(b * t, d.d_model)
            L.att_out(att, residual=x, out=x)
            # convolution module
            g = L.pw1(x, CK.PRE_LAYERNORM, L.norm_conv.weight, L.norm_conv.bias, L.norm_conv.eps)
            g = F.glu(g.to(self.dtype).view(b, t, 2 * d.d_model), dim=-1).transpose(1, 2)
            if padded:
                g = g.masked_fill(pad_mask[:, None, :], 0.0)
            g = F.conv1d(g, L.dw_w, L.dw_b, padding=(d.conv_kernel - 1) // 2, groups=d.d_model)
            g = g.transpose(1, 2).reshape(b * t, d.d_model)
            L.pw2(g, CK.PRE_SILU, residual=x, out=x)
            # feed-forward 2
            hid = L.ff2_l1(x, CK.PRE_LAYERNORM, L.norm_ff2.weight, L.norm_ff2.bias, L.norm_ff2.eps)
            L.ff2_l2(hid, CK.PRE_SILU, residual=x, out=x)
            x = F.layer_norm(x, (d.d_model,), L.norm_out.weight, L.norm_out.bias, L.norm_out.eps)
        enc = self.proj(x).float().view(b, t, -1)
        return enc, lens

    def _audio_forward(self, audio, lengths, always_mask=True):
        feats, flens = self.features(audio, lengths)
        return self.encode(feats, flens, always_mask=always_mask)

    @torch.inference_mode()
    def run(self, audios: torch.Tensor, lengths: torch.Tensor, use_graph: bool = True):
        """Waveforms [B, N] -> (audio embeddings [B, T, hidden] fp32, valid lengths [B]).

        The waveform is zero padded to a whole number of ``BUCKET_SAMPLES`` (masked like NeMo's batch
        padding, so the valid frames match an unpadded run) and each (B, padded length) runs as a
        CUDA graph: every new shape would otherwise make cuDNN build new convolution plans, which cost
        more than the whole encoder, and the graph removes about a thousand kernel launches per chunk.
        """
        b, n = audios.shape
        nb = max(1, (n + self.BUCKET_SAMPLES - 1) // self.BUCKET_SAMPLES) * self.BUCKET_SAMPLES
        if not use_graph or self.device.type != "cuda":
            padded = torch.zeros((b, nb), device=self.device, dtype=torch.float32)
            padded[:, :n] = audios.to(self.device, torch.float32)
            with torch.backends.cudnn.flags(enabled=True, benchmark=False):
                return self._audio_forward(padded, lengths.to(self.device), always_mask=True)
        key = (b, nb)
        g = self._graphs.pop(key, None)
        if g is None:
            while len(self._graphs) >= self.MAX_GRAPHS:
                self._graphs.pop(next(iter(self._graphs)))
            g = _EncoderGraph(self, b, nb)
        self._graphs[key] = g  # most recently used last
        return g(audios, lengths)

    BUCKET_SAMPLES = 16000
    MAX_GRAPHS = 16  # live microphone previews grow a buffer up to 15 s, one bucket per second


class _EncoderGraph:
    """Encoder (preprocessor to projected embeddings) captured for one (batch, padded samples) shape."""

    def __init__(self, enc: EncoderRuntime, b: int, n: int):
        dev = enc.device
        self.audio = torch.zeros((b, n), device=dev, dtype=torch.float32)
        self.lengths = torch.full((b,), n, device=dev, dtype=torch.long)
        with torch.backends.cudnn.flags(enabled=True, benchmark=False):
            stream = torch.cuda.Stream(device=dev)
            stream.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(stream):
                for _ in range(2):  # Triton autotuning, cuDNN plans, cuFFT plans
                    enc._audio_forward(self.audio, self.lengths)
            torch.cuda.current_stream(dev).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, capture_error_mode=CAPTURE_MODE):
                self.out, self.out_lens = enc._audio_forward(self.audio, self.lengths)

    def __call__(self, audios: torch.Tensor, lengths: torch.Tensor):
        n = audios.shape[1]
        self.audio[:, :n].copy_(audios, non_blocking=True)
        self.audio[:, n:].zero_()
        self.lengths.copy_(lengths, non_blocking=True)
        self.graph.replay()
        return self.out, self.out_lens


# ----------------------------------------------------------------------------------------------
# LLM (Qwen3)
# ----------------------------------------------------------------------------------------------

class LLMLayer:
    __slots__ = ("in_norm", "qkv", "q_norm", "k_norm", "o", "post_norm", "gate_up", "down")


class LLMRuntime:
    def __init__(self, t: dict, dims: CanaryDims, device, dtype=torch.float16, weight_only: bool = True):
        self.dims = dims
        self.device = torch.device(device)
        self.dtype = dtype  # float weights; activations of float runs
        self.attn_dtype = torch.float16 if dtype != torch.float32 else torch.float32
        dev = self.device
        eps = dims.rms_eps
        self.embed = t["embed_tokens.weight"].to(dev, torch.bfloat16 if dtype != torch.float32 else torch.float32)
        self.layers = []
        for i in range(dims.llm_layers):
            p = f"llm.model.layers.{i}."
            L = LLMLayer()
            L.in_norm = Norm(t[p + "input_layernorm.weight"].to(dev), eps=eps)
            L.qkv = _cat_linear([load_linear(t, p + f"self_attn.{n}_proj", dev, dtype) for n in "qkv"])
            L.q_norm = t[p + "self_attn.q_norm.weight"].float().to(dev)
            L.k_norm = t[p + "self_attn.k_norm.weight"].float().to(dev)
            L.o = load_linear(t, p + "self_attn.o_proj", dev, dtype)
            L.post_norm = Norm(t[p + "post_attention_layernorm.weight"].to(dev), eps=eps)
            L.gate_up = _cat_linear([load_linear(t, p + f"mlp.{n}_proj", dev, dtype) for n in ("gate", "up")])
            L.down = load_linear(t, p + "mlp.down_proj", dev, dtype)
            self.layers.append(L)
        self.norm = Norm(t["llm.model.norm.weight"].to(dev), eps=eps)
        hd = dims.head_dim
        self.inv_freq = (1.0 / (dims.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.int64).float() / hd))).to(dev)
        self.lm_head = None  # optional INT8 ConvRot copy of the tied LM head (weight-only GEMM, fp32 logits)
        self.set_weight_only(weight_only)

    @torch.inference_mode()
    def set_lm_head_int8(self, flag: bool, rows_per_chunk: int = 8192):
        """Quantize the tied LM head (round-to-nearest in the rotated basis) for the logits GEMM; the
        token embedding itself stays BF16."""
        if not flag:
            self.lm_head = None
            return
        if self.dtype == torch.float32:
            raise ValueError("the INT8 LM head needs the fp16 runtime")
        qs, ss = [], []
        for i in range(0, self.embed.shape[0], rows_per_chunk):
            q, s = WK.quantize_weight_rowwise(WK.rotate_weight(self.embed[i:i + rows_per_chunk]))
            qs.append(q)
            ss.append(s)
        self.lm_head = Linear(torch.cat(qs), None, torch.cat(ss))
        self.lm_head.weight_only = True

    def linears(self):
        for L in self.layers:
            for n in ("qkv", "o", "gate_up", "down"):
                yield getattr(L, n)

    def set_weight_only(self, flag: bool):
        self.weight_only = bool(flag)
        for lin in self.linears():
            lin.weight_only = lin.quantized and self.weight_only
            lin.group_scales = True

    @property
    def fast(self) -> bool:
        """flash-attn + Triton path (INT8 or fp16/bf16 weights); fp32 runs use the reference path."""
        return self.dtype != torch.float32 and flash_attn_with_kvcache is not None

    @torch.inference_mode()
    def to(self, device):
        device = torch.device(device)
        self.embed = _move(self.embed, device)
        if self.lm_head is not None:
            self.lm_head.to(device)
        self.norm.to(device)
        self.inv_freq = _move(self.inv_freq, device)
        for L in self.layers:
            for n in LLMLayer.__slots__:
                v = getattr(L, n)
                if isinstance(v, torch.Tensor):
                    setattr(L, n, _move(v, device))
                else:
                    v.to(device)
        self.device = device

    def qk(self, L: LLMLayer, qkv: torch.Tensor, pos: torch.Tensor, out=None):
        d = self.dims
        if self.fast:
            return CK.qk_norm_rope(qkv, pos, self.inv_freq, L.q_norm, L.k_norm, d.n_q_heads, d.n_kv_heads,
                                   d.head_dim, d.rms_eps, out=out, out_dtype=self.attn_dtype)
        return CK.torch_qk_norm_rope(qkv, pos, self.inv_freq, L.q_norm, L.k_norm, d.n_q_heads, d.n_kv_heads,
                                     d.head_dim, d.rms_eps).to(self.attn_dtype)

    def run_layers(self, x: torch.Tensor, pos: torch.Tensor, attn):
        """All decoder blocks on the fp32 residual ``x`` [M, hidden] in place.

        ``attn(l, q [M, Hq, D], k [M, Hk, D], v [M, Hk, D])`` returns the attention output [M, Hq*D].
        """
        d = self.dims
        nq, nk, hd = d.n_q_heads, d.n_kv_heads, d.head_dim
        for li, L in enumerate(self.layers):
            qkv = L.qkv(x, CK.PRE_RMSNORM, L.in_norm.weight, None, L.in_norm.eps)
            qk = self.qk(L, qkv, pos)
            v = qkv[:, (nq + nk) * hd:].view(-1, nk, hd)
            if v.dtype != self.attn_dtype:
                v = v.to(self.attn_dtype)
            att = attn(li, qk[:, :nq], qk[:, nq:], v)
            L.o(att, residual=x, out=x)
            gu = L.gate_up(x, CK.PRE_RMSNORM, L.post_norm.weight, None, L.post_norm.eps)
            L.down(gu, CK.PRE_SWIGLU, residual=x, out=x)
        return x

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Final RMSNorm + tied LM head, fp32 logits."""
        if self.lm_head is not None:
            out = torch.empty((x.shape[0], self.dims.vocab), device=x.device, dtype=torch.float32)
            return self.lm_head(x, CK.PRE_RMSNORM, self.norm.weight, None, self.norm.eps, out=out)
        xn = CK.torch_pre(x, CK.PRE_RMSNORM, self.norm.weight, None, self.norm.eps)
        if self.embed.dtype == torch.float32:
            return F.linear(xn, self.embed)
        return torch.mm(xn.to(self.embed.dtype), self.embed.t(), out_dtype=torch.float32)

    def embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.embed).float()


def attend_reference(q, k_new, v_new, k_cache, v_cache, seqlens, rows):
    """fp32/any-dtype attention over a KV cache (reference path, not capturable).

    q [B, P, Hq, D], k_new/v_new [B, P, Hk, D] are appended at ``seqlens`` (python ints) of rows ``rows``.
    """
    b, p, hq, dd = q.shape
    hk = k_new.shape[2]
    out = torch.empty_like(q)
    for i in range(b):
        r = rows[i]
        s0 = int(seqlens[i])
        k_cache[r, s0:s0 + p] = k_new[i]
        v_cache[r, s0:s0 + p] = v_new[i]
        kk = k_cache[r, : s0 + p].transpose(0, 1).repeat_interleave(hq // hk, dim=0)
        vv = v_cache[r, : s0 + p].transpose(0, 1).repeat_interleave(hq // hk, dim=0)
        qq = q[i].transpose(0, 1)
        qpos = torch.arange(s0, s0 + p, device=q.device)[:, None]
        kpos = torch.arange(0, s0 + p, device=q.device)[None, :]
        mask = kpos <= qpos
        o = F.scaled_dot_product_attention(qq.float(), kk.float(), vv.float(), attn_mask=mask)
        out[i] = o.transpose(0, 1).to(out.dtype)
    return out


class DecodeSession:
    """KV cache for ``rows`` sequences of up to ``max_len`` tokens and a CUDA graph for one decode step."""

    def __init__(self, llm: LLMRuntime, rows: int, max_len: int, use_graph: bool = True):
        d = llm.dims
        self.llm = llm
        self.rows = rows
        self.max_len = max_len
        dev = llm.device
        self.kv = torch.zeros((2, d.llm_layers, rows, max_len, d.n_kv_heads, d.head_dim), device=dev,
                              dtype=llm.attn_dtype)
        self.ids = torch.zeros((rows,), device=dev, dtype=torch.long)
        self.seqlens = torch.zeros((rows,), device=dev, dtype=torch.int32)
        self.x = torch.zeros((rows, d.hidden), device=dev, dtype=torch.float32)
        self.logits_out = torch.empty((rows, d.vocab), device=dev, dtype=torch.float32)
        self.use_graph = use_graph and dev.type == "cuda" and llm.fast
        self.graph = None
        self._warm = 0
        self.host_seqlens = [0] * rows
        self._prefill_graphs: dict = {}

    PREFILL_BUCKET = 32
    MAX_PREFILL_GRAPHS = 12

    @torch.inference_mode()
    def prefill(self, x_emb: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        """Prompt embeddings [B=rows, P, hidden] (right padded) -> fp32 logits at each row's last position.

        With CUDA graphs the prompt is zero padded to a multiple of ``PREFILL_BUCKET`` positions and the
        prefill of each padded length runs as a graph. Causal attention keeps the padding out of the real
        positions, and the cache entries it writes past a row's length are overwritten by decoding.
        """
        b, p, hdim = x_emb.shape
        pb = ((p + self.PREFILL_BUCKET - 1) // self.PREFILL_BUCKET) * self.PREFILL_BUCKET
        if not self.use_graph or pb > self.max_len:
            return self._prefill_eager(x_emb, lengths)
        g = self._prefill_graphs.pop(pb, None)
        if g is None:
            while len(self._prefill_graphs) >= self.MAX_PREFILL_GRAPHS:
                self._prefill_graphs.pop(next(iter(self._prefill_graphs)))
            g = _PrefillGraph(self, pb)
        self._prefill_graphs[pb] = g  # most recently used last
        g.x[:, :p].copy_(x_emb)
        g.x[:, p:].zero_()
        g.last.copy_(torch.tensor([i * pb + n - 1 for i, n in enumerate(lengths)], dtype=torch.long))
        g.graph.replay()
        self.last_hidden = g.x
        self.seqlens.copy_(torch.tensor(lengths, dtype=torch.int32))
        self.host_seqlens = list(lengths)
        return g.logits

    def _prefill_layers(self, x: torch.Tensor, b: int, p: int):
        llm = self.llm
        d = llm.dims
        nq, nk, hd = d.n_q_heads, d.n_kv_heads, d.head_dim
        kc, vc = self.kv[0], self.kv[1]
        pos = torch.arange(p, device=x.device, dtype=torch.int32).repeat(b)
        zeros = torch.zeros((b,), device=x.device, dtype=torch.int32)

        def attn(l, q, k, v):
            o = flash_attn_with_kvcache(q.reshape(b, p, nq, hd), kc[l], vc[l], k=k.reshape(b, p, nk, hd),
                                        v=v.reshape(b, p, nk, hd), cache_seqlens=zeros, causal=True)
            return o.reshape(b * p, nq * hd)

        llm.run_layers(x, pos, attn)

    def _prefill_eager(self, x_emb: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        llm = self.llm
        d = llm.dims
        b, p, hdim = x_emb.shape
        x = x_emb.reshape(b * p, hdim).float().contiguous()
        if llm.fast:
            self._prefill_layers(x, b, p)
        else:
            pos = torch.arange(p, device=x.device, dtype=torch.int32).repeat(b)
            nq, nk, hd = d.n_q_heads, d.n_kv_heads, d.head_dim
            kc, vc = self.kv[0], self.kv[1]

            def attn(l, q, k, v):
                o = attend_reference(q.reshape(b, p, nq, hd), k.reshape(b, p, nk, hd), v.reshape(b, p, nk, hd),
                                     kc[l], vc[l], [0] * b, list(range(b)))
                return o.reshape(b * p, nq * hd)

            llm.run_layers(x, pos, attn)
        self.last_hidden = x.view(b, p, hdim)
        last = torch.tensor([i * p + n - 1 for i, n in enumerate(lengths)], device=x.device)
        self.seqlens.copy_(torch.tensor(lengths, dtype=torch.int32))
        self.host_seqlens = list(lengths)
        return llm.logits(x.index_select(0, last))

    def _step_impl(self):
        llm = self.llm
        d = llm.dims
        rows = self.rows
        nq, nk, hd = d.n_q_heads, d.n_kv_heads, d.head_dim
        kc, vc = self.kv[0], self.kv[1]
        self.x.copy_(llm.embed_ids(self.ids))
        if llm.fast:
            def attn(l, q, k, v):
                o = flash_attn_with_kvcache(q.reshape(rows, 1, nq, hd), kc[l], vc[l], k=k.reshape(rows, 1, nk, hd),
                                            v=v.reshape(rows, 1, nk, hd), cache_seqlens=self.seqlens, causal=True,
                                            num_splits=1)
                return o.reshape(rows, nq * hd)
        else:
            seq = list(self.host_seqlens)

            def attn(l, q, k, v):
                o = attend_reference(q.reshape(rows, 1, nq, hd), k.reshape(rows, 1, nk, hd),
                                     v.reshape(rows, 1, nk, hd), kc[l], vc[l], seq, list(range(rows)))
                return o.reshape(rows, nq * hd)

        llm.run_layers(self.x, self.seqlens, attn)
        self.logits_out.copy_(llm.logits(self.x))
        self.seqlens.add_(1)

    @torch.inference_mode()
    def step(self) -> torch.Tensor:
        """One decode step for ``self.ids`` at positions ``self.seqlens``; returns fp32 logits."""
        if self.graph is not None:
            self.graph.replay()
        else:
            self._step_impl()
            if self.use_graph:
                self._warm += 1
                if self._warm >= 2:
                    self._capture()
        self.host_seqlens = [s + 1 for s in self.host_seqlens]
        return self.logits_out

    def _capture(self):
        # Capture records the kernels without executing them, so the decoding state is untouched.
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode=CAPTURE_MODE):
            self._step_impl()
        torch.cuda.synchronize()
        self.graph = graph


class _PrefillGraph:
    """Prefill of ``rows`` prompts padded to ``p`` positions, captured as one CUDA graph."""

    def __init__(self, sess: DecodeSession, p: int):
        llm = sess.llm
        dev = llm.device
        b = sess.rows
        self.x = torch.zeros((b, p, llm.dims.hidden), device=dev, dtype=torch.float32)
        self.last = torch.arange(b, device=dev, dtype=torch.long) * p + p - 1
        flat = self.x.view(b * p, -1)
        stream = torch.cuda.Stream(device=dev)
        stream.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(stream):
            for _ in range(2):  # Triton autotuning for these row counts
                sess._prefill_layers(flat, b, p)
                llm.logits(flat.index_select(0, self.last))
        torch.cuda.current_stream(dev).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, capture_error_mode=CAPTURE_MODE):
            sess._prefill_layers(flat, b, p)
            self.logits = llm.logits(flat.index_select(0, self.last))
