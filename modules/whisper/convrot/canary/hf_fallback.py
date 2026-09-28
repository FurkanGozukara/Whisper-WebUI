"""Hugging Face ``generate`` on the ConvRot runtime's weights.

Beam search, sampling, n-gram blocking and custom generation configs run exactly as in NeMo's
``SALM.generate``: a ``Qwen3ForCausalLM`` receives the left-padded prompt embeddings and an
attention mask. Its linear layers are thin modules over the runtime's fused INT8 (or float)
linears, so no weights are duplicated; only the small norm weights are copied to BF16.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from modules.whisper.convrot import kernels as WK

from . import kernels as CK


class ConvRotLinearModule(nn.Module):
    """``nn.Linear``-compatible view of rows ``[start, stop)`` of a runtime ``Linear``."""

    def __init__(self, lin, start: int = 0, stop: int | None = None):
        super().__init__()
        self.lin = lin
        self.start = start
        self.stop = lin.out_features if stop is None else stop
        self.in_features = lin.in_features
        self.out_features = self.stop - self.start

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lin = self.lin
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        w = lin.weight[self.start:self.stop]
        b = lin.bias[self.start:self.stop] if lin.bias is not None else None
        if lin.quantized:
            xr = CK.rotate_pre(x2, CK.PRE_NONE, out_mode=CK.OUT_ROT)
            y = WK.w8a16_linear(xr, w, lin.scale[self.start:self.stop], bias=b)
        else:
            y = F.linear(x2.to(lin.dtype), w, b)
        return y.to(x.dtype).reshape(*shape[:-1], self.out_features)


def build_hf_model(engine):
    """Qwen3ForCausalLM (BF16) sharing the engine's LLM weights."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    llm = engine.llm
    d = llm.dims
    dev = llm.device
    cfg = Qwen3Config(**{k: v for k, v in engine.llm_cfg.items() if k not in ("architectures", "transformers_version")})
    cfg.torch_dtype = torch.bfloat16
    with torch.device("meta"):
        model = Qwen3ForCausalLM(cfg)

    def param(t):
        return nn.Parameter(t.to(dev, torch.bfloat16), requires_grad=False)

    embed = llm.embed if llm.embed.dtype == torch.bfloat16 else llm.embed.to(torch.bfloat16)
    model.model.embed_tokens.weight = nn.Parameter(embed, requires_grad=False)
    model.lm_head.weight = model.model.embed_tokens.weight
    model.model.norm.weight = param(llm.norm.weight)
    q, kv = d.n_q_heads * d.head_dim, d.n_kv_heads * d.head_dim
    for layer, L in zip(model.model.layers, llm.layers):
        layer.input_layernorm.weight = param(L.in_norm.weight)
        layer.post_attention_layernorm.weight = param(L.post_norm.weight)
        layer.self_attn.q_norm.weight = param(L.q_norm)
        layer.self_attn.k_norm.weight = param(L.k_norm)
        layer.self_attn.q_proj = ConvRotLinearModule(L.qkv, 0, q)
        layer.self_attn.k_proj = ConvRotLinearModule(L.qkv, q, q + kv)
        layer.self_attn.v_proj = ConvRotLinearModule(L.qkv, q + kv, q + 2 * kv)
        layer.self_attn.o_proj = ConvRotLinearModule(L.o)
        layer.mlp.gate_proj = ConvRotLinearModule(L.gate_up, 0, d.intermediate)
        layer.mlp.up_proj = ConvRotLinearModule(L.gate_up, d.intermediate, 2 * d.intermediate)
        layer.mlp.down_proj = ConvRotLinearModule(L.down)
    model.model.rotary_emb = Qwen3RotaryEmbedding(config=cfg).to(dev)  # (device= is deprecated in Transformers 5.18)
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if t.is_meta:
            raise RuntimeError(f"Canary-Qwen fallback model: {name} was not initialized")
    model.eval()
    return model


def hf_generate(engine, x: torch.Tensor, lengths: list[int], generation_config, generation_kwargs: dict):
    """NeMo ``SALM.generate`` decoding on right-padded prompt embeddings ``x`` [B, P, hidden]."""
    from transformers import GenerationConfig

    model = getattr(engine, "_hf_model", None)
    if model is None:
        model = engine._hf_model = build_hf_model(engine)
    b, p, h = x.shape
    emb = torch.zeros((b, p, h), device=x.device, dtype=torch.bfloat16)
    mask = torch.zeros((b, p), device=x.device, dtype=torch.bool)
    for i, n in enumerate(lengths):  # left padding, as NeMo builds it
        emb[i, p - n:] = x[i, :n]
        mask[i, p - n:] = True
    if generation_config is None:
        generation_config = GenerationConfig(bos_token_id=None, eos_token_id=engine.text_eos_id,
                                             pad_token_id=engine.text_pad_id)
    return model.generate(inputs_embeds=emb, attention_mask=mask, generation_config=generation_config,
                          **generation_kwargs)
