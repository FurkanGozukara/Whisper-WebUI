"""Calibration statistics for INT8 ConvRot Canary-Qwen conversion.

The float model runs on calibration audio; for every linear we accumulate the Gram matrix X^T X
of its input after the pre-op (LayerNorm, RMSNorm, SiLU or SwiGLU: exactly what the INT8 kernel
rotates). Encoder statistics come from the encoder forward, LLM statistics from teacher-forced
prefill of the prompt, the audio embeddings and the float model's own greedy transcript. GPTQ
(``modules.whisper.convrot.calibrate``) then rounds each rotated weight with the Hessian rotated
into the ConvRot basis.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch

from . import kernels as CK

ENC_LINEARS = ("ff1_l1", "ff1_l2", "qkv", "att_out", "pw1", "pw2", "ff2_l1", "ff2_l2")
LLM_LINEARS = ("qkv", "o", "gate_up", "down")

# runtime linear -> checkpoint linears sharing its input (rows concatenated in this order)
ENC_PARTS = {
    "ff1_l1": ["feed_forward1.linear1"], "ff1_l2": ["feed_forward1.linear2"],
    "qkv": ["self_attn.linear_q", "self_attn.linear_k", "self_attn.linear_v"], "att_out": ["self_attn.linear_out"],
    "pw1": ["conv.pointwise_conv1"], "pw2": ["conv.pointwise_conv2"],
    "ff2_l1": ["feed_forward2.linear1"], "ff2_l2": ["feed_forward2.linear2"],
}
LLM_PARTS = {
    "qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "o": ["self_attn.o_proj"],
    "gate_up": ["mlp.gate_proj", "mlp.up_proj"], "down": ["mlp.down_proj"],
}


def gram_groups(dims) -> list[tuple[str, list[str]]]:
    """(gram key, checkpoint linear prefixes) for every quantized input group."""
    out = []
    for i in range(dims.enc_layers):
        for n in ENC_LINEARS:
            out.append((f"enc.{i}.{n}", [f"perception.encoder.layers.{i}.{p}" for p in ENC_PARTS[n]]))
    for i in range(dims.llm_layers):
        for n in LLM_LINEARS:
            out.append((f"llm.{i}.{n}", [f"llm.model.layers.{i}.{p}" for p in LLM_PARTS[n]]))
    return out


class _GramLinear:
    """Wraps a float Linear and accumulates X^T X of its (pre-processed) input."""

    def __init__(self, base, store: dict, key: str, row_filter: Optional[Callable] = None):
        self.base = base
        self.store = store
        self.key = key
        self.row_filter = row_filter

    def __call__(self, x, pre=CK.PRE_NONE, nw=None, nb=None, eps=1e-5, residual=None, out=None):
        xf = CK.torch_pre(x, pre, nw, nb, eps)
        x2 = xf.reshape(-1, xf.shape[-1])
        if self.row_filter is not None:
            x2 = self.row_filter(x2)
        entry = self.store.get(self.key)
        if entry is None:
            entry = self.store[self.key] = [torch.zeros((x2.shape[1], x2.shape[1]), device=x2.device,
                                                        dtype=torch.float32), 0]
        entry[0].addmm_(x2.t(), x2)
        entry[1] += x2.shape[0]
        return self.base(x, pre, nw, nb, eps, residual=residual, out=out)


def wrap(engine, store: dict, encoder: bool = True, llm: bool = True):
    saved = []
    if encoder:
        for i, layer in enumerate(engine.encoder.layers):
            for n in ENC_LINEARS:
                base = getattr(layer, n)
                saved.append((layer, n, base))
                setattr(layer, n, _GramLinear(base, store, f"enc.{i}.{n}"))
    if llm:
        for i, layer in enumerate(engine.llm.layers):
            for n in LLM_LINEARS:
                base = getattr(layer, n)
                saved.append((layer, n, base))
                setattr(layer, n, _GramLinear(base, store, f"llm.{i}.{n}"))
    return saved


def unwrap(saved):
    for layer, n, base in saved:
        setattr(layer, n, base)


@torch.inference_mode()
def collect_gram(engine, windows: Iterable[torch.Tensor], max_new_tokens: int = 256,
                 progress: Optional[Callable[[str], None]] = None, store: Optional[dict] = None,
                 transcripts: Optional[list] = None) -> dict:
    """Accumulate X^T X for every linear of a float ``CanaryConvRot`` over single-window batches.

    Each window: (1) greedy transcript from the float model (no statistics), (2) encoder forward and
    teacher-forced LLM prefill of prompt + audio + transcript + <|im_end|> with the Gram wrappers.
    """
    store = {} if store is None else store
    prompt = [{"role": "user", "content": f"Transcribe the following: {engine.audio_locator_tag}"}]
    prompt_ids = engine.encode_prompt(prompt)
    for wi, audio in enumerate(windows):
        audios = audio.reshape(1, -1).float()
        lens = torch.tensor([audios.shape[1]], dtype=torch.long)
        out = engine.generate([prompt], audios=audios, audio_lens=lens, max_new_tokens=max_new_tokens)
        text_ids = [int(t) for t in out[0].tolist() if int(t) not in (engine.text_eos_id, engine.text_pad_id)]
        if transcripts is not None:
            transcripts.append(text_ids)
        saved = wrap(engine, store)
        try:
            embs = engine.encode_audio(audios, lens)
            ids = prompt_ids + text_ids + [engine.text_eos_id]
            x, lengths = engine._assemble([ids], embs)
            sess = engine._session(1, lengths[0] + 1)
            sess.prefill(x, lengths)
        finally:
            unwrap(saved)
        if progress:
            progress(f"calibration window {wi + 1}: {audios.shape[1] / 16000:.1f}s audio, {embs[0].shape[0]} audio "
                     f"frames, {len(text_ids)} text tokens")
    return store
