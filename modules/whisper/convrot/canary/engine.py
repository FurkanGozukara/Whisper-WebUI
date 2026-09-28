"""Canary-Qwen model backed by the ConvRot runtime, with the subset of NeMo ``SALM`` the app uses.

``CanaryConvRot.generate(prompts=..., audios=..., audio_lens=..., **generation_kwargs)`` follows
``SALM.generate``: the Qwen prompt format, the audio embeddings in place of the audio locator tag,
Hugging Face ``generate`` semantics and its output layout (new tokens only, finished rows padded
with the pad id). Greedy decoding (optionally with a repetition penalty) runs on the fast path:
batched prefill into a static KV cache and CUDA-graph decode steps. Other decoding strategies
(beam search, sampling, n-gram blocking, custom generation configs) run Hugging Face ``generate``
on a Qwen3 model that shares the same INT8 layers (``hf_fallback``).
"""

from __future__ import annotations

import json
import os
import threading
from typing import Optional

import torch

from .model import CanaryDims, DecodeSession, EncoderRuntime, LLMRuntime, canonical_tensors

AUDIO_TAG = "<|audioplaceholder|>"
QWEN_BOT = "<|im_start|>"
QWEN_EOT = "<|im_end|>"

# Runtime defaults (stored in each converted model's config.json). Weight-only INT8 everywhere: the
# encoder runs as a CUDA graph where INT8 activations (W8A8) buy no measurable speed but 2.5x the
# next-token KL, and decoding is bound by weight bytes. The tied LM head stays BF16 (INT8 saves
# about 10 % per decode step but nearly doubles the KL).
DEFAULT_RUNTIME = {"encoder_w8a8": False, "encoder_group_scales": True, "llm_weight_only": True,
                   "lm_head_int8": False}

FAST_KWARGS = {"max_new_tokens", "num_beams", "do_sample", "temperature", "length_penalty", "repetition_penalty",
               "no_repeat_ngram_size", "top_k", "top_p", "max_length"}


def load_safetensors(path: str, device: str = "cpu") -> tuple[dict, dict]:
    from safetensors import safe_open

    tensors = {}
    with safe_open(path, framework="pt", device=device) as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors, metadata


class CanaryTokenizer:
    """Qwen3 tokenizer (``tokenizers``) plus the audio locator tag, with NeMo's ``ids_to_text``."""

    def __init__(self, tokenizer_json: str):
        import tokenizers

        self.tok = tokenizers.Tokenizer.from_file(tokenizer_json)
        self.tok.add_special_tokens([tokenizers.AddedToken(AUDIO_TAG, special=True, normalized=False)])
        self.audio_id = self.tok.token_to_id(AUDIO_TAG)
        self.eos_id = self.tok.token_to_id(QWEN_EOT)
        self.pad_id = self.tok.token_to_id("<|endoftext|>")

    def text_to_ids(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def ids_to_text(self, ids, remove_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tok.decode([int(i) for i in ids], skip_special_tokens=remove_special_tokens)

    def token_to_id(self, token: str) -> Optional[int]:
        return self.tok.token_to_id(token)


def format_qwen_dialog(turns: list[dict]) -> str:
    """NeMo's "qwen" prompt format for inference (user/assistant turns + the assistant prefix)."""
    text = ""
    for turn in turns:
        role = turn.get("role")
        if role not in ("user", "assistant"):
            raise ValueError(f"Unsupported prompt role for Canary-Qwen: {role!r}")
        message = turn.get("content", turn.get("slots", {}).get("message", ""))
        text += f"{QWEN_BOT}{role}\n{message}{QWEN_EOT}\n"
    if turns and turns[-1].get("role") != "assistant":
        text += f"{QWEN_BOT}assistant\n"
    return text


class CanaryConvRot:
    audio_locator_tag = AUDIO_TAG

    def __init__(self, tensors: dict, salm_cfg: dict, llm_cfg: dict, tokenizer_json: str, device_index: int = 0,
                 dtype: torch.dtype = torch.float16, use_cuda_graphs: bool = True, runtime: Optional[dict] = None,
                 model_path: Optional[str] = None):
        if not torch.cuda.is_available():
            raise RuntimeError("The Canary-Qwen ConvRot runtime requires an NVIDIA CUDA GPU")
        self._torch_device = torch.device("cuda", int(device_index))
        self.device = self._torch_device
        self.model_path = model_path
        self.salm_cfg = salm_cfg
        self.llm_cfg = llm_cfg
        self.dims = CanaryDims.from_configs(salm_cfg, llm_cfg)
        self.dtype = dtype
        self.runtime_cfg = {**DEFAULT_RUNTIME, **(runtime or {})}
        self.tokenizer = CanaryTokenizer(tokenizer_json)
        with torch.cuda.device(self._torch_device):
            self.encoder = EncoderRuntime(tensors, self.dims, self._torch_device, dtype=dtype,
                                          w8a8=self.runtime_cfg["encoder_w8a8"],
                                          group_scales=self.runtime_cfg["encoder_group_scales"])
            self.llm = LLMRuntime(tensors, self.dims, self._torch_device, dtype=dtype,
                                  weight_only=self.runtime_cfg["llm_weight_only"])
            if self.runtime_cfg.get("lm_head_int8"):
                self.llm.set_lm_head_int8(True)
        self.use_cuda_graphs = use_cuda_graphs
        self._sessions: dict = {}
        self._loaded = True
        # One caller at a time: sessions, CUDA graphs and their static buffers are shared state.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------------------------------
    @classmethod
    def from_folder(cls, model_path: str, **kwargs) -> "CanaryConvRot":
        """A converted INT8 ConvRot model folder."""
        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        tensors, _ = load_safetensors(os.path.join(model_path, "model.safetensors"))
        runtime = {**config.get("convrot", {}).get("runtime", {}), **kwargs.pop("runtime", {})}
        return cls(tensors, config["salm"], config["llm"], os.path.join(model_path, "tokenizer.json"),
                   runtime=runtime, model_path=model_path, **kwargs)

    @classmethod
    def from_source(cls, source_dir: str, llm_dir: str, dtype: torch.dtype = torch.float32, **kwargs):
        """The original NeMo checkpoint (BF16 weights, LoRA merged) as a float model."""
        with open(os.path.join(source_dir, "config.json"), "r", encoding="utf-8") as f:
            salm_cfg = json.load(f)
        with open(os.path.join(llm_dir, "config.json"), "r", encoding="utf-8") as f:
            llm_cfg = json.load(f)
        sd, _ = load_safetensors(os.path.join(source_dir, "model.safetensors"))
        lora = salm_cfg.get("lora") or {}
        scale = float(lora.get("lora_alpha", 1)) / float(lora.get("r", 1)) if lora else 0.0
        tensors = canonical_tensors(sd, scale)
        del sd
        return cls(tensors, salm_cfg, llm_cfg, os.path.join(llm_dir, "tokenizer.json"), dtype=dtype, **kwargs)

    # ------------------------------------------------------------------------------------------
    # NeMo SALM-compatible surface used by the app
    # ------------------------------------------------------------------------------------------
    def eval(self):
        return self

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        return self.tokenizer.pad_id

    @property
    def audio_locator_tag_id(self) -> int:
        return self.tokenizer.audio_id

    def to(self, device=None, dtype=None, **_ignored):
        """Move the weights (``"cpu"`` parks them in pinned RAM; CUDA graphs and KV caches are dropped)."""
        if device is None:
            return self
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = self._torch_device
        with self._lock, torch.cuda.device(self._torch_device):
            # The beam/sampling adapter holds Parameters sharing the current
            # CUDA embedding storage, plus copied norm weights. Moving the
            # runtime replaces its tensors, so keeping the adapter both leaks
            # VRAM while offloaded and leaves stale parameters on reload.
            # Rebuild this inexpensive view on the next fallback generation.
            self.__dict__.pop("_hf_model", None)
            self._sessions.clear()
            self.encoder.to(device)
            self.llm.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self._loaded = device.type == "cuda"
        if not self._loaded:
            import gc

            gc.collect()
            torch.cuda.empty_cache()
        return self

    def release_cuda_graphs(self):
        """Drop the cached CUDA graphs and KV caches; generate() captures the ones it needs again.

        Every graph keeps its own activation memory, which no other allocation can use while the graph exists.
        """
        with self._lock:
            self._sessions.clear()
            self.encoder.release_graphs()

    # ------------------------------------------------------------------------------------------
    # prompts and inputs
    # ------------------------------------------------------------------------------------------
    def encode_prompt(self, turns) -> list[int]:
        if isinstance(turns, torch.Tensor):
            return [int(i) for i in turns.tolist() if int(i) != self.text_pad_id]
        return self.tokenizer.text_to_ids(format_qwen_dialog([dict(t) for t in turns]))

    def _assemble(self, prompt_ids: list[list[int]], audio_embs: list[torch.Tensor]):
        """Token embeddings with each audio locator replaced by its audio embeddings; right padded."""
        rows, lengths = [], []
        ai = 0
        dev = self._torch_device
        for ids in prompt_ids:
            parts = []
            start = 0
            for pos, tok in enumerate(ids):
                if tok == self.tokenizer.audio_id:
                    if pos > start:
                        parts.append(self.llm.embed_ids(torch.tensor(ids[start:pos], device=dev)))
                    parts.append(audio_embs[ai].float())
                    ai += 1
                    start = pos + 1
            if start < len(ids):
                parts.append(self.llm.embed_ids(torch.tensor(ids[start:], device=dev)))
            row = torch.cat(parts, dim=0)
            rows.append(row)
            lengths.append(row.shape[0])
        if ai != len(audio_embs):
            raise ValueError(f"Expected {len(audio_embs)} audio locator tags in the prompts, found {ai}")
        p = max(lengths)
        x = torch.zeros((len(rows), p, self.dims.hidden), device=dev, dtype=torch.float32)
        for i, row in enumerate(rows):
            x[i, : row.shape[0]] = row
        return x, lengths

    @torch.inference_mode()
    def encode_audio(self, audios: torch.Tensor, audio_lens: torch.Tensor) -> list[torch.Tensor]:
        """Per-row audio embeddings (views of a buffer the next call overwrites)."""
        enc, elens = self.encoder.run(audios, audio_lens, use_graph=self.use_cuda_graphs)
        return [enc[i, :n] for i, n in enumerate(elens.tolist())]

    # ------------------------------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------------------------------
    def _session(self, rows: int, need: int) -> DecodeSession:
        max_len = ((need + 255) // 256) * 256
        key = (rows, max_len)
        sess = self._sessions.get(key)
        if sess is None:
            if len(self._sessions) >= 2:
                self._sessions.pop(next(iter(self._sessions)))
            sess = DecodeSession(self.llm, rows, max_len, use_graph=self.use_cuda_graphs)
            self._sessions[key] = sess
        return sess

    def _options(self, generation_config, kwargs: dict) -> tuple[dict, bool]:
        """Effective decoding options, and whether anything outside the fast path's scope was requested."""
        opts = {}
        other = False
        if generation_config is not None:
            for name, value in generation_config.to_diff_dict().items():
                if name in ("transformers_version", "_from_model_config", "bos_token_id"):
                    continue
                if name == "eos_token_id":
                    other |= value not in (self.text_eos_id, [self.text_eos_id])
                elif name == "pad_token_id":
                    other |= value != self.text_pad_id
                elif name in FAST_KWARGS:
                    opts[name] = value
                else:
                    other = True
        opts.update(kwargs)
        other |= any(k not in FAST_KWARGS for k in opts)
        return opts, other

    @staticmethod
    def fast_path_supported(opts: dict) -> bool:
        if int(opts.get("num_beams", 1) or 1) != 1 or bool(opts.get("do_sample", False)):
            return False
        if int(opts.get("no_repeat_ngram_size", 0) or 0) > 0:
            return False
        return True

    @torch.inference_mode()
    def generate(self, prompts, audios: torch.Tensor = None, audio_lens: torch.Tensor = None,
                 generation_config=None, enable_thinking: Optional[bool] = None, **generation_kwargs) -> torch.Tensor:
        del enable_thinking  # the "qwen" prompt format has no thinking switch (same as NeMo)
        opts, other = self._options(generation_config, generation_kwargs)
        with self._lock, torch.cuda.device(self._torch_device):
            if not self._loaded:
                self.to(self._torch_device)
            prompt_ids = [self.encode_prompt(p) for p in prompts]
            audio_embs = self.encode_audio(audios, audio_lens) if audios is not None else []
            x, lengths = self._assemble(prompt_ids, audio_embs)
            if not other and self.fast_path_supported(opts):
                return self._greedy(x, lengths, opts)
            from .hf_fallback import hf_generate

            return hf_generate(self, x, lengths, generation_config, generation_kwargs)

    @staticmethod
    def _max_new(opts: dict, prompt_len: int) -> int:
        if opts.get("max_new_tokens") is not None:
            return max(1, int(opts["max_new_tokens"]))
        if opts.get("max_length") is not None:  # Hugging Face counts the prompt embeddings in max_length
            return max(1, int(opts["max_length"]) - prompt_len)
        return max(1, 20 - prompt_len)  # GenerationConfig default max_length

    def _greedy(self, x: torch.Tensor, lengths: list[int], opts: dict) -> torch.Tensor:
        rows = x.shape[0]
        max_new = self._max_new(opts, x.shape[1])
        penalty = float(opts.get("repetition_penalty", 1.0) or 1.0)
        eos, pad = self.text_eos_id, self.text_pad_id
        sess = self._session(rows, max(lengths) + max_new + 1)
        logits = sess.prefill(x, lengths)
        dev = x.device
        out = torch.full((rows, max_new), pad, device=dev, dtype=torch.long)
        finished = torch.zeros((rows,), device=dev, dtype=torch.bool)
        seen = torch.zeros((rows, self.dims.vocab), device=dev, dtype=torch.bool) if penalty != 1.0 else None
        n = 0
        for t in range(max_new):
            if seen is not None:
                logits = torch.where(seen, torch.where(logits < 0, logits * penalty, logits / penalty), logits)
            nxt = logits.argmax(-1)
            nxt = torch.where(finished, torch.full_like(nxt, pad), nxt)
            out[:, t] = nxt
            finished |= nxt == eos
            n = t + 1
            if seen is not None:
                seen.scatter_(1, nxt[:, None], True)
            if n == max_new or bool(finished.all()):
                break
            sess.ids.copy_(nxt)
            logits = sess.step()
        return out[:, :n]
