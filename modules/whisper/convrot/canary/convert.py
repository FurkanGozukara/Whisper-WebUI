"""Convert NVIDIA Canary-Qwen-2.5B (NeMo SALM checkpoint) to an INT8 ConvRot model folder.

Output folder:
  model.safetensors  INT8 ConvRot linears (ComfyUI native ``int8_tensorwise`` + ``convrot``
                     layout) for every attention/MLP/pointwise-convolution linear of the
                     FastConformer encoder and the Qwen3 LLM (LoRA merged); everything else
                     (subsampling, depthwise convolutions, norms, relative-position projections,
                     audio projection, token embedding / tied LM head) stays BF16 as in the source
  config.json        {"salm": SALM config, "llm": Qwen3 config, "canary_dims": ..., "convrot": provenance + runtime}
  tokenizer files    from Qwen/Qwen3-1.7B (the audio locator tag is added at load time)
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil

import torch
from safetensors.torch import save_file

from modules.whisper.convrot import kernels as WK

from .calibrate import gram_groups
from .engine import DEFAULT_RUNTIME, load_safetensors
from .model import QUANT_FORMAT, CanaryDims, canonical_tensors

CONVERTER_VERSION = "1.0"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quantize_group(weights: list[torch.Tensor], gram, device="cuda", gptq_block=128, gptq_damp=0.01):
    """GPTQ (or round-to-nearest without a Gram) of row-concatenated weights sharing one input."""
    from modules.whisper.convrot.calibrate import gptq_quantize, hessian_rotated, output_error

    w = torch.cat([x.reshape(x.shape[0], -1) for x in weights], dim=0).to(device)
    w_rot = WK.rotate_weight(w)  # fp64
    q_rtn, s_rtn = WK.quantize_weight_rowwise(w_rot)
    if gram is None:
        deq = WK.rotate_weight(q_rtn.to(torch.float64) * s_rtn.to(w_rot.device, torch.float64)[:, None])
        rel = ((deq - w.double()).norm() / w.double().norm()).item()
        return q_rtn.cpu(), s_rtn.cpu(), rel, rel
    hess = hessian_rotated(gram.to(device, torch.float64))
    q, scale = gptq_quantize(w_rot, hess, block=gptq_block, damp=gptq_damp)
    rel = output_error(w_rot, q, scale, hess)
    rel_rtn = output_error(w_rot, q_rtn.to(device), s_rtn.to(device), hess)
    return q.cpu(), scale.cpu(), rel, rel_rtn


def convert(source_dir: str, llm_dir: str, out_dir: str, model_name: str, grams: dict | None = None,
            device: str = "cuda", gptq_block: int = 128, gptq_damp: float = 0.01, keep_float: set | None = None,
            calibration: dict | None = None, runtime: dict | None = None, notes: str | None = None,
            progress=None) -> dict:
    keep_float = set(keep_float or [])
    with open(os.path.join(source_dir, "config.json"), "r", encoding="utf-8") as f:
        salm_cfg = json.load(f)
    with open(os.path.join(llm_dir, "config.json"), "r", encoding="utf-8") as f:
        llm_cfg = json.load(f)
    dims = CanaryDims.from_configs(salm_cfg, llm_cfg)
    src_path = os.path.join(source_dir, "model.safetensors")
    source_sha = sha256_file(src_path)
    sd, _ = load_safetensors(src_path)
    lora = salm_cfg.get("lora") or {}
    lora_scale = float(lora.get("lora_alpha", 1)) / float(lora.get("r", 1)) if lora else 0.0
    t = canonical_tensors(sd, lora_scale)
    del sd

    comfy_quant = torch.tensor(list(json.dumps(QUANT_FORMAT).encode("utf-8")), dtype=torch.uint8)
    tensors: dict[str, torch.Tensor] = {}
    layers_meta: dict[str, dict] = {}
    stats = []
    quantized = set()
    for gi, (key, prefixes) in enumerate(gram_groups(dims)):
        if any(p in keep_float for p in prefixes):
            continue
        weights = [t[p + ".weight"].float() for p in prefixes]
        gram = grams[key][0] if grams is not None else None
        q, scale, rel, rel_rtn = quantize_group(weights, gram, device=device, gptq_block=gptq_block,
                                                gptq_damp=gptq_damp)
        row = 0
        for p, w in zip(prefixes, weights):
            n = w.shape[0]
            tensors[p + ".weight"] = q[row:row + n].contiguous()
            tensors[p + ".weight_scale"] = scale[row:row + n].reshape(-1, 1).contiguous()
            tensors[p + ".comfy_quant"] = comfy_quant.clone()
            layers_meta[p] = dict(QUANT_FORMAT)
            quantized.add(p)
            row += n
        stats.append((key, rel, rel_rtn))
        if progress and (gi % 16 == 0):
            progress(f"quantized {gi + 1} groups; {key}: rel output error {rel:.4f} (round-to-nearest {rel_rtn:.4f})")
    for name, value in t.items():
        prefix = name[: -len(".weight")] if name.endswith(".weight") else None
        if prefix in quantized:
            continue
        if value.is_floating_point():
            value = value.to(torch.bfloat16)
        tensors[name] = value.contiguous()

    os.makedirs(out_dir, exist_ok=True)
    provenance = {
        "converter": "WhisperWebUI modules/whisper/convrot/canary/convert.py",
        "converter_version": CONVERTER_VERSION,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "source_model": "nvidia/canary-qwen-2.5b",
        "source_checkpoint": "model.safetensors",
        "source_sha256": source_sha,
        "lora": f"merged into q_proj/v_proj (scale lora_alpha/r = {lora_scale:g})",
        "quantization": "INT8 ConvRot: weights rotated per 256-group with the regular Hadamard matrix and quantized per "
                        "output channel (symmetric, " + ("GPTQ rounding with calibration Hessians" if grams is not None
                                                         else "round-to-nearest") + "); activations rotated online",
        "calibration": calibration or {},
        "quantized_linears": len(quantized),
        "float_linears": sorted(keep_float),
        "notes": notes or "",
    }
    metadata = {
        "canary_dims": dims.to_json(),
        "_quantization_metadata": json.dumps({"format_version": "1.0", "layers": layers_meta}),
        "convrot_canary": json.dumps(provenance),
    }
    save_file(tensors, os.path.join(out_dir, "model.safetensors"), metadata=metadata)
    provenance["runtime"] = {**DEFAULT_RUNTIME, **(runtime or {})}
    config = {"convrot": provenance, "canary_dims": json.loads(dims.to_json()), "salm": salm_cfg, "llm": llm_cfg}
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    for name in TOKENIZER_FILES:
        src = os.path.join(llm_dir, name)
        if os.path.isfile(src):
            shutil.copyfile(os.path.realpath(src), os.path.join(out_dir, name))
    for lic in ("LICENSES", "LICENSE"):
        if os.path.isfile(os.path.join(source_dir, lic)):
            shutil.copyfile(os.path.join(source_dir, lic), os.path.join(out_dir, lic))
    worst = sorted(stats, key=lambda s: -s[1])[:8]
    info = {"out_dir": out_dir, "quantized_linears": len(quantized), "groups": len(stats), "source_sha256": source_sha,
            "mean_rel_err": sum(s[1] for s in stats) / max(1, len(stats)),
            "mean_rel_err_rtn": sum(s[2] for s in stats) / max(1, len(stats)), "worst": worst}
    return info
