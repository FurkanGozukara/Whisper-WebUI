"""Convert an OpenAI Whisper checkpoint to an INT8 ConvRot model folder.

Output (a faster-whisper style folder the app can load):
  model.safetensors   INT8 ConvRot linears (ComfyUI native int8_tensorwise + convrot
                      layout) and FP16 everything else
  config.json         CTranslate2 Whisper config (alignment heads, language ids,
                      suppress ids) plus a "convrot" section
  tokenizer.json, vocabulary.*, preprocessor_config.json  copied from the matching
                      Systran faster-whisper model

Usage:
  python -m modules.whisper.convrot.convert --checkpoint large-v3.pt \
      --ct2-dir <Systran faster-whisper-large-v3 snapshot> --out <folder>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shutil

import torch
from safetensors.torch import save_file

from . import kernels as K
from .engine import DEFAULT_RUNTIME
from .model import QUANT_FORMAT, WhisperDims

CONVERTER_VERSION = "1.0"
LINEAR_RE = re.compile(r"^(encoder|decoder)\.blocks\.(\d+)\.(attn\.(query|key|value|out)|"
                       r"cross_attn\.(query|key|value|out)|mlp\.0|mlp\.2)\.weight$")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def linear_prefixes(state_dict: dict) -> list[str]:
    return sorted(k[:-len(".weight")] for k in state_dict if LINEAR_RE.match(k))


def quantize_linear(weight: torch.Tensor, device: str = "cuda", clip_search: bool = False):
    """Rotate (W @ H per 256-group) and quantize per output channel to INT8.

    With ``clip_search`` each row picks the clipping ratio in [0.80, 1.00] that
    minimizes the weight reconstruction MSE in the rotated basis.
    """
    w = weight.to(device)
    w_rot = K.rotate_weight(w)  # fp64
    if not clip_search:
        return K.quantize_weight_rowwise(w_rot)
    amax = w_rot.abs().amax(dim=1).clamp_min(1e-12)
    best_err = None
    best_ratio = torch.ones_like(amax)
    for ratio in torch.linspace(0.80, 1.00, 41, dtype=torch.float64):
        scale = amax * ratio / 127.0
        q = torch.clamp(torch.round(w_rot / scale[:, None]), -127, 127)
        err = ((q * scale[:, None] - w_rot) ** 2).sum(dim=1)
        if best_err is None:
            best_err = err
            best_ratio[:] = ratio
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_ratio = torch.where(better, ratio.to(best_ratio.dtype), best_ratio)
    return K.quantize_weight_rowwise(w_rot, clip_ratio=best_ratio)


def gram_key(prefix: str) -> str:
    """OpenAI linear prefix -> calibration key used by calibrate.collect_gram."""
    m = re.match(r"^(encoder|decoder)\.blocks\.(\d+)\.(.+)$", prefix)
    side, idx, rest = m.group(1), m.group(2), m.group(3)
    name = {"attn.query": "qkv", "attn.key": "qkv", "attn.value": "qkv", "attn.out": "out",
            "cross_attn.query": "cross_q", "cross_attn.key": "cross_kv", "cross_attn.value": "cross_kv",
            "cross_attn.out": "cross_out", "mlp.0": "fc1", "mlp.2": "fc2"}[rest]
    return f"{side}.{idx}.{name}"


def default_policy(prefix: str) -> bool:
    """Which linears are quantized. Everything else (convs, embeddings, norms) stays FP16."""
    return True


def convert(checkpoint: str, ct2_dir: str, out_dir: str, model_name: str, device: str = "cuda",
            keep_fp16: set[str] | None = None, clip_search: bool = False, notes: str | None = None,
            grams: dict | None = None, gptq_block: int = 128, gptq_damp: float = 0.01,
            calibration: dict | None = None, runtime: dict | None = None) -> dict:
    keep_fp16 = set(keep_fp16 or [])
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    dims = WhisperDims(**ckpt["dims"])
    sd = ckpt["model_state_dict"]
    source_sha = sha256_file(checkpoint)

    tensors: dict[str, torch.Tensor] = {}
    layers_meta: dict[str, dict] = {}
    quant_prefixes = [p for p in linear_prefixes(sd) if p not in keep_fp16 and default_policy(p)]
    quant_set = set(quant_prefixes)
    stats = []
    comfy_quant = torch.tensor(list(json.dumps(QUANT_FORMAT).encode("utf-8")), dtype=torch.uint8)
    for key, value in sd.items():
        prefix = key[:-len(".weight")] if key.endswith(".weight") else None
        if prefix in quant_set:
            if grams is not None:
                from .calibrate import gptq_quantize, hessian_rotated, output_error
                gram = grams[gram_key(prefix)][0].to(device, torch.float64)
                hess = hessian_rotated(gram)
                w_rot = K.rotate_weight(value.to(device))
                q_rtn, s_rtn = K.quantize_weight_rowwise(w_rot)
                q, scale = gptq_quantize(w_rot, hess, block=gptq_block, damp=gptq_damp)
                rel = output_error(w_rot, q, scale, hess)
                rel_rtn = output_error(w_rot, q_rtn.to(device), s_rtn.to(device), hess)
                stats.append((prefix, rel, rel_rtn))
            else:
                q, scale = quantize_linear(value, device=device, clip_search=clip_search)
                deq = K.rotate_weight(q.to(torch.float64) * scale.to(q.device, torch.float64)[:, None])  # H is its own inverse
                ref = value.to(deq.device, torch.float64)
                rel = ((deq - ref).norm() / ref.norm()).item()
                stats.append((prefix, rel))
            tensors[prefix + ".weight"] = q.cpu().contiguous()
            tensors[prefix + ".weight_scale"] = scale.reshape(-1, 1).cpu().contiguous()
            tensors[prefix + ".comfy_quant"] = comfy_quant.clone()
            layers_meta[prefix] = dict(QUANT_FORMAT)
        else:
            tensors[key] = value.to(torch.float16).contiguous() if value.is_floating_point() else value.contiguous()

    os.makedirs(out_dir, exist_ok=True)
    provenance = {
        "converter": "WhisperWebUI modules/whisper/convrot/convert.py",
        "converter_version": CONVERTER_VERSION,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "source_checkpoint": os.path.basename(checkpoint),
        "source_sha256": source_sha,
        "quantization": "W8A8 INT8 ConvRot: weights rotated per 256-group with the regular Hadamard "
                       "matrix and quantized per output channel (symmetric, "
                       + ("GPTQ rounding with calibration Hessians" if grams is not None else
                          "round-to-nearest" + (", per-row MSE clip search" if clip_search else "")) +
                       "); activations rotated online and quantized dynamically (symmetric)",
        "calibration": calibration or {},
        "quantized_linears": len(quant_prefixes),
        "fp16_linears": sorted(keep_fp16),
        "notes": notes or "",
    }
    metadata = {
        "whisper_dims": dims.to_json(),
        "_quantization_metadata": json.dumps({"format_version": "1.0", "layers": layers_meta}),
        "convrot_whisper": json.dumps(provenance),
    }
    save_file(tensors, os.path.join(out_dir, "model.safetensors"), metadata=metadata)

    with open(os.path.join(ct2_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    provenance["runtime"] = {**DEFAULT_RUNTIME, **(runtime or {})}
    config["convrot"] = provenance
    config["whisper_dims"] = json.loads(dims.to_json())
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    for name in ("tokenizer.json", "vocabulary.json", "vocabulary.txt", "preprocessor_config.json"):
        src = os.path.join(ct2_dir, name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(out_dir, name))
    worst = sorted(stats, key=lambda s: -s[1])[:5]
    info = {"out_dir": out_dir, "quantized": len(quant_prefixes), "source_sha256": source_sha,
            "mean_rel_err": sum(s[1] for s in stats) / max(1, len(stats)), "worst": worst}
    if grams is not None:
        info["mean_rel_err_rtn"] = sum(s[2] for s in stats) / max(1, len(stats))
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ct2-dir", required=True, help="faster-whisper (CTranslate2) folder of the same model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--keep-fp16", default="", help="comma separated linear prefixes to keep in FP16")
    ap.add_argument("--clip-search", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    keep = {p.strip() for p in args.keep_fp16.split(",") if p.strip()}
    info = convert(args.checkpoint, args.ct2_dir, args.out, args.name or os.path.basename(args.out),
                   device=args.device, keep_fp16=keep, clip_search=args.clip_search)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
