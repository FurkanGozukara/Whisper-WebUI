"""Hosted INT8 ConvRot models: model-list entries, folder detection and download.

Kept free of torch/Triton imports so the faster-whisper and Canary-Qwen backends can list and
detect these models cheaply; the engines are imported only when one is loaded.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Optional

CONVROT_MARKER = "convrot"

# Offered in the faster-whisper model list and downloaded from Hugging Face on first use.
HOSTED_CONVROT_MODELS = {
    "large-v3-int8-convrot": ("MonsterMMORPG/Wan_GGUF", "Whisper_INT8_ConvRot/large-v3-int8-convrot"),
    "large-v1-int8-convrot": ("MonsterMMORPG/Wan_GGUF", "Whisper_INT8_ConvRot/large-v1-int8-convrot"),
}
# Regular faster-whisper model used when this system cannot run the ConvRot engine.
CONVROT_FALLBACK_MODELS = {
    "large-v3-int8-convrot": "large-v3",
    "large-v1-int8-convrot": "large-v1",
}

# Offered in the Canary-Qwen model list, downloaded the same way.
HOSTED_CANARY_CONVROT_MODELS = {
    "canary-qwen-2.5b-int8-convrot": ("MonsterMMORPG/Wan_GGUF", "Canary_INT8_ConvRot/canary-qwen-2.5b-int8-convrot"),
}
# NeMo model used when this system cannot run the ConvRot engine.
CANARY_CONVROT_FALLBACK_MODELS = {
    "canary-qwen-2.5b-int8-convrot": "nvidia/canary-qwen-2.5b",
}


def convrot_runtime_supported(device_index: int = 0) -> tuple[bool, str]:
    """Whether the ConvRot engine can run here (CUDA GPU, Ampere or newer, flash-attn, Triton)."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is always installed with the app
        return False, f"PyTorch unavailable ({exc})"
    if not torch.cuda.is_available():
        return False, "no NVIDIA CUDA GPU"
    major, minor = torch.cuda.get_device_capability(device_index)
    if (major, minor) < (8, 0):
        return False, f"GPU compute capability {major}.{minor}; RTX 30 series (8.0) or newer is required"
    for module in ("triton", "flash_attn"):
        try:
            __import__(module)
        except Exception as exc:
            return False, f"{module} is not available ({type(exc).__name__})"
    return True, ""


def _read_config(path: str) -> Optional[dict]:
    if not path or not os.path.isdir(path):
        return None
    config_path = os.path.join(path, "config.json")
    if not (os.path.isfile(config_path) and os.path.isfile(os.path.join(path, "model.safetensors"))):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_convrot_model_dir(path: str) -> bool:
    """True for a folder produced by an INT8 ConvRot converter (Whisper or Canary-Qwen)."""
    config = _read_config(path)
    return config is not None and CONVROT_MARKER in config


def is_canary_convrot_model_dir(path: str) -> bool:
    """True for a folder produced by the INT8 ConvRot Canary-Qwen converter."""
    config = _read_config(path)
    return config is not None and CONVROT_MARKER in config and "salm" in config


def download_convrot_model(name: str, target_dir: str, tqdm_class=None, token: Optional[str] = None) -> str:
    """Download a hosted model folder into ``target_dir``.

    Files are fetched into a hidden staging folder next to the target (an
    interrupted download resumes from there) and the finished folder is moved
    into place, so a partial download never looks like a usable model.
    """
    from huggingface_hub import snapshot_download

    repo_id, subfolder = {**HOSTED_CONVROT_MODELS, **HOSTED_CANARY_CONVROT_MODELS}[name]
    staging = os.path.join(os.path.dirname(os.path.abspath(target_dir)), f".download-{name}")
    kwargs = {"tqdm_class": tqdm_class} if tqdm_class is not None else {}
    snapshot_download(repo_id=repo_id, allow_patterns=[f"{subfolder}/*"], local_dir=staging, token=token, **kwargs)
    downloaded = os.path.join(staging, *subfolder.split("/"))
    if not is_convrot_model_dir(downloaded):
        raise RuntimeError(f"Downloaded files for '{name}' from {repo_id}/{subfolder} are incomplete: {downloaded}")
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    os.replace(downloaded, target_dir)
    shutil.rmtree(staging, ignore_errors=True)
    return target_dir
