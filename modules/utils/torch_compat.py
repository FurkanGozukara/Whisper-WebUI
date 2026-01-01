"""
PyTorch compatibility helpers.

Why this exists:
- In PyTorch 2.6, `torch.load(..., weights_only=...)` changed default to `weights_only=True`.
- Some popular checkpoints (including OpenAI Whisper `.pt` files) contain simple metadata
  objects like `torch.__version__` which is an instance of `torch.torch_version.TorchVersion`.
- With `weights_only=True`, PyTorch uses a restricted unpickler and will error unless such
  globals are allowlisted.

This module makes model loading robust across PyTorch versions without broadly disabling
 the security benefits of `weights_only=True`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional, Type


def _get_torchversion_cls() -> Optional[Type[object]]:
    """Best-effort lookup for TorchVersion class across torch versions."""
    try:
        import torch  # noqa: F401
    except Exception:
        return None

    try:
        # Preferred location recommended by the PyTorch error message.
        from torch.torch_version import TorchVersion  # type: ignore

        return TorchVersion
    except Exception:
        pass

    try:
        import torch

        return getattr(torch, "TorchVersion", None)
    except Exception:
        return None


def enable_torch_2_6_weights_only_compat() -> None:
    """
    Globally allowlist known-safe globals needed to load common checkpoints under
    `torch.load(weights_only=True)` (PyTorch 2.6+ default).
    """
    try:
        import torch

        serialization = getattr(torch, "serialization", None)
        add_safe_globals = getattr(serialization, "add_safe_globals", None)
        if add_safe_globals is None:
            return

        torchversion_cls = _get_torchversion_cls()
        if torchversion_cls is None:
            return

        # Idempotent in practice; safe to call multiple times.
        add_safe_globals([torchversion_cls])
    except Exception:
        # Best-effort: never crash the app just because compat patching failed.
        return


@contextmanager
def torch_load_safe_globals() -> Iterator[None]:
    """
    Context manager that temporarily allowlists known-safe globals for weights-only loading.

    Use this around code paths that may trigger `torch.load()` internally (e.g. `whisper.load_model`)
    on PyTorch 2.6+, to avoid the "Weights only load failed" error.
    """
    try:
        import torch

        serialization = getattr(torch, "serialization", None)
        safe_globals_cm = getattr(serialization, "safe_globals", None)
        torchversion_cls = _get_torchversion_cls()

        if safe_globals_cm is None or torchversion_cls is None:
            # Fallback to a global patch (or a no-op on older torch versions).
            enable_torch_2_6_weights_only_compat()
            yield
            return

        with safe_globals_cm([torchversion_cls]):
            yield
    except Exception:
        # Best-effort: do not block execution if torch isn't available for some reason.
        yield


