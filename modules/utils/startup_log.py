"""Timestamped startup lines for CMD, so a slow first start does not look frozen.

The worker process that reads the GPU and model lists inherits the start time through
WHISPER_WEBUI_START_TIME, so its lines count from the same moment as the app's.
"""
import importlib.util
import os
import sys
import time

_START_TIME = float(os.environ.setdefault("WHISPER_WEBUI_START_TIME", repr(time.time())))


def startup_log(message: str) -> None:
    print(f"[Startup {time.time() - _START_TIME:5.1f}s] {message}", file=sys.stderr, flush=True)


def is_first_start() -> bool:
    """True while Python has not compiled PyTorch yet: the first start after an install or update."""
    try:
        origin = importlib.util.find_spec("torch").origin
        return not os.path.exists(importlib.util.cache_from_source(origin))
    except Exception:
        return False
