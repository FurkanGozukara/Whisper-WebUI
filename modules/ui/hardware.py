"""Conservative preset batches from measured Canary ConvRot memory use.

Peak process memory (CUDA context included) of the INT8 model with automatic chunking (up to 40 s in one
piece, 15-30 s pieces for long recordings), over a mix of short and hour-long English recordings on an RTX
A6000: batch 1/2/4/8/16 = 4.9/5.3/6.0/8.2/11.5 GiB, at 44/64/90/101/110x real time on long recordings (batch
24 was no faster). Its cached CUDA graphs grow with the number of different recordings; each tier's batch ran a
1,402-file English session (14.8 hours) without a failed file with PyTorch capped to the card's size minus 0.9
GiB (6 GB: batch 2 at 48x real time, batch 1 at 36x; the engine frees its graphs and retries when memory runs
out). The free-memory checks are compared with the memory free at startup, so the desktop and other
applications count. These are capacity estimates, not a claim of testing every GPU model. Other engines retain
batch one: batched Whisper decoding lost words in English testing.
"""

from dataclasses import dataclass
import math


VRAM_TIERS_GIB = (6, 8, 10, 12, 16, 24, 32)
CANARY_TIER_BATCHES = {6: 2, 8: 4, 10: 8, 12: 8, 16: 16, 24: 16, 32: 16}
# Full-process NVML peaks on mixed-length English, plus working headroom (batch 2: the simulated 6 GB card).
# Free memory can be much lower than capacity when another application is open.
CANARY_REQUIRED_FREE_GIB = {2: 5.0, 4: 6.6, 8: 8.9, 16: 12.3}


@dataclass(frozen=True)
class HardwarePreset:
    tier_gib: int | None
    canary_batch_size: int = 1
    free_gib: float | None = None


def _memory(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def select_hardware_preset(total_gib=None, free_gib=None, convrot_supported=False):
    total = _memory(total_gib)
    free = _memory(free_gib)
    if total is None:
        return HardwarePreset(None, free_gib=free)
    # Driver-reserved memory makes an advertised 8 GiB card report a little less.
    tier = max((value for value in VRAM_TIERS_GIB if value <= total + 0.5), default=None)
    if tier is None or not convrot_supported or free is None:
        return HardwarePreset(tier, free_gib=free)
    available = min(total, free)
    ceiling = CANARY_TIER_BATCHES[tier]
    batch = max(
        (size for size, required in CANARY_REQUIRED_FREE_GIB.items()
         if size <= ceiling and required <= available),
        default=1,
    )
    return HardwarePreset(tier, batch, free)


def preset_batch_size(hardware, whisper_type, model_size):
    # The original NeMo model has a different memory profile; never apply the
    # INT8 measurements to it, other engines, or an unrecognized custom model.
    name = str(model_size or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if whisper_type == "canary-qwen" and name == "canary-qwen-2.5b-int8-convrot":
        return hardware.canary_batch_size
    return 1
