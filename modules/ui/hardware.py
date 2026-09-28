"""Conservative preset batches from measured Canary ConvRot memory use.

The tiers are capacity estimates, not a claim of testing every GPU model. Other
engines retain batch one: increasing Whisper encoder batches can change words.
"""

from dataclasses import dataclass
import math


VRAM_TIERS_GIB = (6, 8, 10, 12, 16, 24, 32)
CANARY_TIER_BATCHES = {6: 1, 8: 2, 10: 4, 12: 8, 16: 8, 24: 16, 32: 16}
# Full-process NVML peaks on mixed-length English, plus working headroom.
# Free memory can be much lower than capacity when another application is open.
CANARY_REQUIRED_FREE_GIB = {1: 5.5, 2: 7.0, 4: 8.5, 8: 10.5, 16: 16.0}


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
