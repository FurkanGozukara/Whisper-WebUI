import pytest

from modules.ui.hardware import preset_batch_size, select_hardware_preset


@pytest.mark.parametrize("capacity,batch", [(6, 1), (8, 2), (10, 4), (12, 8), (16, 8), (24, 16), (32, 16)])
def test_tiers_keep_memory_headroom(capacity, batch):
    result = select_hardware_preset(capacity - 0.2, capacity - 0.3, True)
    assert result.tier_gib == capacity
    assert result.canary_batch_size == batch


def test_busy_gpu_uses_free_memory_instead_of_capacity():
    result = select_hardware_preset(48, 7.4, True)
    assert result.tier_gib == 32
    assert result.canary_batch_size == 2
    assert select_hardware_preset(48, 4, True).canary_batch_size == 1
    assert select_hardware_preset(4, 3.8, True).tier_gib is None


@pytest.mark.parametrize("total,free,supported", [(None, None, False), (8, None, True), (48, 40, False), (float('nan'), 40, True), (48, float('inf'), True)])
def test_unknown_or_unsupported_hardware_keeps_batch_one(total, free, supported):
    assert select_hardware_preset(total, free, supported).canary_batch_size == 1


def test_int8_tiers_do_not_apply_to_original_canary_or_whisper():
    hardware = select_hardware_preset(32, 30, True)
    assert preset_batch_size(hardware, "canary-qwen", "canary-qwen-2.5b-int8-convrot") == 16
    assert preset_batch_size(hardware, "canary-qwen", r"C:\models\canary-qwen-2.5b-int8-convrot") == 16
    assert preset_batch_size(hardware, "canary-qwen", "nvidia/canary-qwen-2.5b") == 1
    assert preset_batch_size(hardware, "faster-whisper", "large-v3-int8-convrot") == 1
