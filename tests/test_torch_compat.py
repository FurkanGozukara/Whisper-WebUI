import pickle

import pytest
import torch

from modules.utils.torch_compat import enable_torchaudio_2_9_compat, torch_load_safe_globals


def _build_pyannote_metadata_payload():
    enable_torchaudio_2_9_compat()
    pyannote_task = pytest.importorskip("pyannote.audio.core.task")

    return {
        "version": torch.__version__,
        "spec": pyannote_task.Specifications(
            problem=pyannote_task.Problem.MONO_LABEL_CLASSIFICATION,
            resolution=pyannote_task.Resolution.FRAME,
            duration=1.0,
        ),
    }


def test_torch_load_safe_globals_allows_pyannote_checkpoint_metadata(tmp_path):
    payload = _build_pyannote_metadata_payload()
    checkpoint_path = tmp_path / "pyannote-metadata.pt"
    torch.save(payload, checkpoint_path)

    original_safe_globals = list(torch.serialization.get_safe_globals())
    torch.serialization.clear_safe_globals()

    try:
        with pytest.raises(pickle.UnpicklingError):
            torch.load(checkpoint_path)

        with torch_load_safe_globals():
            loaded = torch.load(checkpoint_path)
    finally:
        torch.serialization.clear_safe_globals()
        if original_safe_globals:
            torch.serialization.add_safe_globals(original_safe_globals)

    assert str(loaded["version"]) == str(payload["version"])
    assert loaded["spec"] == payload["spec"]


def test_torch_load_safe_globals_preserves_body_exceptions():
    with pytest.raises(RuntimeError, match="boom"):
        with torch_load_safe_globals():
            raise RuntimeError("boom")
