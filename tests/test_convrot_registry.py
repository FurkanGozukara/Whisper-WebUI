"""Model-list and folder detection for the hosted INT8 ConvRot Whisper models (no GPU needed)."""

import json
import os
from types import SimpleNamespace

from modules.whisper.convrot.registry import HOSTED_CONVROT_MODELS, is_convrot_model_dir
from modules.whisper.faster_whisper_inference import FasterWhisperInference


def _make_model_dir(path, marker=True):
    os.makedirs(path, exist_ok=True)
    config = {"alignment_heads": []}
    if marker:
        config["convrot"] = {"model_name": os.path.basename(path)}
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f)
    with open(os.path.join(path, "model.safetensors"), "wb") as f:
        f.write(b"\0")


def test_detection_requires_marker_and_weights(tmp_path):
    good = tmp_path / "good"
    _make_model_dir(str(good))
    assert is_convrot_model_dir(str(good))
    plain = tmp_path / "plain"
    _make_model_dir(str(plain), marker=False)
    assert not is_convrot_model_dir(str(plain))
    os.remove(good / "model.safetensors")
    assert not is_convrot_model_dir(str(good))


def test_hosted_models_listed_before_download_and_hidden_dirs_skipped(tmp_path):
    (tmp_path / ".download-large-v3-int8-convrot").mkdir()
    (tmp_path / "my-custom-model").mkdir()
    paths = FasterWhisperInference.get_model_paths(SimpleNamespace(model_dir=str(tmp_path)))
    for name in HOSTED_CONVROT_MODELS:
        assert paths[name] == os.path.join(str(tmp_path), name)
    assert "my-custom-model" in paths
    assert not any(name.startswith(".") for name in paths)


def test_downloaded_convrot_folder_counts_as_complete(tmp_path):
    folder = tmp_path / "large-v3-int8-convrot"
    _make_model_dir(str(folder))
    assert FasterWhisperInference.has_downloaded_model_files(str(folder))
