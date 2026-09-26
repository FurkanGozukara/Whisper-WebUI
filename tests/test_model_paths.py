"""Model downloads and loads stay inside Whisper-WebUI/models."""

import os
import urllib.request

from modules.utils import paths

CACHE_VARIABLES = ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "TORCH_HOME", "NEMO_CACHE_DIR",
                   "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_TOKEN_PATH")


def inside_models_dir(path: str) -> bool:
    models_dir = os.path.normcase(os.path.abspath(paths.MODELS_DIR))
    return os.path.commonpath([os.path.normcase(os.path.abspath(path)), models_dir]) == models_dir


def test_model_cache_env_overrides_global_caches(monkeypatch, tmp_path):
    for name in CACHE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "global_hf"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "global_transformers"))

    paths.configure_model_cache_env()

    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "TORCH_HOME", "NEMO_CACHE_DIR"):
        assert inside_models_dir(os.environ[name]), name
    assert "TRANSFORMERS_CACHE" not in os.environ
    assert "HF_TOKEN_PATH" not in os.environ


def test_model_cache_env_keeps_a_saved_login_token(monkeypatch, tmp_path):
    for name in CACHE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "token").write_text("hf_test", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    paths.configure_model_cache_env()

    assert os.environ["HF_TOKEN_PATH"] == str(tmp_path / "token")


def test_insanely_fast_whisper_never_loads_from_global_hf_caches(monkeypatch, tmp_path):
    from modules.whisper.insanely_fast_whisper_inference import InsanelyFastWhisperInference

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "global_hub"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "user"))

    candidates = InsanelyFastWhisperInference.candidate_hf_cache_dirs()

    assert candidates
    assert all(inside_models_dir(candidate) for candidate in candidates)


def test_uvr_model_in_models_dir_is_not_downloaded_again(monkeypatch, tmp_path):
    import modules.uvr.music_separator  # noqa: F401  (installs the local-first download)
    import uvr.models as uvr_models

    (tmp_path / "UVR-MDX-NET-Inst_HQ_4.onnx").write_bytes(b"onnx")

    def fail_download(*_args, **_kwargs):
        raise AssertionError("downloaded again")

    monkeypatch.setattr(urllib.request, "urlretrieve", fail_download)

    saved = uvr_models.download_model(
        model_name="UVR-MDX-NET-Inst_HQ_4",
        model_arch="MDX",
        model_path=["https://example.invalid/UVR-MDX-NET-Inst_HQ_4.onnx"],
        save_path=str(tmp_path),
    )

    assert saved == str(tmp_path)
