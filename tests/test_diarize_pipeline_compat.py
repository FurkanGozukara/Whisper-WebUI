from types import SimpleNamespace

from modules.diarize.diarizer import Diarizer
from modules.diarize import diarize_pipeline


def _write_config(tmp_path, clustering="VBxClustering"):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "pipeline:",
                "  params:",
                f"    clustering: {clustering}",
            ]
        ),
        encoding="utf-8",
    )
    return str(config_path)


def test_pyannote_4_token_keyword_is_used(tmp_path):
    config_path = _write_config(tmp_path)
    calls = []

    class Pipeline:
        @staticmethod
        def from_pretrained(checkpoint, token=None, cache_dir=None):
            calls.append(
                {
                    "checkpoint": checkpoint,
                    "token": token,
                    "cache_dir": cache_dir,
                }
            )
            return SimpleNamespace(legacy=False)

    pipeline = diarize_pipeline._load_pyannote_pipeline(
        Pipeline,
        config_path,
        use_auth_token="hf_test",
        cache_dir="models",
    )

    assert calls == [
        {
            "checkpoint": config_path,
            "token": "hf_test",
            "cache_dir": "models",
        }
    ]
    assert pipeline.legacy is True


def test_legacy_pyannote_use_auth_token_keyword_is_used(tmp_path):
    config_path = _write_config(tmp_path)
    calls = []

    class Pipeline:
        @staticmethod
        def from_pretrained(checkpoint, use_auth_token=None, cache_dir=None):
            calls.append(
                {
                    "checkpoint": checkpoint,
                    "use_auth_token": use_auth_token,
                    "cache_dir": cache_dir,
                }
            )
            return SimpleNamespace(legacy=False)

    diarize_pipeline._load_pyannote_pipeline(
        Pipeline,
        config_path,
        use_auth_token="hf_test",
        cache_dir="models",
    )

    assert calls == [
        {
            "checkpoint": config_path,
            "use_auth_token": "hf_test",
            "cache_dir": "models",
        }
    ]


def test_non_vbx_config_skips_unused_plda(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, clustering="AgglomerativeClustering")
    original_get_plda = lambda *args, **kwargs: "original"
    fake_speaker_diarization = SimpleNamespace(get_plda=original_get_plda)

    def fake_import_module(name):
        assert name == "pyannote.audio.pipelines.speaker_diarization"
        return fake_speaker_diarization

    monkeypatch.setattr(diarize_pipeline.importlib, "import_module", fake_import_module)

    class Pipeline:
        @staticmethod
        def from_pretrained(checkpoint, cache_dir=None):
            assert checkpoint == config_path
            assert cache_dir == "models"
            assert fake_speaker_diarization.get_plda("unused") is None
            return SimpleNamespace(legacy=False)

    pipeline = diarize_pipeline._load_pyannote_pipeline(
        Pipeline,
        config_path,
        use_auth_token=None,
        cache_dir="models",
    )

    assert pipeline.legacy is True
    assert fake_speaker_diarization.get_plda is original_get_plda


def test_diarizer_failure_message_points_to_offline_bundle(monkeypatch, capsys):
    class BrokenPipeline:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("missing config")

    monkeypatch.setattr(diarize_pipeline, "DiarizationPipeline", BrokenPipeline, raising=False)

    diarizer = Diarizer(model_dir="models/Diarization")
    diarizer.update_pipe(device="cpu")

    output = capsys.readouterr().out

    assert "offline diarization bundle" in output
    assert "DownloadModels.py" in output
    assert "token" not in output.lower()
