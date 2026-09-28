from types import SimpleNamespace

import numpy as np
import pytest

from modules.whisper.data_classes import WhisperParams
from modules.whisper.faster_whisper_inference import FasterWhisperInference


def resolve(samples, *, model="large-v3-int8-convrot", language="en", condition=True, chunk=30):
    inference = object.__new__(FasterWhisperInference)
    inference.model = SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000))
    params = WhisperParams(model_size=model, lang=language, condition_on_previous_text=condition, chunk_length=chunk)
    audio = np.zeros(samples, dtype=np.float32)
    _, effective = inference.resolve_standard_audio_and_params(audio, params, log_console=False)
    return params, effective


@pytest.mark.parametrize("samples, expected", [(0, True), (29 * 16000, True), (30 * 16000, True), (30 * 16000 + 1, False), (60 * 16000, False)])
def test_english_v3_safeguard_uses_actual_30_second_boundary(samples, expected):
    requested, effective = resolve(samples)
    assert effective.condition_on_previous_text is expected
    assert requested.condition_on_previous_text is True


@pytest.mark.parametrize("model", ["large-v3", "large-v3-int8-convrot", "Systran/faster-whisper-large-v3",
                                  "/opt/webui/models/Whisper/faster-whisper/large-v3-int8-convrot",
                                  r"C:\Apps\Whisper\models\Whisper\faster-whisper\large-v3",
                                  r"C:\Apps\models--Systran--faster-whisper-large-v3\snapshots\abc123"])
def test_full_v3_aliases_and_cross_platform_model_paths(model):
    assert resolve(31 * 16000, model=model)[1].condition_on_previous_text is False


@pytest.mark.parametrize("model", ["large-v1", "large-v1-int8-convrot", "large-v2", "large-v3-turbo", "distil-large-v3", "custom-large-v3"])
def test_other_model_families_keep_existing_context_policy(model):
    assert resolve(60 * 16000, model=model)[1].condition_on_previous_text is True


@pytest.mark.parametrize("repository, expected", [
    ("Systran/faster-whisper-large-v3", False),
    ("Systran/faster-whisper-large-v2", True),
    ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", True),
])
def test_large_alias_follows_the_model_registry(monkeypatch, repository, expected):
    monkeypatch.setattr(FasterWhisperInference, "hf_repo_id_for_model_size", staticmethod(lambda _: repository))
    assert resolve(31 * 16000, model="large")[1].condition_on_previous_text is expected


@pytest.mark.parametrize("language", ["en", "english", "English"])
def test_explicit_english_language_forms(language):
    assert resolve(31 * 16000, language=language)[1].condition_on_previous_text is False


@pytest.mark.parametrize("language", [None, "fr", "French"])
def test_other_or_automatic_language_settings_are_not_changed(language):
    assert resolve(31 * 16000, language=language)[1].condition_on_previous_text is True


def test_explicit_context_off_is_preserved_for_short_audio():
    requested, effective = resolve(16000, condition=False)
    assert effective is requested
    assert effective.condition_on_previous_text is False


def test_short_audio_with_custom_chunks_keeps_context():
    assert resolve(20 * 16000, chunk=10)[1].condition_on_previous_text is True


def test_legacy_long_form_guard_is_preserved_for_v1():
    # A tiny sample rate keeps this boundary test lightweight.
    inference = object.__new__(FasterWhisperInference)
    params = WhisperParams(model_size="large-v1", lang="en", condition_on_previous_text=True, chunk_length=30)
    _, before = inference.resolve_standard_audio_and_params(np.zeros(1770, dtype=np.float32), params, sampling_rate=1, log_console=False)
    _, after = inference.resolve_standard_audio_and_params(np.zeros(1771, dtype=np.float32), params, sampling_rate=1, log_console=False)
    assert before.condition_on_previous_text is True
    assert after.condition_on_previous_text is False
