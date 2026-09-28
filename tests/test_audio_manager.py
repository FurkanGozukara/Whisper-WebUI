from types import SimpleNamespace

import numpy as np
import soundfile as sf

from modules.utils.audio_manager import coerce_audio_input_path, is_digital_silence


def test_coerce_audio_input_path_accepts_common_gradio_shapes(tmp_path):
    audio_file = tmp_path / "mic.wav"
    audio_file.write_bytes(b"")

    assert coerce_audio_input_path(str(audio_file)) == str(audio_file)
    assert coerce_audio_input_path({"path": str(audio_file)}) == str(audio_file)
    assert coerce_audio_input_path({"name": str(audio_file)}) == str(audio_file)
    assert coerce_audio_input_path(SimpleNamespace(path=str(audio_file))) == str(audio_file)
    assert coerce_audio_input_path(SimpleNamespace(name=str(audio_file))) == str(audio_file)


def test_coerce_audio_input_path_returns_none_for_missing_payload():
    assert coerce_audio_input_path(None) is None
    assert coerce_audio_input_path("") is None
    assert coerce_audio_input_path({}) is None


def test_digital_silence_preserves_even_very_quiet_nonzero_audio(tmp_path):
    silent = np.zeros(16000, dtype=np.float32)
    assert is_digital_silence(silent)
    silent[-1] = 1e-8
    assert not is_digital_silence(silent)
    path = tmp_path / "quiet.wav"
    sf.write(path, silent, 16000, subtype="FLOAT")
    assert not is_digital_silence(path)
    sf.write(path, np.zeros(16000), 16000)
    assert is_digital_silence(path)


def test_digital_silence_does_not_hide_invalid_audio(tmp_path):
    assert not is_digital_silence(np.array([], dtype=np.float32))
    assert not is_digital_silence(np.array([np.nan]))
    path = tmp_path / "corrupt.wav"
    path.write_text("not audio", encoding="utf-8")
    assert not is_digital_silence(path)
