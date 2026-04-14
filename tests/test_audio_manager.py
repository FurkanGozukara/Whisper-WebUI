from types import SimpleNamespace

from modules.utils.audio_manager import coerce_audio_input_path


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
