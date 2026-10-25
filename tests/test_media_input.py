import asyncio
from fractions import Fraction
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import av
import numpy as np
import pytest

from modules.utils.audio_manager import decode_audio, validate_audio
from modules.utils.media import open_media_input


@pytest.mark.parametrize("version,metadata_options", [
    ("11.0.0", {"metadata_errors": "ignore"}),
    ("18.1.0", {"metadata_errors": "ignore"}),
    ("19.0.0.dev0", {}),
    ("19.0.2", {}),
    ("20.0.0", {}),
])
def test_open_uses_the_matching_pyav_api_once(monkeypatch, version, metadata_options):
    source = BytesIO(b"media")
    opener = Mock(return_value=object())
    monkeypatch.setattr(av, "__version__", version)
    monkeypatch.setattr(av, "open", opener)

    assert open_media_input(source) is opener.return_value
    opener.assert_called_once_with(source, mode="r", **metadata_options)


def write_synthetic_media(path, container_format, codec, with_video=False):
    """Small zero-filled fixtures only; never use a user's recording in these tests."""
    with av.open(str(path), mode="w", format=container_format) as container:
        if with_video:
            video = container.add_stream("mpeg4", rate=10)
            video.width = video.height = 16
            video.pix_fmt = "yuv420p"

        stream = container.add_stream(codec, rate=48000)
        stream.layout = "stereo"
        if codec == "wmav2":
            stream.bit_rate = 128000
        # All streams must exist before muxing starts and writes the header.
        container.start_encoding()
        if with_video:
            frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
            frame.pts = 0
            for packet in video.encode(frame):
                container.mux(packet)
            for packet in video.encode(None):
                container.mux(packet)

        frame = av.AudioFrame.from_ndarray(np.zeros((2, 4800), dtype=np.float32), format="fltp", layout="stereo")
        frame.sample_rate = 48000
        frame.time_base = Fraction(1, 48000)
        frame.pts = 0
        resampler = av.AudioResampler(format=stream.codec_context.format.name, layout="stereo", rate=48000)
        for converted in [*resampler.resample(frame), *resampler.resample(None)]:
            for packet in stream.encode(converted):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


@pytest.fixture(params=[
    ("wav", "wav", "pcm_s16le"),
    ("flac", "flac", "flac"),
    ("mp3", "mp3", "libmp3lame"),
    ("m4a", "mp4", "aac"),
    ("mp4", "mp4", "aac"),
    ("aac", "adts", "aac"),
    ("opus", "ogg", "libopus"),
    ("webm", "webm", "libopus"),
    ("wma", "asf", "wmav2"),
    ("mkv", "matroska", "pcm_s16le"),
], ids=lambda spec: spec[0])
def synthetic_media(request, tmp_path):
    extension, container_format, codec = request.param
    path = tmp_path / f"Türkçe media.{extension}"
    write_synthetic_media(path, container_format, codec, with_video=extension == "mp4")
    return path


def test_original_media_decodes_in_memory_without_conversion(synthetic_media, monkeypatch):
    # Initialize the pinned decoder's helpers before guarding against external tools.
    import faster_whisper.audio
    import subprocess
    import tempfile

    def no_conversion(*args, **kwargs):
        pytest.fail("Opening media must not launch a converter or write a temporary input file")

    monkeypatch.setattr(subprocess, "Popen", no_conversion)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", no_conversion)
    assert validate_audio({"path": str(synthetic_media)})
    samples = decode_audio(synthetic_media)
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert samples.size > 0
    assert np.isfinite(samples).all()

    # File-like uploads use exactly the same path through the decoder.
    samples = decode_audio(BytesIO(synthetic_media.read_bytes()), sampling_rate=8000)
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert samples.size > 0


def test_validation_reads_only_the_first_audio_frame(tmp_path, monkeypatch):
    from modules.utils import audio_manager

    path = tmp_path / "input.mp4"
    path.touch()

    def frames(*args, **kwargs):
        assert args == () and kwargs == {"audio": 0}
        yield object()
        pytest.fail("Validation must not decode the rest of the file")

    container = SimpleNamespace(streams=SimpleNamespace(audio=[object()]), decode=frames)
    context = Mock()
    context.__enter__ = Mock(return_value=container)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(audio_manager, "open_media_input", lambda path: context)

    assert validate_audio(path)
    context.__exit__.assert_called_once()


@pytest.mark.parametrize("error", [TypeError("decoder API mismatch"), RuntimeError("decoder setup failed")])
def test_validation_does_not_label_application_failures_as_corrupt_files(tmp_path, monkeypatch, error):
    from modules.utils import audio_manager

    path = tmp_path / "valid.mp4"
    path.touch()
    opener = Mock(side_effect=error)
    monkeypatch.setattr(audio_manager, "open_media_input", opener)
    with pytest.raises(type(error), match=str(error)):
        validate_audio(path)
    opener.assert_called_once()


def test_invalid_media_still_fails_validation(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a media container")
    assert not validate_audio(path)
    assert not validate_audio(tmp_path / "missing.mp4")


def test_rest_upload_uses_the_compatible_decoder(tmp_path):
    from fastapi import UploadFile
    from backend.common.audio import read_audio

    path = tmp_path / "upload.mp4"
    write_synthetic_media(path, "mp4", "aac", with_video=True)
    upload = UploadFile(file=BytesIO(path.read_bytes()), filename=path.name)
    samples, info = asyncio.run(read_audio(file=upload))
    assert samples.dtype == np.float32
    assert samples.size > 0
    assert info.duration > 0


def test_music_separator_pyav_fallback_reads_original_media(synthetic_media, monkeypatch):
    from modules.uvr import music_separator

    def soundfile_cannot_read(*args, **kwargs):
        raise RuntimeError("Exercise the PyAV fallback for this container")

    monkeypatch.setattr(music_separator.sf, "read", soundfile_cannot_read)
    samples, sample_rate = music_separator.MusicSeparator.read_audio_file(str(synthetic_media))
    assert samples.dtype == np.float32
    assert samples.shape[0] == 2
    assert samples.shape[1] > 0
    assert sample_rate == 48000
