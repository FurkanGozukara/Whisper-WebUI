from typing import Optional, Union, Any
import soundfile as sf
import os
import numpy as np

from modules.utils.files_manager import is_video
from modules.utils.logger import get_logger

logger = get_logger()


def decode_audio(input_file: Any, sampling_rate: int = 16000) -> np.ndarray:
    """faster_whisper.audio.decode_audio (PyAV, mono, s16 at sampling_rate, as float32) without the full
    garbage collection it runs after every file for a resampler leak that PyAV no longer has: in the app's
    process that collection took about 0.26 s per file, more than decoding a 30 second clip."""
    import io

    import av
    from faster_whisper.audio import _group_frames, _ignore_invalid_frames, _resample_frames

    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    raw_buffer = io.BytesIO()
    dtype = None
    with av.open(input_file, mode="r", metadata_errors="ignore") as container:
        frames = _resample_frames(_group_frames(_ignore_invalid_frames(container.decode(audio=0)), 500000), resampler)
        for frame in frames:
            array = frame.to_ndarray()
            dtype = array.dtype
            raw_buffer.write(array)
    del resampler
    if dtype is None:
        return np.zeros(0, dtype=np.float32)
    audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)
    return audio.astype(np.float32) / 32768.0


def is_digital_silence(audio: Any) -> bool:
    """Recognize exactly zero samples without classifying quiet speech as silence.

    Stop at the first nonzero frame for ordinary media. Do not decode an entire
    long speech recording just to check for silence, and do not mask read errors.
    """
    if isinstance(audio, np.ndarray):
        return audio.size > 0 and not np.any(audio)
    path = coerce_audio_input_path(audio)
    if path is None:
        return False
    try:
        import av

        has_samples = False
        with av.open(path, mode="r", metadata_errors="ignore") as container:
            for frame in container.decode(audio=0):
                samples = frame.to_ndarray()
                if np.any(samples):
                    return False
                has_samples = has_samples or samples.size > 0
        return has_samples
    except Exception:
        # Normal validation/inference reports unreadable inputs to the user.
        return False


def coerce_audio_input_path(audio: Any) -> Optional[str]:
    """Best-effort normalization for Gradio audio inputs."""
    if audio is None or isinstance(audio, np.ndarray):
        return None

    if isinstance(audio, os.PathLike):
        return os.fspath(audio)

    if isinstance(audio, str):
        audio = audio.strip()
        return audio or None

    if isinstance(audio, dict):
        for key in ("path", "name"):
            value = audio.get(key)
            if isinstance(value, (str, os.PathLike)):
                value = os.fspath(value).strip()
                if value:
                    return value
        return None

    for attr in ("path", "name"):
        value = getattr(audio, attr, None)
        if isinstance(value, (str, os.PathLike)):
            value = os.fspath(value).strip()
            if value:
                return value

    return None


def validate_audio(audio: Optional[Union[str, Any]] = None):
    """Validate audio file and check if it's corrupted"""
    if isinstance(audio, np.ndarray):
        return True

    audio_path = coerce_audio_input_path(audio)
    if audio_path is None:
        logger.info("No audio input was provided.")
        return False

    if not os.path.exists(audio_path):
        logger.info(f"The file {audio_path} does not exist. Please check the path.")
        return False

    try:
        import av

        # Opening the file and decoding its first audio frame tells an unreadable file apart; the full decode
        # done here before was thrown away and repeated by the transcription right after.
        with av.open(audio_path, mode="r", metadata_errors="ignore") as container:
            if not container.streams.audio:
                raise ValueError("the file has no audio stream")
            for _frame in container.decode(audio=0):
                return True
        raise ValueError("the audio stream has no decodable frames")
    except Exception as e:
        logger.info(f"The file {audio_path} is not able to open or corrupted. Please check the file. {e}")
        return False
