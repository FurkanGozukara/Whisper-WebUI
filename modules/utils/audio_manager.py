from typing import Optional, Union, Any
import soundfile as sf
import os
import numpy as np

from modules.utils.files_manager import is_video
from modules.utils.logger import get_logger

logger = get_logger()


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
