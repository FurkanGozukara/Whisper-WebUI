from typing import Optional, Union, Any
import soundfile as sf
import os
import numpy as np

from modules.utils.files_manager import is_video
from modules.utils.logger import get_logger

logger = get_logger()


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
        from faster_whisper.audio import decode_audio

        decode_audio(audio_path)
        return True
    except Exception as e:
        logger.info(f"The file {audio_path} is not able to open or corrupted. Please check the file. {e}")
        return False
