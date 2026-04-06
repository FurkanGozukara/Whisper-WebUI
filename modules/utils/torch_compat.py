"""
PyTorch compatibility helpers.

Why this exists:
- In PyTorch 2.6, `torch.load(..., weights_only=...)` changed default to `weights_only=True`.
- Some popular checkpoints (including OpenAI Whisper `.pt` files) contain simple metadata
  objects like `torch.__version__` which is an instance of `torch.torch_version.TorchVersion`.
- With `weights_only=True`, PyTorch uses a restricted unpickler and will error unless such
  globals are allowlisted.

This module makes model loading robust across PyTorch versions without broadly disabling
 the security benefits of `weights_only=True`.

- In TorchAudio 2.9, legacy top-level APIs such as `torchaudio.info`,
  `torchaudio.AudioMetaData`, and backend helpers were removed, while `torchaudio.load`
  now requires `torchcodec`. This app still relies on the legacy API surface.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional, Type


_TORCHAUDIO_COMPAT_BACKEND = "soundfile"


def _get_torchversion_cls() -> Optional[Type[object]]:
    """Best-effort lookup for TorchVersion class across torch versions."""
    try:
        import torch  # noqa: F401
    except Exception:
        return None

    try:
        # Preferred location recommended by the PyTorch error message.
        from torch.torch_version import TorchVersion  # type: ignore

        return TorchVersion
    except Exception:
        pass

    try:
        import torch

        return getattr(torch, "TorchVersion", None)
    except Exception:
        return None


def enable_torch_2_6_weights_only_compat() -> None:
    """
    Globally allowlist known-safe globals needed to load common checkpoints under
    `torch.load(weights_only=True)` (PyTorch 2.6+ default).
    """
    try:
        import torch

        serialization = getattr(torch, "serialization", None)
        add_safe_globals = getattr(serialization, "add_safe_globals", None)
        if add_safe_globals is None:
            return

        torchversion_cls = _get_torchversion_cls()
        if torchversion_cls is None:
            return

        # Idempotent in practice; safe to call multiple times.
        add_safe_globals([torchversion_cls])
    except Exception:
        # Best-effort: never crash the app just because compat patching failed.
        return


@contextmanager
def torch_load_safe_globals() -> Iterator[None]:
    """
    Context manager that temporarily allowlists known-safe globals for weights-only loading.

    Use this around code paths that may trigger `torch.load()` internally (e.g. `whisper.load_model`)
    on PyTorch 2.6+, to avoid the "Weights only load failed" error.
    """
    try:
        import torch

        serialization = getattr(torch, "serialization", None)
        safe_globals_cm = getattr(serialization, "safe_globals", None)
        torchversion_cls = _get_torchversion_cls()

        if safe_globals_cm is None or torchversion_cls is None:
            # Fallback to a global patch (or a no-op on older torch versions).
            enable_torch_2_6_weights_only_compat()
            yield
            return

        with safe_globals_cm([torchversion_cls]):
            yield
    except Exception:
        # Best-effort: do not block execution if torch isn't available for some reason.
        yield


def _reset_seekable_stream(uri: object) -> None:
    """Best-effort rewind for file-like inputs used with audio loaders."""
    seek = getattr(uri, "seek", None)
    if callable(seek):
        try:
            seek(0)
        except Exception:
            pass


def _bits_per_sample_from_subtype(subtype: Optional[str]) -> int:
    """Infer bit depth from a soundfile subtype string."""
    if not subtype:
        return 0

    subtype = subtype.upper()
    digits = "".join(ch for ch in subtype if ch.isdigit())
    if digits:
        try:
            return int(digits)
        except ValueError:
            pass

    if subtype == "FLOAT":
        return 32
    if subtype == "DOUBLE":
        return 64
    return 0


def _torchaudio_info_with_soundfile(uri: object, backend: Optional[str] = None):
    """Compatibility replacement for `torchaudio.info` using soundfile."""
    del backend

    import soundfile as sf
    import torchaudio

    try:
        info = sf.info(uri)
        return torchaudio.AudioMetaData(
            sample_rate=int(info.samplerate),
            num_frames=int(info.frames),
            num_channels=int(info.channels),
            bits_per_sample=_bits_per_sample_from_subtype(info.subtype),
            encoding=info.subtype or "",
        )
    finally:
        _reset_seekable_stream(uri)


def _torchaudio_info_with_wave(uri: object, backend: Optional[str] = None):
    """WAV-only fallback for environments where soundfile probing fails."""
    del backend

    import os
    import wave
    import torchaudio

    wav_uri = os.fspath(uri) if isinstance(uri, os.PathLike) else uri
    try:
        with wave.open(wav_uri, "rb") as wav_file:
            return torchaudio.AudioMetaData(
                sample_rate=int(wav_file.getframerate()),
                num_frames=int(wav_file.getnframes()),
                num_channels=int(wav_file.getnchannels()),
                bits_per_sample=int(wav_file.getsampwidth() * 8),
                encoding="PCM_S",
            )
    finally:
        _reset_seekable_stream(uri)


def _torchaudio_info_compat(uri: object, backend: Optional[str] = None):
    """Best-effort legacy `torchaudio.info` implementation."""
    try:
        return _torchaudio_info_with_soundfile(uri, backend=backend)
    except Exception:
        return _torchaudio_info_with_wave(uri, backend=backend)


def _torchaudio_load_compat(
    uri,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: Optional[str] = None,
    buffer_size: int = 4096,
    backend: Optional[str] = None,
):
    """Compatibility replacement for `torchaudio.load` using soundfile."""
    del format, buffer_size, backend

    import numpy as np
    import soundfile as sf
    import torch

    dtype = "float32"
    if not normalize:
        try:
            audio_info = _torchaudio_info_compat(uri)
            if 0 < audio_info.bits_per_sample <= 16:
                dtype = "int16"
            elif 16 < audio_info.bits_per_sample <= 32:
                dtype = "int32"
        except Exception:
            dtype = "float32"

    try:
        waveform, sample_rate = sf.read(
            uri,
            start=frame_offset,
            frames=num_frames,
            dtype=dtype,
            always_2d=True,
        )
    finally:
        _reset_seekable_stream(uri)

    if channels_first:
        waveform = np.ascontiguousarray(waveform.T)
    else:
        waveform = np.ascontiguousarray(waveform)

    return torch.from_numpy(waveform), int(sample_rate)


def _soundfile_subtype_from_params(
    encoding: Optional[str], bits_per_sample: Optional[int]
) -> Optional[str]:
    """Map legacy torchaudio save params to soundfile subtype names."""
    normalized_encoding = encoding.upper() if encoding else None

    if normalized_encoding == "PCM_S" and bits_per_sample in {8, 16, 24, 32}:
        return f"PCM_{bits_per_sample}"
    if normalized_encoding == "PCM_U" and bits_per_sample == 8:
        return "PCM_U8"
    if normalized_encoding in {"FLOAT", "DOUBLE", "ULAW", "ALAW", "VORBIS"}:
        return normalized_encoding
    if bits_per_sample in {8, 16, 24, 32}:
        return f"PCM_{bits_per_sample}"
    return None


def _torchaudio_save_compat(
    uri,
    src,
    sample_rate: int,
    channels_first: bool = True,
    format: Optional[str] = None,
    encoding: Optional[str] = None,
    bits_per_sample: Optional[int] = None,
    buffer_size: int = 4096,
    backend: Optional[str] = None,
    compression: Optional[float | int] = None,
) -> None:
    """Compatibility replacement for `torchaudio.save` using soundfile."""
    del buffer_size, backend, compression

    import numpy as np
    import soundfile as sf
    import torch

    if isinstance(src, torch.Tensor):
        waveform = src.detach().cpu().numpy()
    else:
        waveform = np.asarray(src)

    if waveform.ndim == 2 and channels_first:
        waveform = waveform.T

    sf.write(
        uri,
        waveform,
        sample_rate,
        format=format,
        subtype=_soundfile_subtype_from_params(encoding, bits_per_sample),
    )


def enable_torchaudio_2_9_compat() -> None:
    """
    Restore the legacy TorchAudio API surface expected by this app and pyannote.audio
    when running on TorchAudio 2.9+ without torchcodec.
    """
    try:
        import importlib.util
        from typing import NamedTuple

        import torchaudio
    except Exception:
        return

    needs_legacy_api = any(
        not hasattr(torchaudio, attribute)
        for attribute in ("AudioMetaData", "info", "list_audio_backends")
    )
    torchcodec_missing = importlib.util.find_spec("torchcodec") is None

    if not needs_legacy_api and not torchcodec_missing:
        return

    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData(NamedTuple):
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int
            encoding: str

        torchaudio.AudioMetaData = AudioMetaData

    global _TORCHAUDIO_COMPAT_BACKEND

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: _TORCHAUDIO_COMPAT_BACKEND

    if not hasattr(torchaudio, "set_audio_backend"):
        def _set_audio_backend(backend: Optional[str]) -> None:
            global _TORCHAUDIO_COMPAT_BACKEND

            if backend is None:
                _TORCHAUDIO_COMPAT_BACKEND = "soundfile"
                return

            available_backends = torchaudio.list_audio_backends()
            if backend not in available_backends:
                raise RuntimeError(
                    f"Unsupported torchaudio backend '{backend}'. "
                    f"Available backends: {available_backends}"
                )
            _TORCHAUDIO_COMPAT_BACKEND = backend

        torchaudio.set_audio_backend = _set_audio_backend

    if not hasattr(torchaudio, "info"):
        torchaudio.info = _torchaudio_info_compat

    if torchcodec_missing:
        torchaudio.load = _torchaudio_load_compat
        torchaudio.save = _torchaudio_save_compat

    exported = getattr(torchaudio, "__all__", None)
    if isinstance(exported, list):
        for name in (
            "AudioMetaData",
            "info",
            "list_audio_backends",
            "get_audio_backend",
            "set_audio_backend",
        ):
            if name not in exported:
                exported.append(name)


