# Adapted from https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py

import numpy as np
import pandas as pd
import os
import builtins
import importlib
import inspect
import threading
from contextlib import contextmanager
from typing import Optional, Union
import torch
import huggingface_hub

from modules.whisper.data_classes import *
from modules.utils.paths import DIARIZATION_MODELS_DIR
from modules.utils.torch_compat import enable_torchaudio_2_9_compat, torch_load_safe_globals
from modules.diarize.audio_loader import load_audio, SAMPLE_RATE

enable_torchaudio_2_9_compat()


DEFAULT_DIARIZATION_REPO_ID = "MonsterMMORPG/Wan_GGUF"
DEFAULT_DIARIZATION_SUBFOLDER = "Speaker_Diarization_3_1"
_PYANNOTE_LOAD_LOCK = threading.RLock()


def _is_pipeline_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.yaml"))


def _find_local_pipeline_dir(
    model_name: str,
    cache_dir: str,
    subfolder: str,
) -> Optional[str]:
    candidates = []

    if model_name:
        candidates.append(model_name)

    if cache_dir:
        candidates.append(cache_dir)
        candidates.append(os.path.join(cache_dir, subfolder))

    for candidate in candidates:
        if _is_pipeline_dir(candidate):
            return candidate

    return None


def _resolve_pipeline_dir(
    model_name: str,
    cache_dir: str,
    use_auth_token: Optional[str] = None,
    subfolder: str = DEFAULT_DIARIZATION_SUBFOLDER,
) -> str:
    """
    Resolve a pyannote Pipeline directory.

    Resolution order:
    - an explicit local pipeline directory in `model_name`
    - an offline bundle stored in `cache_dir` or `cache_dir/subfolder`
    - Hugging Face cache/download fallback for repo ids
    """
    local_pipeline_dir = _find_local_pipeline_dir(model_name, cache_dir, subfolder)
    if local_pipeline_dir is not None:
        return local_pipeline_dir

    repo_id = model_name

    # Prefer using existing cache without hitting the network, then fallback to download.
    snapshot_path: Optional[str] = None
    try:
        snapshot_path = huggingface_hub.snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            allow_patterns=[f"{subfolder}/**"],
            token=use_auth_token or None,
            local_files_only=True,
        )
    except TypeError:
        # Some older huggingface_hub versions don't support local_files_only/token/allow_patterns.
        snapshot_path = None
    except Exception:
        snapshot_path = None

    if snapshot_path is None:
        try:
            snapshot_path = huggingface_hub.snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                allow_patterns=[f"{subfolder}/**"],
                token=use_auth_token or None,
            )
        except TypeError:
            # Avoid downloading the whole repo (it may contain very large GGUF files).
            # Minimal fallback: download just the pipeline config and use its directory.
            config_path = huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename=f"{subfolder}/config.yaml",
                cache_dir=cache_dir,
                token=use_auth_token or None,
            )
            pipeline_dir = os.path.dirname(config_path)
            if not os.path.isfile(os.path.join(pipeline_dir, "config.yaml")):
                raise FileNotFoundError(
                    f"Diarization pipeline config.yaml not found after hf_hub_download: {pipeline_dir}"
                )
            return pipeline_dir

    pipeline_dir = os.path.join(snapshot_path, subfolder)
    if not _is_pipeline_dir(pipeline_dir):
        raise FileNotFoundError(
            f"Diarization pipeline config.yaml not found after download: {pipeline_dir}"
        )

    return pipeline_dir


def _read_pipeline_params(pipeline_config_path: str) -> dict:
    try:
        import yaml

        with open(pipeline_config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
    except Exception:
        return {}

    pipeline = config.get("pipeline", {})
    if not isinstance(pipeline, dict):
        return {}

    params = pipeline.get("params", {})
    return params if isinstance(params, dict) else {}


@contextmanager
def _optional_nemo_import_disabled(disabled: bool):
    """
    Make pyannote treat NeMo speaker embeddings as unavailable when unused.

    pyannote.audio imports optional NeMo speaker-verification support at module
    import time. In this environment, NeMo's telemetry package can fail import
    with a TypeError instead of ImportError. The bundled diarization pipeline
    uses WeSpeaker embeddings, so disabling that optional import avoids an
    unrelated initialization failure without affecting Canary-Qwen's NeMo path.
    """
    if not disabled:
        yield
        return

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and (
            name == "nemo.collections.asr.models"
            or name.startswith("nemo.collections.asr.models.")
        ):
            raise ImportError("Optional NeMo speaker embedding disabled.")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    try:
        yield
    finally:
        builtins.__import__ = original_import


@contextmanager
def _unused_pyannote_plda_skipped(skip_plda: bool):
    """
    Skip pyannote.audio 4.x PLDA loading when clustering does not use PLDA.

    The offline bundle uses AgglomerativeClustering. pyannote.audio 4.x still
    loads the default VBx PLDA dependency before checking the clustering type,
    which triggers a gated Hugging Face download. Returning None is safe for
    non-VBx clustering because that value is never consumed.
    """
    if not skip_plda:
        yield
        return

    speaker_diarization = importlib.import_module(
        "pyannote.audio.pipelines.speaker_diarization"
    )
    original_get_plda = speaker_diarization.get_plda

    def get_unused_plda(*args, **kwargs):
        return None

    speaker_diarization.get_plda = get_unused_plda
    try:
        yield
    finally:
        speaker_diarization.get_plda = original_get_plda


def _load_pyannote_pipeline(
    pipeline_cls,
    pipeline_config_path: str,
    use_auth_token: Optional[str],
    cache_dir: str,
):
    """
    Load a pyannote pipeline across pyannote.audio API versions.

    pyannote.audio 4.x renamed `use_auth_token` to `token`. Older versions still
    require `use_auth_token`, so choose the supported keyword at runtime.
    """
    kwargs = {}
    try:
        parameters = inspect.signature(pipeline_cls.from_pretrained).parameters
    except (TypeError, ValueError):
        parameters = {}

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    def accepts(name: str) -> bool:
        return accepts_kwargs or name in parameters

    if accepts("cache_dir"):
        kwargs["cache_dir"] = cache_dir

    if use_auth_token:
        if accepts("token"):
            kwargs["token"] = use_auth_token
        elif accepts("use_auth_token"):
            kwargs["use_auth_token"] = use_auth_token

    pipeline_params = _read_pipeline_params(pipeline_config_path)
    embedding_config = str(pipeline_params.get("embedding", "")).lower()
    clustering = pipeline_params.get("clustering", "VBxClustering")
    disable_optional_nemo = "nvidia" not in embedding_config
    skip_plda = clustering != "VBxClustering"

    with _PYANNOTE_LOAD_LOCK:
        with _optional_nemo_import_disabled(disable_optional_nemo):
            with _unused_pyannote_plda_skipped(skip_plda):
                pipeline = pipeline_cls.from_pretrained(pipeline_config_path, **kwargs)

    # pyannote.audio 4.x defaults to returning DiarizeOutput, while this app
    # expects the legacy Annotation result used by pyannote.audio 3.x.
    if hasattr(pipeline, "legacy"):
        pipeline.legacy = True

    return pipeline


def _itertracks_compatible(diarization_result):
    if hasattr(diarization_result, "itertracks"):
        return diarization_result.itertracks(yield_label=True)

    speaker_diarization = getattr(diarization_result, "speaker_diarization", None)
    if speaker_diarization is not None and hasattr(speaker_diarization, "itertracks"):
        return speaker_diarization.itertracks(yield_label=True)

    raise TypeError(
        "Unsupported diarization output type: "
        f"{type(diarization_result).__name__}"
    )


class DiarizationPipeline:
    def __init__(
        self,
        model_name: str = DEFAULT_DIARIZATION_REPO_ID,
        cache_dir: str = DIARIZATION_MODELS_DIR,
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        from pyannote.audio import Pipeline

        if isinstance(device, str):
            device = torch.device(device)
        pipeline_dir = _resolve_pipeline_dir(
            model_name=model_name,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            subfolder=DEFAULT_DIARIZATION_SUBFOLDER,
        )
        pipeline_config_path = os.path.join(pipeline_dir, "config.yaml")
        with torch_load_safe_globals():
            self.model = _load_pyannote_pipeline(
                Pipeline,
                pipeline_config_path,
                use_auth_token=use_auth_token,
                cache_dir=cache_dir,
            ).to(device)

    def __call__(self, audio: Union[str, np.ndarray], min_speakers=None, max_speakers=None):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }
        segments = self.model(audio_data, min_speakers=min_speakers, max_speakers=max_speakers)
        diarize_df = pd.DataFrame(_itertracks_compatible(segments), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        return diarize_df


def assign_word_speakers(diarize_df, transcript_result, fill_nearest=False):
    transcript_segments = transcript_result["segments"]
    if transcript_segments and isinstance(transcript_segments[0], Segment):
        transcript_segments = [seg.model_dump() for seg in transcript_segments]
    for seg in transcript_segments:
        # assign speaker to segment (if any)
        diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'],
                                                                                            seg['start'])
        diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - np.minimum(diarize_df['start'], seg['start'])

        intersected = diarize_df[diarize_df["intersection"] > 0]

        speaker = None
        if len(intersected) > 0:
            # Choosing most strong intersection
            speaker = intersected.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
        elif fill_nearest:
            # Otherwise choosing closest
            speaker = diarize_df.sort_values(by=["intersection"], ascending=False)["speaker"].values[0]

        if speaker is not None:
            seg["speaker"] = speaker

        # assign speaker to words
        if 'words' in seg and seg['words'] is not None:
            for word in seg['words']:
                if 'start' in word:
                    diarize_df['intersection'] = np.minimum(diarize_df['end'], word['end']) - np.maximum(
                        diarize_df['start'], word['start'])
                    diarize_df['union'] = np.maximum(diarize_df['end'], word['end']) - np.minimum(diarize_df['start'],
                                                                                                  word['start'])

                    intersected = diarize_df[diarize_df["intersection"] > 0]

                    word_speaker = None
                    if len(intersected) > 0:
                        # Choosing most strong intersection
                        word_speaker = \
                            intersected.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                    elif fill_nearest:
                        # Otherwise choosing closest
                        word_speaker = diarize_df.sort_values(by=["intersection"], ascending=False)["speaker"].values[0]

                    if word_speaker is not None:
                        word["speaker"] = word_speaker

    return {"segments": transcript_segments}


class DiarizationSegment:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker
