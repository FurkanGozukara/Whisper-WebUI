# Adapted from https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py

import numpy as np
import pandas as pd
import os
from pyannote.audio import Pipeline
from typing import Optional, Union
import torch
import huggingface_hub

from modules.whisper.data_classes import *
from modules.utils.paths import DIARIZATION_MODELS_DIR
from modules.diarize.audio_loader import load_audio, SAMPLE_RATE


DEFAULT_DIARIZATION_REPO_ID = "MonsterMMORPG/Wan_GGUF"
DEFAULT_DIARIZATION_SUBFOLDER = "Speaker_Diarization_3_1"


def _resolve_pipeline_dir(
    model_name: str,
    cache_dir: str,
    use_auth_token: Optional[str] = None,
    subfolder: str = DEFAULT_DIARIZATION_SUBFOLDER,
) -> str:
    """
    Resolve a pyannote Pipeline directory.

    - If `model_name` is a local path, it is returned as-is.
    - Otherwise, `model_name` is treated as a HF repo id and `subfolder` is downloaded.
    """
    if os.path.isdir(model_name):
        return model_name

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
    if not os.path.isdir(pipeline_dir):
        raise FileNotFoundError(
            f"Diarization pipeline subfolder not found after download: {pipeline_dir}"
        )

    config_path = os.path.join(pipeline_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Diarization pipeline config.yaml not found: {config_path}"
        )

    return pipeline_dir


class DiarizationPipeline:
    def __init__(
        self,
        model_name: str = DEFAULT_DIARIZATION_REPO_ID,
        cache_dir: str = DIARIZATION_MODELS_DIR,
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        pipeline_dir = _resolve_pipeline_dir(
            model_name=model_name,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            subfolder=DEFAULT_DIARIZATION_SUBFOLDER,
        )
        self.model = Pipeline.from_pretrained(
            pipeline_dir,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir
        ).to(device)

    def __call__(self, audio: Union[str, np.ndarray], min_speakers=None, max_speakers=None):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }
        segments = self.model(audio_data, min_speakers=min_speakers, max_speakers=max_speakers)
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
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
