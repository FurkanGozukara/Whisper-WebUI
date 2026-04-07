import os
import subprocess
import tempfile

import gradio as gr
import pytest
from faster_whisper.audio import decode_audio

from modules.utils.files_manager import load_yaml, save_yaml
from modules.utils.paths import DEFAULT_PARAMETERS_CONFIG_PATH
from modules.utils.paths import WEBUI_DIR
from modules.whisper.data_classes import (
    BGMSeparationParams,
    DiarizationParams,
    TranscriptionPipelineParams,
    VadParams,
    WhisperImpl,
    WhisperParams,
)
from modules.whisper.whisper_factory import WhisperFactory


REAL_TEST_FILE_PATH = os.path.join(WEBUI_DIR, "a.mp4")
RUN_REAL_TESTS = os.getenv("WHISPERWEBUI_RUN_REAL_TESTS") == "1"


def load_audio_excerpt(file_path: str, seconds: int = 60, target_sample_rate: int = 16000):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                "0",
                "-i",
                file_path,
                "-t",
                str(seconds),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(target_sample_rate),
                temp_path,
            ],
            check=True,
        )
        return decode_audio(temp_path, sampling_rate=target_sample_rate)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def normalize_segments(segments):
    return [
        (
            round(segment.start or 0.0, 3),
            round(segment.end or 0.0, 3),
            " ".join((segment.text or "").split()),
        )
        for segment in segments
        if (segment.text or "").strip()
    ]


def select_model(available_models):
    for candidate in ("large-v3", "large-v1", "large-v2", "medium", "small", "tiny"):
        if candidate in available_models:
            return candidate
    return list(available_models)[0]


def run_excerpt_transcription(inferencer, audio_excerpt, model_size: str, batch_size: int):
    pipeline_params = TranscriptionPipelineParams(
        whisper=WhisperParams(
            model_size=model_size,
            compute_type=inferencer.current_compute_type,
            chunk_length=15,
            use_batched_inference=True,
            batch_size=batch_size,
            enable_offload=False,
        ),
        vad=VadParams(vad_filter=False),
        bgm_separation=BGMSeparationParams(is_separate_bgm=False),
        diarization=DiarizationParams(is_diarize=False),
    ).to_list()

    return inferencer.run(
        audio_excerpt,
        gr.Progress(),
        "SRT",
        False,
        None,
        *pipeline_params,
    )


@pytest.mark.skipif(not os.path.exists(REAL_TEST_FILE_PATH), reason="a.mp4 is not available in the repo root.")
@pytest.mark.skipif(not RUN_REAL_TESTS, reason="Set WHISPERWEBUI_RUN_REAL_TESTS=1 to run the real batching parity test.")
def test_real_batch_processing_matches_batch_size_one():
    inferencer = WhisperFactory.create_whisper_inference(
        whisper_type=WhisperImpl.FASTER_WHISPER.value,
    )
    model_size = select_model(inferencer.available_models)
    audio_excerpt = load_audio_excerpt(REAL_TEST_FILE_PATH, seconds=60)
    cached_defaults = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)

    try:
        batch1_segments, batch1_elapsed = run_excerpt_transcription(
            inferencer=inferencer,
            audio_excerpt=audio_excerpt,
            model_size=model_size,
            batch_size=1,
        )
        batch4_segments, batch4_elapsed = run_excerpt_transcription(
            inferencer=inferencer,
            audio_excerpt=audio_excerpt,
            model_size=model_size,
            batch_size=4,
        )
    finally:
        inferencer.offload()
        save_yaml(cached_defaults, DEFAULT_PARAMETERS_CONFIG_PATH)

    normalized_batch1 = normalize_segments(batch1_segments)
    normalized_batch4 = normalize_segments(batch4_segments)

    print(
        f"\nReal batch benchmark on first 60s of a.mp4 with model {model_size}: "
        f"batch_size=1 -> {batch1_elapsed:.2f}s, "
        f"batch_size=4 -> {batch4_elapsed:.2f}s, "
        f"speedup={batch1_elapsed / max(batch4_elapsed, 1e-6):.2f}x"
    )

    assert normalized_batch1
    assert normalized_batch1 == normalized_batch4
