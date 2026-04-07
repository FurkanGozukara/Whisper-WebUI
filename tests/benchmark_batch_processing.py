import argparse
import os
import subprocess
import sys
import tempfile
from difflib import SequenceMatcher

import gradio as gr
from faster_whisper.audio import decode_audio

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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


def load_audio_excerpt(file_path: str, seconds: int, target_sample_rate: int = 16000):
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


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[-1][-1] / len(ref_words)


def select_model(available_models, requested_model: str | None):
    if requested_model:
        return requested_model

    for candidate in ("large-v3", "large-v1", "large-v2", "medium", "small", "tiny"):
        if candidate in available_models:
            return candidate
    return list(available_models)[0]


def run_transcription(inferencer, audio_excerpt, model_size: str, chunk_length: int, batch_size: int):
    pipeline_params = TranscriptionPipelineParams(
        whisper=WhisperParams(
            model_size=model_size,
            compute_type=inferencer.current_compute_type,
            chunk_length=chunk_length,
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark faster-whisper chunk batching on a real media file.")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(WEBUI_DIR, "a.mp4"),
        help="Path to the media file to benchmark.",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=60,
        help="How many seconds to read from the beginning of the file.",
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=15,
        help="Audio window size in seconds.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional faster-whisper model name.",
    )
    parser.add_argument(
        "--max-wer",
        type=float,
        default=0.02,
        help="Maximum acceptable word error rate when comparing batch_size=1 and batch_size=4.",
    )
    parser.add_argument(
        "--min-char-ratio",
        type=float,
        default=0.95,
        help="Minimum acceptable character-level similarity ratio between transcripts.",
    )
    args = parser.parse_args()

    inferencer = WhisperFactory.create_whisper_inference(
        whisper_type=WhisperImpl.FASTER_WHISPER.value,
    )
    model_size = select_model(inferencer.available_models, args.model)
    audio_excerpt = load_audio_excerpt(args.input, seconds=args.seconds)
    cached_defaults = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)

    try:
        batch1_segments, batch1_elapsed = run_transcription(
            inferencer=inferencer,
            audio_excerpt=audio_excerpt,
            model_size=model_size,
            chunk_length=args.chunk_length,
            batch_size=1,
        )
        batch4_segments, batch4_elapsed = run_transcription(
            inferencer=inferencer,
            audio_excerpt=audio_excerpt,
            model_size=model_size,
            chunk_length=args.chunk_length,
            batch_size=4,
        )
    finally:
        inferencer.offload()
        save_yaml(cached_defaults, DEFAULT_PARAMETERS_CONFIG_PATH)

    normalized_batch1 = normalize_segments(batch1_segments)
    normalized_batch4 = normalize_segments(batch4_segments)
    exact_match = normalized_batch1 == normalized_batch4
    speedup = batch1_elapsed / max(batch4_elapsed, 1e-6)
    batch1_text = " ".join(segment[2] for segment in normalized_batch1)
    batch4_text = " ".join(segment[2] for segment in normalized_batch4)
    transcript_wer = word_error_rate(batch1_text, batch4_text)
    char_ratio = SequenceMatcher(None, batch1_text, batch4_text).ratio()
    parity_ok = (
        len(normalized_batch1) == len(normalized_batch4)
        and transcript_wer <= args.max_wer
        and char_ratio >= args.min_char_ratio
    )

    print(f"Input: {args.input}")
    print(f"Excerpt length: {args.seconds}s")
    print(f"Model: {model_size}")
    print(f"Chunk length: {args.chunk_length}s")
    print(f"Effective parallel span at batch_size=4: {args.chunk_length * 4}s")
    print(f"Batch size 1: {batch1_elapsed:.2f}s")
    print(f"Batch size 4: {batch4_elapsed:.2f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Exact segment match: {exact_match}")
    print(f"Segment count batch_size=1: {len(normalized_batch1)}")
    print(f"Segment count batch_size=4: {len(normalized_batch4)}")
    print(f"Transcript WER: {transcript_wer:.6f}")
    print(f"Character similarity ratio: {char_ratio:.6f}")
    print(f"Parity OK: {parity_ok}")

    if not parity_ok:
        raise SystemExit("Batch output drift exceeded the configured parity thresholds.")


if __name__ == "__main__":
    main()
