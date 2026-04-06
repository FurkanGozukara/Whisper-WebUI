import argparse
import json
import os
import re
import sys
import traceback
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import jiwer
import numpy as np
import torch
from faster_whisper.audio import decode_audio
import whisper

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.utils.subtitle_manager import WriteSRT, generate_file
from modules.whisper.data_classes import (
    BGMSeparationParams,
    DiarizationParams,
    TranscriptionPipelineParams,
    VadParams,
    WhisperImpl,
    WhisperParams,
)
from modules.whisper.whisper_factory import WhisperFactory


REFERENCE_WRITER = WriteSRT(output_dir=".")


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def append_jsonl(path: str, payload: Any) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    value = value.strip(".-")
    return value or "run"


def normalize_app_language(value: Optional[str]) -> Optional[str]:
    if value in (None, "", "auto", "Automatic Detection"):
        return None

    value = str(value).strip().lower()
    if value in whisper.tokenizer.LANGUAGES:
        return whisper.tokenizer.LANGUAGES[value]
    if value in whisper.tokenizer.TO_LANGUAGE_CODE:
        code = whisper.tokenizer.TO_LANGUAGE_CODE[value]
        return whisper.tokenizer.LANGUAGES[code]
    return value


def coerce_optional_float(value: Optional[Any]) -> Optional[float]:
    if value in (None, "", "none", "null"):
        return None
    return float(value)


def coerce_optional_int(value: Optional[Any]) -> Optional[int]:
    if value in (None, "", "none", "null"):
        return None
    return int(value)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ")
    text = text.replace("Â", "")
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def loose_tokenize(text: str) -> List[str]:
    text = normalize_text(text).lower()
    return re.findall(r"[a-z0-9]+(?:[./-][a-z0-9]+)*", text)


def loose_text(text: str) -> str:
    return " ".join(loose_tokenize(text))


def flatten_segments_text(segments) -> str:
    parts = []
    for segment in segments:
        text = getattr(segment, "text", None)
        if text:
            parts.append(text)
    return normalize_text(" ".join(parts))


def word_error_rate(reference: str, hypothesis: str) -> float:
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    return float(jiwer.wer(reference, hypothesis))


def char_error_rate(reference: str, hypothesis: str) -> float:
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    return float(jiwer.cer(reference, hypothesis))


def parse_reference_segments(reference_path: str):
    return REFERENCE_WRITER.to_segments(reference_path)


def merge_dict(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = deepcopy(base)
    if not overrides:
        return merged
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def derive_run_name(spec: Dict[str, Any]) -> str:
    if spec.get("name"):
        return sanitize_name(spec["name"])

    whisper_cfg = spec["whisper"]
    vad_cfg = spec["vad"]
    language = whisper_cfg.get("lang") or "auto"
    parts = [
        whisper_cfg["model_size"],
        f"lang-{language}",
        f"chunk-{whisper_cfg['chunk_length']}",
        f"beam-{whisper_cfg['beam_size']}",
        f"cond-{int(bool(whisper_cfg['condition_on_previous_text']))}",
        f"wt-{int(bool(whisper_cfg['word_timestamps']))}",
        f"vad-{int(bool(vad_cfg['vad_filter']))}",
    ]
    return sanitize_name("__".join(parts))


def build_pipeline_params(spec: Dict[str, Any], default_compute_type: str) -> TranscriptionPipelineParams:
    whisper_cfg = deepcopy(spec["whisper"])
    whisper_cfg["compute_type"] = whisper_cfg.get("compute_type") or default_compute_type

    return TranscriptionPipelineParams(
        whisper=WhisperParams(**whisper_cfg),
        vad=VadParams(**spec["vad"]),
        bgm_separation=BGMSeparationParams(**spec["bgm_separation"]),
        diarization=DiarizationParams(**spec["diarization"]),
    )


def build_base_spec(args: argparse.Namespace) -> Dict[str, Any]:
    language = normalize_app_language(args.lang)
    return {
        "name": args.name,
        "whisper": {
            "model_size": args.model,
            "lang": language,
            "is_translate": False,
            "beam_size": args.beam_size,
            "log_prob_threshold": args.log_prob_threshold,
            "no_speech_threshold": args.no_speech_threshold,
            "compute_type": args.compute_type,
            "best_of": args.best_of,
            "patience": args.patience,
            "condition_on_previous_text": args.condition_on_previous_text,
            "prompt_reset_on_temperature": args.prompt_reset_on_temperature,
            "initial_prompt": args.initial_prompt,
            "temperature": args.temperature,
            "compression_ratio_threshold": args.compression_ratio_threshold,
            "length_penalty": args.length_penalty,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "prefix": args.prefix,
            "suppress_blank": args.suppress_blank,
            "suppress_tokens": args.suppress_tokens,
            "max_initial_timestamp": args.max_initial_timestamp,
            "word_timestamps": args.word_timestamps,
            "prepend_punctuations": args.prepend_punctuations,
            "append_punctuations": args.append_punctuations,
            "max_new_tokens": coerce_optional_int(args.max_new_tokens),
            "chunk_length": args.chunk_length,
            "hallucination_silence_threshold": coerce_optional_float(args.hallucination_silence_threshold),
            "hotwords": args.hotwords,
            "language_detection_threshold": coerce_optional_float(args.language_detection_threshold),
            "language_detection_segments": args.language_detection_segments,
            "batch_size": args.batch_size,
            "enable_offload": False,
        },
        "vad": {
            "vad_filter": args.vad_filter,
            "threshold": args.vad_threshold,
            "min_speech_duration_ms": args.vad_min_speech_duration_ms,
            "max_speech_duration_s": args.vad_max_speech_duration_s,
            "min_silence_duration_ms": args.vad_min_silence_duration_ms,
            "speech_pad_ms": args.vad_speech_pad_ms,
        },
        "bgm_separation": {
            "is_separate_bgm": False,
            "uvr_model_size": "UVR-MDX-NET-Inst_HQ_4",
            "uvr_device": "cuda",
            "segment_size": 256,
            "save_file": False,
            "enable_offload": True,
        },
        "diarization": {
            "is_diarize": False,
            "diarization_device": "cuda",
            "hf_token": "",
            "enable_offload": True,
        },
    }


def load_audio_array(input_path: str, cache_dir: str, sampling_rate: int = 16000) -> np.ndarray:
    ensure_dir(cache_dir)
    cache_file = os.path.join(cache_dir, f"{sanitize_name(Path(input_path).stem)}_{sampling_rate}hz.npy")

    if os.path.exists(cache_file):
        return np.load(cache_file)

    audio = decode_audio(input_path, sampling_rate=sampling_rate)
    audio = np.ascontiguousarray(audio.squeeze(), dtype=np.float32)
    np.save(cache_file, audio)
    return audio


def audio_duration_seconds(audio: np.ndarray, sampling_rate: int = 16000) -> float:
    if audio.size == 0:
        return 0.0
    return float(audio.shape[-1]) / float(sampling_rate)


@dataclass
class EvaluationContext:
    reference_segments: List[Any]
    reference_raw_text: str
    reference_loose_text: str
    reference_duration: float
    audio: np.ndarray
    audio_duration: float


def build_eval_context(reference_path: str, input_path: str, cache_dir: str) -> EvaluationContext:
    reference_segments = parse_reference_segments(reference_path)
    reference_raw_text = flatten_segments_text(reference_segments)
    reference_loose = loose_text(reference_raw_text)
    audio = load_audio_array(input_path=input_path, cache_dir=cache_dir)
    ref_duration = reference_segments[-1].end if reference_segments else 0.0
    return EvaluationContext(
        reference_segments=reference_segments,
        reference_raw_text=reference_raw_text,
        reference_loose_text=reference_loose,
        reference_duration=float(ref_duration or 0.0),
        audio=audio,
        audio_duration=audio_duration_seconds(audio),
    )


def evaluate_prediction(reference_raw: str, reference_loose: str, predicted_segments) -> Dict[str, Any]:
    predicted_raw = flatten_segments_text(predicted_segments)
    predicted_loose = loose_text(predicted_raw)
    predicted_duration = float(predicted_segments[-1].end or 0.0) if predicted_segments else 0.0

    wer = word_error_rate(reference_loose, predicted_loose)
    cer = char_error_rate(reference_loose, predicted_loose)
    similarity = SequenceMatcher(None, reference_loose, predicted_loose).ratio()
    strict_similarity = SequenceMatcher(
        None,
        normalize_text(reference_raw).lower(),
        normalize_text(predicted_raw).lower(),
    ).ratio()

    return {
        "reference_text": reference_raw,
        "prediction_text": predicted_raw,
        "reference_loose_text": reference_loose,
        "prediction_loose_text": predicted_loose,
        "word_error_rate": wer,
        "char_error_rate": cer,
        "loose_similarity": similarity,
        "strict_similarity": strict_similarity,
        "reference_token_count": len(reference_loose.split()),
        "prediction_token_count": len(predicted_loose.split()),
        "predicted_segment_count": len(predicted_segments),
        "predicted_duration_seconds": predicted_duration,
    }


def rank_key(result: Dict[str, Any]) -> Tuple[float, float, float, float]:
    metrics = result["metrics"]
    return (
        float(metrics["word_error_rate"]),
        float(metrics["char_error_rate"]),
        -float(metrics["loose_similarity"]),
        float(result.get("elapsed_seconds", float("inf"))),
    )


def load_run_specs(matrix_file: Optional[str], base_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not matrix_file:
        return [base_spec]

    payload = read_json(matrix_file)
    if not isinstance(payload, list):
        raise ValueError("Matrix file must contain a JSON array of run specs.")

    specs = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each run spec in the matrix file must be a JSON object.")
        specs.append(merge_dict(base_spec, item))
    return specs


def save_run_artifact(output_dir: str, run_name: str, segments) -> str:
    _, srt_path = generate_file(
        output_dir=output_dir,
        output_file_name=run_name,
        output_format="srt",
        result=segments,
        add_timestamp=False,
    )
    return srt_path


def format_metric_summary(result: Dict[str, Any]) -> str:
    metrics = result["metrics"]
    return (
        f"WER={metrics['word_error_rate']:.4f} | "
        f"CER={metrics['char_error_rate']:.4f} | "
        f"LooseSim={metrics['loose_similarity']:.4f} | "
        f"Elapsed={result['elapsed_seconds']:.2f}s | "
        f"Speed={result['speed_x']:.2f}x"
    )


def print_leaderboard(results: List[Dict[str, Any]], top_n: int = 10) -> None:
    ranked = sorted((result for result in results if result.get("status") == "ok"), key=rank_key)
    if not ranked:
        print("\nNo successful runs to rank.")
        return

    print("\nLeaderboard")
    print("-" * 80)
    for index, result in enumerate(ranked[:top_n], start=1):
        print(f"{index:02d}. {result['run_name']}: {format_metric_summary(result)}")


def run_benchmark(
    input_path: str,
    session_dir: str,
    specs: List[Dict[str, Any]],
    context: EvaluationContext,
) -> List[Dict[str, Any]]:
    ensure_dir(session_dir)
    artifacts_dir = ensure_dir(os.path.join(session_dir, "artifacts"))
    results_path = os.path.join(session_dir, "results.jsonl")

    inferencer = WhisperFactory.create_whisper_inference(
        whisper_type=WhisperImpl.FASTER_WHISPER.value,
    )
    results: List[Dict[str, Any]] = []

    try:
        for index, raw_spec in enumerate(specs, start=1):
            spec = deepcopy(raw_spec)
            run_name = derive_run_name(spec)
            pipeline_params = build_pipeline_params(spec, inferencer.current_compute_type)
            audio_input = input_path if spec["bgm_separation"].get("is_separate_bgm") else context.audio
            run_started_at = datetime.now().isoformat(timespec="seconds")

            print("\n" + "=" * 100)
            print(f"[{index}/{len(specs)}] {run_name}")
            print(json.dumps(spec, indent=2, ensure_ascii=False))
            print("=" * 100)

            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                segments, elapsed_seconds = inferencer.run(
                    audio_input,
                    gr.Progress(),
                    "SRT",
                    False,
                    None,
                    *pipeline_params.to_list(),
                )
                srt_path = save_run_artifact(output_dir=artifacts_dir, run_name=run_name, segments=segments)
                metrics = evaluate_prediction(
                    reference_raw=context.reference_raw_text,
                    reference_loose=context.reference_loose_text,
                    predicted_segments=segments,
                )
                gpu_peak_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

                result = {
                    "status": "ok",
                    "run_name": run_name,
                    "run_started_at": run_started_at,
                    "elapsed_seconds": elapsed_seconds,
                    "speed_x": context.audio_duration / elapsed_seconds if elapsed_seconds else 0.0,
                    "audio_duration_seconds": context.audio_duration,
                    "reference_duration_seconds": context.reference_duration,
                    "reference_segment_count": len(context.reference_segments),
                    "srt_path": srt_path,
                    "gpu_peak_bytes": gpu_peak_bytes,
                    "params": spec,
                    "metrics": metrics,
                }
                print(format_metric_summary(result))
            except Exception as exc:
                inferencer.offload()
                result = {
                    "status": "error",
                    "run_name": run_name,
                    "run_started_at": run_started_at,
                    "params": spec,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                print(f"ERROR: {result['error']}")

            results.append(result)
            append_jsonl(results_path, result)
    finally:
        inferencer.offload()

    summary = {
        "session_dir": session_dir,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": input_path,
        "results_count": len(results),
        "successful_runs": sum(1 for result in results if result.get("status") == "ok"),
        "best_run": min((result for result in results if result.get("status") == "ok"), key=rank_key, default=None),
    }
    write_json(os.path.join(session_dir, "summary.json"), summary)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Whisper-WebUI faster-whisper transcription accuracy against a reference SRT.",
    )
    parser.add_argument("--input", default=os.path.join(ROOT_DIR, "test.mp4"))
    parser.add_argument("--reference", default=os.path.join(ROOT_DIR, "test.srt"))
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--matrix-file", default=None)
    parser.add_argument("--name", default=None)

    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--lang", default="English")
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-length", type=int, default=20)
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--log-prob-threshold", type=float, default=-1.0)
    parser.add_argument("--no-speech-threshold", type=float, default=0.6)
    parser.add_argument("--best-of", type=int, default=5)
    parser.add_argument("--patience", type=float, default=1.5)
    parser.add_argument("--condition-on-previous-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-reset-on-temperature", type=float, default=0.5)
    parser.add_argument("--initial-prompt", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--compression-ratio-threshold", type=float, default=2.4)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--suppress-blank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--suppress-tokens", default=[-1])
    parser.add_argument("--max-initial-timestamp", type=float, default=1.0)
    parser.add_argument("--word-timestamps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prepend-punctuations", default="\"'([{-")
    parser.add_argument("--append-punctuations", default="\"'.,!?:)]}")
    parser.add_argument("--max-new-tokens", default=None)
    parser.add_argument("--hallucination-silence-threshold", default=None)
    parser.add_argument("--hotwords", default=None)
    parser.add_argument("--language-detection-threshold", default=0.5)
    parser.add_argument("--language-detection-segments", type=int, default=1)

    parser.add_argument("--vad-filter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-speech-duration-ms", type=int, default=250)
    parser.add_argument("--vad-max-speech-duration-s", type=float, default=9999.0)
    parser.add_argument("--vad-min-silence-duration-ms", type=int, default=2000)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=400)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    outputs_root = ensure_dir(os.path.join(ROOT_DIR, "outputs", "benchmarks"))
    session_name = sanitize_name(args.session_name or f"transcription_accuracy_{timestamp_slug()}")
    session_dir = ensure_dir(os.path.join(outputs_root, session_name))
    cache_dir = ensure_dir(os.path.join(outputs_root, "_audio_cache"))

    base_spec = build_base_spec(args)
    specs = load_run_specs(args.matrix_file, base_spec)
    context = build_eval_context(reference_path=args.reference, input_path=args.input, cache_dir=cache_dir)

    metadata = {
        "session_dir": session_dir,
        "input": os.path.abspath(args.input),
        "reference": os.path.abspath(args.reference),
        "audio_duration_seconds": context.audio_duration,
        "reference_duration_seconds": context.reference_duration,
        "reference_segment_count": len(context.reference_segments),
        "spec_count": len(specs),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(os.path.join(session_dir, "metadata.json"), metadata)

    print(json.dumps(metadata, indent=2))
    results = run_benchmark(
        input_path=os.path.abspath(args.input),
        session_dir=session_dir,
        specs=specs,
        context=context,
    )
    print_leaderboard(results)

    best_result = min((result for result in results if result.get("status") == "ok"), key=rank_key, default=None)
    if best_result:
        print("\nBest run")
        print("-" * 80)
        print(best_result["run_name"])
        print(format_metric_summary(best_result))
        print(best_result["srt_path"])
    else:
        raise SystemExit("All benchmark runs failed.")


if __name__ == "__main__":
    main()
