import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from difflib import SequenceMatcher
from pathlib import Path

import gradio as gr

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.utils.files_manager import load_yaml, save_yaml
from modules.utils.paths import DEFAULT_PARAMETERS_CONFIG_PATH
from modules.whisper.data_classes import (
    BGMSeparationParams,
    DiarizationParams,
    TranscriptionPipelineParams,
    VadParams,
    WhisperImpl,
    WhisperParams,
)
from modules.whisper.whisper_factory import WhisperFactory


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


def normalize_segments(segments):
    normalized = []
    for segment in segments:
        text = " ".join((segment.text or "").split())
        if not text:
            continue
        normalized.append(
            {
                "start": round(segment.start or 0.0, 3),
                "end": round(segment.end or 0.0, 3),
                "text": text,
            }
        )
    return normalized


def query_gpu_memory_mb(gpu_index: int) -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu_index),
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None

    try:
        return int(float(lines[0]))
    except ValueError:
        return None


def wait_for_gpu_idle(gpu_index: int, timeout_sec: float = 30.0, stable_samples: int = 5) -> int | None:
    deadline = time.time() + timeout_sec
    observed = []
    best = None

    while time.time() < deadline:
        mem = query_gpu_memory_mb(gpu_index)
        if mem is not None:
            observed.append(mem)
            observed = observed[-stable_samples:]
            best = mem if best is None else min(best, mem)
            if len(observed) == stable_samples and max(observed) - min(observed) <= 16:
                return min(observed)
        time.sleep(0.5)

    return best


def select_model(available_models, requested_model: str | None, default_model: str | None):
    if requested_model:
        return requested_model
    if default_model and default_model in available_models:
        return default_model

    for candidate in ("large-v3", "large-v2", "large-v1", "medium", "small", "tiny"):
        if candidate in available_models:
            return candidate
    return list(available_models)[0]


def build_pipeline_params(default_whisper: dict, inferencer, model_size: str, batch_size: int):
    whisper_defaults = dict(default_whisper)
    whisper_defaults.update(
        {
            "model_size": model_size,
            "compute_type": whisper_defaults.get("compute_type", inferencer.current_compute_type),
            "use_batched_inference": False,
            "batch_size": int(batch_size),
            "enable_offload": False,
        }
    )
    return TranscriptionPipelineParams(
        whisper=WhisperParams(**whisper_defaults),
        vad=VadParams(vad_filter=False),
        bgm_separation=BGMSeparationParams(is_separate_bgm=False),
        diarization=DiarizationParams(is_diarize=False),
    ).to_list()


def run_child(args):
    cached_defaults = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
    defaults = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH) or {}
    default_whisper = defaults.get("whisper", {})
    inferencer = WhisperFactory.create_whisper_inference(
        whisper_type=WhisperImpl.FASTER_WHISPER.value,
    )

    try:
        model_size = select_model(
            inferencer.available_models,
            args.model,
            default_whisper.get("model_size"),
        )
        pipeline_params = build_pipeline_params(
            default_whisper=default_whisper,
            inferencer=inferencer,
            model_size=model_size,
            batch_size=args.batch_size,
        )
        segments, elapsed = inferencer.run(
            args.input,
            gr.Progress(),
            "SRT",
            False,
            None,
            *pipeline_params,
        )
        normalized = normalize_segments(segments)
        payload = {
            "batch_size": args.batch_size,
            "model_size": model_size,
            "elapsed_time_sec": elapsed,
            "segment_count": len(normalized),
            "segments": normalized,
            "transcript": " ".join(segment["text"] for segment in normalized),
        }
        Path(args.result_json).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    finally:
        try:
            inferencer.offload()
        finally:
            save_yaml(cached_defaults, DEFAULT_PARAMETERS_CONFIG_PATH)


def run_parent(args):
    results = []
    python_exe = args.python or sys.executable
    script_path = str(Path(__file__).resolve())

    def persist_summary():
        summary = {
            "input": args.input,
            "gpu_index": args.gpu_index,
            "results": results,
        }
        summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
        if args.summary_json:
            Path(args.summary_json).write_text(summary_json, encoding="utf-8")
        return summary_json

    for batch_size in range(args.min_batch_size, args.max_batch_size + 1):
        idle_memory = wait_for_gpu_idle(args.gpu_index)
        result_file = tempfile.NamedTemporaryFile(prefix=f"batch_{batch_size}_", suffix=".json", delete=False)
        result_file.close()

        child_cmd = [
            python_exe,
            script_path,
            "--child-run",
            "--input",
            args.input,
            "--batch-size",
            str(batch_size),
            "--result-json",
            result_file.name,
        ]
        if args.model:
            child_cmd.extend(["--model", args.model])

        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_index))
        child_stdout_file = tempfile.NamedTemporaryFile(prefix=f"batch_{batch_size}_stdout_", suffix=".log", delete=False)
        child_stderr_file = tempfile.NamedTemporaryFile(prefix=f"batch_{batch_size}_stderr_", suffix=".log", delete=False)
        child_stdout_path = child_stdout_file.name
        child_stderr_path = child_stderr_file.name
        child_stdout_file.close()
        child_stderr_file.close()

        start_time = time.perf_counter()
        with open(child_stdout_path, "w", encoding="utf-8") as child_stdout, open(
            child_stderr_path, "w", encoding="utf-8"
        ) as child_stderr:
            process = subprocess.Popen(
                child_cmd,
                cwd=str(ROOT_DIR),
                stdout=child_stdout,
                stderr=child_stderr,
                text=True,
                env=env,
            )

            peak_memory = idle_memory
            while process.poll() is None:
                current_memory = query_gpu_memory_mb(args.gpu_index)
                if current_memory is not None:
                    peak_memory = current_memory if peak_memory is None else max(peak_memory, current_memory)
                time.sleep(args.poll_interval_sec)

            process.wait()

        wall_time_sec = time.perf_counter() - start_time
        current_memory = query_gpu_memory_mb(args.gpu_index)
        if current_memory is not None:
            peak_memory = current_memory if peak_memory is None else max(peak_memory, current_memory)
        stdout = Path(child_stdout_path).read_text(encoding="utf-8", errors="replace").strip()
        stderr = Path(child_stderr_path).read_text(encoding="utf-8", errors="replace").strip()

        result = {
            "batch_size": batch_size,
            "idle_gpu_memory_mb": idle_memory,
            "peak_gpu_memory_mb": peak_memory,
            "peak_gpu_memory_delta_mb": None
            if idle_memory is None or peak_memory is None
            else max(0, peak_memory - idle_memory),
            "wall_time_sec": wall_time_sec,
            "returncode": process.returncode,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
        }

        if process.returncode == 0 and os.path.exists(result_file.name):
            child_payload = json.loads(Path(result_file.name).read_text(encoding="utf-8"))
            result.update(child_payload)
        else:
            result["error"] = stderr.strip() or stdout.strip() or "Benchmark run failed."

        results.append(result)
        persist_summary()

        try:
            os.remove(result_file.name)
        except OSError:
            pass
        for child_log_path in (child_stdout_path, child_stderr_path):
            try:
                os.remove(child_log_path)
            except OSError:
                pass

    baseline = next((result for result in results if result["batch_size"] == 1 and result.get("returncode") == 0), None)
    if baseline:
        baseline_segments = baseline.get("segments", [])
        baseline_transcript = baseline.get("transcript", "")
        for result in results:
            if result.get("returncode") != 0 or result["batch_size"] == 1:
                continue
            transcript = result.get("transcript", "")
            result["wer_vs_batch_1"] = word_error_rate(baseline_transcript, transcript)
            result["char_ratio_vs_batch_1"] = SequenceMatcher(None, baseline_transcript, transcript).ratio()
            result["exact_segment_match_vs_batch_1"] = baseline_segments == result.get("segments", [])
            result["segment_count_delta_vs_batch_1"] = result.get("segment_count", 0) - baseline.get("segment_count", 0)

    summary_json = persist_summary()
    print(summary_json)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark standard faster-whisper batch sizes on a real media file.")
    parser.add_argument("--input", required=True, help="Path to the media file to benchmark.")
    parser.add_argument("--model", default=None, help="Optional faster-whisper model size override.")
    parser.add_argument("--python", default=None, help="Python executable to use for child runs.")
    parser.add_argument("--gpu-index", type=int, default=0, help="Physical GPU index to monitor with nvidia-smi.")
    parser.add_argument("--min-batch-size", type=int, default=1, help="First batch size to test.")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Last batch size to test.")
    parser.add_argument("--poll-interval-sec", type=float, default=0.25, help="GPU memory polling interval.")
    parser.add_argument("--summary-json", default=None, help="Optional path for the summary JSON output.")
    parser.add_argument("--child-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--result-json", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.child_run:
        if not args.result_json:
            raise SystemExit("--result-json is required for --child-run")
        run_child(args)
        return

    run_parent(args)


if __name__ == "__main__":
    main()
