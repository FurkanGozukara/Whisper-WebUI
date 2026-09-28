"""Run the application's English Whisper presets against a corpus manifest.

Example (one GPU per process):
 CUDA_VISIBLE_DEVICES=1 venv/bin/python tests/english_benchmark_whisper.py \
   --model large-v1-int8-convrot --configs baseline,beam1,no_context --split tuning

Cold includes model load and compilation; measured rows follow one warmup.
Later new shapes may still compile, which is recorded in ordinary wall time.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.utils.paths import configure_model_cache_env
configure_model_cache_env()

import numpy as np
import torch
from faster_whisper.audio import decode_audio
from modules.whisper.data_classes import WhisperParams, TranscriptionPipelineParams, VadParams, DiarizationParams, BGMSeparationParams
from modules.whisper.faster_whisper_inference import FasterWhisperInference
from modules.whisper.convrot import triton_status
from english_benchmark_metrics import evaluate

CONFIGS = {
    "baseline": {},
    "encoder2": {"batch_size": 2},
    "encoder4": {"batch_size": 4},
    "encoder8": {"batch_size": 8},
    "encoder16": {"batch_size": 16},
    "encoder32": {"batch_size": 32},
    "beam1": {"beam_size": 1},
    "beam3": {"beam_size": 3},
    "beam8": {"beam_size": 8, "patience": 1.5},
    "no_context": {"condition_on_previous_text": False},
    "batched8": {"use_batched_inference": True, "batch_size": 8},
    "batched16": {"use_batched_inference": True, "batch_size": 16},
    "chunk20": {"chunk_length": 20},
    "silence1": {"hallucination_silence_threshold": 1.0},
    "silence2": {"hallucination_silence_threshold": 2.0},
}


class LegacyContextWhisperInference(FasterWhisperInference):
    """Freeze the pre-tuning 60-window guard for reproducible A/B comparisons.

    This benchmark-only override never changes the application's runtime class.
    It lets condition=true remain a historical baseline after the application
    adds a model-aware English long-form safeguard.
    """
    def resolve_standard_audio_and_params(self, audio, params, sampling_rate=None, log_console=True):
        if not params.condition_on_previous_text:
            return audio, params
        sampling_rate = sampling_rate or self.model.feature_extractor.sampling_rate
        audio_array = self.prepare_audio_array(audio, sampling_rate)
        windows = self.estimate_chunk_windows(audio_array, params.chunk_length, sampling_rate)
        if windows >= 60:
            params = params.model_copy(update={"condition_on_previous_text": False})
        return audio_array, params


class ProcessMemory:
    """NVML catches CTranslate2 allocations that torch counters do not see."""
    def __init__(self):
        self.peak = 0
        self.stop = threading.Event()
        self.thread = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:
            self.nvml = None

    def sample(self):
        if not self.nvml:
            return
        while not self.stop.is_set():
            for handle in self.handles:
                try:
                    for process in self.nvml.nvmlDeviceGetComputeRunningProcesses(handle):
                        if process.pid == os.getpid():
                            self.peak = max(self.peak, int(process.usedGpuMemory))
                except Exception:
                    pass
            self.stop.wait(0.15)

    def __enter__(self):
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=2)


def append(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/benchmarks/english_corpus/manifest.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--configs", default="baseline")
    parser.add_argument("--split", default="tuning", choices=["tuning", "heldout", "confirmation", "all"])
    parser.add_argument("--category", choices=["short", "long", "all"], default="all")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--condition-on-previous-text", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--context-policy", choices=["legacy", "shipped"], default="legacy",
                        help="legacy preserves the original >=60-window safeguard for reproducible A/B comparisons; shipped exercises the current app policy")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260928)
    np.random.seed(20260928)
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [r for r in records if (args.split == "all" or r["split"] == args.split)
               and (args.category == "all" or r["category"] == args.category)
               and (not args.dataset or r["dataset"] == args.dataset)
               and (not args.ids or r["id"] in args.ids.split(","))]
    records.sort(key=lambda r: (r["category"] != "short", r["id"]))
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise SystemExit("No matching evaluation records")
    name = args.run_name or f"{args.model}_{args.split}_{args.configs.replace(',', '-')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    folder = ROOT / "outputs/benchmarks/whisper" / name
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "results.jsonl"
    base = json.loads((ROOT / "modules/ui/defaults/presets/fast_whisper_best_quality.json").read_text(encoding="utf-8"))["file_tab"]["whisper"]
    base.update(model_size=args.model, lang="english", offload_to_ram=False, enable_offload=False)
    # Pin the experimental baseline so later preset edits cannot contaminate runs.
    base.update(batch_size=1, beam_size=5, patience=1.0, condition_on_previous_text=True,
                chunk_length=30, word_timestamps=True, use_batched_inference=False)
    if args.condition_on_previous_text is not None:
        base["condition_on_previous_text"] = args.condition_on_previous_text
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "model": args.model,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": torch.cuda.get_device_name(),
                "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory, "torch": torch.__version__,
                "manifest": str(args.manifest), "ids": [r["id"] for r in records], "configs": args.configs,
                "baseline": base, "context_policy": args.context_policy, "command": sys.argv,
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  [Path(__file__), ROOT / "modules/whisper/faster_whisper_inference.py",
                                   ROOT / "modules/whisper/convrot/model.py", ROOT / "modules/whisper/convrot/engine.py"]},
                "metrics": "OpenAI EnglishTextNormalizer WER; word-aligned .,!?;: punctuation F1 (not the published PER metric)",
                "timing": "Wall time includes decode, app preprocessing/inference/postprocessing. cold.json contains first model load + compilation; rows are subsequent runs. Shared GPUs have unrelated ComfyUI allocations, no control over contention."}
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    inferencer = (LegacyContextWhisperInference if args.context_policy == "legacy" else FasterWhisperInference)()
    # Keep the application's actual pipeline, suppress only per-segment console text.
    transcribe = inferencer.transcribe
    def quiet_transcribe(*positional, **kwargs):
        return transcribe(*positional, **kwargs, log_console=False, log_model_banner=False)
    inferencer.transcribe = quiet_transcribe
    for log_name in ("whisper-webui", "faster_whisper", "httpx", "httpcore"):
        logging.getLogger(log_name).setLevel(logging.WARNING)

    def run(record, config_name, phase, iteration=0):
        params = WhisperParams(**{**base, **CONFIGS[config_name]})
        pipeline = TranscriptionPipelineParams(whisper=params, vad=VadParams(vad_filter=False),
                                             diarization=DiarizationParams(is_diarize=False),
                                             bgm_separation=BGMSeparationParams(is_separate_bgm=False))
        row = {"id": record["id"], "split": record["split"], "category": record["category"],
               "dataset": record["dataset"], "model": args.model, "config": config_name,
               "phase": phase, "iteration": iteration, "duration_seconds": record["duration_seconds"],
               "params": params.model_dump(), "punctuation_reference": record.get("punctuation_reference", False)}
        start = time.perf_counter()
        before_tuning = triton_status.counts()
        torch.cuda.reset_peak_memory_stats()
        with ProcessMemory() as memory:
            try:
                segments, reported = inferencer.run(record["audio_path"], lambda *a, **kw: None, "SRT", False, None, *pipeline.to_list())
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                text = " ".join(s.text.strip() for s in segments if s.text)
                row.update(status="ok", elapsed_seconds=elapsed, app_elapsed_seconds=reported,
                           speed_x=record["duration_seconds"] / elapsed,
                           prediction_text=text, segments=[s.model_dump() for s in segments],
                           metrics=evaluate(record["reference_text"], text, record.get("punctuation_reference", False)),
                           torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - start)
        row["process_gpu_peak_bytes"] = memory.peak or None
        after_tuning = triton_status.counts()
        row["triton_cache_activity"] = {key: after_tuning[key] - before_tuning[key] for key in after_tuning}
        print(json.dumps({k: row[k] for k in ("id", "model", "config", "phase", "status", "elapsed_seconds")}, ensure_ascii=False), flush=True)
        if row["status"] == "error":
            print(row["traceback"], flush=True)
        else:
            print("WER", row["metrics"]["wer"], "punctuation_f1", row["metrics"].get("punctuation_f1"), "VRAM", row["process_gpu_peak_bytes"], flush=True)
        return row

    try:
        cold = run(records[0], args.configs.split(",")[0], "cold")
        (folder / "cold.json").write_text(json.dumps(cold, ensure_ascii=False, indent=2), encoding="utf-8")
        if cold["status"] == "error":
            raise RuntimeError(cold["error"])
        for config in args.configs.split(","):
            for iteration in range(args.repeat):
                for record in records:
                    row = run(record, config, "warm", iteration)
                    append(result_path, row)
    finally:
        inferencer.offload()
    print("RESULTS", result_path, flush=True)


if __name__ == "__main__":
    main()
