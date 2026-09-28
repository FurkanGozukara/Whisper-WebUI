"""Reproducible Canary app-path benchmark; original precision is a smoke check only.

Run with CUDA_VISIBLE_DEVICES set before Python. References come from the shared
English corpus manifest. The final pass is summarized separately from cold runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
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

import torch
from modules.whisper.canary_qwen_inference import CanaryQwenInference
from modules.whisper.data_classes import WhisperParams
from english_benchmark_metrics import evaluate


class ProcessMemory:
    """Sample this worker's full CUDA allocation, including library workspaces."""
    def __init__(self):
        self.peak = 0
        self.stop = threading.Event()
        self.thread = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(str(torch.cuda.get_device_properties(0).uuid))
        except Exception:
            self.nvml = None

    def sample(self):
        while not self.stop.is_set():
            if self.nvml:
                try:
                    for process in self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
                        value = process.usedGpuMemory
                        if process.pid == os.getpid() and value is not None and 0 <= value < 2**60:
                            self.peak = max(self.peak, int(value))
                except Exception:
                    # NVML per-process accounting is unavailable on some WDDM drivers.
                    pass
            self.stop.wait(.1)

    def start(self):
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=2)


def summarize(records):
    counts = Counter()
    for record in records:
        for key in ("hits", "substitutions", "deletions", "insertions", "reference_words",
                    "punctuation_matched", "punctuation_reference_count", "punctuation_hypothesis_count"):
            counts[key] += record.get("metrics", {}).get(key, 0)
    counts["wer"] = (counts["substitutions"] + counts["deletions"] + counts["insertions"]) / max(1, counts["reference_words"])
    counts["punctuation_precision"] = counts["punctuation_matched"] / max(1, counts["punctuation_hypothesis_count"])
    counts["punctuation_recall"] = counts["punctuation_matched"] / max(1, counts["punctuation_reference_count"])
    counts["punctuation_f1"] = 2 * counts["punctuation_matched"] / max(1, counts["punctuation_hypothesis_count"] + counts["punctuation_reference_count"])
    counts["audio_seconds"] = sum(r["duration_seconds"] for r in records)
    counts["elapsed_seconds"] = sum(r["elapsed_seconds"] for r in records)
    counts["realtime_factor"] = counts["elapsed_seconds"] / max(1, counts["audio_seconds"])
    counts["speed_x_realtime"] = counts["audio_seconds"] / max(.001, counts["elapsed_seconds"])
    counts["max_allocated_gib"] = max((r.get("max_allocated_gib", 0) for r in records), default=0)
    counts["max_reserved_gib"] = max((r.get("max_reserved_gib", 0) for r in records), default=0)
    counts["process_gpu_peak_gib"] = max((r.get("process_gpu_peak_gib", 0) for r in records), default=0)
    counts["files"] = len(records)
    return dict(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "outputs/benchmarks/english_corpus/manifest.json"))
    ap.add_argument("--configs", required=True, help="JSON list of name and WhisperParams overrides")
    ap.add_argument("--output", required=True)
    ap.add_argument("--split", default="tuning")
    ap.add_argument("--ids", default="")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--category", default="")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--original", action="store_true")
    args = ap.parse_args()
    rows = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    rows = [r for r in rows if not args.split or r["split"] == args.split]
    if args.dataset:
        rows = [r for r in rows if args.dataset.lower() in r["dataset"].lower()]
    if args.category:
        rows = [r for r in rows if args.category == r["category"]]
    if args.ids:
        rows = [r for r in rows if r["id"] in args.ids.split(",")]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("No matching corpus samples")
    configs = json.loads(Path(args.configs).read_text(encoding='utf-8'))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    model_name = "nvidia/canary-qwen-2.5b" if args.original else "canary-qwen-2.5b-int8-convrot"
    p = CanaryQwenInference()
    t0 = time.perf_counter()
    p.update_model(model_name, "bfloat16", progress=lambda *a, **k: None)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - t0
    all_records = []
    for config in configs:
        config = dict(config)
        name = config.pop("name")
        if p.is_convrot_engine():
            p.model._sessions.clear()
            p.model.encoder._graphs.clear()
            if hasattr(p.model, "_hf_model"):
                del p.model._hf_model
            gc.collect()
            torch.cuda.empty_cache()
        options = dict(model_size=model_name, lang="english", compute_type="bfloat16", beam_size=1,
                       temperature=0, word_timestamps=False, condition_on_previous_text=False)
        options.update({k: v for k, v in config.items() if k != "prompt_template"})
        if "prompt_template" in config:
            template = config["prompt_template"]
            p.build_asr_prompt = lambda params, previous_text="", t=template: [
                {"role": "user", "content": t.format(audio_tag=p.model.audio_locator_tag)}]
        else:
            p.build_asr_prompt = CanaryQwenInference.build_asr_prompt.__get__(p)
        params = WhisperParams(**options)
        for pass_index in range(args.passes):
            for row in rows:
                record = {"config": name, "parameters": config, "pass": pass_index,
                          "id": row["id"], "split": row["split"], "category": row["category"],
                          "dataset": row["dataset"], "duration_seconds": row["duration_seconds"],
                          "model": model_name, "load_seconds": load_seconds,
                          "gpu": torch.cuda.get_device_name(), "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                          "reference": row["reference_text"]}
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                memory = ProcessMemory()
                memory.start()
                started = time.perf_counter()
                try:
                    segments, app_elapsed = p.transcribe(row["audio_path"], lambda *a, **k: None, None,
                                               *params.to_list(), log_console=False, log_model_banner=False)
                    torch.cuda.synchronize()
                    # Freeze inference timing before reference scoring. The old
                    # experiment files included CPU scoring in their wall time.
                    record.update(elapsed_seconds=time.perf_counter() - started,
                                  app_elapsed_seconds=app_elapsed,
                                  timing_scope="synchronized_transcription_only")
                    scoring_started = time.perf_counter()
                    hypothesis = " ".join(s.text for s in segments)
                    record.update(hypothesis=hypothesis, metrics=evaluate(row["reference_text"], hypothesis,
                                                                      row.get("punctuation_reference", False)),
                                  segments=[{"start": s.start, "end": s.end, "text": s.text} for s in segments])
                    record["scoring_seconds"] = time.perf_counter() - scoring_started
                except Exception:
                    record["error"] = traceback.format_exc()
                    print(record["error"], flush=True)
                record.setdefault("elapsed_seconds", time.perf_counter() - started)
                record.update(max_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                              max_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
                memory.finish()
                record["process_gpu_peak_gib"] = memory.peak / 2**30
                all_records.append(record)
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                display = {k: record.get(k) for k in ("config", "pass", "id", "elapsed_seconds", "metrics", "max_reserved_gib", "error")}
                if display["metrics"]:
                    display["metrics"] = {k: v for k, v in display["metrics"].items() if not k.startswith("normalized_")}
                print(json.dumps(display), flush=True)
    summary = {"model": model_name, "load_seconds": load_seconds, "configs": {}}
    for name in dict.fromkeys(r["config"] for r in all_records):
        current = [r for r in all_records if r["config"] == name and r["pass"] == args.passes - 1 and "error" not in r]
        summary["configs"][name] = {"overall": summarize(current)}
        for category in ("short", "long"):
            summary["configs"][name][category] = summarize([r for r in current if r["category"] == category])
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
