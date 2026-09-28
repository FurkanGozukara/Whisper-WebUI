"""Run the app's own transcription pipeline (BaseTranscriptionPipeline.run, the path every UI job uses) on a
benchmark manifest with one configuration, and append one JSON line per file (resumable).

Configuration JSON (inline or a file):
  {"name": "canary", "preset": "canary_qwen_best_quality",   # shipped UI preset (file_tab section)
   "whisper": {"batch_size": 16},                            # overrides of preset values
   "overrides": {"modules.whisper.faster_whisper_inference.FasterWhisperInference.SOME_CONSTANT": 1}}

    CUDA_VISIBLE_DEVICES=0 python tests/asr_benchmark/run_benchmark.py --config cfg.json \
        --manifest outputs/asr_benchmark/corpus/manifest_short.jsonl --split test --out results.jsonl

Set APP_DIR to benchmark another copy of the app.
"""
import argparse
import json
import os
import sys
import threading
import time
import traceback

APP_DIR = os.environ.get("APP_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)

from modules.utils.paths import configure_model_cache_env  # noqa: E402

configure_model_cache_env()


class NvmlPeak:
    """Samples this process's GPU memory (all allocators, CUDA context included) every 20 ms."""

    def __init__(self):
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(int(visible))
            self._pynvml = pynvml
        except Exception:
            self._handle = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _sample(self):
        if self._handle is None:
            return 0.0
        try:
            procs = self._pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
        except Exception:
            return 0.0
        pid = os.getpid()
        for p in procs:
            if p.pid == pid and p.usedGpuMemory:
                return p.usedGpuMemory / 1024 ** 2
        return 0.0

    def _run(self):
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self._sample())
            time.sleep(0.05)

    def reset(self):
        self.peak_mb = self._sample()


def noop_progress(*args, **kwargs):
    return None


def build_params(cfg):
    from modules.whisper.data_classes import (TranscriptionPipelineParams, WhisperParams, VadParams,
                                              DiarizationParams, BGMSeparationParams)
    preset_name = cfg.get("preset")
    base_whisper, base_vad = {}, {}
    if preset_name:
        path = os.path.join(APP_DIR, "modules", "ui", "defaults", "presets", preset_name + ".json")
        preset = json.load(open(path, encoding="utf-8"))
        tab = preset.get("file_tab", {})
        base_whisper = dict(tab.get("whisper", {}))
        base_vad = dict(tab.get("vad", {}))
    base_whisper.update(cfg.get("whisper", {}))
    base_vad.update(cfg.get("vad", {}))
    # benchmark throughput like a batch job: keep the model on the GPU between files
    base_whisper.setdefault("enable_offload", False)
    base_whisper["offload_to_ram"] = False
    base_whisper["start_as_subprocess"] = False
    whisper = WhisperParams(**base_whisper)
    vad = VadParams(**base_vad)
    return TranscriptionPipelineParams(whisper=whisper, vad=vad, diarization=DiarizationParams(),
                                       bgm_separation=BGMSeparationParams())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON file or inline JSON")
    ap.add_argument("--manifest", required=True, action="append")
    ap.add_argument("--split", default=None)
    ap.add_argument("--datasets", default=None, help="comma separated dataset names")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1, help="passes over the files (timing)")
    args = ap.parse_args()

    cfg = json.loads(args.config) if args.config.strip().startswith("{") else json.load(open(args.config))
    items = []
    for m in args.manifest:
        with open(m, encoding="utf-8") as f:
            items.extend(json.loads(line) for line in f if line.strip())
    if args.split:
        items = [it for it in items if it["split"] == args.split]
    if args.datasets:
        keep = set(args.datasets.split(","))
        items = [it for it in items if it["dataset"] in keep]
    # longest first for better load balance across shards, then round-robin sharding
    items.sort(key=lambda it: -it["duration"])
    k, n = (int(x) for x in args.shard.split("/"))
    items = items[k::n]
    if args.limit:
        items = items[: args.limit]

    cfg_name = cfg["name"]
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("config") == cfg_name:
                        done.add((r["id"], r.get("pass", 0)))
                except Exception:
                    pass

    params = build_params(cfg)
    from modules.whisper.whisper_factory import WhisperFactory
    import importlib
    import torch

    # "overrides": {"modules.whisper.faster_whisper_inference.FasterWhisperInference.SOME_CONSTANT": value}
    for dotted, value in (cfg.get("overrides") or {}).items():
        mod_name, cls_name, attr = dotted.rsplit(".", 2)
        setattr(getattr(importlib.import_module(mod_name), cls_name), attr, value)
        print(f"[worker] override {cls_name}.{attr} = {value!r}", flush=True)

    inf = WhisperFactory.create_whisper_inference(whisper_type=params.whisper.whisper_type)
    peak = NvmlPeak()
    param_list = params.to_list()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"[worker] config={cfg['name']} files={len(items)} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}",
          flush=True)
    for p in range(args.repeat):
        for it in items:
            if (it["id"], p) in done:
                continue
            peak.reset()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            error = None
            segments = []
            try:
                segments, _ = inf.run(it["path"], noop_progress, "SRT", False, None, *param_list)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            texts = [(s.text or "").strip() for s in segments if s is not None and (s.text or "").strip()]
            rec = {"id": it["id"], "dataset": it["dataset"], "split": it["split"], "config": cfg["name"],
                   "pass": p, "hyp": " ".join(texts), "n_segments": len(texts), "seconds": round(dt, 3),
                   "duration": it["duration"], "peak_mb": round(peak.peak_mb, 1), "error": error,
                   "segments": [[round(s.start or 0, 2), round(s.end or 0, 2), (s.text or "").strip()]
                                for s in segments if s is not None and (s.text or "").strip()]}
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[worker] {it['id']} {it['duration']:.1f}s audio in {dt:.2f}s peak={peak.peak_mb:.0f}MB"
                  f"{' ERROR ' + error if error else ''}", flush=True)
    print("[worker] DONE", flush=True)


if __name__ == "__main__":
    main()
