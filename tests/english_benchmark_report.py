"""Aggregate raw English benchmark rows without averaging per-file WERs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(rows):
    valid = [r for r in rows if r.get("status") == "ok"]
    words = sum(r["metrics"]["reference_words"] for r in valid)
    errors = sum(sum(r["metrics"][key] for key in ("substitutions", "deletions", "insertions")) for r in valid)
    ref_punct = sum(r["metrics"].get("punctuation_reference_count", 0) for r in valid)
    hyp_punct = sum(r["metrics"].get("punctuation_hypothesis_count", 0) for r in valid)
    matched = sum(r["metrics"].get("punctuation_matched", 0) for r in valid)
    seconds = sum(r["elapsed_seconds"] for r in valid)
    duration = sum(r["duration_seconds"] for r in valid)
    return {"files": len(valid), "failed_files": len(rows) - len(valid), "reference_words": words,
            "word_errors": errors, "micro_wer": errors / words if words else None,
            "punctuation_f1": 2 * matched / (ref_punct + hyp_punct) if ref_punct + hyp_punct else None,
            "punctuation_reference_count": ref_punct, "punctuation_hypothesis_count": hyp_punct,
            "punctuation_matched": matched, "elapsed_seconds": seconds, "audio_seconds": duration,
            "speed_x": duration / seconds if seconds else None,
            "process_gpu_peak_bytes": max([r.get("process_gpu_peak_bytes") or 0 for r in valid], default=0),
            "torch_peak_allocated_bytes": max([r.get("torch_peak_allocated_bytes") or 0 for r in valid], default=0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw", action="store_true", help="Report original unvalidated references; not for final conclusions")
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in args.paths:
        files = [path] if path.is_file() else list(path.rglob("results.jsonl" if args.raw else "results.audited.jsonl"))
        for file in files:
            for line in file.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                for category in ("all", row["category"], row["dataset"] + "/" + row["category"]):
                    key = (file.parent.name, row["model"], row["config"], row["split"], category)
                    groups[key].append(row)
    output = [{"run": k[0], "model": k[1], "config": k[2], "split": k[3], "category": k[4], **summarize(v)} for k, v in groups.items()]
    text = json.dumps(output, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
