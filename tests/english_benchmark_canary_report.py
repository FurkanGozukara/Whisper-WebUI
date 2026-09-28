"""Rescore saved Canary hypotheses against the audited human references.

No inference or reference text is generated here. Invalid alignment sources are
excluded symmetrically by using the audited manifest's IDs for every setting.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from english_benchmark_metrics import evaluate

BASE = ROOT / "outputs/benchmarks/canary"
CORPUS = ROOT / "outputs/benchmarks/english_corpus"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(file, config=None, pass_index=1):
    path = BASE / f"{file}.jsonl"
    if not path.exists():
        return []
    return [r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if r["pass"] == pass_index and (config is None or r["config"] == config)]


def aggregate(rows):
    totals = {key: sum(r["metrics"].get(key, 0) for r in rows) for key in (
        "hits", "substitutions", "deletions", "insertions", "reference_words",
        "punctuation_reference_count", "punctuation_hypothesis_count", "punctuation_matched")}
    totals["files"] = len(rows)
    totals["wer"] = sum(totals[k] for k in ("substitutions", "deletions", "insertions")) / max(1, totals["reference_words"])
    denom = totals["punctuation_reference_count"] + totals["punctuation_hypothesis_count"]
    totals["punctuation_f1"] = 2 * totals["punctuation_matched"] / denom if denom else None
    totals["elapsed_seconds"] = sum(r["elapsed_seconds"] for r in rows)
    totals["duration_seconds"] = sum(r["duration_seconds"] for r in rows)
    totals["speed_x_realtime"] = totals["duration_seconds"] / max(.001, totals["elapsed_seconds"])
    totals["process_gpu_peak_gib"] = max((r.get("process_gpu_peak_gib", 0) for r in rows), default=0)
    scopes = {r.get("timing_scope", "includes_reference_scoring") for r in rows}
    totals["timing_scope"] = next(iter(scopes)) if len(scopes) == 1 else ("mixed" if scopes else "unavailable")
    return totals


def main():
    reference_rows = (read(CORPUS / "manifest.json") + read(CORPUS / "manifest_tedlium.json")
                      + read(CORPUS / "manifest_boundaries.json"))
    confirmation_manifest = CORPUS / "manifest_confirmation.json"
    confirmation_refs = read(confirmation_manifest) if confirmation_manifest.exists() else []
    reference_rows += confirmation_refs
    refs = {r["id"]: r for r in reference_rows}

    def rescore(rows):
        result = []
        for row in rows:
            if row["id"] not in refs:
                continue
            if "error" in row:
                raise RuntimeError(f"Cannot score failed case: {row['config']} {row['id']}")
            row = dict(row)
            ref = refs[row["id"]]
            row["reference"] = ref["reference_text"]
            row["metrics"] = evaluate(ref["reference_text"], row["hypothesis"], ref["punctuation_reference"])
            row["reference_audit_version"] = 2
            result.append(row)
        return result

    train = {}
    for chunk in (10, 12, 15, 20, 30, 40):
        if chunk == 10:
            rows = records("sweep5_stable", "chunk10_b1") + records("ted_tuning", "chunk10_b8")
        elif chunk in (12, 15):
            rows = records("intermediate_earnings", f"chunk{chunk}_b8") + records("intermediate_ted", f"chunk{chunk}_b8")
        else:
            rows = records("sweep7_stable", f"chunk{chunk}_b8") + records("ted_tuning", f"chunk{chunk}_b8")
        train[f"chunk{chunk}"] = rescore(rows)

    baseline = rescore(records("heldout_main", "baseline_10s_b1") + records("heldout_ted", "baseline_10s_b1"))
    heldout = {
        "baseline_10s_b1": baseline,
        "rejected_30s_b8": rescore(records("heldout_main", "candidate_30s_b8") + records("heldout_ted", "candidate_30s_b8")),
        "candidate_12s_b8": rescore(records("heldout_final12", "candidate_12s_b8")),
    }
    for batch in (2, 4, 8, 16):
        heldout[f"chunk10_b{batch}"] = rescore(records("ten_second_batches_heldout", f"chunk10_b{batch}"))
    for batch in (1, 2, 4, 16):
        file = "final12_batches_small" if batch <= 2 else "final12_batches_large"
        heldout[f"chunk12_b{batch}"] = rescore(records(file, f"chunk12_b{batch}"))
    for batch in (1, 2, 4, 8, 16):
        group = "small" if batch <= 2 else ("medium" if batch <= 8 else "large")
        heldout[f"auto_b{batch}"] = rescore(records(f"auto_batches_{group}", f"auto_b{batch}"))

    report = {"reference_audit": read(CORPUS / "reference_audit.json"), "tuning": {}, "heldout": {}, "batch_memory": {}}
    for name in ("confirmation_reference_audit", "confirmation_protocol_frozen"):
        path = CORPUS / f"{name}.json"
        if path.exists():
            report[name] = read(path)
    for label, datasets in (("tuning", train), ("heldout", heldout)):
        for name, rows in datasets.items():
            report[label][name] = {"overall": aggregate(rows)}
            for category in ("short", "long"):
                report[label][name][category] = aggregate([r for r in rows if r["category"] == category])
            report[label][name]["main_without_ted"] = aggregate([r for r in rows if "ted" not in r["dataset"].lower()])
            for category in ("short", "long"):
                report[label][name][f"main_{category}"] = aggregate([
                    r for r in rows if r["category"] == category and "ted" not in r["dataset"].lower()])
            for dataset in sorted({r["dataset"] for r in rows}):
                report[label][name][dataset] = aggregate([r for r in rows if r["dataset"] == dataset])
            if label == "heldout":
                paired = {r["id"]: r for r in baseline}
                matched = [r for r in rows if r["id"] in paired]
                report[label][name]["paired_files"] = len(matched)
                report[label][name]["exact_text_matches_to_baseline"] = sum(
                    r["hypothesis"] == paired[r["id"]]["hypothesis"] for r in matched)
                report[label][name]["baseline_same_ids"] = aggregate([paired[r["id"]] for r in matched])
                report[label][name]["complete"] = {r["id"] for r in rows} == set(paired)

    for batch in (1, 2, 4, 8, 16):
        name = f"chunk12_b{batch}"
        file = "heldout_final12" if batch == 8 else ("final12_batches_small" if batch <= 2 else "final12_batches_large")
        config = "candidate_12s_b8" if batch == 8 else name
        cold_warm = rescore(records(file, config, pass_index=0) + records(file, config, pass_index=1))
        rows = heldout[config]
        single = {r["id"]: r for r in heldout["chunk12_b1"]}
        matched = [r for r in rows if r["id"] in single]
        report["batch_memory"][name] = {
            "nvml_peak_gib": max((r.get("process_gpu_peak_gib", 0) for r in cold_warm), default=0),
            "torch_peak_reserved_gib": max((r.get("max_reserved_gib", 0) for r in cold_warm), default=0),
            "warm": aggregate(rows),
            "complete": {r["id"] for r in rows} == {r["id"] for r in baseline},
            "exact_text_matches_to_batch1": sum(r["hypothesis"] == single[r["id"]]["hypothesis"] for r in matched),
            "batch1_same_ids": aggregate([single[r["id"]] for r in matched]),
        }

    for batch in (1, 2, 4, 8, 16):
        name = f"auto_b{batch}"
        group = "small" if batch <= 2 else ("medium" if batch <= 8 else "large")
        file = f"auto_batches_{group}"
        cold_warm = rescore(records(file, name, pass_index=0) + records(file, name, pass_index=1))
        rows = heldout[name]
        single = {r["id"]: r for r in heldout["auto_b1"]}
        matched = [r for r in rows if r["id"] in single]
        report["batch_memory"][name] = {
            "nvml_peak_gib": max((r.get("process_gpu_peak_gib", 0) for r in cold_warm), default=0),
            "torch_peak_reserved_gib": max((r.get("max_reserved_gib", 0) for r in cold_warm), default=0),
            "warm": aggregate(rows),
            "complete": {r["id"] for r in rows} == {r["id"] for r in baseline},
            "exact_text_matches_to_batch1": sum(r["hypothesis"] == single[r["id"]]["hypothesis"] for r in matched),
            "batch1_same_ids": aggregate([single[r["id"]] for r in matched]),
        }

    for batch in (1, 2, 4, 8, 16, 32):
        name = f"chunk30_b{batch}"
        rows = records("batch_tiers", name, pass_index=0) + records("batch_tiers", name, pass_index=1)
        report["batch_memory"][name] = {"nvml_peak_gib": max((r.get("process_gpu_peak_gib", 0) for r in rows), default=0),
                                              "warm": aggregate(rescore([r for r in rows if r["pass"] == 1]))}

    boundaries = {
        "baseline_10s_b8": rescore(records("boundary_validation", "boundary10_b8")),
        "fixed_12s_b8": rescore(records("boundary_validation", "boundary12_b8")),
        "auto_b8": rescore(records("auto_boundary", "auto_b8")),
    }
    expected_boundaries = {r["id"] for r in read(CORPUS / "manifest_boundaries.json")}
    report["boundaries"] = {}
    for name, rows in boundaries.items():
        report["boundaries"][name] = {
            "overall": aggregate(rows),
            "tuning": aggregate([r for r in rows if r["split"] == "tuning"]),
            "heldout": aggregate([r for r in rows if r["split"] == "heldout"]),
            "at_most_30_seconds": aggregate([r for r in rows if r["duration_seconds"] <= 30]),
            "above_30_seconds": aggregate([r for r in rows if r["duration_seconds"] > 30]),
            "complete": {r["id"] for r in rows} == expected_boundaries,
        }

    confirmation = {
        "baseline_10s_b1": rescore(records("confirmation", "baseline_10s_b1")),
        "auto_b8": rescore(records("confirmation", "auto_b8")),
        "auto_b16": rescore(records("confirmation_b16", "auto_b16")),
    }
    report["confirmation"] = {}
    expected_confirmation = {r["id"] for r in confirmation_refs}
    for name, rows in confirmation.items():
        report["confirmation"][name] = {"overall": aggregate(rows),
                                         "complete": bool(expected_confirmation) and {r["id"] for r in rows} == expected_confirmation}
        for category in ("short", "long"):
            report["confirmation"][name][category] = aggregate([r for r in rows if r["category"] == category])
        for dataset in sorted({r["dataset"] for r in rows}):
            report["confirmation"][name][dataset] = aggregate([r for r in rows if r["dataset"] == dataset])

    (BASE / "audited_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (BASE / "audited_predictions.jsonl").open("w", encoding="utf-8") as f:
        for phase, datasets in (("tuning", train), ("heldout", heldout), ("boundaries", boundaries),
                                ("confirmation", confirmation)):
            for name, rows in datasets.items():
                for row in rows:
                    f.write(json.dumps({"phase": phase, "comparison": name, **row}, ensure_ascii=False) + "\n")
    for phase in ("tuning", "heldout", "confirmation"):
        print(phase)
        for name, result in report[phase].items():
            a = result["overall"]
            print(name, "files", a["files"], "WER", round(100*a["wer"], 3), "punctF1",
                  round(100*a["punctuation_f1"], 2) if a["punctuation_f1"] is not None else None,
                  "seconds", round(a["elapsed_seconds"], 2), "complete", result.get("complete", True))


if __name__ == "__main__":
    main()
