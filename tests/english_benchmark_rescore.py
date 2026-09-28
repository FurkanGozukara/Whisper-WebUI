"""Rescore saved Whisper hypotheses against audited human references.

Original results.jsonl files remain untouched. No additional inference and no
machine-generated reference text are used. Exclusions are symmetric by ID.
"""
import argparse
import json
from pathlib import Path

from english_benchmark_corpus import OUT, ROOT
from english_benchmark_metrics import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "outputs/benchmarks/whisper")
    args = parser.parse_args()
    audit = json.loads((OUT / "reference_audit.json").read_text(encoding="utf-8"))
    excluded = set(audit["excluded_ids"])
    references = {}
    for filename in ("manifest.json", "manifest_boundaries.json", "manifest_tedlium.json", "manifest_confirmation.json"):
        path = OUT / filename
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            references[row["id"]] = row
    count = 0
    for source in args.results_root.rglob("results.jsonl"):
        rows, omissions = [], []
        # A running worker might currently be appending its final line.
        text = source.read_text(encoding="utf-8")
        lines = text.splitlines()
        if text and not text.endswith("\n"):
            lines = lines[:-1]
        for line in lines:
            row = json.loads(line)
            if row["id"] in excluded:
                omissions.append({"id": row["id"], "config": row["config"], "reason": "Source forced alignment failed reference audit"})
                continue
            if row["id"] not in references:
                raise RuntimeError(f"No audited reference for {row['id']} in {source}")
            reference = references[row["id"]]
            if row.get("status") == "ok":
                row["metrics"] = evaluate(reference["reference_text"], row["prediction_text"], reference.get("punctuation_reference", False))
            row["reference_revision"] = reference.get("reference_revision", 1)
            row["reference_audit_version"] = audit["version"]
            rows.append(row)
        target = source.with_name("results.audited.jsonl")
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        temp.replace(target)
        source.with_name("reference_rescore.json").write_text(json.dumps({"audit_version": audit["version"],
            "source": str(source), "retained_rows": len(rows), "excluded_rows": omissions,
            "note": "Original raw results preserved; hypotheses unchanged; references restored to original human text."}, indent=2), encoding="utf-8")
        count += len(rows)
    print("Audited scored rows", count)


if __name__ == "__main__":
    main()
