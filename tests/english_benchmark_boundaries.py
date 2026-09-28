"""Contiguous human-reference clips around the proposed 30-second policy boundary.

Auxiliary clips overlap existing source recordings and MUST NOT be counted as
additional independent speakers or pooled with the main corpus for inference.
"""
import csv
import json

from english_benchmark_corpus import OUT, punctuation_text, render, write_json


def main():
    manifest = json.loads((OUT / "manifest_earnings.json").read_text(encoding="utf-8"))
    sources = {row["source_recording"]: row for row in manifest}
    rows = []
    for call_id, base in sources.items():
        folder = OUT / "earnings22"
        with (folder / (call_id + ".aligned.nlp")).open(encoding="utf-8", newline="") as handle:
            timed = list(csv.DictReader(handle, delimiter="|"))
        last_time = max(float(r.get("endTs") or 0) for r in timed)
        starts = [0] + [i + 1 for i, row in enumerate(timed[:-1]) if row.get("punctuation") in {".", "?", "!"}]
        for label, minimum, maximum, fraction in [("below30", 28, 29.9, .34), ("above30", 30.1, 34, .36), ("mid50", 45, 60, .74)]:
            candidates = []
            for start_index in starts:
                first = timed[start_index]
                if not first.get("ts"):
                    continue
                for end_index in range(start_index + 10, min(start_index + 220, len(timed))):
                    last = timed[end_index]
                    if not last.get("endTs") or last.get("punctuation") not in {".", "?", "!"}:
                        continue
                    start, end = float(first["ts"]), float(last["endTs"])
                    # Equal padding leaves the spoken duration test simple.
                    start = max(0, start - .04)
                    end += .04
                    if not minimum <= end - start <= maximum:
                        continue
                    span = timed[start_index:end_index + 1]
                    if any("<" in r["token"] for r in span):
                        continue
                    if sum(bool(r.get("ts") and r.get("endTs")) for r in span) / len(span) < .98:
                        continue
                    candidates.append((abs(start - last_time * fraction), start, end, span))
            if not candidates:
                print("No qualified span", call_id, label)
                continue
            _, start, end, span = min(candidates, key=lambda x: x[0])
            path = folder / (call_id + "_" + label + ".wav")
            duration = render(folder / (call_id + ".mp3"), path, start, end)
            record = {**base, "id": f"earnings22_{call_id}_{label}", "audio_path": str(path),
                      "category": "short" if duration <= 30 else "long", "duration_seconds": duration,
                      "reference_text": punctuation_text(span), "excerpt_start_seconds": start,
                      "excerpt_end_seconds": end, "analysis_group": "duration_boundary",
                      "boundary_band": label, "sampling_note": "Auxiliary correlated excerpts, not additional independent corpus examples."}
            rows.append(record)
    path = OUT / "manifest_boundaries.json"
    write_json(path, rows)
    from english_benchmark_reference_audit import main as audit_references
    audit_references()
    print(path, len(rows), sum(r["duration_seconds"] for r in rows))


if __name__ == "__main__":
    main()
