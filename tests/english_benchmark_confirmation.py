"""Build an untouched confirmation set after freezing the measured defaults.

Selection uses metadata and reference structure only, never ASR predictions.
"""
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json

from english_benchmark_corpus import OUT, ROOT, earnings_call, librispeech, punctuation_text, sha256, write_json
from english_benchmark_reference_audit import find_span, load_source, original_span


def audit_call(records):
    call_id = records[0]["source_recording"]
    original, aligned, operations = load_source(call_id)
    timed = [r for r in aligned if r.get("ts") and r.get("endTs")]
    reversals = sum(float(b["ts"]) < float(a["ts"]) for a, b in zip(timed, timed[1:]))
    if len(timed) / len(aligned) < .95 or reversals:
        raise ValueError("Source failed predeclared alignment criterion")
    audit = {"source_recording": call_id, "timed_fraction": len(timed) / len(aligned),
             "timestamp_reversals": reversals, "excerpts": []}
    for row in records:
        row["reference_revision"] = 2
        if row["excerpt_end_seconds"] is None:
            assert row["reference_text"] == punctuation_text(original)
            continue
        start, end = find_span(aligned, row["reference_text"], row["excerpt_start_seconds"])
        first, last = original_span(operations, start, end)
        span = aligned[start:end + 1]
        first_padding = float(span[0]["ts"]) - row["excerpt_start_seconds"]
        last_padding = row["excerpt_end_seconds"] - float(span[-1]["endTs"])
        if abs(first_padding) > .11 or abs(last_padding) > .11:
            raise ValueError("Excerpt failed predeclared crop-boundary criterion")
        row["reference_text"] = punctuation_text(original[first:last])
        row["reference_provenance"] = "Original human .nlp contiguous span; aligned verbalizations determine crop boundaries only."
        audit["excerpts"].append({"id": row["id"], "original_start": first, "original_stop": last,
                                  "first_padding_seconds": first_padding, "last_padding_seconds": last_padding,
                                  "max_word_seconds": max(float(r["endTs"]) - float(r["ts"]) for r in span)})
    return records, audit


def prepare_group(group):
    rejected = []
    for candidate in group:
        call_id = candidate["File ID"]
        records = earnings_call((call_id, "confirmation", True, None), short_fractions=(.15, .35, .55, .75))
        if len(records) != 5:
            rejected.append({"id": call_id, "reason": "Insufficient fully aligned reference spans/source coverage"})
            continue
        try:
            retained, audit = audit_call(records)
            return retained, audit, rejected
        except ValueError as error:
            rejected.append({"id": call_id, "reason": str(error)})
    raise RuntimeError(f"No valid confirmation call in predeclared group: {rejected}")


def main():
    manifest = OUT / "manifest_confirmation.json"
    if manifest.exists():
        print("Frozen confirmation manifest already exists:", manifest)
        return
    existing = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    seen_ids = {r["source_recording"] for r in existing if r["dataset"] == "Earnings22"} | {"4475604"}
    excluded_speakers = {r["speaker_id"] for r in existing if r["dataset"] == "LibriSpeech"}
    with (OUT / "earnings22_metadata.csv").open(encoding="utf-8", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    seen_groups = {r["Language Family + Area Based"] for r in metadata if r["File ID"] in seen_ids}
    candidates = [r for r in metadata if r["File ID"] not in seen_ids
                  and r["Language Family + Area Based"] not in seen_groups
                  and 600 <= int(r["File Length (seconds)"]) <= 1800]
    groups = sorted({r["Language Family + Area Based"] for r in candidates})[:2]
    ordered = [sorted([r for r in candidates if r["Language Family + Area Based"] == group],
                      key=lambda r: r["File ID"]) for group in groups]
    freeze = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "selection": "First two lexicographic previously unseen company language-family groups with 600–1800s calls; ascending source IDs within each. Structural reference failures alone permit next-ID fallback. Four short spans at .15/.35/.55/.75 duration. Twelve LibriSpeech test clips from new speakers, seed20260929.",
              "candidate_groups": ordered, "excluded_speakers": sorted(excluded_speakers),
              "whisper": {"models": ["large-v1-int8-convrot", "large-v3-int8-convrot"], "beam": 5, "batch": 1,
                          "baseline_context": "legacy >=60 windows", "final_context": "shipped English full v3 >30s off, otherwise legacy"},
              "canary": {"baseline": {"chunk_seconds": 10, "batch": 1, "tokens": 256},
                         "final": {"chunk_seconds": "Auto:10<=30s else12", "batch": 8, "tokens": 256}},
              "policy_file_sha256": {name: sha256(ROOT / name) for name in (
                  "modules/whisper/faster_whisper_inference.py", "modules/whisper/canary_qwen_inference.py")},
              "rule": "Do not tune settings or select recordings using these ASR results."}
    write_json(OUT / "confirmation_protocol_frozen.json", freeze)
    with ThreadPoolExecutor(4) as pool:
        call_jobs = [pool.submit(prepare_group, group) for group in ordered]
        libri_jobs = [pool.submit(librispeech, subset, 6, excluded_speakers=excluded_speakers,
                                  seed=20260929, split="confirmation") for subset in ("test-clean", "test-other")]
        records, audits, rejected = [], [], []
        for job in call_jobs:
            selected, audit, exclusions = job.result()
            records.extend(selected)
            audits.append(audit)
            rejected.extend(exclusions)
        for job in libri_jobs:
            records.extend(job.result())
    assert len(records) == 22
    write_json(OUT / "confirmation_reference_audit.json", {"sources": audits, "structural_rejections": rejected,
        "limitation": "Human published references with structural alignment audit; not an independent listening certification. Short excerpts overlap the two long calls."})
    write_json(manifest, records)
    print(manifest, len(records), sum(r["duration_seconds"] for r in records) / 60, "minutes")


if __name__ == "__main__":
    main()
