"""Audit original human references and preserve rejected manifest revisions.

Structural/provenance checks are not an independent human listening review.
Forced-alignment verbalizations are mapped back to original written references;
ASR output is never used to author a replacement reference.
"""
import bisect
import csv
from difflib import SequenceMatcher
import json

from english_benchmark_corpus import OUT, punctuation_text


def load_source(call_id):
    folder = OUT / "earnings22"
    with (folder / (call_id + ".nlp")).open(encoding="utf-8", newline="") as handle:
        original = list(csv.DictReader(handle, delimiter="|"))
    with (folder / (call_id + ".aligned.nlp")).open(encoding="utf-8", newline="") as handle:
        aligned = list(csv.DictReader(handle, delimiter="|"))
    operations = SequenceMatcher(None, [r["token"] for r in original], [r["token"] for r in aligned], autojunk=False).get_opcodes()
    return original, aligned, operations


def find_span(aligned, text, approximate_start):
    tokens = [r.get("prepunctuation", "") + r["token"] + r.get("punctuation", "") for r in aligned]
    joined = " ".join(tokens)
    offsets, offset = [], 0
    for token in tokens:
        offsets.append(offset)
        offset += len(token) + 1
    matches, found = [], joined.find(text)
    while found >= 0:
        start = bisect.bisect_left(offsets, found)
        end = bisect.bisect_right(offsets, found + len(text) - 1) - 1
        first_time = aligned[start].get("ts")
        if offsets[start] == found and (first_time or (start == 0 and approximate_start == 0)) and aligned[end].get("endTs"):
            matches.append((abs(float(first_time or 0) - approximate_start), start, end))
        found = joined.find(text, found + 1)
    if not matches:
        raise RuntimeError("Cannot locate original aligned excerpt span")
    _, start, end = min(matches)
    return start, end


def original_span(operations, start, end):
    original_indices = []
    stop = end + 1
    for kind, left, right, astart, astop in operations:
        if kind == "delete":
            if start < astart < stop:
                original_indices.extend(range(left, right))
            continue
        overlap_start, overlap_stop = max(start, astart), min(stop, astop)
        if overlap_start >= overlap_stop:
            continue
        if kind == "equal":
            original_indices.extend(range(left + overlap_start - astart, left + overlap_stop - astart))
        elif kind == "replace":
            if not (start <= astart and astop <= stop):
                raise RuntimeError("Excerpt starts/ends inside a verbalized human token")
            original_indices.extend(range(left, right))
        # An inserted verbalization such as 'dollars' is already expressed by
        # the original currency symbol mapped in the surrounding span.
    if not original_indices:
        raise RuntimeError("No original human words mapped to excerpt")
    return min(original_indices), max(original_indices) + 1


def main():
    source_manifest = OUT / "manifest_earnings_v1_unvalidated.json"
    if not source_manifest.exists():
        source_manifest = OUT / "manifest_earnings.json"
    source_rows = json.loads(source_manifest.read_text(encoding="utf-8"))
    sources, excluded_sources, loaded = {}, [], {}
    for call_id in dict.fromkeys(r["source_recording"] for r in source_rows):
        original, aligned, operations = load_source(call_id)
        loaded[call_id] = original, aligned, operations
        timed = [r for r in aligned if r.get("ts") and r.get("endTs")]
        fraction = len(timed) / len(aligned)
        reversals = sum(float(b["ts"]) < float(a["ts"]) for a, b in zip(timed, timed[1:]))
        invalid = fraction < .95 or reversals > 0
        sources[call_id] = {"original_human_tokens": len(original), "verbalized_alignment_tokens": len(aligned),
                            "timed_alignment_tokens": len(timed), "aligned_fraction": fraction,
                            "timestamp_reversals": reversals, "excluded": invalid}
        if invalid:
            excluded_sources.append(call_id)

    excerpts, excluded_ids, updated = {}, set(), {}
    for filename in ("manifest_earnings.json", "manifest.json", "manifest_boundaries.json"):
        path = OUT / filename
        if not path.exists():
            continue
        prior = path.with_name(path.stem + "_v1_unvalidated.json")
        if not prior.exists():
            prior.write_bytes(path.read_bytes())
        records = json.loads(prior.read_text(encoding="utf-8"))
        retained = []
        for row in records:
            call_id = row.get("source_recording")
            if row.get("dataset") != "Earnings22":
                retained.append(row)
                continue
            if call_id in excluded_sources:
                excluded_ids.add(row["id"])
                continue
            original, aligned, operations = loaded[call_id]
            original_row = dict(row)
            if row.get("excerpt_end_seconds") is not None:
                start, end = find_span(aligned, row["reference_text"], row["excerpt_start_seconds"])
                first, last = original_span(operations, start, end)
                row["reference_text"] = punctuation_text(original[first:last])
                span = aligned[start:end + 1]
                timed = [r for r in span if r.get("ts") and r.get("endTs")]
                first_offset = float(span[0].get("ts") or 0) - row["excerpt_start_seconds"]
                last_offset = row["excerpt_end_seconds"] - float(span[-1]["endTs"])
                if row["excerpt_start_seconds"] > 0 and (abs(first_offset) > .11 or abs(last_offset) > .11):
                    raise RuntimeError(f"Incorrect excerpt crop bounds: {row['id']}")
                excerpts[row["id"]] = {"contiguous_original_human_reference": True,
                    "original_token_start": first, "original_token_stop": last,
                    "aligned_token_start": start, "aligned_token_stop": end + 1,
                    "first_boundary_padding_seconds": first_offset, "last_boundary_padding_seconds": last_offset,
                    "words": last - first, "duration_seconds": row["duration_seconds"],
                    "words_per_minute": (last - first) * 60 / row["duration_seconds"],
                    "max_aligned_word_seconds": max(float(r["endTs"]) - float(r["ts"]) for r in timed),
                    "max_between_word_gap_seconds": max([float(b["ts"]) - float(a["endTs"]) for a, b in zip(timed, timed[1:])], default=0),
                    "local_alignment_fraction": len(timed) / len(span),
                    "reference_restored_from_verbalized_alignment": row["reference_text"] != original_row["reference_text"]}
            else:
                assert row["reference_text"] == punctuation_text(original)
            row["reference_revision"] = 2
            row["source_aligned_word_fraction"] = sources[call_id]["aligned_fraction"]
            row["reference_provenance"] = "Original published human .nlp transcript; forced-aligned verbalizations used only to map crop boundaries back to the original human token span."
            retained.append(row)
        updated[path] = retained
    # Do not replace any manifest until the complete audit succeeds.
    for path, retained in updated.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(retained, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    audit = {"version": 2,
        "criterion": "Reject source recordings with <95% timed alignment words or timestamp reversals. Map every retained excerpt to a contiguous ORIGINAL human .nlp token span; require nonzero crop-boundary padding <=110ms. Aligned files verbalize human numbers/acronyms and are not used as final references.",
        "excluded_source_ids": excluded_sources, "excluded_ids": sorted(excluded_ids), "sources": sources,
        "retained_excerpt_audits": excerpts,
        "observed_invalid_reference": "4475604_short2 assigned a six-word greeting to audio at941.77–965.31s. Independent model outputs exposed the mismatch; global21.29%alignment independently fails the source-quality criterion. Entire source excluded symmetrically.",
        "limitations": "Structural/provenance checks cannot certify all human words or automatic boundaries are perfect. This is not a listening audit. No machine transcription replaces human references."}
    path = OUT / "reference_audit.json"
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(path, "excluded", sorted(excluded_ids), "retained excerpts", len(excerpts))


if __name__ == "__main__":
    main()
