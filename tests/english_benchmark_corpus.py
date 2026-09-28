"""Reproducible English evaluation material with published human references.

Audio/references stay under outputs/benchmarks, not in the source distribution.
This intentionally does not claim that a published reference is error-free.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
import subprocess
import tarfile

import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/benchmarks/english_corpus"
EARNINGS_REVISION = "c05ab6fd8b4b627d123c922a22a39e993dd37635"
RAW = f"https://raw.githubusercontent.com/revdotcom/speech-datasets/{EARNINGS_REVISION}/earnings22/"
MEDIA = f"https://media.githubusercontent.com/media/revdotcom/speech-datasets/{EARNINGS_REVISION}/earnings22/media/"


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size:
        return target
    temp = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with temp.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
    temp.replace(target)
    return target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def punctuation_text(rows: list[dict]) -> str:
    return " ".join(
        row.get("prepunctuation", "") + row["token"] + row.get("punctuation", "")
        for row in rows
    )


def render(source: Path, target: Path, start: float = 0, end: float | None = None) -> float:
    if not target.exists():
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source), "-ss", str(start)]
        if end is not None:
            cmd += ["-t", str(end - start)]
        subprocess.run(cmd + ["-ac", "1", "-ar", "16000", str(target)], check=True)
    return float(sf.info(target).duration)


def earnings_call(spec: tuple, short_fractions=(0.25, 0.55)) -> list[dict]:
    call_id, split, include_long, long_end = spec
    folder = OUT / "earnings22"
    folder.mkdir(parents=True, exist_ok=True)
    audio = download(MEDIA + call_id + ".mp3", folder / (call_id + ".mp3"))
    ref = download(RAW + "transcripts/nlp_references/" + call_id + ".nlp", folder / (call_id + ".nlp"))
    aligned = download(RAW + "transcripts/force_aligned_nlp_references/" + call_id + ".aligned.nlp", folder / (call_id + ".aligned.nlp"))
    rows = list(csv.DictReader(io.StringIO(ref.read_text(encoding="utf-8")), delimiter="|"))
    timed = list(csv.DictReader(io.StringIO(aligned.read_text(encoding="utf-8")), delimiter="|"))
    aligned_fraction = sum(bool(row.get("ts") and row.get("endTs")) for row in timed) / max(len(timed), 1)
    if aligned_fraction < 0.95:
        print(f"Excluded {call_id}: only {aligned_fraction:.2%} of human words have usable source alignment", flush=True)
        return []
    with (OUT / "earnings22_metadata.csv").open(encoding="utf-8", newline="") as handle:
        metadata = next(r for r in csv.DictReader(handle) if r["File ID"] == call_id)
    base = {
        "split": split, "dataset": "Earnings22", "source_recording": call_id,
        "source_revision": EARNINGS_REVISION,
        "source_url": "https://github.com/revdotcom/speech-datasets/tree/main/earnings22",
        "audio_source_url": MEDIA + call_id + ".mp3", "reference_source_url": RAW + "transcripts/nlp_references/" + call_id + ".nlp",
        "source_audio_sha256": sha256(audio), "source_reference_sha256": sha256(ref),
        "source_aligned_word_fraction": aligned_fraction,
        "punctuation_reference": True, "reference_provenance": "Published human transcript; timestamps are automatic forced alignment.",
        "accent_group": metadata["Language Family + Area Based"], "company_country": metadata["Country by Ticker"],
        "accent_note": "Dataset country/group metadata describes the company; individual speakers may have different accents.",
        "reference_limitations": "Human references may contain errors or omit fillers; punctuation is stylistic. Inaudible/event tags are removed for WER. Excerpts use automatic alignment, so boundaries are approximate.",
    }
    result = []
    if include_long:
        selected = rows
        end = None
        if long_end is not None:
            candidates = [(i, row) for i, row in enumerate(timed) if row.get("endTs") and float(row["endTs"]) <= long_end and row.get("punctuation") in {".", "?", "!"}]
            index, last = candidates[-1]
            selected = timed[:index + 1]
            end = float(last["endTs"]) + 0.1
        target = folder / (call_id + "_long.wav")
        duration = render(audio, target, 0, end)
        result.append({**base, "id": "earnings22_" + call_id + "_long", "category": "long", "continuous": True,
                       "audio_path": str(target), "duration_seconds": duration, "reference_text": punctuation_text(selected),
                       "excerpt_start_seconds": 0, "excerpt_end_seconds": end})

    # Choose sentence-bounded, fully aligned 8–25 second spans without unreadable tokens.
    # Deterministic time targets are chosen before any model is evaluated.
    duration = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(audio)]))
    sentence_starts = [0] + [i + 1 for i, r in enumerate(timed[:-1]) if r.get("punctuation") in {".", "?", "!"}]
    used = set()
    for number, fraction in enumerate(short_fractions, 1):
        candidates = []
        for start_index in sentence_starts:
            for end_index in range(start_index + 5, min(start_index + 80, len(timed))):
                span = timed[start_index:end_index + 1]
                if not all(r.get("ts") and r.get("endTs") and "<" not in r["token"] for r in span):
                    break
                if span[-1].get("punctuation") not in {".", "?", "!"}:
                    continue
                start, end = float(span[0]["ts"]), float(span[-1]["endTs"])
                if 8 <= end - start <= 25 and start_index not in used:
                    candidates.append((abs(start - duration * fraction), start_index, span, start, end))
                    break
        if not candidates:
            continue
        _, start_index, span, start, end = min(candidates, key=lambda x: x[0])
        used.add(start_index)
        before = float(timed[start_index - 1].get("endTs") or start) if start_index else 0
        start = max(before, start - 0.08)
        end += 0.08
        target = folder / f"{call_id}_short{number}.wav"
        clip_duration = render(audio, target, start, end)
        result.append({**base, "id": f"earnings22_{call_id}_short{number}", "category": "short", "continuous": True,
                       "audio_path": str(target), "duration_seconds": clip_duration, "reference_text": punctuation_text(span),
                       "excerpt_start_seconds": start, "excerpt_end_seconds": end})
    print("Prepared", call_id, len(result), flush=True)
    return result


def librispeech(subset: str, count: int = 6, *, excluded_speakers=None, seed=20260928, split=None) -> list[dict]:
    url = f"https://www.openslr.org/resources/12/{subset}.tar.gz"
    archive = download(url, OUT / "downloads" / f"{subset}.tar.gz")
    target_dir = OUT / "librispeech"
    target_dir.mkdir(exist_ok=True)
    selected = []
    with tarfile.open(archive) as tar:
        # One utterance per speaker, deterministic shuffle, 5–20 seconds.
        refs = {}
        for member in tar.getmembers():
            if member.name.endswith(".trans.txt"):
                for line in tar.extractfile(member).read().decode().splitlines():
                    utterance, text = line.split(" ", 1)
                    refs[utterance] = text
        members = [m for m in tar.getmembers() if m.name.endswith(".flac")]
        random.Random(seed).shuffle(members)
        speakers = set(excluded_speakers or ())
        for member in members:
            utterance = Path(member.name).stem
            speaker = utterance.split("-")[0]
            if speaker in speakers:
                continue
            data = tar.extractfile(member).read()
            duration = sf.info(io.BytesIO(data)).duration
            if not 5 <= duration <= 20:
                continue
            path = target_dir / (utterance + ".flac")
            path.write_bytes(data)
            selected.append({"id": "librispeech_" + utterance, "split": split or ("tuning" if subset.startswith("dev") else "heldout"),
                             "dataset": "LibriSpeech", "subset": subset, "category": "short", "continuous": True,
                             "audio_path": str(path), "duration_seconds": duration, "reference_text": refs[utterance],
                             "speaker_id": speaker, "source_recording": utterance.rsplit("-", 1)[0],
                             "source_url": "https://www.openslr.org/12/", "audio_source_url": url,
                             "audio_sha256": sha256(path), "punctuation_reference": False,
                             "reference_provenance": "Published LibriSpeech human book-text reference, aligned to recorded speech; uppercase without punctuation.",
                             "reference_limitations": "Book-reading domain; speaker accents are not independently labeled; reference errors remain possible."})
            speakers.add(speaker)
            if len(selected) >= count:
                break
    print("Prepared", subset, len(selected), flush=True)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--earnings-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    download(RAW + "metadata.csv", OUT / "earnings22_metadata.csv")
    download(RAW + "LICENSE.md", OUT / "earnings22_LICENSE.md")
    specs = [("4481221", "tuning", True, None), ("4482383", "tuning", True, 600),
             ("4482311", "tuning", False, None), ("4462231", "heldout", True, None),
             ("4469088", "heldout", True, 600), ("4475604", "heldout", False, None)]
    records = []
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        for result in pool.map(earnings_call, specs):
            records.extend(result)
    write_json(OUT / "manifest_earnings.json", records)
    if not args.earnings_only:
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            for result in pool.map(librispeech, ["dev-clean", "dev-other", "test-clean", "test-other"]):
                records.extend(result)
    write_json(OUT / "manifest.json", records)
    # Published forced alignments verbalize numbers/acronyms. Final references
    # must be mapped back to the original human text before anyone scores them.
    from english_benchmark_reference_audit import main as audit_references
    audit_references()
    audited = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({"files": len(audited), "seconds": sum(r["duration_seconds"] for r in audited), "manifest": str(OUT / "manifest.json")}))


if __name__ == "__main__":
    main()
