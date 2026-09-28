"""Build the English benchmark corpus (16 kHz mono PCM16 WAV + JSONL manifests) from public human-labelled sets.

Short-form (Open ASR Leaderboard ESB test sets): librispeech clean/other, common_voice, voxpopuli, earnings22,
ami, gigaspeech, spgispeech, tedlium (release3 test). Long-form: TED-LIUM long-form, Earnings-21, Earnings-22
(leaderboard long-form), Rev16 podcasts, Meanwhile (Colbert monologues).

Every dataset is split into disjoint dev (tuning) and test (confirmation) parts with a fixed seed.

    python tests/asr_benchmark/prepare_corpus.py [short|long|all] [--data-dir ...] [--out-dir ...]
"""
import glob
import io
import json
import os
import random
import re
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(APP_DIR, "outputs", "asr_benchmark", "data")
OUT = os.path.join(APP_DIR, "outputs", "asr_benchmark", "corpus")
SEED = 20260928
N_SHORT = int(os.environ.get("N_SHORT", "150"))  # per dataset per split

ESB = os.path.join(DATA, "hf-audio__esb-datasets-test-only-sorted")
SHORT_SETS = {
    "ls_clean": (os.path.join(ESB, "librispeech", "test.clean-*.parquet"), False),
    "ls_other": (os.path.join(ESB, "librispeech", "test.other-*.parquet"), False),
    "common_voice": (os.path.join(ESB, "common_voice", "*.parquet"), True),
    "voxpopuli": (os.path.join(ESB, "voxpopuli", "*.parquet"), False),
    "earnings22": (os.path.join(ESB, "earnings22", "*.parquet"), True),
    "ami": (os.path.join(ESB, "ami", "*.parquet"), False),
    "gigaspeech": (os.path.join(ESB, "gigaspeech", "*.parquet"), True),
    "spgispeech": (os.path.join(ESB, "spgispeech", "*.parquet"), True),
    "tedlium": (os.path.join(DATA, "distil-whisper__tedlium-prompted", "release3", "test-*.parquet"), False),
}


def to_16k_mono(data, sr):
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr != 16000:
        import soxr
        data = soxr.resample(data, sr, 16000, quality="HQ").astype(np.float32)
    return data


def decode(audio_field):
    raw = audio_field.get("bytes")
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    return to_16k_mono(data, sr)


def write_wav(path, audio):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    sf.write(path, audio, 16000, subtype="PCM_16")


def safe_id(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:120]


def prep_short(name, pattern, punctuated, rng):
    files = sorted(glob.glob(pattern))
    rows = []
    for f in files:
        pf = pq.ParquetFile(f)
        cols = [c for c in pf.schema_arrow.names if c != "audio"]
        t = pf.read(columns=cols)
        for i, r in enumerate(t.to_pylist()):
            rows.append((f, i, r))
    # stable order, then sample 2*N distinct (dev first N, test next N)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    chosen = []
    for k in idx:
        f, i, r = rows[k]
        text = (r.get("text") or "").strip()
        if not text or text in ("ignore_time_segment_in_scoring",):
            continue
        chosen.append((f, i, r))
        if len(chosen) >= 2 * N_SHORT:
            break
    # read audio of chosen rows grouped per file
    by_file = {}
    for c in chosen:
        by_file.setdefault(c[0], []).append(c)
    out = []
    for f, items in by_file.items():
        tab = pq.read_table(f, columns=["audio"])
        audio_col = tab.column("audio")
        for (_f, i, r) in items:
            audio = decode(audio_col[i].as_py())
            rid = r.get("id") or f"{os.path.basename(f)}_{i}"
            split = "dev" if chosen.index((_f, i, r)) < N_SHORT else "test"
            path = os.path.join(OUT, "short", name, safe_id(str(rid)) + ".wav")
            write_wav(path, audio)
            out.append({"id": f"{name}/{safe_id(str(rid))}", "dataset": name, "split": split, "path": path,
                        "ref": r["text"], "duration": round(len(audio) / 16000.0, 3), "punctuated": punctuated,
                        "form": "short"})
    return out


def prep_long():
    out = []
    rng = random.Random(SEED + 1)
    # TED-LIUM long-form: validation talks = dev, test talks = test (unpunctuated references)
    for split_name, split in (("validation", "dev"), ("test", "test")):
        for f in sorted(glob.glob(os.path.join(DATA, "distil-whisper__tedlium-long-form", "data", f"{split_name}-*.parquet"))):
            for r in pq.read_table(f).to_pylist():
                audio = decode(r["audio"])
                rid = safe_id(r["speaker_id"])
                path = os.path.join(OUT, "long", "tedlium_long", rid + ".wav")
                write_wav(path, audio)
                out.append({"id": f"tedlium_long/{rid}", "dataset": "tedlium_long", "split": split, "path": path,
                            "ref": r["text"], "duration": round(len(audio) / 16000.0, 3), "punctuated": False,
                            "form": "long"})
    # Earnings-21 (punctuated): 4 dev calls, 8 test calls, chosen by seed
    e21 = []
    for f in sorted(glob.glob(os.path.join(DATA, "hf-audio__asr-leaderboard-longform", "earnings21", "*.parquet"))):
        pf = pq.ParquetFile(f)
        meta = pf.read(columns=[c for c in pf.schema_arrow.names if c != "audio"]).to_pylist()
        for i, r in enumerate(meta):
            e21.append((f, i, r))
    order = list(range(len(e21)))
    rng.shuffle(order)
    picks = [("dev", k) for k in order[:4]] + [("test", k) for k in order[4:12]]
    for split, k in picks:
        f, i, r = e21[k]
        audio = decode(pq.read_table(f, columns=["audio"]).column("audio")[i].as_py())
        rid = safe_id(f"{r['company_name']}_{r['financial_quarter']}")
        path = os.path.join(OUT, "long", "earnings21", rid + ".wav")
        write_wav(path, audio)
        out.append({"id": f"earnings21/{rid}", "dataset": "earnings21", "split": split, "path": path,
                    "ref": r["text"], "duration": round(len(audio) / 16000.0, 3), "punctuated": True,
                    "form": "long"})
    # Earnings-22 leaderboard long-form (uppercase, unpunctuated; strip the stray leading TOKEN marker)
    e22 = []
    for f in sorted(glob.glob(os.path.join(DATA, "hf-audio__asr-leaderboard-longform", "earnings22", "*.parquet"))):
        n = pq.ParquetFile(f).metadata.num_rows
        for i in range(n):
            e22.append((f, i))
    order = list(range(len(e22)))
    rng.shuffle(order)
    picks = [("dev", k) for k in order[:3]] + [("test", k) for k in order[3:9]]
    for split, k in picks:
        f, i = e22[k]
        tab = pq.read_table(f)
        r = tab.slice(i, 1).to_pylist()[0]
        audio = decode(r["audio"])
        text = re.sub(r"^\s*TOKEN\s+", "", r["text"])
        rid = f"e22_{os.path.basename(f).split('-')[1]}_{i}"
        path = os.path.join(OUT, "long", "earnings22_long", rid + ".wav")
        write_wav(path, audio)
        out.append({"id": f"earnings22_long/{rid}", "dataset": "earnings22_long", "split": split, "path": path,
                    "ref": text, "duration": round(len(audio) / 16000.0, 3), "punctuated": False, "form": "long"})
    # Rev16 whisper subset (punctuated podcasts)
    rev = []
    for f in sorted(glob.glob(os.path.join(DATA, "distil-whisper__rev16", "whisper_subset", "*.parquet"))):
        for r in pq.read_table(f).to_pylist():
            rev.append(r)
    rev.sort(key=lambda r: int(r["file_number"]))
    order = list(range(len(rev)))
    rng.shuffle(order)
    for n, k in enumerate(order):
        r = rev[k]
        split = "dev" if n < 5 else "test"
        audio = decode(r["audio"])
        rid = f"rev16_{r['file_number']}"
        path = os.path.join(OUT, "long", "rev16", rid + ".wav")
        write_wav(path, audio)
        out.append({"id": f"rev16/{rid}", "dataset": "rev16", "split": split, "path": path,
                    "ref": r["transcription"], "duration": round(len(audio) / 16000.0, 3), "punctuated": True,
                    "form": "long"})
    # Meanwhile (~1 minute Colbert monologue segments, punctuated, upper case)
    mw = pq.read_table(glob.glob(os.path.join(DATA, "distil-whisper__meanwhile", "data", "*.parquet"))[0]).to_pylist()
    order = list(range(len(mw)))
    rng.shuffle(order)
    for n, k in enumerate(order):
        r = mw[k]
        split = "dev" if n < 32 else "test"
        audio = decode(r["audio"])
        rid = f"meanwhile_{k:02d}"
        path = os.path.join(OUT, "long", "meanwhile", rid + ".wav")
        write_wav(path, audio)
        out.append({"id": f"meanwhile/{rid}", "dataset": "meanwhile", "split": split, "path": path,
                    "ref": r["text"], "duration": round(len(audio) / 16000.0, 3), "punctuated": True,
                    "form": "medium"})
    return out


def main():
    global DATA, OUT, ESB
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("which", nargs="?", default="all", choices=["short", "long", "all"])
    parser.add_argument("--data-dir", default=DATA)
    parser.add_argument("--out-dir", default=OUT)
    args = parser.parse_args()
    DATA, OUT = args.data_dir, args.out_dir
    ESB = os.path.join(DATA, "hf-audio__esb-datasets-test-only-sorted")
    for name, (pattern, punct) in list(SHORT_SETS.items()):
        SHORT_SETS[name] = (pattern.replace(os.path.join(APP_DIR, "outputs", "asr_benchmark", "data"), DATA), punct)
    os.makedirs(OUT, exist_ok=True)
    which = args.which
    if which in ("all", "short"):
        rng = random.Random(SEED)
        allrows = []
        for name, (pattern, punct) in SHORT_SETS.items():
            rows = prep_short(name, pattern, punct, random.Random(f"{SEED}-{name}"))
            print(name, len(rows), f"{sum(r['duration'] for r in rows)/3600:.2f} h", flush=True)
            allrows.extend(rows)
        with open(os.path.join(OUT, "manifest_short.jsonl"), "w") as f:
            for r in allrows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if which in ("all", "long"):
        rows = prep_long()
        for ds in sorted(set(r["dataset"] for r in rows)):
            sel = [r for r in rows if r["dataset"] == ds]
            print(ds, len(sel), f"dev {sum(r['duration'] for r in sel if r['split']=='dev')/3600:.2f} h",
                  f"test {sum(r['duration'] for r in sel if r['split']=='test')/3600:.2f} h", flush=True)
        with open(os.path.join(OUT, "manifest_long.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
