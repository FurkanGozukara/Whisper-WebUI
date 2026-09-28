"""Score benchmark results: normalized WER (OpenAI Whisper EnglishTextNormalizer, as the Open ASR Leaderboard),
punctuation F1 (comma / period / question mark at aligned words) on punctuated references, speed and VRAM.

    python tests/asr_benchmark/score.py results/*.jsonl [--split test] [--datasets a,b] [--per-file]
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import jiwer
from whisper.normalizers import EnglishTextNormalizer

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORPUS_DIR = os.environ.get("ASR_BENCHMARK_CORPUS", os.path.join(APP_DIR, "outputs", "asr_benchmark", "corpus"))
NORM = EnglishTextNormalizer()

PUNCT_CLASS = {".": "P", "!": "P", "…": "P", ",": "C", ";": "C", ":": "C", "?": "Q"}
WORD_RE = re.compile(r"[A-Za-z0-9À-ɏ][A-Za-z0-9À-ɏ'’%$&@.\-]*|[.,!?;:…]")


def load_manifests():
    refs = {}
    for m in glob.glob(os.path.join(CORPUS_DIR, "manifest_*.jsonl")):
        with open(m, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                refs[r["id"]] = r
    return refs


def clean_ref(text):
    # Rev16 speaker/unknown markup and bracketed events are not words anybody should transcribe
    text = re.sub(r"<unk:([^>]*)>", r"\1", text)
    return text


def punct_tokens(text):
    """Words (lowercased, stripped of outer punctuation) with the punctuation class that follows each."""
    words, classes = [], []
    for tok in WORD_RE.findall(text.replace("’", "'")):
        if len(tok) == 1 and tok in PUNCT_CLASS:
            if words and classes[-1] is None:
                classes[-1] = PUNCT_CLASS[tok]
            continue
        # a trailing period on a word ("etc." / "Inc." / end of sentence) is punctuation, not word
        trail = ""
        while tok and tok[-1] in ".-" and not re.search(r"\d\.\d*$", tok[-3:] if len(tok) > 2 else tok):
            trail = tok[-1] + trail
            tok = tok[:-1]
        w = re.sub(r"[^a-z0-9']", "", tok.lower())
        if not w:
            continue
        words.append(w)
        classes.append(PUNCT_CLASS.get(trail[:1]) if trail and trail[0] == "." else None)
    return words, classes


def punct_counts(ref, hyp):
    rw, rc = punct_tokens(ref)
    hw, hc = punct_tokens(hyp)
    counts = defaultdict(int)
    if not rw or not hw:
        return counts
    out = jiwer.process_words(" ".join(rw), " ".join(hw))
    for chunk in out.alignments[0]:
        if chunk.type in ("equal", "substitute"):
            n = min(chunk.ref_end_idx - chunk.ref_start_idx, chunk.hyp_end_idx - chunk.hyp_start_idx)
            for k in range(n):
                r = rc[chunk.ref_start_idx + k]
                h = hc[chunk.hyp_start_idx + k]
                for c in ("P", "C", "Q"):
                    if r == c and h == c:
                        counts[c + "_tp"] += 1
                    elif h == c and r != c:
                        counts[c + "_fp"] += 1
                    elif r == c and h != c:
                        counts[c + "_fn"] += 1
    return counts


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def wer_counts(ref, hyp):
    r = NORM(ref)
    h = NORM(hyp)
    if not r.strip():
        return None
    if not h.strip():
        n = len(r.split())
        return {"s": 0, "d": n, "i": 0, "n": n}
    o = jiwer.process_words(r, h)
    return {"s": o.substitutions, "d": o.deletions, "i": o.insertions, "n": o.substitutions + o.deletions + o.hits}


def score(files, split=None, datasets=None, passes=(0,), detail=False, per_file=False):
    refs = load_manifests()
    rows = defaultdict(dict)  # config -> id -> rec (first pass)
    timing = defaultdict(lambda: defaultdict(list))
    for fn in files:
        with open(fn, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if split and r["split"] != split:
                    continue
                if datasets and r["dataset"] not in datasets:
                    continue
                timing[r["config"]][r["id"]].append(r["seconds"])
                if r.get("pass", 0) == 0:
                    rows[r["config"]][r["id"]] = r
    results = {}
    for cfg, recs in rows.items():
        agg = defaultdict(lambda: defaultdict(float))
        for rid, r in recs.items():
            ref = refs.get(rid)
            if ref is None:
                continue
            ds = r["dataset"]
            c = wer_counts(clean_ref(ref["ref"]), r["hyp"])
            if c is None:
                continue
            a = agg[ds]
            for k, v in c.items():
                a[k] += v
            a["files"] += 1
            a["audio"] += ref["duration"]
            ts = timing[cfg][rid]
            a["seconds"] += min(ts)  # best pass (warm) when repeated
            a["peak_mb"] = max(a["peak_mb"], r.get("peak_mb") or 0)
            a["errors"] += 1 if r.get("error") else 0
            if ref.get("punctuated"):
                pc = punct_counts(clean_ref(ref["ref"]), r["hyp"])
                for k, v in pc.items():
                    a["p_" + k] += v
            if per_file:
                e = c["s"] + c["d"] + c["i"]
                print(f"{cfg}\t{rid}\t{ref['duration']:.0f}s\tWER {100*e/max(1,c['n']):.2f}% "
                      f"(S{c['s']} D{c['d']} I{c['i']} N{c['n']})")
        results[cfg] = agg
    return results


def summarize(results, show=True):
    table = {}
    for cfg, agg in sorted(results.items()):
        per = {}
        tot = defaultdict(float)
        for ds, a in sorted(agg.items()):
            e = a["s"] + a["d"] + a["i"]
            w = 100 * e / a["n"] if a["n"] else 0
            ptp = sum(a.get(f"p_{c}_tp", 0) for c in "PCQ")
            pfp = sum(a.get(f"p_{c}_fp", 0) for c in "PCQ")
            pfn = sum(a.get(f"p_{c}_fn", 0) for c in "PCQ")
            pf1 = 100 * f1(ptp, pfp, pfn) if (ptp + pfp + pfn) else None
            per[ds] = {"wer": w, "s": a["s"], "d": a["d"], "i": a["i"], "n": a["n"], "files": a["files"],
                       "audio": a["audio"], "seconds": a["seconds"], "rtfx": a["audio"] / a["seconds"] if a["seconds"] else 0,
                       "peak_mb": a["peak_mb"], "errors": a["errors"], "punct_f1": pf1,
                       "comma_f1": 100 * f1(a.get("p_C_tp", 0), a.get("p_C_fp", 0), a.get("p_C_fn", 0)) if pf1 is not None else None,
                       "period_f1": 100 * f1(a.get("p_P_tp", 0), a.get("p_P_fp", 0), a.get("p_P_fn", 0)) if pf1 is not None else None,
                       "question_f1": 100 * f1(a.get("p_Q_tp", 0), a.get("p_Q_fp", 0), a.get("p_Q_fn", 0)) if pf1 is not None else None}
            for k in ("s", "d", "i", "n", "audio", "seconds", "errors", "files"):
                tot[k] += a[k]
            for k in ("P", "C", "Q"):
                for t in ("tp", "fp", "fn"):
                    tot[f"p_{k}_{t}"] += a.get(f"p_{k}_{t}", 0)
            tot["peak_mb"] = max(tot["peak_mb"], a["peak_mb"])
        macro = sum(v["wer"] for v in per.values()) / len(per) if per else 0
        e = tot["s"] + tot["d"] + tot["i"]
        ptp = sum(tot[f"p_{c}_tp"] for c in "PCQ")
        pfp = sum(tot[f"p_{c}_fp"] for c in "PCQ")
        pfn = sum(tot[f"p_{c}_fn"] for c in "PCQ")
        table[cfg] = {"per": per, "macro_wer": macro, "pooled_wer": 100 * e / tot["n"] if tot["n"] else 0,
                      "punct_f1": 100 * f1(ptp, pfp, pfn) if (ptp + pfp + pfn) else None,
                      "audio": tot["audio"], "seconds": tot["seconds"],
                      "rtfx": tot["audio"] / tot["seconds"] if tot["seconds"] else 0, "peak_mb": tot["peak_mb"],
                      "errors": tot["errors"], "files": tot["files"]}
    if show:
        dsets = sorted({ds for v in table.values() for ds in v["per"]})
        head = f"{'config':34s} " + " ".join(f"{d[:11]:>11s}" for d in dsets) + f" {'MACRO':>7s} {'POOLED':>7s} {'PUNCF1':>7s} {'RTFx':>7s} {'peakMB':>7s} {'files':>5s} {'err':>3s}"
        print(head)
        for cfg, v in table.items():
            cells = []
            for d in dsets:
                x = v["per"].get(d)
                cells.append(f"{x['wer']:11.2f}" if x else f"{'-':>11s}")
            pf = f"{v['punct_f1']:7.2f}" if v["punct_f1"] is not None else f"{'-':>7s}"
            print(f"{cfg[:34]:34s} " + " ".join(cells) + f" {v['macro_wer']:7.2f} {v['pooled_wer']:7.2f} {pf} {v['rtfx']:7.1f} {v['peak_mb']:7.0f} {int(v['files']):5d} {int(v['errors']):3d}")
        # punctuation per dataset
        pd = [d for d in dsets if any(v["per"].get(d, {}).get("punct_f1") is not None for v in table.values())]
        if pd:
            print("\npunctuation F1 (overall / comma / period / question) per punctuated dataset")
            for cfg, v in table.items():
                cells = []
                for d in pd:
                    x = v["per"].get(d)
                    if x and x["punct_f1"] is not None:
                        cells.append(f"{d[:10]}:{x['punct_f1']:.1f}/{x['comma_f1']:.1f}/{x['period_f1']:.1f}/{x['question_f1']:.1f}")
                print(f"{cfg[:34]:34s} " + "  ".join(cells))
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--split", default=None)
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--per-file", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    files = []
    for p in a.files:
        files.extend(glob.glob(p))
    res = score(files, split=a.split, datasets=set(a.datasets.split(",")) if a.datasets else None,
                per_file=a.per_file)
    t = summarize(res)
    if a.json:
        json.dump(t, open(a.json, "w"), indent=1)
