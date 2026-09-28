"""Supplement continuous earnings calls with reconstructed public-talk audio.

Uses original human TED-LIUM text, not Distil-Whisper pseudo-labels.
The distributor concatenates original labeled segments; these are explicitly
NOT represented as untouched continuous recordings or punctuation references.
"""
import concurrent.futures
import json
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

from english_benchmark_corpus import OUT, download, sha256, write_json


def prepare(subset):
    repo = "distil-whisper/tedlium-long-form"
    revision = "ea3a78479fb5337761f359abcd4e883d2d6e3c5b"
    filename = {"validation": "data/validation-00000-of-00001-9ed099229d0cbe10.parquet",
                "test": "data/test-00000-of-00001-7a1bb92f62e929b8.parquet"}[subset]
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
    path = download(url, OUT / "downloads" / Path(filename).name)
    records = pq.read_table(path).to_pylist()
    chosen = {"validation": {"Brian_Cox", "Elizabeth_Gilbert"}, "test": {"DanBarber", "JaneMcGonigal"}}[subset]
    folder = OUT / "tedlium"
    folder.mkdir(exist_ok=True)
    output = []
    for row in records:
        if row["speaker_id"] not in chosen:
            continue
        audio = folder / (row["speaker_id"] + ".wav")
        audio.write_bytes(row["audio"]["bytes"])
        duration = sf.info(audio).duration
        output.append({"id": "tedlium_" + row["speaker_id"], "split": "tuning" if subset == "validation" else "heldout",
                       "dataset": "TED-LIUM reconstructed long form", "subset": subset, "category": "long",
                       "audio_path": str(audio), "reference_text": row["text"], "duration_seconds": duration,
                       "source_recording": row["speaker_id"], "speaker_id": row["speaker_id"],
                       "source_url": f"https://huggingface.co/datasets/{repo}", "source_revision": revision,
                       "audio_source_url": url, "audio_sha256": sha256(audio), "punctuation_reference": False,
                       "continuous": False, "construction": "Original human-labeled TED-LIUM segments concatenated in recording order by dataset publisher.",
                       "reference_provenance": "Original TED-LIUM human transcript text; not model-generated pseudo labels.",
                       "reference_limitations": "Concatenation removes unlabeled gaps; lowercase unpunctuated reference; occasional reference errors possible.",
                       "license": "Original TED-LIUM CC BY-NC-ND 3.0; evaluation material kept outside source distribution."})
    return output


if __name__ == "__main__":
    rows = []
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        for result in pool.map(prepare, ["validation", "test"]):
            rows.extend(result)
    path = OUT / "manifest_tedlium.json"
    write_json(path, rows)
    print(path, len(rows), sum(r["duration_seconds"] for r in rows))
