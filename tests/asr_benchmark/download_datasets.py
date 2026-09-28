"""Download the public English test sets used by the ASR benchmark (about 40 GB).

Short form: the Open ASR Leaderboard test sets (hf-audio/esb-datasets-test-only-sorted) and the TED-LIUM 3 test
set (distil-whisper/tedlium-prompted). Long form: TED-LIUM long-form, Earnings-21 and Earnings-22 (Open ASR
Leaderboard long-form), Rev16 podcasts and the Meanwhile monologues. All transcripts are human made.

    python tests/asr_benchmark/download_datasets.py [--data-dir outputs/asr_benchmark/data]
"""
import argparse
import os
import time

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

JOBS = [
    ("hf-audio/esb-datasets-test-only-sorted",
     ["librispeech/*", "voxpopuli/*", "earnings22/*", "common_voice/*", "ami/*", "gigaspeech/*", "spgispeech/*"]),
    ("distil-whisper/tedlium-prompted", ["release3/test-*"]),
    ("distil-whisper/tedlium-long-form", ["data/*"]),
    ("hf-audio/asr-leaderboard-longform", ["earnings21/*", "earnings22/*"]),
    ("distil-whisper/rev16", ["whisper_subset/*"]),
    ("distil-whisper/meanwhile", ["data/*"]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(APP_DIR, "outputs", "asr_benchmark", "data"))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    for repo, patterns in JOBS:
        started = time.time()
        local = os.path.join(args.data_dir, repo.replace("/", "__"))
        snapshot_download(repo, repo_type="dataset", allow_patterns=patterns + ["README.md"], local_dir=local,
                          max_workers=16)
        print(f"{repo}: {time.time() - started:.0f} s -> {local}", flush=True)


if __name__ == "__main__":
    main()
