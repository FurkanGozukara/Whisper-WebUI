# English ASR benchmark

Measures what the app itself produces: `run_benchmark.py` calls `BaseTranscriptionPipeline.run()`, the same
path every File / YouTube / Mic job uses, with a shipped UI preset plus optional overrides. Results and the
method are summarized in [docs/english-benchmarks.md](../../docs/english-benchmarks.md).

## Data

Only public test sets with human transcripts (about 40 GB download, 16 kHz WAV corpus about 5 GB):

| Kind | Sets |
| --- | --- |
| Short (utterances up to 40 s) | LibriSpeech test-clean / test-other, Common Voice, VoxPopuli, TED-LIUM 3, Earnings-22, AMI, GigaSpeech, SPGISpeech (Open ASR Leaderboard test sets) |
| Long (1 minute to 2 hours) | TED-LIUM long-form, Earnings-21, Earnings-22 (Open ASR Leaderboard long-form), Rev16 podcasts, Meanwhile (Colbert monologues) |

Every set is split with a fixed seed into disjoint `dev` (used for tuning) and `test` (used once, to confirm)
halves: 150 + 150 utterances per short set; long-form 52 dev files (12 h) and 68 test files (31 h).

```bash
source venv/bin/activate
python tests/asr_benchmark/download_datasets.py            # -> outputs/asr_benchmark/data
python tests/asr_benchmark/prepare_corpus.py all           # -> outputs/asr_benchmark/corpus
```

## Run and score

```bash
echo '{"name": "canary", "preset": "canary_qwen_best_quality", "whisper": {"batch_size": 16}}' > canary.json
CUDA_VISIBLE_DEVICES=0 python tests/asr_benchmark/run_benchmark.py --config canary.json \
    --manifest outputs/asr_benchmark/corpus/manifest_long.jsonl --split test --out results/canary.jsonl
python tests/asr_benchmark/score.py results/*.jsonl --split test
```

`--shard 0/2` / `--shard 1/2` split one run over two GPUs; `--repeat 2` times a warm second pass. Runs resume
where they stopped. Set `APP_DIR` to benchmark another copy of the app (for example the previous release).

## Metrics

* **WER**: the OpenAI Whisper `EnglishTextNormalizer` on reference and hypothesis (as the Open ASR Leaderboard
  and the Canary-Qwen model card), errors pooled over all words of a set. `MACRO` averages the sets.
* **Punctuation F1**: commas (`, ; :`), periods (`. ! …`) and question marks after each aligned word, on the
  sets whose references are punctuated (Common Voice, Earnings-22, GigaSpeech, SPGISpeech, Earnings-21, Rev16,
  Meanwhile). Casing is not scored.
* **RTFx**: audio seconds per second of processing, **peakMB**: the process's GPU memory (NVML, CUDA context
  included). Speed depends on the GPU and on other load; compare runs made on the same idle GPU.
