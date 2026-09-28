# English transcription validation — 2026-09-28

The measured Whisper defaults remain beam size **5**, standard decoding, encoder batch size **1**, 30-second windows, and word timestamps. For **explicitly selected English and full Whisper large-v3**, the app now disables previous-text conditioning on recordings longer than 30 seconds. Clips of 30 seconds or less keep the selected setting. Large-v1 retains its existing context policy, including the existing safeguard for recordings spanning at least 60 windows. This applies to original-precision and ConvRot INT8 large-v3; turbo, distilled models, and automatic language detection are unchanged.

These settings improve the tested cases; they do not guarantee the minimum possible error on every recording. Corpus quality, punctuation style, accents, and audio conditions still matter.

## Corpus and reference audit

The main corpus has **38 English recordings, 87.43 minutes**: 24 LibriSpeech short clips, 10 Earnings22 short excerpts, and four continuous Earnings22 recordings/excerpts ranging from 9.8 to 41.3 minutes. A separate TED-LIUM supplement has four reconstructed long recordings totaling 55.15 minutes. The publisher concatenated labeled segments, so these are distinguished from continuous recordings. Fifteen additional contiguous Earnings22 clips, totaling 9.30 minutes, test the 30-second boundary; these overlap source recordings and are not independent extra test data.

- [LibriSpeech](https://www.openslr.org/12/) supplies published English audiobook references, without punctuation. Six clips per dev-clean/dev-other/test-clean/test-other subset were selected deterministically, with distinct speakers within each subset.
- [Earnings22](https://github.com/revdotcom/speech-datasets/tree/main/earnings22) supplies human transcripts with punctuation. Retained companies are based in India, Australia, France, the United States, and South Africa. Company geography is not a verified accent label for every speaker. Tuning and held-out calls are distinct.
- [TED-LIUM long form](https://huggingface.co/datasets/distil-whisper/tedlium-long-form) supplies original human TED-LIUM text, not Whisper pseudo-labels. Its references are unpunctuated. Validation talks were used for tuning and test talks for held-out checks.

The supplied Earnings22 alignment for recording **4475604** was invalid: only 21.29% of words had timestamps, and one short excerpt assigned a six-word greeting to 23.54 seconds of unrelated later speech. Every excerpt from that recording was excluded symmetrically from baseline and candidate scores. The other five sources have 96.7–98.8% timed alignment tokens and no reversed timestamps.

Forced-alignment files also verbalize numbers and acronyms. Final references were restored to exact contiguous spans of the **original human transcript**, using alignment only to find audio boundaries. All 27 retained derived excerpts passed this mapping and crop-boundary checks. Original manifests, predictions, and exclusions remain available for audit. This is a structural/provenance audit, not a claim that independent human listening established error-free captions.

Audio is kept under `outputs/benchmarks/english_corpus`, outside the source distribution. Source revisions and hashes are recorded. Respect the source licenses: LibriSpeech CC BY 4.0; Earnings22 transcript text CC BY-SA 4.0; original TED-LIUM CC BY-NC-ND 3.0.

A separate **22-file, 52.47-minute confirmation set** was selected after freezing all final policies. It contains previously unused 21.0/27.9-minute calls from companies based in Russia and Argentina, four short excerpts from each, and six clean/six other LibriSpeech clips from previously unused speakers. New company language-family groups and ascending source IDs determine call selection; two sources failed the predeclared 95% alignment-coverage rule before any ASR and were replaced by the next IDs. The retained sources have 97.21%/99.79% timed words, no reversed timestamps, and 80-ms crop padding. The frozen protocol, separate audit, and manifest are saved as `confirmation_protocol_frozen.json`, `confirmation_reference_audit.json`, and `manifest_confirmation.json`. Confirmation results do not drive further tuning; the short excerpts overlap their corresponding long calls.

## Whisper quality

All figures below use ConvRot INT8 and audited original human references. WER is pooled edit count divided by pooled reference-word count after OpenAI English normalization. Punctuation F1 measures exact `.,!?;:` marks at aligned word boundaries; it is a custom diagnostic, **not** the published Punctuation Error Rate metric. Punctuation is scored only where references contain it.

| Large-v3 evaluation | WER before → final | Punctuation F1 before → final |
|---|---:|---:|
| Main tuning, 18 short clips | 7.11% → **7.11%** | 55.74% → **55.74%** |
| Main held-out, 16 short clips | 6.48% → **6.48%** | 49.12% → **49.12%** |
| Main tuning, 2 continuous long files | 51.20% → **7.29%** | 39.96% → **67.06%** |
| Main held-out, 2 continuous long files | 10.16% → **9.72%** | 50.07% → **49.23%** |
| TED-LIUM tuning, 2 long files | 4.30% → **3.54%** | Not scored |
| TED-LIUM held-out, 2 long files | 4.72% → **2.38%** | Not scored |
| Boundary tuning, 6 clips over 30 seconds | 13.26% → **10.22%** | 47.93% → **45.93%** |
| Boundary held-out, 4 clips over 30 seconds | 9.98% → **9.51%** | 47.51% → **46.99%** |

The large tuning improvement comes mainly from a 20.6-minute call where previous-text conditioning caused repeated passages. Original FP16 large-v3 reproduced the problem, establishing that it was not specific to quantization. The final runtime safeguard was exercised on **all 57 main, TED-LIUM, and boundary inputs**; every resulting transcript exactly matched the intended short-context/long-no-context candidate.

There are tradeoffs: long held-out punctuation F1 decreases slightly, and one extra word error appears among the two held-out 30–34-second clips. The 45–60-second clips improve sufficiently to reduce their combined held-out boundary WER. Short clips preserve their prior transcripts exactly.

The long-input no-context candidate was selected on tuning data. The duration-dependent retention of short context was finalized after examining the short/long results, so the main held-out set is reused validation, not an untouched final test. The additional boundary excerpts check the fixed rule, but share source recordings with the main corpus. A larger independent listening-verified evaluation would be needed for stronger generalization claims.

**Large-v1 should keep context.** Its unchanged default obtains main long WER of 7.44% on tuning and 10.18% held out; short WER is 7.87% and 5.67%. Disabling context worsens TED-LIUM tuning WER from 2.70% to 5.14% and held-out WER from 1.92% to 2.90%. A shared context-off preset for both models is therefore inappropriate.

Beam 3 had a small large-v3 tuning gain but worsened held-out long WER from 9.72% to 10.01%. Beam 8 harmed punctuation and/or short accuracy. Hallucination-silence thresholds of one and two seconds showed no consistent improvement. Batched decoder inference at batch 8 was faster but worsened main long tuning WER to 17.77% for v1 and 16.56% for v3; it is not the quality default.

### Untouched confirmation after policy freeze

All 88 runs (22 files × two models × baseline/final) completed successfully. No configuration was adjusted using these results.

| Confirmation subset | WER before → final | Punctuation F1 before → final |
|---|---:|---:|
| Large-v1, 20 short clips | 8.78% → **8.78%** | 38.55% → **38.55%** |
| Large-v1, 2 continuous calls | 17.79% → **17.79%** | 46.79% → **46.79%** |
| Large-v3, 20 short clips | 12.40% → **12.40%** | 39.51% → **39.51%** |
| Large-v3, 2 continuous calls | 18.56% → **18.15%** | 38.81% → **45.10%** |

Large-v3 long errors fall from 1,135 to 1,110 over 6,115 reference words. Both new calls improve WER and punctuation individually. All 20 short v3 transcripts and all 22 v1 transcripts are byte-identical to their baselines. These harder calls also show that v1 can remain better in absolute WER; the change does not make v3 universally superior. This set was untouched by this tuning exercise; overlap with model pretraining data is unknown. The original human-reference and overlap limitations still apply.

## Whisper speed and memory

Tests used RTX A6000 48 GB GPUs, with independent Whisper workers on GPUs 1–4 and Canary workers on GPUs 5–7. GPU 0 served browser testing. Existing unrelated GPU applications were left running. Results are measurements on this environment, not guarantees for other cards. The full environment snapshot is saved alongside results: Python 3.12.14, PyTorch 2.13.0+cu130, faster-whisper 1.2.1, CTranslate2 4.8.2, and Triton 3.7.1.

The table uses the **second complete timed pass**, with zero Triton tuning time, over the same 30.44 minutes of continuous tuning audio. Large-v1 uses context; large-v3 uses the chosen no-context behavior. Offloading is disabled. NVML measures this process's GPU memory, including allocations outside PyTorch; it samples memory and may miss brief transients.

| Standard encoder batch | v1 seconds | v3 seconds | Maximum process VRAM across the two models |
|---:|---:|---:|---:|
| **1** | **45.23** | **43.17** | **3.35 GiB** |
| 2 | 48.12 | 42.46 | 3.35 GiB |
| 4 | 48.76 | 43.00 | 3.35 GiB |
| 8 | 56.58 | 50.78 | 3.72 GiB |
| 16 | 63.46 | 59.62 | 4.36 GiB |
| 32 | 71.77 | 68.26 | 6.23 GiB |

Encoder prefetch often prepares windows that cannot be reused after timestamp-based decoding advances by less than 30 seconds. Higher batches therefore did not provide a useful quality-default speedup. Batch 2's small v3 difference is within observed run variability. Batch 4 also changed some words, despite the same v1 aggregate WER. Whisper remains at batch 1 across VRAM tiers; larger memory alone is not a reason to increase it. Physical 6/8/10/12/16/24/32 GB cards were not available for direct validation.

Cold model loading, graph creation, and first-shape Triton tuning are saved separately in each run's `cold.json`; new shapes can also appear later. First-run timings are not presented as steady-state speed gains.

A separate paired, fully warmed large-v3 run measures the context fix itself. On the same 30.44 minutes of continuous tuning audio, the second pass takes **98.58 seconds before and 48.41 seconds after (2.04×)**, with zero Triton tuning in either pass and process peaks of 3.37/3.35 GiB. The speedup is partly the removal of the repeated-passage failure on the Indian call; it should not be generalized to every input. The first warmed pass was 104.75/43.59 seconds, showing the remaining timing variation on the shared machine.

## Canary-Qwen

Canary quality, automatic chunk selection, batch tiers, and their separately audited results are documented in [Canary English benchmarks](canary-english-benchmarks.md). Its final policy preserves the original 10-second chunking on short inputs and uses the measured long-input setting; positive manual chunk values retain their meaning. See that report for final batch memory measurements and short/long tradeoffs.

On the 20-file main-plus-TED held-out validation, final Auto chunking at batch 8 changes WER from **6.852% to 6.720%** and punctuation F1 from **41.03% to 42.22%**. Short WER remains **4.251%** and short punctuation F1 remains **39.22%**; long WER improves **6.940% to 6.803%**. Auto uses 10-second chunks for files up to 30 seconds and 12-second chunks above that boundary, with 256 output tokens. Its batch-8 peak is **7.86 GiB** on these recordings. Historical Canary wall timings included CPU scoring; use the separately corrected paired-transcription measurements in the Canary report for speed claims.

On the **untouched 22-file confirmation**, Canary WER improves **16.509% → 16.433%** and punctuation F1 **32.68% → 36.91%**. Long WER changes **17.155% → 17.073%** and long punctuation **33.20% → 37.56%**. All 20 short transcripts are byte-identical (WER 8.969%, punctuation F1 18.92%). The Russian call regresses slightly in WER while the Argentine call improves, so the aggregate result does not imply every file improves.

The corrected paired measurement, which excludes CPU scoring, takes **90.72 → 28.44 seconds (3.19×)** across the confirmation files, or **67.94 → 15.16 seconds (4.48×)** for the continuous calls. These measurements compare the former 10-second/batch-1 configuration with frozen Auto/batch-8 settings on the same test GPU. The **48 GiB hardware default is batch 16**: it produces the same quality but takes **51.05 seconds overall and 18.49 seconds for the long calls**, or **1.78× and 3.67×** relative to that baseline. Its short-file timing was slower, and it ran on a different GPU under changing shared-machine load. Larger batches do not guarantee lower latency. No further parameter choice was made from confirmation results.

## Reproduction and artifacts

Use the application's virtual environment with FFmpeg available. Optional benchmark dependencies include `jiwer`, `requests`, `soundfile`, `pyarrow` for TED-LIUM, and `nvidia-ml-py` for process-memory measurements. The following commands use the Linux environment; on Windows substitute `venv\Scripts\python.exe` and set `CUDA_VISIBLE_DEVICES` using the shell's environment-variable syntax.

```bash
venv/bin/python tests/english_benchmark_corpus.py
venv/bin/python tests/english_benchmark_tedlium.py
venv/bin/python tests/english_benchmark_boundaries.py
venv/bin/python tests/english_benchmark_confirmation.py

# Historical baseline, preserving the old >=60-window context guard:
CUDA_VISIBLE_DEVICES=1 venv/bin/python tests/english_benchmark_whisper.py \
  --model large-v3-int8-convrot --configs baseline --context-policy legacy --split heldout

# Actual current application behavior, including the English >30-second guard:
CUDA_VISIBLE_DEVICES=1 venv/bin/python tests/english_benchmark_whisper.py \
  --model large-v3-int8-convrot --configs baseline --context-policy shipped --split heldout

venv/bin/python tests/english_benchmark_rescore.py
venv/bin/python tests/english_benchmark_report.py outputs/benchmarks/whisper \
  --output outputs/benchmarks/whisper/summary_audited.json
```

Use `--manifest outputs/benchmarks/english_corpus/manifest_tedlium.json` or `manifest_boundaries.json` for supplemental checks. Standard batch sweeps use `--configs baseline,encoder2,encoder4,encoder8,encoder16,encoder32 --category long --repeat 2`. `--no-condition-on-previous-text` fixes the v3 candidate during parameter sweeps. Never use original unvalidated metrics for final conclusions; reporting defaults to `results.audited.jsonl` and preserves raw `results.jsonl` files.

For the untouched confirmation, use `--manifest outputs/benchmarks/english_corpus/manifest_confirmation.json --split confirmation`, testing `--context-policy legacy` and `--context-policy shipped` once for each Whisper model. The confirmation builder keeps an existing frozen manifest unchanged.

Local evidence is under `outputs/benchmarks`:

- `english_corpus/reference_audit.json`, audited manifests, and `*_v1_unvalidated.json` preserve reference provenance and exclusions.
- `whisper/v3_final_shipped_*` contains the actual final-policy checks; `v3_tuning_decoding`, `v3_heldout_context_fixed`, and `v3_tedlium_context` contain historical comparisons.
- `whisper/v1_confirmation_*`, `v3_confirmation_*`, and `confirmation_comparison.json` contain the untouched confirmation after policy freeze.
- `whisper/v1_standard_batch_profile_fixed` and `v3_standard_batch_profile` contain repeated warm batch/memory measurements.
- `whisper/v3_warm_context_repeat` contains the paired context-fix timing, with two warmed passes per configuration in one process.
- `whisper/original_v1_smoke`, `original_v3_smoke`, and `original_v3_long_diagnostic` verify original precision. Original and INT8 downloads are audited in `whisper/model_download_verification.json`: all physical model files are inside installation-relative `models/Whisper/faster-whisper`. FP16 checkpoints occupy about 2.88 GiB each; INT8 checkpoints about 1.51 GiB each.

Focused tests cover the exact 30-second boundary, model families, English selection, Windows/Linux paths, preservation of the old guard, WER/punctuation calculations, and mapping aligned verbalizations back to human references.
