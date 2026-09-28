# Canary-Qwen English validation

The default uses greedy ConvRot INT8 inference, 256 output tokens, and automatic
chunking: 10-second chunks for inputs up to 30 seconds, 12-second chunks for longer
inputs. Batch size follows the conservative GPU memory tiers. Automatic chunking
is represented by `chunk_length=0`; positive values remain explicit overrides,
and `None` retains the previous 40-second fallback.

The shorter setting preserves the measured short-speech punctuation advantage.
Canary still sometimes omits punctuation, including on the JFK smoke recording.
Neither the chosen settings nor this corpus establish a universally lowest WER.

## Frozen-policy confirmation

After freezing the policy, a new 22-input set supplied 52.47 minutes of audio:
two continuous Earnings22 calls, eight excerpts from those calls, and 12
LibriSpeech recordings from previously unused speakers. These recordings were
untouched by this experiment before confirmation. Source selection and alignment
rejections occurred before ASR. No settings were changed after viewing these
results. See
[`confirmation_protocol_frozen.json`](../outputs/benchmarks/english_corpus/confirmation_protocol_frozen.json)
and [`confirmation_reference_audit.json`](../outputs/benchmarks/english_corpus/confirmation_reference_audit.json).

Baseline and automatic batch-8 runs used the same GPU sequentially, two passes
each. The table uses the second pass and stops timing immediately after
synchronized app transcription, before reference scoring.

| Confirmation subset | Original 10-second/batch-1 WER | Auto/batch-8 WER | Punctuation F1 before → after | Transcription seconds before → after |
| --- | ---: | ---: | ---: | ---: |
| All 22 inputs; 6,639 words | 16.509% | 16.433% | 32.68% → 36.91% | 90.72 → 28.44 |
| 20 short inputs; 524 words | 8.969% | 8.969% | 18.92% → 18.92% | 22.78 → 13.28 |
| 2 continuous calls; 6,115 words | 17.155% | 17.073% | 33.20% → 37.56% | 67.94 → 15.16 |

All 20 short transcripts were byte-identical. The first call's WER increased
slightly, 11.940% → 12.017%; the second improved, 20.981% → 20.783%. Aggregate
accuracy improved, but difficult calls still have substantial errors. Observed
batch-8 speedups were 3.19× overall and 4.48× on long audio.

The 48 GiB test hardware selects **batch 16 automatically**; batch 8 is a
manual choice or a lower-tier default. Batch 16 produced identical aggregate
quality and identical short transcripts, but took 51.05 seconds overall (18.49
long, 32.56 short) in its separate GPU run. Its observed speedup over the original
baseline was **1.78× overall and 3.67× on long files**, with slower short-file
timing. The 3.19× result above belongs specifically to batch 8. Larger batches
were not uniformly faster. Shared-machine load and short-file
overhead make latency variable; these measurements do not promise the same
speedup on every GPU or recording.

## Evidence and selection

All optimization experiments used `canary-qwen-2.5b-int8-convrot`. Original
`nvidia/canary-qwen-2.5b` was downloaded and exercised as a functional smoke check,
not tuned. Both models produced all 22 normalized JFK reference words correctly.
Original precision also passed the parent task's Chrome transcription check.

The main audited corpus contains 38 files: 24 LibriSpeech short recordings and 14
Earnings22 excerpts/full calls. The held-out comparison contains 18 main files
plus two supplemental TED-LIUM talks, totaling 87.6 minutes and 15,163 normalized
reference words. The continuous held-out calls are approximately 41 and 10
minutes. TED-LIUM talks are publisher reconstructions from labeled segments,
not uninterrupted original recordings; their references have no punctuation gold.
Company country metadata does not prove an individual speaker's accent.

References are published human transcripts. The structural audit rejected every
excerpt from Earnings22 source `4475604`, which had only 21.29% usable word times.
Retained source alignment coverage was 96.7–98.8%, with no timestamp reversals.
Excerpt text was restored from original human transcripts; alignment determines
crop boundaries only. This is not a claim that every word was independently
checked by human listening. See
[`reference_audit.json`](../outputs/benchmarks/english_corpus/reference_audit.json).

WER uses the shared Whisper English normalizer and word-weighted error totals.
Punctuation F1 counts `.,!?;:` at aligned word boundaries and is computed only
where human punctuation is available. Decimals and apostrophes remain inside
words. Punctuation in disfluencies, numbers, and transcript editorial choices
can affect this measure. The authoritative scorer is
[`english_benchmark_canary_report.py`](../tests/english_benchmark_canary_report.py),
which rescored stored hypotheses against audited reference version 2. Earlier
raw JSONL metrics are retained for traceability and should not replace this report.

Initial tuning compared 10, 20, 30, and 40-second chunks. Thirty seconds improved
tuning WER and punctuation, but worsened held-out WER from 6.852% to 7.241%; it was
rejected. A subsequent tuning-only comparison of 12 and 15 seconds selected 12.
The fixed 12-second candidate then improved held-out aggregate accuracy:

| Held-out subset | 10 seconds, batch 1 WER | 12 seconds, batch 8 WER | Punctuation F1 before → after | Historical benchmark-wall seconds before → after |
| --- | ---: | ---: | ---: | ---: |
| All 20 files | 6.852% | 6.720% | 41.03% → 42.07% | 94.57 → 30.77 |
| 16 short files | 4.251% | 4.251% | 39.22% → 29.17% | 6.17 → 6.85 |
| 4 long files | 6.940% | 6.803% | 41.06% → 42.27% | 88.39 → 23.92 |
| 18 main files, excluding TED | 10.391% | 10.176% | 41.03% → 42.07% | 56.87 → 20.48 |
| 2 supplemental TED talks | 1.886% | 1.870% | unavailable | 37.70 → 10.29 |

Short punctuation fell on both tuning (47.06% → 34.04%) and held-out data; batching
alone preserved short punctuation. The duration-dependent policy retains
10-second chunks for short inputs. This policy was finalized after inspecting
held-out results, so the main held-out data are no longer a pristine single-use
test set. A separate set of 15 previously untested 28–58-second excerpts checks
the fixed duration boundary. On those excerpts, automatic chunking improved WER
from 10.345% to 9.842% and punctuation F1 from 33.41% to 35.59% at batch 8. The six
held-out-source excerpts improved from 10.788% to 10.274% WER and 37.45% to 38.14%
punctuation F1. Individual recordings can still regress.

The implemented automatic policy was then run, rather than inferred from those
results, on all 20 main-plus-TED held-out files. At batches 8 and 16, WER was
6.720% and punctuation F1 42.22%, versus 6.852% and 41.03% for the original
10-second/batch-1 setting. The 16 short files retained their original 4.251% WER
and 39.22% punctuation F1. Long-file WER improved from 6.940% to 6.803%.

The native ASR prompt is retained. An added punctuation instruction worsened the
tuning results. Beam search gave negligible word changes while running roughly
15 times slower in the tested long recordings. Batch 32 had diminishing speed
returns and much higher memory use. These alternatives were not adopted.

## Memory and timing

Measurements used Linux, NVIDIA RTX A6000 48 GiB cards, driver 580.65.06,
PyTorch 2.13.0+cu130, Transformers 5.17.0, NeMo 3.1.0+cf724ac33, and Triton 3.7.1.
Canary experiments ran on GPUs 5–7 while the other agents used the remaining
devices. Existing unrelated processes used about 13 GiB on each GPU during the
earlier profiles and were left untouched; those allocations were no longer
present at final confirmation. These are shared-machine measurements, not isolated laboratory
latency guarantees.

Each configuration has a cold pass and a second, warm pass. Historical runtime
in the tables above includes CPU reference scoring after transcription; these
figures must not be compared directly to inference-only timings. The final
confirmation run stops the timer immediately after synchronized transcription
and records reference-scoring time separately. All timings exclude model
download/load and first-time kernel compilation. The first INT8 JFK call took
about 81 seconds during kernel tuning, versus about 0.60 seconds warmed up.
The original-model first transcription took about 33 seconds, excluding load.

The final automatic, 256-token validation exercised every held-out file at every
batch below. VRAM is peak own-process NVML memory across both passes, including
library allocations; it is not total device usage or just PyTorch tensor memory.

| Batch | Peak process GiB | WER | Punctuation F1 |
| ---: | ---: | ---: | ---: |
| 1 | 4.42 | 6.720% | 42.22% |
| 2 | 5.28 | 6.727% | 42.28% |
| 4 | 6.24 | 6.720% | 42.22% |
| 8 | 7.86 | 6.720% | 42.22% |
| 16 | 11.18 | 6.720% | 42.22% |

Batching can change floating-point decoding ties: texts are not guaranteed byte
identical. Batch 2 had one additional deletion out of 15,163 reference words.
Longer manually selected chunks can consume more memory: the 30-second tuning
profile peaked at 4.89/6.22/7.33/8.88/13.76/22.45 GiB for batches 1/2/4/8/16/32.
Other long recordings raised the batch-8 peak further, to about 11 GiB.

The automatic capacity tiers are 6/8/10/12/16/24/32 GiB → batches 1/2/4/8/8/16/16.
The free-memory checks additionally require 5.5/7/8.5/10.5/16 GiB for batches
1/2/4/8/16, and reduce the selected batch when another application occupies VRAM.
These estimates include headroom and apply only to the recognized INT8 Canary
model. Native 6–32 GiB cards and Windows GPU inference were not directly tested;
the measured hardware had 48 GiB. Available VRAM and manual parameter changes
still matter.

## Reliability and local model paths

The original weights are under
`models/Whisper/canary-qwen/nvidia--canary-qwen-2.5b` (5,119,161,060 bytes), and
INT8 weights under `models/Whisper/canary-qwen/canary-qwen-2.5b-int8-convrot`
(2,907,359,925 bytes). The original model's auxiliary Hugging Face files are also
under the installation's `models` directory. All inspected files and symlink
targets resolve inside it; see
[`model_inventory.json`](../outputs/benchmarks/canary/model_inventory.json).

The tests reproduced and fixed a CUDA graph lifetime crash: evicting one graph
could invalidate another graph's cuBLAS workspace when they shared the default
capture stream. Each graph now owns a native nonblocking stream, waits for
pending work before cleanup, resets its graph, and then releases its stream.
The helper serves both Canary and Whisper; the underlying issue is documented
in [PyTorch issue 193402](https://github.com/pytorch/pytorch/issues/193402).

The fallback Hugging Face model adapter is discarded before offloading, avoiding
stale CUDA weight references and about 0.58 GiB retained after beam inference.
Structured generation outputs are unpacked correctly, and multiple returned
hypotheses are rejected instead of being mapped to the wrong audio chunks.
Exactly zero audio chunks are skipped without discarding quiet nonzero audio
or shifting later timestamps. Real-model smoke checks confirm empty output for
digital silence and correct JFK words in both native and structured-output modes.
The original-precision model also completed two passes of the automatic policy
on both JFK and a 46.7-second human-reference excerpt, exercising both duration
branches without a download or inference error.

The final targeted suite passed **41 tests**. GPU checks cover graph eviction,
40 unique native streams, synchronization before release, release while another
device is current, offload/reload after beam generation, and mixed batch shapes.
Unit checks cover automatic chunk boundaries, explicit settings, silence,
structured outputs, and mocked Windows/Linux CUDA driver loading. Four Triton
deprecation warnings remain. Native Windows execution was not performed.

## Reproduction and artifacts

Use the repository virtual environment. Corpus preparation and the reference
audit are documented in [`english-benchmarks.md`](english-benchmarks.md). Example
configuration JSON for the final policy:

```json
[{"name":"auto_b8","chunk_length":0,"batch_size":8,"max_new_tokens":256}]
```

```sh
python tests/english_benchmark_canary.py --manifest outputs/benchmarks/english_corpus/manifest.json --split heldout --configs config.json --output results.jsonl --passes 2
python tests/english_benchmark_canary_report.py
python -m pytest tests/test_canary_qwen_inference.py tests/test_canary_convrot.py tests/test_convrot_cuda_graph.py tests/test_convrot_cuda_driver.py -q
```

Set `CUDA_VISIBLE_DEVICES` before launching Python to choose the GPU. The report
script consumes the named run files from this experiment; an arbitrary new
`results.jsonl` must be included explicitly before making a new comparison.

Canonical evidence lives under `outputs/benchmarks/canary/`:
`audited_results.json`, `audited_predictions.jsonl`, `heldout_main.jsonl`,
`heldout_ted.jsonl`, `heldout_final12.jsonl`, `final12_batches_*.jsonl`,
`auto_batches_*.jsonl`, `boundary_validation.jsonl`, `auto_boundary.jsonl`, and
`original_smoke.jsonl`. Final independent evidence is in `confirmation.jsonl`
(baseline and batch 8) and `confirmation_b16.jsonl`, with matching config JSON.
Files ending in `_448_probe` are additional runs with a
larger output-token cap; final default comparisons use 256 tokens. Earlier
reproduction failure logs remain as evidence of the bugs, not current failures.

Model behavior and the native prompt follow the
[official NVIDIA model card](https://huggingface.co/nvidia/canary-qwen-2.5b).
