# English benchmarks (version 12.11, 28 September 2026)

Word error rate (WER), punctuation, speed and VRAM of the three recommended English models, measured through the
app itself: [`tests/asr_benchmark/run_benchmark.py`](../tests/asr_benchmark/README.md) calls the same
`BaseTranscriptionPipeline.run()` as the File tab, with the shipped presets (Whisper: *Fast Whisper Best Quality*,
beam 5, word timestamps, standard decoding; Canary-Qwen: *Canary Qwen Best Quality*, automatic chunk length, batch
16 as selected on the 48 GB test GPUs). All tuning used the ConvRot INT8 models (`large-v3-int8-convrot`,
`large-v1-int8-convrot`, `canary-qwen-2.5b-int8-convrot`). The original-precision models were downloaded into the
installation and used in Chrome ([Chrome verification](chrome-verification.md)); in 12.10 original FP16 large-v3
had the same end-of-file problem as INT8 (short dev WER 8.44% vs 8.28%). This report replaces the 12.10 reports,
which used a 38-file corpus.

## Summary

Whisper large-v3 and Canary-Qwen had a much higher WER than their published numbers, and large-v1 lost
accuracy on very long recordings. The four causes, all fixed in 12.11:

1. **Whisper invented words at the end of files.** After its last 30-second window, Whisper decoded the remaining
   fraction of a second on its own and, on silence, invented "you", "Thank you." or "For more information visit
   www.fema.org"; on LibriSpeech test-other large-v3 had 9.0% WER instead of the published 3.9%. The remaining
   tail is now skipped unless the voice detector hears at least 0.3 s of speech after its first half second:
   the last window sometimes also stops before the last words ("... a loss of life for opinion's sake." ended at
   "for" when every tail was skipped).
2. **Canary-Qwen cut recordings into 10-12 second pieces**, splitting sentences in the middle: words and most of
   the punctuation at the cut were lost. Recordings up to 40 seconds are now transcribed in one piece, as the model
   was trained, and longer recordings are cut at the clearest pause (Silero voice detection) into 15-30 second
   pieces.
3. **Canary-Qwen sometimes looped** on music, chants or noise ("Kwame Kwame Kwame ..." for hundreds of words).
   A piece whose output fills the token limit or compresses like a loop is transcribed again as two halves.
4. **Whisper large-v1 lost the previous-text context on recordings over 30 minutes**, which cost it accuracy
   (large-v3 still turns context off for English recordings over 30 seconds, where it made v3 repeat passages).

Held-out test half, 12.10 → 12.11 (lower WER is better, higher punctuation F1 is better):

| Model (ConvRot INT8) | Short clips, average WER | Long recordings, average WER | Punctuation F1, short / long |
| --- | ---: | ---: | ---: |
| Whisper large-v3 | 8.37% → **7.45%** | 8.88% → **8.84%** | 69.8 → 69.8 / 52.8 → 52.8 |
| Whisper large-v1 | 8.03% → **7.99%** | 9.55% → **9.46%** | 69.7 → 69.7 / 59.4 → 58.9 |
| Canary-Qwen-2.5B | 5.89% → **5.62%** | 10.47% → **8.59%** | 60.2 → 62.8 / 44.2 → 54.6 |

With 12.11 the INT8 models reach the published accuracy of the original models on the same test sets (see
[Compared with the published numbers](#compared-with-the-published-numbers)).

## Test data

Only public test sets with human transcripts, English with many accents (Common Voice volunteers from around the
world, European Parliament speakers in VoxPopuli, company calls from many countries in Earnings-21/22, meetings
with non-native speakers in AMI, podcasts, talks and read speech):

| Kind | Sets | Files | Audio |
| --- | --- | ---: | ---: |
| Short (0.1-69 s) | LibriSpeech test-clean and test-other, Common Voice, VoxPopuli, TED-LIUM 3, Earnings-22, AMI, GigaSpeech, SPGISpeech (the Open ASR Leaderboard test sets) | 2,700 | 5.3 h |
| Long (24 s-2.2 h) | TED-LIUM long-form talks, Earnings-21 and Earnings-22 full calls, Rev16 podcasts, Meanwhile (Colbert monologues) | 120 | 43.2 h |

Every set was split with a fixed seed into a **dev** half, used for all tuning, and a **test** half that was run
at the end with both 12.10 and 12.11: 150 + 150 utterances per short set, 52 dev (12.1 h) and 68 test (31.1 h)
long recordings. YouTube downloads were not needed (YouTube also refuses this server's address).

**WER** uses the OpenAI Whisper `EnglishTextNormalizer` on reference and output, as the Open ASR Leaderboard and
the Canary-Qwen model card do; errors are pooled over all words of a set. **Average** is the mean over the sets
and **Pooled** pools all words. **Punctuation F1** scores commas, periods and question marks after each aligned
word on the sets whose references are punctuated (Common Voice, Earnings-22, GigaSpeech, SPGISpeech, Earnings-21,
Rev16, Meanwhile). The sets punctuate differently, so compare it between versions rather than between sets.

The INT8 engines are not bit-for-bit deterministic between processes (about 1% of files change a word or a comma
between identical runs), so differences of a few hundredths of a point are noise.

## Held-out test results

**Short clips** (1334 files, 2.7 hours), WER in %:

| Model | Version | LS clean | LS other | Common Voice | VoxPopuli | TED-LIUM | Earnings-22 | AMI | GigaSpeech | SPGISpeech | **Average** | Pooled | Punctuation F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| large-v3 INT8 | 12.10 | 4.38 | 9.02 | 9.38 | 7.16 | 3.99 | 10.28 | 18.36 | 9.32 | 3.43 | 8.37 | 7.27 | 69.8 |
|  | 12.11 | 1.88 | 3.40 | 9.32 | 7.16 | 3.99 | 10.24 | 18.36 | 9.32 | 3.41 | **7.45** | 6.29 | 69.8 |
| large-v1 INT8 | 12.10 | 2.37 | 5.01 | 8.91 | 7.71 | 3.99 | 10.46 | 18.63 | 10.91 | 4.23 | 8.03 | 6.94 | 69.7 |
|  | 12.11 | 2.37 | 5.01 | 8.91 | 7.71 | 3.99 | 10.46 | 18.63 | 10.91 | 3.87 | **7.99** | 6.89 | 69.7 |
| Canary INT8 | 12.10 | 1.91 | 3.40 | 5.76 | 7.49 | 2.76 | 10.24 | 10.06 | 8.65 | 2.77 | 5.89 | 5.43 | 60.2 |
|  | 12.11 | 1.52 | 3.33 | 5.63 | 6.28 | 3.04 | 9.87 | 10.06 | 8.26 | 2.58 | **5.62** | 5.11 | 62.8 |

**Long recordings** (68 files, 31.1 hours), WER in %:

| Model | Version | TED-LIUM | Earnings-21 | Earnings-22 | Rev16 | Meanwhile | **Average** | Pooled | Punctuation F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| large-v3 INT8 | 12.10 | 3.43 | 11.20 | 14.28 | 10.34 | 5.17 | 8.88 | 10.62 | 52.8 |
|  | 12.11 | 3.43 | 11.20 | 14.07 | 10.34 | 5.17 | **8.84** | 10.58 | 52.8 |
| large-v1 INT8 | 12.10 | 3.08 | 12.51 | 15.41 | 11.73 | 5.04 | 9.55 | 11.78 | 59.4 |
|  | 12.11 | 3.08 | 11.98 | 15.29 | 11.92 | 5.04 | **9.46** | 11.70 | 58.9 |
| Canary INT8 | 12.10 | 2.42 | 11.26 | 22.43 | 10.82 | 5.42 | 10.47 | 12.28 | 44.2 |
|  | 12.11 | 2.39 | 10.87 | 14.73 | 10.27 | 4.68 | **8.59** | 10.50 | 54.6 |

## Compared with the published numbers

The Open ASR Leaderboard (results revision of 12 May 2026, before its normalizer change) evaluates the same eight
short test sets (without Common Voice) with the original models in full precision on the complete sets; our test
half has 150 utterances per set, so single sets move by about ±1 point, but the averages are comparable. WER in %:

| Model | AMI | Earnings-22 | GigaSpeech | LS clean | LS other | SPGISpeech | TED-LIUM | VoxPopuli | **Average** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| large-v3, published (full precision) | 15.95 | 11.29 | 10.02 | 2.01 | 3.91 | 2.94 | 3.86 | 9.54 | 7.44 |
| large-v3 INT8, app 12.10 | 18.36 | 10.28 | 9.32 | 4.38 | 9.02 | 3.43 | 3.99 | 7.16 | 8.24 |
| large-v3 INT8, app 12.11 | 18.36 | 10.24 | 9.32 | 1.88 | 3.40 | 3.41 | 3.99 | 7.16 | **7.22** |
| large-v1, published (full precision) | 16.73 | 12.91 | 10.76 | 2.73 | 5.54 | 3.20 | 3.91 | 7.76 | 7.94 |
| large-v1 INT8, app 12.10 | 18.63 | 10.46 | 10.91 | 2.37 | 5.01 | 4.23 | 3.99 | 7.71 | 7.92 |
| large-v1 INT8, app 12.11 | 18.63 | 10.46 | 10.91 | 2.37 | 5.01 | 3.87 | 3.99 | 7.71 | **7.87** |
| Canary, published (full precision) | 10.19 | 10.45 | 9.43 | 1.61 | 3.10 | 1.90 | 2.71 | 5.66 | 5.63 |
| Canary INT8, app 12.10 | 10.06 | 10.24 | 8.65 | 1.91 | 3.40 | 2.77 | 2.76 | 7.49 | 5.91 |
| Canary INT8, app 12.11 | 10.06 | 9.87 | 8.26 | 1.52 | 3.33 | 2.58 | 3.04 | 6.28 | **5.62** |

For long recordings the leaderboard's long-form track reports large-v3 9.71 / 13.16 / 3.15 and Canary-Qwen 9.60 /
13.59 / 2.57 on Earnings-21 / Earnings-22 / TED-LIUM. The app on the same sets (dev and test halves together):

large-v3 10.71 / 12.98 / 3.43; Canary-Qwen 10.42 / 13.02 / 2.45 (12 Earnings-21 calls, 9 Earnings-22 calls and 19 TED talks: a subset of the leaderboard's files, so only roughly comparable).

## Tuning on the dev half

**Short clips** (1338 files, 2.6 hours), WER in %:

| Model | Version | LS clean | LS other | Common Voice | VoxPopuli | TED-LIUM | Earnings-22 | AMI | GigaSpeech | SPGISpeech | **Average** | Pooled | Punctuation F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| large-v3 INT8 | 12.10 | 5.46 | 6.16 | 10.59 | 7.86 | 3.77 | 10.49 | 16.93 | 10.30 | 2.95 | 8.28 | 7.13 | 72.4 |
|  | 12.11 | 1.51 | 2.51 | 10.24 | 7.38 | 3.80 | 10.45 | 16.93 | 10.33 | 2.64 | **7.31** | 6.08 | 72.5 |
| large-v1 INT8 | 12.10 | 2.23 | 4.60 | 10.66 | 7.29 | 4.25 | 10.64 | 17.12 | 10.40 | 3.20 | 7.82 | 6.60 | 70.7 |
|  | 12.11 | 2.23 | 4.60 | 10.45 | 7.29 | 4.22 | 10.60 | 17.12 | 10.40 | 3.26 | **7.80** | 6.59 | 70.8 |
| Canary INT8 | 12.10 | 1.43 | 2.70 | 8.19 | 7.41 | 2.34 | 10.26 | 9.56 | 10.23 | 2.23 | 6.04 | 5.35 | 60.8 |
|  | 12.11 | 1.43 | 2.44 | 7.98 | 6.02 | 2.10 | 9.35 | 9.46 | 9.99 | 2.06 | **5.65** | 4.93 | 64.5 |

**Long recordings** (52 files, 12.1 hours), WER in %:

| Model | Version | TED-LIUM | Earnings-21 | Earnings-22 | Rev16 | Meanwhile | **Average** | Pooled | Punctuation F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| large-v3 INT8 | 12.10 | 3.44 | 8.96 | 10.60 | 11.47 | 5.19 | 7.93 | 9.39 | 57.9 |
|  | 12.11 | 3.44 | 8.96 | 10.60 | 11.47 | 5.19 | **7.93** | 9.39 | 57.9 |
| large-v1 INT8 | 12.10 | 3.22 | 9.81 | 11.74 | 12.29 | 4.88 | 8.39 | 10.09 | 63.5 |
|  | 12.11 | 3.22 | 9.11 | 10.97 | 10.44 | 4.88 | **7.72** | 9.03 | 66.5 |
| Canary INT8 | 12.10 | 2.62 | 9.29 | 9.65 | 11.59 | 5.85 | 7.80 | 9.21 | 42.4 |
|  | 12.11 | 2.53 | 8.84 | 9.25 | 11.45 | 4.96 | **7.41** | 8.93 | 57.8 |

Tried on the dev half and not adopted:

| Change | Result |
| --- | --- |
| large-v3 with previous-text context on long recordings | much worse on the 8 longest recordings (14.3% vs 11.2% pooled): it repeats passages |
| Beam size 1 instead of 5 | large-v3 short 7.44% vs 7.34%, long 8.40% vs 7.93% (better punctuation); about 10% faster |
| Use Batched Inference (batch 16) as the default | large-v3 long 8.93% vs 7.93%, twice as fast; large-v1 long 11.62% vs 7.72% (12.10 batched code) |
| Initial prompt "Hello, welcome to my lecture." | worse on 8 of the 9 short sets (average 9.11% vs 7.34%) |
| A disfluent initial prompt ("Um, well, I- I think, uh, ...") | more verbatim: AMI meetings 16.9% → 12.8%, but Common Voice 10.5% → 11.8% and SPGISpeech 2.6% → 5.0%; pooled WER no better (6.14% vs 6.10%) |
| Word timestamps off | large-v3 short 7.54% vs 7.34% |
| Skipping every end-of-file tail | dropped real last words ("... a loss of life for opinion's sake." ended at "for"); large-v1 short 7.84% vs 7.80% |
| Decoding every tail with 0.15-0.3 s of detected speech (first half second counted) | the invented "you" came back after the last word (large-v3, first 324 short files: 5.99% / 5.46% vs 4.68%) |
| Canary pieces of up to 20 / 25 / 35 s instead of 30 s | long 7.52% / 7.45% / 7.41% vs 7.40% |
| Skipping Canary pieces without detected speech | dropped real short replies ("Thanks.", "Yeah,") |

## Speed

Warm second pass, one RTX A6000 per model with both versions on the same GPU (the other GPUs were running
benchmarks, so absolute speed on an idle GPU is higher), the same files for both versions: 3 long recordings
(Earnings-21 call 45 min, Rev16 podcast 40 min, TED talk 14 min) and 60 short clips of 0.1-26 s:

| Model | Version | 3 long recordings (99.6 min) | 60 short clips (6.8 min) | Peak VRAM |
| --- | --- | ---: | ---: | ---: |
| Whisper large-v3 INT8 | 12.10 | 161.1 s (37x real time) | 39.0 s (0.65 s per clip) | 3.5 GB |
|  | 12.11 | 162.5 s (37x real time) | 23.6 s (0.39 s per clip) | 3.5 GB |
| Whisper large-v1 INT8 | 12.10 | 172.9 s (35x real time) | 37.5 s (0.63 s per clip) | 3.5 GB |
|  | 12.11 | 181.0 s (33x real time) | 21.9 s (0.37 s per clip) | 3.5 GB |
| Canary-Qwen INT8, batch 16 | 12.10 | 21.9 s (273x real time) | 28.5 s (0.48 s per clip) | 10.4 GB |
|  | 12.11 | 26.3 s (227x real time) | 16.9 s (0.28 s per clip) | 12.0 GB |

* Short clips are about 1.7x faster: faster-whisper's audio loading ran a full Python garbage collection for every
  file (0.2-0.3 s); the app now decodes audio itself, with bit-identical samples.
* large-v1 is slower on the two recordings over 30 minutes because it now keeps the previous-text context there.
* Canary-Qwen pieces are up to 30 s instead of 12 s, which costs some speed on long recordings but lowers their
  WER from 10.47% to 8.59% on the test half. Finding the pauses (Silero voice detection, several seconds of CPU
  per hour of audio) now runs while the first pieces are transcribed: before, it made long recordings 1.8x slower
  (40.0 s instead of 26.3 s here).
* Whisper's standard decoding does not get faster with a larger encoder batch, so Whisper keeps batch 1 on every
  GPU; beam 1 would be about 10% faster but less accurate (see above).

## VRAM tiers

When the app starts, it reads the GPU's capacity and free memory and sets the Canary-Qwen INT8 preset's batch size
(the Batch Size help text shows the choice, for example "batch 16 (32 GB tier)"):

| GPU memory | Canary-Qwen INT8 batch | Free memory needed at startup |
| --- | ---: | ---: |
| 6 GB | 2 | 5.0 GB |
| 8 GB | 4 | 6.6 GB |
| 10 and 12 GB | 8 | 8.9 GB |
| 16, 24, 32 GB and more | 16 | 12.3 GB |

With less free memory (other programs on the GPU), the largest batch that fits is used, and batch 1 below 5.0 GB.
The original NeMo Canary model starts at batch 1; Whisper uses 3.5 GB on every GPU (standard decoding, batch 1).

Measured process peaks (CUDA context included) of Canary-Qwen INT8 on a mix of short and hour-long recordings:
batch 1 / 2 / 4 / 8 / 16 = 4.9 / 5.3 / 6.0 / 8.2 / 11.5 GB, at 44 / 64 / 90 / 101 / 110x real time on long
recordings. Over a long session its cached CUDA graphs add memory (batch 16: up to 14.4 GB after 52 long
recordings), so each tier was also run as a 1,402-file session (every dev recording, 14.8 hours, shuffled) with
PyTorch limited to the card's size minus 0.9 GB for the driver, CUDA context and desktop:

| Simulated GPU | Batch | Files that failed | Out-of-memory recoveries | Speed |
| --- | ---: | ---: | ---: | ---: |
| 6 GB | 1 | 0 | 35 | 36x |
| 6 GB | **2** | 0 | 46 | 49x |
| 8 GB | **4** | 0 | 0 | 72x |
| 8 GB | 8 | 0 | 28 | 69x |
| 10 GB | **8** | 0 | 0 | 80x |
| 12 GB | **8** | 0 | 0 | 77x |
| 16 GB | **16** | 0 | 0 | 81x |

**Fixed:** before 12.11, one out-of-memory error made every later file fail, even 20-second clips (a simulated
8 GB card at batch 8: 14 of 17 files failed), because the cached CUDA graphs kept their memory. Now the engine
frees its graphs and KV caches and retries; a batch that still does not fit is split in half, and later batches
keep the smaller size until a model is loaded again. Physical 6-16 GB GPUs and Windows (whose desktop may use more
VRAM) were not available for testing; the simulation limits PyTorch's allocator, not the whole card.

**Use Batched Inference** (Whisper, not the default) needs 8.1 / 14.3 / 25.8 GB at batch 4 / 8 / 16 on a
75-minute recording.

Memory is given in GiB (1024³ bytes), as Windows Task Manager and PyTorch report it.

## Reproduce

See [tests/asr_benchmark/README.md](../tests/asr_benchmark/README.md): `download_datasets.py` (about 40 GB),
`prepare_corpus.py` (16 kHz WAV corpus of about 5 GB), `run_benchmark.py` with a preset and overrides, and
`score.py`. Set `APP_DIR` to benchmark another copy of the app, for example the previous release. The older
small-corpus scripts (`tests/english_benchmark_*.py`) remain in the repository; their 12.10 results are superseded.
