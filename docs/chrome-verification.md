# Google Chrome feature verification — 2026-09-28

The application was exercised through installed, headed **Google Chrome on Linux**, using its rendered controls, file chooser, media players, downloads and confirmation dialogs. Tests used `http://127.0.0.1:7860`, GPU 0, real model inference, and several application restarts. GPU workers 1–7 performed separate English quality experiments. This report records observed workflows, not a guarantee that every possible parameter combination or recording will succeed.

Browser snapshots, screenshots, downloaded artifacts and the app log are retained under `output/playwright/` and `output/verification/`. These temporary artifacts and the downloaded benchmark corpus are ignored by Git. Reproducible benchmark scripts and regression tests are included in the repository.

## Completed workflows

| Feature | Observed result |
| --- | --- |
| Startup and navigation | File, YouTube, Mic, T2T Translation and BGM Separation tabs render; accordions, expand/collapse, light/dark theme and reload work. |
| File input | Native file upload, quoted absolute path, relative path and `file://` path work. Audio and MKV video preview play in Chrome. Missing input produces an actionable message. |
| Original Whisper models | Large-v1 and large-v3 load and transcribe the official JFK fixture through faster-whisper at FP16. Both also download/load/run through the Transformers backend. |
| ConvRot INT8 models | Large-v1, large-v3 and Canary-Qwen load and transcribe through their actual INT8 engines. Repeat runs reuse local model/kernel caches. Original BF16 Canary also transcribes successfully. |
| Local model storage | All eight tested engine/model folders resolve inside installation `models/`, including their files and symlinks. Final inventory is `output/verification/final-model-inventory.json`. |
| Subtitle outputs | SRT, WebVTT, TXT, LRC, JSON and TSV generated together; Chrome ZIP download contains all six and passes CRC checks. Filename timestamps and word-timestamp normalization run. |
| Folder batches | Six valid inputs and one corrupt input: six succeed, the corrupt file is reported, and outputs remain available. Uppercase extensions, Unicode/spaces, recursive folders, equal stems across formats and repeated basenames in separate folders work. A 36-file ZIP preserves folder structure and passes CRC checks. |
| Skip/overwrite controls | With overwrite disabled, all 36 prior output mtimes stay unchanged; enabling it updates all 36. |
| Presets | Save/load/reset/delete work. Decimal settings survive reload; dismissing delete preserves the preset, accepting removes it. A saved original-Canary model with custom batch 3 survives application restart. Temporary QA presets were removed. |
| Engine/model switching | Controls follow the selected engine; Canary remains English-only. Switching one tab preserves other tabs. Automatic Canary batch is 16 on this A6000; selecting original Canary resets it to 1. Saved custom batches remain unchanged. |
| Advanced decoding | Actual standard and batched-decoder paths run. Beam/context/chunk/prompt and other supported decoding options are additionally covered by parameter experiments and backend tests. Batched decoding remains optional because it worsened measured long-form accuracy. |
| Long audio in Chrome | A 9.8-minute Whisper run produced 141 segments in 21 seconds of transcription. A 41.3-minute Canary run with automatic chunks/batch 16 produced 237 chronological segments covering 0–2480.448 seconds in 22 seconds. Loading/offloading are additional; these are workflow smoke timings, not controlled comparisons. |
| Cancellation/recovery | Dismissing the confirmation keeps a running job; accepting terminates its worker process. A subsequent job succeeds. Output-less callback warnings were fixed. Session ownership is covered by runtime regressions. |
| GPU cleanup and queues | RAM parking/reload and full model offload execute. Jobs from separate Chrome sessions queue and finish; a live capture during a longer combined job retains complete audio. |
| Record Then Generate | Browser microphone capture saves a WAV, generates subtitles and exposes downloads. After completion, the recorder returns to its idle state. |
| Live Mic | With preview off, a 13-second capture saves 12.92 seconds / 206,720 samples at 16 kHz. With preview on, a complete 32.936-second recording is saved, preview text updates and final subtitles contain the complete repeated passage. Restarted recording works; final UI says audio was uploaded. |
| Silence | An exact-zero five-second WAV produces empty subtitles in all six formats, instead of the previous invented “Thank you.” |
| VAD/no speech | Real nonzero background noise with Silero enabled produces zero segments and no invented text. Speech input still transcribes. |
| Combined filters | A 90-second real two-speaker English call passes BGM removal, VAD, word timing, diarization and full offload together. Final subtitles contain SPEAKER_00 and SPEAKER_01 at the handover, preserve source timing and extend to 89.44 seconds. |
| BGM Separation tab | Both HQ4 and Inst_3 download/load/run on CUDA. Eleven-second input produces finite stereo stems at 44.1 kHz, each exactly 11 seconds. Chrome playback, vocal WAV download and output-folder action work. CPU WAV/video checks are documented separately. |
| NLLB translation | English-to-English input preserves all words in all six subtitle formats. All six downloads are now listed even when stems match; duplicate/prior filenames receive distinct output names. Real NLLB 600M loading/generation was also tested on CPU before identifying the unnecessary same-language paraphrasing. |
| Output folders | Main and BGM folder actions launch the system file manager. |
| Errors | Missing files/languages/API key, corrupt media, blocked YouTube metadata/channel access and cancellation produce readable results; failed batch inputs do not erase successful output. |

The microphone input in these Chrome tests is a virtual capture device fed real English speech, so recordings are repeatable. It exercises browser capture, buffering, uploads and inference, but does not establish the quality of physical microphone hardware. JavaScript tests additionally cover permission denial, stalled previews, final PCM/WAV integrity and upload retries.

## Fixed failures found during this work

- Gradio's live streaming recorder dropped roughly two thirds of recorded PCM even with transcription disabled. The application now retains complete PCM in its own browser recorder; preview delivery cannot truncate the final WAV.
- Exact silence and VAD-rejected background noise could reach the decoder and generate speech. They now return empty results with cleanup intact.
- UVR relabeled 16/48 kHz audio instead of resampling it to the model's trained 44.1 kHz rate. It now resamples correctly and preserves video stereo and duration.
- Mixed-length ConvRot CUDA graphs could lose workspace ownership when another graph was destroyed. Dedicated driver streams and synchronized cleanup now pass repeated eviction/offload tests.
- Translation result dictionaries discarded same-stem files from the download list; filename collisions could overwrite outputs. Both translation backends now retain every result with unique names.
- Original-Canary selection in a saved preset was replaced on startup. It is now restored with the user's batch value.
- YouTube's blocked metadata could form `/channel/None` and produce a misleading 404. The app now preserves the actual availability error.
- Subtitle imports, portable filenames, relocated diarization paths, launchers/installers and optional API database handling required additional fixes; see the compatibility report.

## Verification limits

YouTube rejected requests from this server with **BotDetection**. Metadata and single/channel failure paths were exercised, including the corrected channel error; a successful service download remains unverified. No DeepL credential was available, so only its validation and mocked response/file handling were tested. No non-English quality tests were run.

Native Windows, a clean installer run, physical 6–32 GB GPUs and physical microphone devices were not available. Windows paths, reserved filenames, launch scripts and CUDA DLL branches were reviewed/tested where possible, but Linux execution cannot certify native Windows operation. Diarization and music separation have execution/output checks, not comprehensive reference-based accuracy benchmarks.

See [English benchmark evidence](english-benchmarks.md), [Canary benchmark evidence](canary-english-benchmarks.md), and [compatibility verification](compatibility-verification.md) for detailed measurements, test counts and reproduction.
