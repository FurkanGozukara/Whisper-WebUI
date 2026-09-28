# Compatibility verification

Verified on Linux x86-64 on 2026-09-28 with Python 3.12.14, Gradio 6.28.0,
PyTorch 2.13.0+cu130 and installed Google Chrome 135.0.7049.52.
**Windows was reviewed and tested through portable-path/DLL mocks; it was not
run on native Windows. No fresh installer was executed during this audit.**

## Measured checks

| Check | Result |
| --- | --- |
| CPU regression suite | 320 passed; 22 CUDA-dependent cases skipped. Includes presets/VRAM tiers, English context boundaries, reference auditing, audio, subtitles, translation filenames, runtime, VAD, UVR and path handling. |
| Final digital-silence cleanup check | 14 targeted tests passed after adding cached-model offload checks; this run is separate from the 320-test sweep. |
| Recorder JavaScript suite | 5 passed: every PCM sample retained, full/recent WAV contents, stalled preview uploads, retry and denied permissions. |
| Chrome live microphone | Auto-off capture saved 12.92s / 206,720 samples at 16k; auto-on capture saved the full 32.936s, updated previews and generated final subtitles. |
| Linux launcher from another directory | Real `start-webui.sh --help` succeeded from `/tmp`. An isolated installation path containing spaces preserved arguments and propagated child exit code 7. |
| Model/cache placement | Seven cache roots resolved below the installation's `models/`. Fourteen model/cache directories were inspected; no model symlink escaped that directory. |
| Standalone API | CPU transcription and VAD requests passed. BGM upload, task polling and ZIP download passed. Default SQLite database remained in `backend/records.db` when launched from `/tmp`. |
| Optional local models | Cached Silero VAD, offline diarization, UVR HQ4 and NLLB distilled 600M loaded and ran on CPU. Same-language English translation preserves source cues verbatim. |
| UVR rate/duration | Real 16k and 48k WAVs both produced exactly 11s at 44.1k; MP4 duration was retained within one output sample. Both stems were finite and the subsequent 16k ASR resample retained duration. |
| No-speech VAD | Real nonzero background noise returned no words, wrote empty SRT/TXT files and completed cleanup without invoking the transcription decoder. |
| Final Chrome translation/BGM checks | All six English subtitle formats remained downloadable and matched the source words. CUDA UVR HQ4 and Inst3 produced finite 44.1k stereo outputs of 11s with playback/download. |

Detailed logs, XML results, the exact CPU test selection and environment versions
are retained in `output/verification/`. GPU/model quality and speed results are
described in [English benchmarks](english-benchmarks.md).

## Distribution fixes covered

- Linux and Windows launchers/installers anchor paths to their own directory,
  handle missing environments and preserve failure status. Repository installers
  reference the actual distribution requirements/build constraints and check
  Python 3.12.
- Model downloads and framework caches resolve inside the installation. Backend
  database paths no longer depend on the caller's current directory. Task
  timestamps use timezone-aware UTC for current SQLModel.
- Bundled diarization checkpoints resolve after the installation is moved,
  including saved paths using Windows separators. Output filenames handle Windows
  reserved names and trailing dots/spaces.
- TXT/SRT/VTT/LRC/TSV/JSON handling covers empty results and round trips; adjacent
  VTT cues retain their text, and English-to-English translation preserves words,
  punctuation and cue timing. NLLB and DeepL return every translated file when
  stems match across formats; duplicate names in different directories and prior
  plain exports receive distinct filenames instead of being overwritten.
- Live recording uses a repository-shipped AudioWorklet and supported Gradio HTML
  events/uploads. Complete audio remains in the browser independently of preview
  requests, with local download and upload retry. Stock streaming previously lost
  about two thirds of the captured audio in Chrome.
- UVR receives actual 44.1k resampled PCM for its trained frequency bins, preserves
  video stereo and labels output rates correctly. Empty VAD results do not fall
  back to transcribing the rejected noise.
- YouTube requests have bounded socket timeouts and clear metadata errors;
  failures clean up temporary downloads. Missing channel IDs no longer generate
  a misleading request to `/channel/None`. Output-less Cancel buttons discard
  their internal boolean result while retaining session-specific cancellation.

## Remaining verification limits

Native Windows installation, NVIDIA driver loading, Chrome microphone capture and
CUDA inference still require a Windows machine. Linux tests cover Windows path
forms, reserved filenames and the `nvcuda.dll`/`WinDLL` ABI branch; these do not
establish native Windows execution.

YouTube rejected live requests with `BotDetection`; metadata/channel error paths
were verified, but a successful live download cannot be claimed. No DeepL API
credential was available. Diarization and BGM checks establish execution/output
validity, not multi-speaker accuracy or separation quality. Tests here exercised
English only.
