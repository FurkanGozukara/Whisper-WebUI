# Google Chrome feature verification (28 September 2026, version 12.11)

Every feature below was used in installed Google Chrome 135 on Linux (headed, on the desktop), driving the rendered
page like a user: clicking the controls, uploading files through the page's file inputs, typing paths, reading the
Live Transcription / Output boxes, downloading the ZIP, accepting and dismissing confirmation dialogs. The app ran
with `python app.py` and `./start-webui.sh` on one RTX A6000 (48 GB), restarted after each round of code changes.
The Chrome microphone was a fake capture device playing a LibriSpeech sentence, so microphone results are
repeatable (it does not test physical microphone hardware).

## Model downloads (first use, empty model folders)

All six models were removed from the installation first. Selecting each one and clicking GENERATE SUBTITLE FILE
downloaded it into the installation's own `models` folder and transcribed the JFK test clip correctly.

| Model | Folder (inside `Whisper-WebUI`) | Size |
| --- | --- | --- |
| large-v3-int8-convrot | `models/Whisper/faster-whisper/large-v3-int8-convrot` | 1.6 GB |
| large-v1-int8-convrot | `models/Whisper/faster-whisper/large-v1-int8-convrot` | 1.6 GB |
| large-v3 (original, FP16) | `models/Whisper/faster-whisper/large-v3` | 2.9 GB |
| large-v1 (original, FP16) | `models/Whisper/faster-whisper/large-v1` | 2.9 GB |
| canary-qwen-2.5b-int8-convrot | `models/Whisper/canary-qwen/canary-qwen-2.5b-int8-convrot` | 2.8 GB |
| nvidia/canary-qwen-2.5b (original) | `models/Whisper/canary-qwen/nvidia--canary-qwen-2.5b` (+ Qwen3 tokenizer in `canary-qwen/hub`, 16 MB) | 4.8 GB |

The live box shows the download and loading steps. The original NeMo Canary model takes about 1.5-2 minutes to
load on first use (NeMo builds the model before loading the weights); the INT8 models load in seconds.

## Workflows

| Feature | Result |
| --- | --- |
| File tab | Single and multiple uploads; all 6 formats (SRT, WebVTT, txt, LRC, JSON, TSV) with the timestamp suffix; Download Transcription ZIP holds all 6 files. A 2-file job now ends with "Done! 2 files, 4 segments in 10 seconds." (before: the last file's message only). |
| Load From File Path | Quoted absolute path loads and previews a WAV. **Fixed**: a wrong path showed nothing (HTTP 500 in the browser console); it now shows "File not found: ...". |
| Batch processing | Folder with a subfolder, a name with spaces, an upper-case `.MP3` and a corrupt WAV: 4 outputs in the same subfolder structure, the corrupt file listed as failed. Running again without Overwrite skips all 4 (modification times unchanged). |
| Presets | Save (beam 3) → Reset Defaults (beam 5) → load the saved preset (beam 3) → Delete (confirmation dialog, file removed, settings kept). |
| Base models | Whisper / Insanely Fast Whisper / Canary-Qwen switch their model lists and options; Canary selects batch 16 on this 48 GB card. |
| Advanced parameters | Beam size and Use Batched Inference change the run; batched runs cut windows at pauses. |
| Filters | Background Music Remover + Voice Detection + Diarization on a 2-speaker recording with background music: music removed, SPEAKER_00 / SPEAKER_01 exactly on the four speaker turns. |
| Cancel Generation | With Start As Sub Process: confirmation dialog, the subprocess is terminated, "Cancelled. Running subprocess was terminated." |
| Mic tab | Live Mic: previews while recording, the complete recording transcribed after Stop. Record Then Generate: recording transcribed. |
| T2T Translation | NLLB (distilled 600M) English→German SRT translated with timings kept; DeepL without a key says what is missing. |
| BGM Separation tab | Speech + synthetic music: vocals and instrumental players; the vocals stem correlates 0.993 with the clean speech, the instrumental 0.03. |
| Page controls | Light / dark theme, Open / close all sections, OPEN OUTPUTS FOLDER (opens the file manager). |
| Launcher | **Fixed**: `start-webui.sh` and `Install.sh` were stored without the executable bit (`./start-webui.sh`: Permission denied). |

## Found and fixed while testing

* Whisper invented a word at the end of almost every short file ("you", "the", "Thank you for watching",
  "For more information, visit www.fema.org"); seen in the UI on LibriSpeech clips and a microphone recording.
* Canary cut short recordings into 10 s pieces (the JFK clip became two pieces without punctuation between them).
* Load From File Path gave no feedback for a wrong path; multi-file jobs reported only the last file.

## Final check with the released code

After the last changes (end-of-file speech check, Canary out-of-memory recovery, voice detection running during
Canary transcription, 6 GB tier) the app was restarted and used in Chrome again: the title shows V12.11 and the
new help texts; large-v3 INT8 ends a LibriSpeech clip with "... oh how ugly that is" (12.10 added "you");
large-v1 INT8 keeps the last words of "... a loss of life for opinion's sake." (dropped when every end-of-file
tail was skipped); loading the Canary Qwen Best Quality preset selects the INT8 model, batch 16 ("32 GB tier" on
the 48 GB card) and automatic chunks; a 3-minute TED talk was cut at pauses into 15-30 s pieces at 51x real time
and the 3-file job ended with "Done! 3 files, 10 segments in 11 seconds."

## Not verifiable here

* YouTube refuses this server's address (pytubefix: BotDetection; yt-dlp: "Sign in to confirm you're not a bot").
  The app shows the reason in the description box; a successful download could not be tested.
* No DeepL API key was available; physical microphones, native Windows and smaller GPUs were not available
  (smaller cards were simulated by capping the process's GPU memory, see the benchmark report).
