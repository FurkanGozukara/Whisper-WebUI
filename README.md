# Whisper-WebUI Premium made for SECourses Patreon followers only : https://www.patreon.com/posts/145395299

## Download Installers and App

> https://www.patreon.com/posts/145395299

**Whisper-WebUI Premium turns any video or audio into accurate subtitles and transcripts on your own PC.** On an RTX 5090, our own INT8 ConvRot engine transcribed a 1 hour 29 minute lecture in 32 seconds. Every setting comes ready with researched best-quality presets. Below you will see every feature with real screenshots from the app.

## Why Whisper-WebUI Premium

- **Our own INT8 ConvRot engines:** Whisper large-v3 runs 3.4x faster and NVIDIA Canary-Qwen 2.5B runs 8.9x faster than the standard models, on the same video and the same GPU.
- **Same accuracy as the original models:** measured through the app on 14 public English test sets, 48 hours of audio.
- **Real speed:** a 10 minute video in 9 seconds, a 1.5 hour lecture in 32 seconds.
- **3 engines in one app:** Whisper, Insanely Fast Whisper and NVIDIA Canary-Qwen 2.5B, with 21 Whisper models and 100 languages.
- **Ready presets:** researched best-quality settings for every engine, plus your own saved presets.
- **6 output formats in one run:** SRT, WebVTT, TXT, LRC, JSON and TSV, with a one-click ZIP download.
- **Everything in one place:** batch folders, YouTube links and whole channels, live microphone, speaker labels, background music remover, voice detection filter and subtitle translation to 200 languages.
- **1-click installers:** Windows, RunPod, SimplePod, Massed Compute and Linux, with PyTorch 2.13, CUDA 13 and precompiled Flash Attention, xFormers, SageAttention and Triton.
- **Automatic model downloads** with live progress, **frequent updates** and support.

Here is the app right after a job. A 10 minute 54 second video was transcribed in 9 seconds, 70 times faster than real time.

![Whisper-WebUI Premium main screen after transcribing a 10 minute video in 9 seconds](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/WiVw0_0XI6sZy69wl853g.png)

1: pick a ready preset. 2: download all subtitle files in one ZIP. 3: preview your video instantly. 4: watch the transcription live. 5: choose from 3 engines and our INT8 ConvRot models.

## Speed: Our INT8 ConvRot Engines

We built INT8 ConvRot versions of Whisper large-v3, Whisper large-v1 and NVIDIA Canary-Qwen 2.5B, and our own GPU engine to run them. The model files are smaller too: 1.6 GB instead of 3.1 GB for Whisper, and 2.9 GB instead of 5.1 GB for Canary-Qwen. Here is the same video on the same RTX 5090 with the same settings:

![Speed chart: Whisper large-v3 31 s vs 9 s, Canary-Qwen 71 s vs 8 s on the same video and GPU](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/Ku3fPZK9eVTFtq0ZvfAPm.png)

Long files are just as fast. Our 1 hour 29 minute lecture took 1 minute 15 seconds with our INT8 Whisper large-v3 (71x real time) and only 32 seconds with Canary-Qwen (168x real time). This is the Canary-Qwen run in CMD:

![CMD log: a 1 hour 29 minute lecture transcribed in 32 seconds](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/8rc0vV6JbCEn3vhUGgzYI.png)

After each job the model waits in RAM and your VRAM is free. The next job moves it back to the GPU in under a second, so it starts right away.

## Same Accuracy as the Original Models

Speed only matters with accuracy. We measured every model through the app on 14 public English test sets with human transcripts: 2,700 short clips and 120 long recordings, 48 hours in total.

![Word error rate chart: our INT8 models match the published full-precision accuracy](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/6RpxsrpyYEVRkwifcL3SI.png)

On the Open ASR Leaderboard's 8 English test sets, our INT8 models reach the published accuracy of the full-precision originals. Canary-Qwen 2.5B scored 5.62% word error rate against the published 5.63%, and Whisper large-v3 scored 7.22% against 7.44%.

## 1-Click Installation

### What You Download

You get one small zip file with the installers for every platform.

![The files inside WhisperWeb_UI_v13.zip](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/KfwaHvJPmSPSUoI0Ln5l8.png)

Extract it into any folder. On Windows, double-click Windows_Install_Update.bat, then start the app with Windows_Start_app.bat. Run the same installer again at any time to update.

### Latest PyTorch, CUDA 13 and Precompiled Libraries

The installer makes its own Python 3.12 virtual environment, so your other apps stay untouched. It installs PyTorch 2.13 with CUDA 13 and our precompiled Flash Attention, xFormers, SageAttention and Triton for Windows. You never compile anything.

![Installer log: PyTorch 2.13 with CUDA 13, Transformers, Triton for Windows and precompiled xFormers](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/QQZCx7bo_Q4AECx9iDYp-.png)

Every package version is tested with the app. This is the end of a real fresh install on our PC:

![Installer log: speaker models downloaded automatically, installation finished](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/TrZCXxI-LwYQ0HtAqnCt2.png)

At the end, the installer downloads the speaker label models from our mirror. You do not need a Hugging Face token or any model approval.

### Start the App

Double-click Windows_Start_app.bat. CMD shows every startup step with its time.

![App startup log: ready in 6.5 seconds](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/cGGIfyZD6zLjXUEoeske6.png)

On our RTX 5090 the app was ready in 6.5 seconds. Open the local address in your browser and start transcribing.

### Cloud GPUs: RunPod, SimplePod and Massed Compute

You can also run the app on a cloud GPU. The cloud installers set up Python 3.12, FFmpeg n9.0 and everything else with one command, and a Gradio share link lets you use the app from any device.

- **SimplePod:** [register here](https://simplepod.ai/ref?user=secourses) and use [this template](https://dash.simplepod.ai/account/explore/100/ref-secourses/).
- **RunPod:** [register here](https://get.runpod.io/955rkuppqv4h) and use [this template](https://get.runpod.io/SECourses_CU13).
- **Massed Compute:** [register here](https://vm.massedcompute.com/signup?linkId=lp_034338&sourceId=secourses&tenantId=massed-compute) and use our coupon **SECourses**.

The step-by-step commands are in the instruction files inside the zip.

### Requirements

On Windows you need Python 3.12, Git, FFmpeg, CUDA 13, cuDNN 9.17 and Visual Studio with C++ tools. [This tutorial](https://youtu.be/DrhUHnYfwC0) shows every step, and the same setup runs all our AI apps. The app runs on NVIDIA GPUs from the GTX 16 and RTX 20 series up to the RTX 50 series. Our INT8 ConvRot engines use the RTX 30 series and newer, and other GPUs switch to the standard models automatically.

## Config Presets

Every engine comes with a locked best-quality preset. We researched each value on real test sets, so you get top results without changing a single setting.

**Where to find it:** Config Presets sits at the top of the page and stays there on every tab.

![Where to find Config Presets: the top of the page, above the tabs](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/RrDpnQC9IimrUsF7yxser.png)

Open the Select Preset list to see every preset:

![Config Presets: built-in best-quality presets and a saved user preset](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/U13I6OBIa67p0yl5rLh9w.png)

Change anything you like, type a name and click Save to keep your own preset. The app remembers your last preset and loads it at every start.

## Three Engines and 21 Whisper Models

Choose the engine in Base Model: Whisper (faster-whisper), Insanely Fast Whisper (Transformers) or NVIDIA Canary-Qwen 2.5B. Each engine loads its best settings when you select it.

**Where to find it:** open the File tab and scroll to Base Model. The Model list is right under it, and the Youtube and Mic tabs have the same controls.

![Where to find Base Model and Model: File tab, below the upload and batch area](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/GzX13DmX8A3YNbVJj8SJq.png)

Open the Model list to see every model:

![Base Model engines and the Whisper model list with our INT8 ConvRot models](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/ca0KWkCbeJVFzGfKoJz7E.png)

You get every Whisper model from tiny to large-v3, plus turbo, distil and our two INT8 ConvRot models. A model downloads automatically the first time you use it.

### NVIDIA Canary-Qwen 2.5B

Canary-Qwen 2.5B is one of the most accurate open English speech models on the Open ASR Leaderboard (September 2026). Our INT8 ConvRot build runs it with Triton kernels and CUDA graphs.

**Where to find it:** pick Canary Qwen Best Quality in Select Preset, or choose Canary-Qwen (NVIDIA NeMo) in Base Model.

![Where to select Canary-Qwen: the Select Preset list or Base Model](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/R_-GrMoXmsmJopoK7chdt.png)

The Canary settings then load automatically:

![Canary-Qwen engine with the INT8 ConvRot model, batch 16 and a 1.5 hour lecture done in 36 seconds](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/f8IvYe3EuuNYll85ayADg.png)

The app reads your GPU memory at startup and picks the batch size for you: batch 2 on 6 GB, 4 on 8 GB, 8 on 10 to 12 GB and 16 on 16 GB and more. Here it chose batch 16 on the 32 GB RTX 5090 and finished the 1 hour 29 minute lecture in 36 seconds.

### Automatic Model Downloads

You never download models by hand. The first time you use a model, the app downloads it and shows the progress in Live Transcription and in CMD.

![Live Transcription showing the Canary-Qwen model download progress](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/qEAtR8vkgn_Ma2X0JZ1ZG.png)

All models and caches stay inside the app's own models folder, so the rest of your PC stays clean.

## 100 Languages and Translation to English

Whisper understands 100 languages, and Automatic Detection finds the language for you.

**Where to find it:** in the File tab, the Language list and the Translate to English checkbox sit next to the Model list.

![Where to find Language and Translate to English: File tab, next to the Model list](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/qR-ZrOVkbp-e0S5Dfx30E.png)

Open the Language list to see every language:

![Language list with 100 languages and Translate to English](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/PCeqYkmNS7NCYkEYaqqTh.png)

Turn on Translate to English and Whisper writes English subtitles directly from speech in any language.

## Output Formats and Run Controls

All the main controls sit in one clear panel under the model settings.

**Where to find it:** in the File tab, right under the Model row. Tick your formats, then click GENERATE SUBTITLE FILE.

![Where to find File Formats and GENERATE SUBTITLE FILE: File tab, under the Model row](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/DMniPxbKoWBYZULjoP6Dj.png)

Here are the formats and run controls up close:

![File formats, Generate and Cancel buttons, batch size and extra settings](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/7A19brZz1Xv_JMbmmXsxW.png)

1: tick as many formats as you like. 2 and 3: start and stop jobs. 4: previous-text context is tuned for long files automatically. 5: open the extra sections for advanced settings, the music remover, the voice filter and speaker labels. 6: every job ends with a clear summary. One run writes all six files, named after your input file:

![SRT, WebVTT, TXT, LRC, JSON and TSV output of one run with speaker labels](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/RUNMacJnj7vgZdvtKd4bd.png)

The best-quality presets use word timestamps. You get clean sentence-level subtitles, or word-level highlighted SRT and WebVTT when you want them.

## Advanced Parameters

**Where to find it:** open the File tab, scroll down and click Advanced Parameters. The Open / close all sections button at the top opens it too, together with every other section.

![Where to find Advanced Parameters: File tab, then the Advanced Parameters section near the bottom](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/eCkJJ860iIrfJp1IjR9he.png)

The panel opens with every decoding setting, each with a short explanation under it:

![Advanced Parameters panel with beam size, prompts, word timestamps, hotwords and RAM offload](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/DQ4KZie5FTBIGmghPwz73.png)

Use Hotwords or the Initial Prompt for names and terms. Turn on Use Batched Inference for extra speed on long files. Offload Models to RAM When Idle keeps your models ready while your VRAM stays free.

## Music Remover, Voice Filter and Speaker Labels

Three built-in filters prepare your audio before transcription.

**Where to find it:** in the File tab, scroll down. The three sections sit right under Advanced Parameters; click a title to open it.

![Where to find the music remover, voice filter and Diarization: File tab, under Advanced Parameters](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/9BXfvdCES5POLaZFm0l40.png)

Here are the three sections opened:

![Background Music Remover, Silero voice detection and Diarization settings](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/DT3n7S1ip7raHI02MfU8j.png)

The Background Music Remover separates speech from music with UVR MDX-Net. The Silero voice filter skips silence. Diarization adds speaker labels such as SPEAKER_00 and SPEAKER_01 without a Hugging Face token. In our test it labelled every turn of a two-person interview correctly.

## Batch Processing Whole Folders

Transcribe a whole folder in one click, subfolders included.

**Where to find it:** in the File tab, on the right side, next to the upload box.

![Where to find Batch Processing: File tab, right side, next to the upload box](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/ND4_QO4sHtxIjOtoL3N6u.png)

Tick Enable Batch Processing and set your folders:

![Batch processing: 3 videos in subfolders transcribed in 26 seconds](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/ZRMfIVJjkmuqvTI753UOn.png)

Here 3 videos in 3 subfolders, 18 minutes in total, were finished in 26 seconds. The output folder mirrors your folder tree, finished files are skipped on the next run, and one broken file never stops the batch.

## YouTube Videos and Whole Channels

Paste a YouTube link and the app loads the thumbnail, title and description.

**Where to find it:** click the Youtube tab. The Youtube Link box is at the top.

![Where to find the YouTube features: the Youtube tab](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/Ixfg7O_sBC9CTOKVN4QLn.png)

The link loads the video details right away:

![YouTube tab with the video link, thumbnail, title and channel mass transcribe option](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/q1n1Zi6M5LtOTDt90Opl7.png)

Click Generate and the app downloads the audio and transcribes it. Our 12 minute video went from link to finished subtitles in about 15 seconds. Turn on Mass Transcribe Latest Channel Videos to process the latest videos of a whole channel, one after another.

## Live Microphone

Speak and watch the text appear. The Mic tab has two modes.

**Where to find it:** click the Mic tab. Live Mic is on the left and Record Then Generate on the right.

![Where to find the microphone features: the Mic tab](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/7MsPWZ5j97F8IJSAedk0v.png)

Press Live Mic Record and start speaking:

![Live Mic with the live transcription preview](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/rg3AYAng0YzGpnSIEDLoQ.png)

Live Mic shows a preview while you speak and updates it every 2 seconds. When you stop, the whole recording is transcribed with the full model and saved as subtitle files. Record Then Generate lets you record first and create the subtitles after.

## Subtitle Translation to 200 Languages

Translate your subtitle files with Meta NLLB, right inside the app.

**Where to find it:** click the T2T Translation tab, drop your subtitle files, then open the NLLB tab. DeepL API is right next to it.

![Where to find subtitle translation: the T2T Translation tab, then NLLB or DeepL API](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/L0cLk4koKD9U3JhxURIUE.png)

Choose the languages and click TRANSLATE SUBTITLE FILE:

![NLLB translation of an English subtitle file to Spanish with speaker labels kept](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/vCDHVd046bmHV_pa1ZS2C.png)

Choose the source and target language, click Translate and download the new file. Timestamps and speaker labels stay in place. A DeepL API tab is included too, for your own DeepL key.

## Background Music Separation

The BGM Separation tab splits any audio file into music and voice.

**Where to find it:** click the BGM Separation tab, drop your audio files and click SEPARATE BACKGROUND MUSIC.

![Where to find background music separation: the BGM Separation tab](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/oIBPR0xY9Iq4jMKOvxpeU.png)

After the separation you get two tracks:

![BGM Separation tab with separate music and voice tracks](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/9Q6j5ONNJaoPMO-CseyW2.png)

You get a music-only track and a clean voice track, both saved in your outputs folder.

## Light and Dark Theme

The app starts in dark mode, and one click switches to the light theme.

![The app in light theme](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/FsRcW-BcDVWcsA8Lkd_sq.png)

The Open / close all sections button opens or closes every panel at once.

## Latest Updates

Whisper-WebUI Premium gets frequent updates. Versions 12.4 to 13 came out between 27 and 29 September 2026:

- New INT8 ConvRot models for Whisper large-v3, large-v1 and Canary-Qwen 2.5B, downloaded automatically.
- Lower word error rate in English: the INT8 models now match the published accuracy of the originals.
- Canary-Qwen transcribes recordings up to 40 seconds in one piece and cuts longer ones at pauses.
- Canary batch size is picked from your GPU memory, with automatic recovery when VRAM runs short.
- Models wait in RAM between jobs, so the next job starts in under a second.
- Faster start, with every startup step shown in CMD.
- Batch processing keeps your subfolders, skips finished files and lists any failed files at the end.
- All GPU jobs wait in one queue and run one after another, so two jobs never clash.
- A redesigned interface with light and dark themes.

## Get Whisper-WebUI Premium

Download the latest zip file attached to this post, extract it and run the installer for your platform. To update, get the newest zip, overwrite the old files and run Windows_Install_Update.bat again.

- **Full tutorial video:** [watch it on YouTube](https://www.youtube.com/watch?v=4lAk6sf1qF8).
- **Requirements tutorial:** [Python, Git, FFmpeg, CUDA and C++ tools step by step](https://youtu.be/DrhUHnYfwC0).
- **Support:** [our Discord channel](https://discord.com/channels/772774097734074388/1079506787734134844).
