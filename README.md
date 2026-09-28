# Whisper-WebUI Premium made for SECourses Patreon followers only : https://www.patreon.com/posts/145395299

## Download Installers and App

> https://www.patreon.com/posts/145395299

## Quick Info
- This app has the combination of perfect implementation of the following repos and their advanced forks with so many additional new features and improvements (models auto downloaded, everything automatically installed into Python 3.12 venv, best quality presets fully ready):
  -   Whisper from OpenAI : [https://github.com/openai/whisper](https://github.com/openai/whisper)
  -   NVIDIA NeMo Canary-Qwen-2.5B : [https://huggingface.co/nvidia/canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b)
-   Full tutorial video (2 May 2026) : [https://www.youtube.com/watch?v=4lAk6sf1qF8](https://www.youtube.com/watch?v=4lAk6sf1qF8)

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/ffd01d11-ba2d-48a4-b5b0-be723218e38b" />

### 28 September 2026 - Version 12.10

- New INT8 ConvRot Canary-Qwen model: `canary-qwen-2.5b-int8-convrot`
  - About 10x faster than the NeMo model with the same accuracy: 5 hours of test videos took 158 seconds instead of 1679 seconds (28 minutes), with 10.82% WER instead of 10.80% (RTX 5090, Canary Qwen Best Quality preset)
  - A 10 second chunk takes about 0.07 seconds instead of 1.1 seconds, and a 1 hour video about 40 seconds
  - Closer to the full-precision model than the BF16 model used before (next-token KL divergence against FP32: 0.00003 instead of 0.00024)
  - Model file is 2.9 GB instead of 5.1 GB
  - Automatically downloaded from Hugging Face the first time you select it
  - Canary Qwen Best Quality preset and the Canary-Qwen default model now use it
  - Needs an RTX 3000 series (Ampere) or newer GPU; on older GPUs the app automatically uses `nvidia/canary-qwen-2.5b`
  - Beam search, sampling and Canary Generation Kwargs keep working
  - The first transcription on a new PC compiles and tunes its GPU kernels once (about 30 seconds, shown in CMD and Live Transcription); later runs reuse them

### 28 September 2026 - Version 12.9

- Starting two jobs at once (two tabs, two users, or a new job right after Cancel) no longer crashes: all GPU jobs (File, YouTube, Mic, BGM Separation, NLLB translation) now wait in one queue and run one after another
- Batch processing
  - One broken file no longer stops the batch: the other files are transcribed and the failed files are listed at the end, and the files that finished stay downloadable
  - A file that cannot be opened is reported as failed instead of writing empty subtitle files
  - Folder paths copied with Explorer's "Copy as path" (with quotes) now work, also for OPEN OUTPUTS FOLDER
  - Files with the same name (talk.mp3 and talk.wav) no longer overwrite each other's subtitles (talk.srt and talk_wav.srt), .webm files are no longer transcribed twice, and upper-case extensions such as .MP4 are found on Linux too
  - An existing lecture-2.srt no longer makes lecture.mp4 count as already done
- Use Batched Inference no longer stops with an error, and its INT8 memory use no longer grows with every file: it reached 64 GB of VRAM after 7 files at batch size 16, now 15.6 GB, at the same speed
- The first INT8 transcription after starting the app no longer spends about 16 seconds compiling GPU code again (a 30 second file: 18.4 → 1.5 seconds): the compiled code is now kept in models\cuda_cache, so the first few starts after this update are still slow while it fills
- Long files start sooner: the check that a file can be opened no longer decodes the whole file before the transcription decodes it again
- Canary-Qwen
  - Audio is cut into chunks at pauses instead of every 10 seconds exactly, so words are no longer split at the cuts: word error rate 11.8% → 8.2% on our 5 minute test video
  - Loading no longer puts the 32-bit model (about 10 GB) on the GPU before converting it, which ran out of memory on 8-12 GB GPUs
- Switching the Base Model loads its best quality model settings into that tab only: before, it also changed the Base Model of the other tabs and reset settings such as the output folder
- Changing the model or the Base Model unloads the previous model first, and only the engine in use keeps its models, so memory no longer adds up when you switch models
- Saved presets keep decimal values (Hallucination Silence Threshold 0.5 was saved as 0)
- Diarization keeps the word timestamps (word-level and highlighted subtitles work with speaker labels) and labels every subtitle with a speaker: "None|" is gone
- Background Music Remover
  - Separates .m4a, .aac, .wma and .opus files
  - Changing the UVR model or Segment Size now takes effect, and the model is no longer reloaded for every file
  - A video after an audio file in the same batch is no longer transcribed at the wrong speed
  - The SEPARATE BACKGROUND MUSIC button explains what is missing instead of showing only "Error"
- Subtitles
  - Chinese, Japanese and Thai subtitles no longer get spaces between words when word timestamps are on
  - Highlighted word subtitles no longer contain every line twice
  - Silent files no longer break the txt and TSV outputs
- Temperature fallback now works (a window whose output repeats is decoded again with more randomness), and Repeat Initial Prompt Every Window now really repeats the prompt
- Insanely Fast Whisper no longer fails on RTX 20 and GTX 16 GPUs (it used flash-attention 2, which needs an RTX 30 or newer)
- Translation
  - YouTube captions (.vtt) translate correctly, including their first lines
  - An interrupted NLLB model download is resumed on the next run instead of failing every time; Santali translates, and two languages that NLLB cannot translate were removed from the list
- Cancel Generation stops only your own job, not the jobs of other users
- Diarization works on PCs without an NVIDIA GPU (the device was stuck on cuda); Language Detection Threshold 0 no longer fails; CMD shows non-English file names and errors from sub process jobs correctly
- INT8 models waiting in RAM use about 530 MB less RAM

### 28 September 2026 - Version 12.8

- New defaults in all built-in presets: Offload Models to RAM When Idle is on, Start As Sub Process and Offload model when finished are off
  - Jobs run inside the app and the models wait in RAM between jobs: from the second job on, a 5 minute file takes 7 seconds instead of about 20 seconds per job before (RTX 5090)
  - Cancel Generation can only stop a running job when Start As Sub Process is enabled
- Faster app start, and CMD now shows every startup step with its time instead of staying empty until the web address appears
  - Normal start: 9.2 → 5.6 seconds; first start after an install or update: 21.6 → 11.7 seconds (RTX 5090 test PC)
  - The Background Music Remover, Insanely Fast Whisper (Transformers) and NLLB libraries are loaded only when they are used
  - The installers now compile the Python libraries during install, so the first start no longer looks frozen: use the latest installer files and run Windows_Install_Update.bat to update
- Batch processing keeps the models loaded until the whole batch is done: before, the Whisper model (and the Background Music Remover and Diarization models when enabled) was unloaded and reloaded for every file
- Download Transcription
  - The zip is made once, when the job finishes: before, a new and bigger zip was written into outputs\_download_bundles on every live update of a batch, and none were ever deleted (zips older than a day are now removed)
  - The zip keeps the batch subfolders (partA/segment.srt and partB/segment.srt instead of segment.srt and segment_2.srt)
- The first download of an Insanely Fast Whisper or NLLB model now shows its progress in CMD (before, up to 17.6 GB were downloaded with no output)
- While a model downloads or loads, Live Transcription says so ("Still working... downloading the model") instead of "waiting for the first segment", and these lines no longer break the download bar in CMD
- Presets
  - Deleting a preset asks for confirmation, keeps your current settings and shows "Deleted preset" (before, every setting was reset to the defaults and the message was replaced by "No preset selected")
  - Loading a saved preset of another Base Model keeps its settings (its model was reset to the default of that Base Model)
  - Choosing a Base Model or a preset no longer starts several competing updates, which could leave a setting such as Condition On Previous Text different from the preset
- Canary-Qwen no longer moves the Hugging Face cache of the other models into its own folder
- The startup message names the real default Base Model (faster-whisper), the classic console no longer shows "�" instead of emoji, and --allowed_paths is no longer read with eval()
- YouTube: ffmpeg no longer prints about 50 lines of build information into CMD for every video

### 27 September 2026 - Version 12.7

- Fixed the installer stopping with `CERTIFICATE_VERIFY_FAILED ... self-signed certificate in certificate chain` while downloading the diarization models, on PCs with antivirus HTTPS scanning (for example Kaspersky) or a company proxy
  - Downloads are now verified with the Windows certificate store, as browsers and pip do: use the latest installer files and run Windows_Install_Update.bat to update
- Insanely Fast Whisper: `large-v1` and `turbo` no longer fail with "404 Repository Not Found"
  - `large-v1` now downloads a 3.1 GB FP16 copy of OpenAI's large-v1 (the original is 6.2 GB FP32) from our Hugging Face repo
  - `turbo` and `large` now load `large-v3-turbo` and `large-v3`, the same models these names mean in the Whisper (faster-whisper) list
- Batch processing with an Output Folder now keeps the input subfolders: before, two files with the same name in different subfolders shared one output, and the second one was skipped as "outputs already exist"

### 27 September 2026 - Version 12.6

- Fixed Canary-Qwen on new installs: a new fsspec release made the installer pick a 2020 version of `datasets`, and Canary-Qwen stopped with `module 'pyarrow' has no attribute 'PyExtensionType'`
  - The requirements now require `datasets` 4.0 or newer: use the latest installer files and run Windows_Install_Update.bat to update
- Live Mic now keeps the whole recording: before, about half of the audio was lost (7 seconds saved from a 14 second recording) and the button showed "Waiting" instead of "Stop"
  - The live preview runs in the background, so it never interrupts the recording
  - With Auto transcribe while recording turned off, Stop still saves and transcribes the whole recording
- YouTube tab: Live Transcription now shows the download, every segment and the result as they happen, also for Mass Transcribe Latest Channel Videos
- NLLB translation
  - No more repeating lines such as "İran'ın, İran'ın, İran'ın…" (beam search and a length limit tied to each line)
  - Speaker labels from Diarization (`SPEAKER_00|`) are kept in the translated subtitles
  - Clicking Translate without a file or languages now tells you what is missing instead of doing nothing
- DeepL: a missing or wrong API key now shows a clear message (for example HTTP 403) instead of failing silently
- The first use of a model now shows "Downloading model ..." in Live Transcription instead of only "waiting for the first segment"
- When a job fails, the full error is now printed in CMD too, so saved console logs show the cause
- A file that cannot be opened (corrupted or unsupported) now shows a warning instead of "Done! 0 segments"
- A Hugging Face token saved with `hf auth login` is now used, so model downloads are no longer unauthenticated
- YouTube audio is downloaded into a temporary folder that is removed after each job (it was written into the Whisper-WebUI folder)
- Advanced Parameters: decimal settings such as No Speech Threshold now change in 0.05 or 0.1 steps with the arrow keys (they jumped from 0.6 to 1)

### 27 September 2026 - Version 12.5

- Redesigned interface in the style of the IndexTTS app
  - Every button has its own color and icon, dark theme by default with a Light / dark theme switch, and an Open / close all sections button
  - Smoother: no constantly animated buttons and lighter page scripts
- Uploaded videos are previewed directly, without any conversion
  - MKV (H.264, VP9, HEVC) plays as soon as it is uploaded; formats the browser cannot play (such as AVI) show a note and still transcribe normally
  - Load From File Path no longer copies the file: a 3 GB MKV loads in 0.2 seconds instead of 6 seconds
- Fixed Insanely Fast Whisper hallucinations: audio is now decoded in windows of up to 30 seconds that end in a pause (test video WER 57% → 16%)
- Fixed Canary-Qwen on new installs (the latest NeMo needs lhotse 2.0.0a6)
- Fixed NLLB subtitle translation with Transformers 5, and it no longer downloads a second, unused copy of each NLLB model
- INT8 ConvRot models: the one-time Triton kernel tuning now shows its progress in CMD and in Live Transcription, and the tuned kernels are cached in `Whisper-WebUI\models\triton_cache` and reused by every later run (a 5 minute file then takes about 7 seconds on an RTX 5090)
- All output formats of one run now share the same timestamp in their file names, and durations read like "1 minute 6 seconds"
- Requirements updated (Transformers 5.17.0, lhotse 2.0.0a6): use the latest installer files and run Windows_Install_Update.bat to update

### 27 September 2026 - Version 12.4

- New INT8 ConvRot Whisper models: `large-v3-int8-convrot` and `large-v1-int8-convrot`
  - About 3x faster than the standard models with the same accuracy: on held-out test videos Large-v3 finished in 104 seconds instead of 327 seconds, with 7.92% WER instead of 8.21%
  - Model files are 1.6 GB instead of 3.1 GB
  - Automatically downloaded from Hugging Face the first time you select them
  - Fast Whisper Best Quality preset now uses `large-v3-int8-convrot` by default
  - Needs an RTX 3000 series (Ampere) or newer GPU; on older GPUs the app automatically uses the standard model
- New option: Offload Models to RAM When Idle
  - After each job the loaded models move to system RAM and free the VRAM; the next job moves them back to the GPU in under a second instead of loading them from disk again
  - Works with Whisper, Insanely Fast Whisper, Canary-Qwen and speaker diarization, with or without Start As Subprocess
- All model downloads and caches now stay inside the `Whisper-WebUI\models` folder
- The background music remover model is no longer downloaded again every time it is used
- Just run Windows_Install_Update.bat to update

### 15 June 2026 - Version 12.3 

- Now when downloading model - when you first time use that model - it will show download progress on CMD
- Now it will show selected and used model on CMD
- Just run Windows_Install_Update.bat to update / install

<img width="3516" height="425" alt="image" src="https://github.com/user-attachments/assets/41c62156-51ac-45c7-bee9-de42f78978f9" />

### 8 June 2026 - Version 12.2 

-   -   This is a major quality upgrade        
    -   Both Whisper and Canary models made more robust        
        -   Thus, if you were getting random errors on random files, it should not happen any more            
        -   Even though they were rare edge cases we fixed this issue            
    -   Whisper models seperated into below 2 model selection        
        -   Whisper (faster-whisper / CTranslate2)            
        -   Insanely Fast Whisper (Transformers)            
    -   Process based executing improved and cancel feature improved, now cancel immediately works properly        
    -   For Whisper models, the displayed messages during processing improved        
        -   Now you will see all messages on CMD and Gradio accurately            
    -   Default system presets updated to Fast Whisper Best Quality, Insane Fast Whisper Best Quality, Canary Qwen Best Quality        
    -   Now default selected preset / model is Fast Whisper Best Quality with Whisper Large-v3        
        -   Whisper Large-v3 model is the most capable robust model that excels at 100+ languages            
    -   I have done a very through research and experimentation to remake these presets with improved accuracy        
        -   Our accuracy is improved over 80% now            
    -   Now Word Timestamps is automatically selected along with new option Normalize Word Timestamp Output        
        -   This ensures that you get sentence level accurate subtitles / transcription not just 30 second long speeches            
    -   You can see newest quality research and new preset results tested on very hard to transcribe audio files as below        
    
    <img height="600" alt="image" src="https://github.com/user-attachments/assets/d5a6b565-6262-4633-bc77-9310a7c4d115" />

### Word Error Rate (WER)
-   -   WER means Word Error Rate.        
        -   It measures word-level transcription mistakes:            
    -   WER = (substituted words + deleted words + inserted words) / reference words        
        -   So if the real subtitle has 100 words and the transcription has 6 total word mistakes, WER is 6%.            
    -   WER catches:        
        -   \- missing words            
        -   \- wrong words            
        -   \- extra/repeated words            

### CER means Character Error Rate.
-   -   It is the same idea, but measured at the character level instead of word level:        
        -   CER = (substituted chars + deleted chars + inserted chars) / reference chars
                -   CER catches smaller spelling/detail mistakes better.        
        -   Example:            
            -   Reference:                
                -   Wan 2.2 training                    
                -   Prediction:                    
                -   One 2.2 trainings                    
        -   WER may be high because Wan became One and training became trainings.            
        -   CER may be lower because most letters are still similar.            
    -   For our use case, WER is the main metric because you care about missing words, wrong words, repeated words, and hallucinated extra words. CER is useful secondary evidence for spelling accuracy.
        
<img height="600" alt="image" src="https://github.com/user-attachments/assets/63c1cee0-4d0b-4e80-9fe9-fe364e8a3f00" />

<img height="600" alt="image" src="https://github.com/user-attachments/assets/c8a05c8a-8f2e-4210-8fd3-ffdfa53bcc75" />

### 26 May 2026 - Version 12.0 
-   With new zip file below errors fixed    
    -   Token: not provided Note: pyannote gated files require --token or HF\_TOKEN on first download. \[ERROR\] Dependency source does not expose pytorch\_model.bin: MonsterMMORPG/Wan\_GGUF/pyannote\_segmentation3
        
### 2 May 2026 - Version 11.1
-   Some installer bugs fixed    
-   Enable Background Music Remover Filter - fixed    
-   Enable Silero VAD Filter - fixed

## 30 April 2026 - Version 10.0

- This is a quite big upgrade to our application

- We now fully support NVIDIA NeMo Canary-Qwen-2.5B is an English speech recognition model : https://huggingface.co/nvidia/canary-qwen-2.5b

- This model is currently State Of The Art (SOTA) Speech to Text model for English language

- I have done extensive research and testing and it is set to best default parameters

- Fully supporting all of the features our Whisper app were already supporting

- Get the zip file, overwrite all previous files and run installer for update / upgrade

- The model will be auto downloaded when you first time run

<img width="3567" height="602" alt="image" src="https://github.com/user-attachments/assets/583647bc-9120-4c6e-ad67-1f5ad1ee24ab" />

- I also have compared with Whisper best configurations are here the comparison results - best results of Whisper taken

<img height="600" alt="image" src="https://github.com/user-attachments/assets/9baabf10-6511-4b63-a4bb-60b4b3c998fc" />

<img height="600" alt="image" src="https://github.com/user-attachments/assets/0a45ac6b-4898-4e15-a629-41381a9d6169" />

<img height="600" alt="image" src="https://github.com/user-attachments/assets/65d60c04-06f1-404f-957d-420420fc664d" />

- As you can see NVIDIA NeMo Canary-Qwen-2.5B is not only significantly better but also faster 


## 15 April 2026 - Version 8.0

- Diarization had some error and this is fixed

- Mic tab completey remade and now both live transcription from microphone and offline transcription from microphone working

  - Live transcription quality is not that great

  - Both live transcription and offline transcription recordings from microphone will be saved in outputs folder

  - Live transcription will auto run but for offline transcription first record voice with microphone and then click Generate Subtitles button

- Don't forget to select your working microphone and give permission for app to use your microphone from your browser

- For update / install get the latest zip file, overwrite older files and run Windows_Install_Update.bat

<img height="600" alt="image" src="https://github.com/user-attachments/assets/961d6b7a-fd78-434c-977a-6785d12148a8" />

<img height="600" alt="image" src="https://github.com/user-attachments/assets/bf4e6bf1-af92-47a7-9df0-b1782bb0bd63" />


## 14 April 2026 - Version 7.0

- Now auto downloads Diarization files and thus you don't need to enter Hugging Face token and get permission

- Now you can copy paste any YouTube link and generate subtitles

  - This was broken and now fixed

  - It will save generated files with same name as the video title

- Now you can batch generate subtitles for YouTube video channels

- Paste the video channel, enable batch and it will generate subtitles for every video

  - Set how many videos you want (scans latest ones)

  - You may get rate limited by YouTube

- For update / install get the latest zip file, overwrite older files and run Windows_Install_Update.bat

<img height="600" alt="image" src="https://github.com/user-attachments/assets/023176ee-146f-4886-b92c-07a7904435eb" />

## 8 April 2026 - Version 5.0 

- This is a massive update with so many new features

  - Get the latest zip file and make a fresh install please > https://www.patreon.com/posts/145395299

  - 1-Click to install on Windows, RunPod, SimplePod, Massed Compute, Linux
 
  - <img height="500" alt="image" src="https://github.com/user-attachments/assets/27909c4a-bd77-408f-824a-ab8fc9837379" />

- New preset save and load system with locked best-quality presets for faster-whisper, Insanely Fast Whisper, and Canary-Qwen

  - Presets are automatically loaded as you change them and also last used preset is remembered when you restart the app

  - Word Timestamps is enabled by default to improve quality but it also generates regular version as well automatically

- Download transcription button 

- Open outputs folder button (all transcriptions automatically saved)

- Load video / audio file directly from path (useful for platforms like RunPod where Gradio upload is slow)

<img height="600" alt="image" src="https://github.com/user-attachments/assets/95b70223-04bc-4ecf-a65e-6af3c025c190" />

- The fast preset uses new custom in house implemented batch size 32 feature and it is literally blazing fast compared to all other existing Whisper apps and repos

- Fully supporting all kind of video and audio formats upload with full preview

- Batch folder processing process given folder all files automatically

- Live transcription Window that shows latest transcription live while processing

- At batch size 1 with best quality, 11x real time transcription speed (depends on GPU)

- At batch size 32 fast preset 15x to 30x real time transcription speed (depends on GPU)

- New feature Repeat Initial Prompt Every Window

<img height="600" alt="image" src="https://github.com/user-attachments/assets/64ec2ff9-bbbe-400b-a26d-5df4edc44a76" />

- Supports all Whisper models like Large V1, Large V3, Turbo, Distill Large, Tiny, etc

- Supports following format outputs you can have checked all so all generated at the same time : SRT, WebVTT, txt, LRC,JSON, TSV

  - All outputs will have the same name as your input file name

- With sub process working system, you can cancel any processing immediately with 0 RAM or VRAM leak

- Fully supports Windows and Linux (use Massed Compute installer)

- Based on Python 3.11 VENV and CUDA 13 and Torch 2.9.1 with pre-compiled libraries like Flash Attention

- If you don't like output, try to enable / disable Condition On Previous Text it makes big difference

<img height="600" alt="image" src="https://github.com/user-attachments/assets/a3f9fc54-11dd-4d94-b8af-72184453b5f3" />

- The app supports 100 languages and 32 models

<img height="600" alt="image" src="https://github.com/user-attachments/assets/0af42f4f-ad2f-4b87-ac1b-d965faf59604" />

<img height="600" alt="image" src="https://github.com/user-attachments/assets/04aedf3e-8d95-48c9-8063-625491534870" />

<img height="600" alt="image" src="https://github.com/user-attachments/assets/c5d3ab44-fb34-479e-b5a6-8cc596a7ee14" />

- Lots of Advanced Parameters and all set to best quality 

- Built in Background Music Remover Filter

- Built in Voice Detection Filter

- <img height="600" alt="image" src="https://github.com/user-attachments/assets/50672e86-d55c-4aba-b761-4f1aacbae020" />

- Fully detailed CMD output to watch entire progress

- Extremely optimized VRAM usage as low as 6 GB GPUs

<img width="1722" height="399" alt="image" src="https://github.com/user-attachments/assets/dd93da42-c52f-42d7-b55f-c2070cb74013" />

- Some other utility features like YouTube, record from a Mic, T2T Translation, BGM Seperation

<img height="600" alt="image" src="https://github.com/user-attachments/assets/f0647197-25f5-4e7b-9ab6-dd3740f743af" />


### Full Page Screenshot

<img height="1200" alt="screencapture-127-0-0-1-7861-2026-05-02-05_09_06" src="https://github.com/user-attachments/assets/78cffef8-e3d1-42dc-a58b-e346cd74dc7e" />




