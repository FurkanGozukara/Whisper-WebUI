# Whisper-WebUI Premium made for SECourses Patreon followers only : https://www.patreon.com/posts/145395299

## Download Installers and App

> https://www.patreon.com/posts/145395299

## Quick Info
- This app has the combination of perfect implementation of the following repos and their advanced forks with so many additional new features and improvements (models auto downloaded, everything automatically installed into Python 3.11 venv, best quality presets fully ready):
  -   Whisper from OpenAI : [https://github.com/openai/whisper](https://github.com/openai/whisper)
  -   NVIDIA NeMo Canary-Qwen-2.5B : [https://huggingface.co/nvidia/canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b)
-   Full tutorial video (2 May 2026) : [https://www.youtube.com/watch?v=4lAk6sf1qF8](https://www.youtube.com/watch?v=4lAk6sf1qF8)

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/ffd01d11-ba2d-48a4-b5b0-be723218e38b" />

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




