# Compatibility verification (version 12.11)

Verified on Linux x86-64 (Ubuntu 22.04) on 28 September 2026 with Python 3.12.14, Gradio 6.28.0, PyTorch
2.13.0+cu130, faster-whisper 1.2.1, CTranslate2 4.8.2, ONNX Runtime 1.30.0, Transformers 5.17.0, NVIDIA driver
580.65.06 on RTX A6000 GPUs, Node.js 22 and installed Google Chrome 135.0.7049.52. **Native Windows was not
available**: Windows behaviour is covered by the path, launcher and DLL tests below and by keeping the new code to
portable APIs, not by running the Windows installer.

## Checks in this release

| Check | Result |
| --- | --- |
| Test suite (`pytest tests/`, one RTX A6000 visible) | 388 passed, 4 skipped: the opt-in real-file batching test (`WHISPERWEBUI_RUN_REAL_TESTS=1`), a test that needs two GPUs, and two DeepL tests without an API key |
| Recorder JavaScript suite (`node tests/test_live_microphone.mjs`) | 5 passed |
| Every UI feature in Google Chrome | see [Chrome verification](chrome-verification.md) |
| Model downloads | large-v3 and large-v1 (original FP16 and ConvRot INT8) and Canary-Qwen (original NeMo and INT8) download into the installation's own `models` folder on first use |
| Linux launcher | `./start-webui.sh` failed with "Permission denied": `start-webui.sh` and `Install.sh` were stored in git without the executable bit (fixed) |
| English accuracy, speed and VRAM | 2,700 short clips and 120 long recordings through the app, see [English benchmarks](english-benchmarks.md) |
| Smaller GPUs | Canary-Qwen INT8 1,402-file sessions on simulated 6, 8, 10, 12 and 16 GB cards (PyTorch memory capped) without a failed file; out-of-memory recovery added |
| faster-whisper version | pinned to 1.2.1 in `requirements_whisper.txt` (used by the Windows and the Linux installers): the app runs its own copy of faster-whisper's decoding loop for the end-of-file check and the INT8 engine |

## Portability of the 12.11 changes

* Audio decoding is faster-whisper's PyAV decoder without its per-file `gc.collect()` (bit-identical samples);
  PyAV ships wheels for Windows and Linux.
* Voice detection (Canary chunk cuts, the Whisper end-of-file check) uses the Silero ONNX model bundled with
  faster-whisper on ONNX Runtime's CPU provider, in a daemon thread that stops when a transcription ends, fails or
  is cancelled. It is loaded once per process with at most 4 threads.
* The Canary out-of-memory recovery only uses PyTorch APIs (`torch.cuda.empty_cache`, dropping CUDA graphs) and
  recognizes both PyTorch's `OutOfMemoryError` and CUDA "out of memory" runtime errors.
* The VRAM tiers read `torch.cuda.mem_get_info()` at startup, so the Windows desktop's own VRAM use counts as used
  memory.

## Remaining verification limits

* Native Windows installation, NVIDIA driver loading, Chrome microphone capture and CUDA inference still need a
  Windows machine. Linux tests cover Windows path forms, reserved file names and the `nvcuda.dll`/`WinDLL` branch.
* Physical 6-16 GB GPUs were not available; they were simulated by limiting PyTorch's allocator to the card's size
  minus 0.9 GB. Memory outside PyTorch's allocator (CUDA context, graph executables) measured about 0.4 GB
  here, on a server without a desktop.
* YouTube refused this server's address ("Sign in to confirm you're not a bot"), so a successful YouTube download
  could not be tested; no DeepL API key was available.
* Only English was tested.
