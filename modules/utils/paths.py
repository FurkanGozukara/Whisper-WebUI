import os

WEBUI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(WEBUI_DIR, "models")
WHISPER_MODELS_DIR = os.path.join(MODELS_DIR, "Whisper")
FASTER_WHISPER_MODELS_DIR = os.path.join(WHISPER_MODELS_DIR, "faster-whisper")
INSANELY_FAST_WHISPER_MODELS_DIR = os.path.join(WHISPER_MODELS_DIR, "insanely-fast-whisper")
CANARY_QWEN_MODELS_DIR = os.path.join(WHISPER_MODELS_DIR, "canary-qwen")
NLLB_MODELS_DIR = os.path.join(MODELS_DIR, "NLLB")
DIARIZATION_MODELS_DIR = os.path.join(MODELS_DIR, "Diarization")
UVR_MODELS_DIR = os.path.join(MODELS_DIR, "UVR", "MDX_Net_Models")
CONFIGS_DIR = os.path.join(WEBUI_DIR, "configs")
UI_DEFAULTS_DIR = os.path.join(WEBUI_DIR, "modules", "ui", "defaults")
UI_SYSTEM_PRESETS_DIR = os.path.join(UI_DEFAULTS_DIR, "presets")
DEFAULT_PARAMETERS_PATH = os.path.join(UI_DEFAULTS_DIR, "default_parameters.yaml")
DEFAULT_PARAMETERS_CONFIG_PATH = DEFAULT_PARAMETERS_PATH
I18N_YAML_PATH = os.path.join(CONFIGS_DIR, "translation.yaml")
OUTPUT_DIR = os.path.join(WEBUI_DIR, "outputs")
TRANSLATION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "translations")
UVR_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "UVR")
UVR_INSTRUMENTAL_OUTPUT_DIR = os.path.join(UVR_OUTPUT_DIR, "instrumental")
UVR_VOCALS_OUTPUT_DIR = os.path.join(UVR_OUTPUT_DIR, "vocals")
PRESETS_DIR = os.path.join(WEBUI_DIR, "presets")
BACKEND_DIR_PATH = os.path.join(WEBUI_DIR, "backend")
SERVER_CONFIG_PATH = os.path.join(BACKEND_DIR_PATH, "configs", "config.yaml")
SERVER_DOTENV_PATH = os.path.join(BACKEND_DIR_PATH, "configs", ".env")
BACKEND_CACHE_DIR = os.path.join(BACKEND_DIR_PATH, "cache")

for dir_path in [MODELS_DIR,
                 WHISPER_MODELS_DIR,
                 FASTER_WHISPER_MODELS_DIR,
                 INSANELY_FAST_WHISPER_MODELS_DIR,
                 CANARY_QWEN_MODELS_DIR,
                 NLLB_MODELS_DIR,
                 DIARIZATION_MODELS_DIR,
                 UVR_MODELS_DIR,
                 CONFIGS_DIR,
                 UI_DEFAULTS_DIR,
                 UI_SYSTEM_PRESETS_DIR,
                 OUTPUT_DIR,
                 TRANSLATION_OUTPUT_DIR,
                 UVR_INSTRUMENTAL_OUTPUT_DIR,
                 UVR_VOCALS_OUTPUT_DIR,
                 BACKEND_CACHE_DIR]:
    os.makedirs(dir_path, exist_ok=True)


def _is_inside(path: str, folder: str) -> bool:
    path = os.path.normcase(os.path.abspath(path))
    folder = os.path.normcase(os.path.abspath(folder))
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:  # different drives
        return False


def configure_model_cache_env() -> None:
    """Point every library cache that can download models (Hugging Face hub and Xet, torch.hub, NeMo) into MODELS_DIR.

    Call it before huggingface_hub is imported (gradio imports it): the hub reads these variables once, at import.
    Values are assigned, not defaulted, so a global HF_HOME or HF_HUB_CACHE cannot send downloads elsewhere.
    """
    default_hf_home = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    previous_hf_home = os.environ.get("HF_HOME")
    token_homes = [previous_hf_home or default_hf_home]
    if previous_hf_home and _is_inside(previous_hf_home, MODELS_DIR):
        # The start scripts set HF_HOME=models, where `hf auth login` never saves its token.
        token_homes.append(default_hf_home)
    if not os.environ.get("HF_TOKEN_PATH"):
        # A token saved by `hf auth login` is a credential, not a model; keep using it.
        for hf_home in token_homes:
            token_path = os.path.join(os.path.abspath(hf_home), "token")
            if os.path.isfile(token_path):
                os.environ["HF_TOKEN_PATH"] = token_path
                break
    for name in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        os.environ.pop(name, None)
    os.environ["HF_HOME"] = MODELS_DIR
    os.environ["HF_HUB_CACHE"] = os.path.join(MODELS_DIR, "hub")
    os.environ["HF_XET_CACHE"] = os.path.join(MODELS_DIR, "xet")
    os.environ["TORCH_HOME"] = os.path.join(MODELS_DIR, "torch")
    os.environ["NEMO_CACHE_DIR"] = os.path.join(MODELS_DIR, "NeMo")
    # Compiled Triton kernels and the INT8 ConvRot GEMM tuning results: kept with the models so
    # every later run reuses them, also on cloud pods where only the app folder persists.
    # An explicitly set TRITON_CACHE_DIR is respected.
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(MODELS_DIR, "triton_cache"))
    # The CUDA driver's cache of GPU code compiled at run time (flash-attn and cuBLAS kernels without a build for
    # the GPU, such as an RTX 50). The per-user cache that all CUDA apps share was full (1 GB) and evicted them,
    # so the first transcription after every start compiled them again: about 13 seconds of the 21 before the
    # first INT8 result. Kept with the models like the Triton cache; explicitly set variables are respected.
    os.environ.setdefault("CUDA_CACHE_PATH", os.path.join(MODELS_DIR, "cuda_cache"))
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(4 * 1024 ** 3))
    # After loading a .bin-only checkpoint (all three NLLB models) Transformers downloads a
    # .safetensors copy from the Hub's conversion PR in the background, which the app never loads:
    # +2.3 GB for nllb-200-distilled-600M, +17.6 GB for nllb-200-3.3B.
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
