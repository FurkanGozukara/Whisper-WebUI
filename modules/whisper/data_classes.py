import gradio as gr
from typing import Any, Optional, Dict, List, Union, NamedTuple
from fastapi import Query
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
from copy import deepcopy
import yaml

from modules.utils.constants import *
from modules.utils.i18n import _
from modules.utils.text import repair_component_text
from modules.utils.whisper_languages import (
    normalize_lang_choice as normalize_whisper_lang_choice,
    normalize_lang_value as normalize_whisper_lang_value,
)


class WhisperImpl(Enum):
    WHISPER = "whisper"
    FASTER_WHISPER = "faster-whisper"
    INSANELY_FAST_WHISPER = "insanely_fast_whisper"
    CANARY_QWEN = "canary-qwen"


class Segment(BaseModel):
    id: Optional[int] = Field(default=None, description="Incremental id for the segment")
    seek: Optional[int] = Field(default=None, description="Seek of the segment from chunked audio")
    text: Optional[str] = Field(default=None, description="Transcription text of the segment")
    start: Optional[float] = Field(default=None, description="Start time of the segment")
    end: Optional[float] = Field(default=None, description="End time of the segment")
    tokens: Optional[List[int]] = Field(default=None, description="List of token IDs")
    temperature: Optional[float] = Field(default=None, description="Temperature used during the decoding process")
    avg_logprob: Optional[float] = Field(default=None, description="Average log probability of the tokens")
    compression_ratio: Optional[float] = Field(default=None, description="Compression ratio of the segment")
    no_speech_prob: Optional[float] = Field(default=None, description="Probability that it's not speech")
    words: Optional[List['Word']] = Field(default=None, description="List of words contained in the segment")

    @classmethod
    def from_faster_whisper(cls,
                            seg: Any):
        if seg.words is not None:
            words = [
                Word(
                    start=w.start,
                    end=w.end,
                    word=w.word,
                    probability=w.probability
                ) for w in seg.words
            ]
        else:
            words = None

        return cls(
            id=seg.id,
            seek=seg.seek,
            text=seg.text,
            start=seg.start,
            end=seg.end,
            tokens=seg.tokens,
            temperature=seg.temperature,
            avg_logprob=seg.avg_logprob,
            compression_ratio=seg.compression_ratio,
            no_speech_prob=seg.no_speech_prob,
            words=words
        )


class Word(BaseModel):
    start: Optional[float] = Field(default=None, description="Start time of the word")
    end: Optional[float] = Field(default=None, description="Start time of the word")
    word: Optional[str] = Field(default=None, description="Word text")
    probability: Optional[float] = Field(default=None, description="Probability of the word")


class BaseParams(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    def to_dict(self) -> Dict:
        return self.model_dump()

    def to_list(self) -> List:
        return list(self.model_dump().values())

    @classmethod
    def from_list(cls, data_list: List) -> 'BaseParams':
        field_names = list(cls.model_fields.keys())
        return cls(**dict(zip(field_names, data_list)))


# Models need to be wrapped with Field(Query()) to fix fastapi doc issue.
# More info : https://github.com/fastapi/fastapi/discussions/8634#discussioncomment-5153136
class VadParams(BaseParams):
    """Voice Activity Detection parameters"""
    vad_filter: bool = Field(default=False, description="Enable voice activity detection to filter out non-speech parts")
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Speech threshold for Silero VAD. Probabilities above this value are considered speech"
    )
    min_speech_duration_ms: int = Field(
        default=250,
        ge=0,
        description="Final speech chunks shorter than this are discarded"
    )
    max_speech_duration_s: float = Field(
        default=float("inf"),
        gt=0,
        description="Maximum duration of speech chunks in seconds"
    )
    min_silence_duration_ms: int = Field(
        default=2000,
        ge=0,
        description="Minimum silence duration between speech chunks"
    )
    speech_pad_ms: int = Field(
        default=400,
        ge=0,
        description="Padding added to each side of speech chunks"
    )

    @classmethod
    def to_gradio_inputs(cls, defaults: Optional[Dict] = None) -> List[gr.components.base.FormComponent]:
        inputs = []
        
        # Row 1: Enable VAD, Speech Threshold, Min Speech Duration
        with gr.Row():
            inputs.append(gr.Checkbox(
                label=_("Enable Silero VAD Filter"),
                value=defaults.get("vad_filter", cls.__fields__["vad_filter"].default),
                interactive=True,
                info="Voice Activity Detection removes silence before transcription. Recommended for cleaner output and faster processing."
            ))
            inputs.append(gr.Slider(
                minimum=0.0, maximum=1.0, step=0.01, label="Speech Threshold",
                value=defaults.get("threshold", cls.__fields__["threshold"].default),
                info="Probability threshold for detecting speech versus silence. Lower values detect quieter speech."
            ))
            inputs.append(gr.Number(
                label="Minimum Speech Duration (ms)", precision=0,
                value=defaults.get("min_speech_duration_ms", cls.__fields__["min_speech_duration_ms"].default),
                info="Discard speech chunks shorter than this duration."
            ))
        
        # Row 2: Max Speech Duration, Min Silence Duration, Speech Padding
        with gr.Row():
            inputs.append(gr.Number(
                label="Maximum Speech Duration (s)",
                value=defaults.get("max_speech_duration_s", GRADIO_NONE_NUMBER_MAX),
                info="Maximum length of continuous speech chunks before forcing a split."
            ))
            inputs.append(gr.Number(
                label="Minimum Silence Duration (ms)", precision=0,
                value=defaults.get("min_silence_duration_ms", cls.__fields__["min_silence_duration_ms"].default),
                info="Silence duration required to split speech chunks."
            ))
            inputs.append(gr.Number(
                label="Speech Padding (ms)", precision=0,
                value=defaults.get("speech_pad_ms", cls.__fields__["speech_pad_ms"].default),
                info="Padding added before and after detected speech so words are not cut off."
            ))
        
        return [repair_component_text(component) for component in inputs]


class DiarizationParams(BaseParams):
    """Speaker diarization parameters"""
    is_diarize: bool = Field(default=False, description="Enable speaker diarization")
    diarization_device: str = Field(default="cuda", description="Device to run Diarization model.")
    hf_token: str = Field(
        default="",
        description="Optional Hugging Face token fallback for diarization"
    )
    enable_offload: bool = Field(
        default=True,
        description="Offload Diarization model after Speaker diarization"
    )

    @classmethod
    def to_gradio_inputs(cls,
                         defaults: Optional[Dict] = None,
                         available_devices: Optional[List] = None,
                         device: Optional[str] = None) -> List[gr.components.base.FormComponent]:
        inputs = []
        
        # Row 1: Enable Diarization, Device, hidden token fallback
        with gr.Row():
            inputs.append(gr.Checkbox(
                label=_("Enable Diarization"),
                value=defaults.get("is_diarize", cls.__fields__["is_diarize"].default),
                info="Speaker diarization identifies who is speaking and adds speaker labels to the transcription."
            ))
            inputs.append(gr.Dropdown(
                label=_("Device"),
                choices=["cpu", "cuda", "xpu"] if available_devices is None else available_devices,
                value=defaults.get("diarization_device", defaults.get("device", device)),
                info="Device for the diarization model. Use CUDA when available for best speed."
            ))
            inputs.append(gr.Textbox(
                label=_("HuggingFace Token"),
                value=defaults.get("hf_token", cls.__fields__["hf_token"].default),
                info="Leave blank. The offline diarization bundle is used automatically.",
                visible=False
            ))
        
        # Row 2: Offload model (single item, but in a row for consistency)
        with gr.Row():
            inputs.append(gr.Checkbox(
                label=_("Offload sub model when finished"),
                value=defaults.get("enable_offload", cls.__fields__["enable_offload"].default),
                info="Unload the diarization model from VRAM after use."
            ))
        
        return [repair_component_text(component) for component in inputs]


class BGMSeparationParams(BaseParams):
    """Background music separation parameters"""
    is_separate_bgm: bool = Field(default=False, description="Enable background music separation")
    uvr_model_size: str = Field(
        default="UVR-MDX-NET-Inst_HQ_4",
        description="UVR model size"
    )
    uvr_device: str = Field(default="cuda", description="Device to run UVR model.")
    segment_size: int = Field(
        default=256,
        gt=0,
        description="Segment size for UVR model"
    )
    save_file: bool = Field(
        default=False,
        description="Whether to save separated audio files"
    )
    enable_offload: bool = Field(
        default=True,
        description="Offload UVR model after transcription"
    )

    @classmethod
    def to_gradio_input(cls,
                        defaults: Optional[Dict] = None,
                        available_devices: Optional[List] = None,
                        device: Optional[str] = None,
                        available_models: Optional[List] = None) -> List[gr.components.base.FormComponent]:
        inputs = []
        
        # Row 1: Enable BGM, Model, Device
        with gr.Row():
            inputs.append(gr.Checkbox(
                label=_("Enable Background Music Remover Filter"),
                value=defaults.get("is_separate_bgm", cls.__fields__["is_separate_bgm"].default),
                interactive=True,
                info="Remove background music or noise before transcription using UVR."
            ))
            inputs.append(gr.Dropdown(
                label=_("Model"),
                choices=["UVR-MDX-NET-Inst_HQ_4",
                         "UVR-MDX-NET-Inst_3"] if available_models is None else available_models,
                value=defaults.get("uvr_model_size", cls.__fields__["uvr_model_size"].default),
                info="UVR model quality preset. Higher quality is usually slower and uses more VRAM."
            ))
            inputs.append(gr.Dropdown(
                label=_("Device"),
                choices=["cpu", "cuda", "xpu"] if available_devices is None else available_devices,
                value=defaults.get("uvr_device", defaults.get("device", device)),
                info="Device for background music separation. Use CUDA when available."
            ))
        
        # Row 2: Segment Size, Save Files, Offload Model
        with gr.Row():
            inputs.append(gr.Number(
                label="Segment Size",
                value=defaults.get("segment_size", cls.__fields__["segment_size"].default),
                precision=0,
                info="Processing chunk size affects quality, speed, and VRAM usage."
            ))
            inputs.append(gr.Checkbox(
                label=_("Save separated files to output"),
                value=defaults.get("save_file", cls.__fields__["save_file"].default),
                info="Save separated vocal and instrumental files to outputs/UVR."
            ))
            inputs.append(gr.Checkbox(
                label=_("Offload sub model when finished"),
                value=defaults.get("enable_offload", cls.__fields__["enable_offload"].default),
                info="Unload the UVR model from VRAM after separation."
            ))
        
        return [repair_component_text(component) for component in inputs]


class WhisperParams(BaseParams):
    """Whisper parameters"""
    model_size: str = Field(default="large-v2", description="Whisper model size")
    lang: Optional[str] = Field(default=None, description="Source language of the file to transcribe")
    is_translate: bool = Field(default=False, description="Translate speech to English end-to-end")
    beam_size: int = Field(default=8, ge=1, description="Beam size for decoding")
    log_prob_threshold: float = Field(
        default=-1.0,
        description="Threshold for average log probability of sampled tokens"
    )
    no_speech_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold for detecting silence"
    )
    compute_type: str = Field(default="bfloat16", description="Computation type for transcription")
    best_of: int = Field(default=5, ge=1, description="Number of candidates when sampling")
    patience: float = Field(default=1.5, gt=0, description="Beam search patience factor")
    condition_on_previous_text: bool = Field(
        default=False,
        description="Use previous output as prompt for the next window."
    )
    start_as_subprocess: bool = Field(
        default=True,
        description="Run transcription in a dedicated subprocess so model memory is fully released when the job ends."
    )
    prompt_reset_on_temperature: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Temperature threshold for resetting prompt"
    )
    initial_prompt: Optional[str] = Field(default=None, description="Initial prompt for first window")
    repeat_initial_prompt_every_window: bool = Field(
        default=False,
        description="Reinject the initial prompt before every faster-whisper window"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        description="Temperature for sampling"
    )
    compression_ratio_threshold: float = Field(
        default=2.4,
        gt=0,
        description="Threshold for gzip compression ratio"
    )
    length_penalty: float = Field(default=1.0, gt=0, description="Exponential length penalty")
    repetition_penalty: float = Field(default=1.0, gt=0, description="Penalty for repeated tokens")
    no_repeat_ngram_size: int = Field(default=0, ge=0, description="Size of n-grams to prevent repetition")
    prefix: Optional[str] = Field(default=None, description="Prefix text for first window")
    suppress_blank: bool = Field(
        default=True,
        description="Suppress blank outputs at start of sampling"
    )
    suppress_tokens: Optional[Union[List[int], str]] = Field(default=[-1], description="Token IDs to suppress")
    max_initial_timestamp: float = Field(
        default=1.0,
        ge=0.0,
        description="Maximum initial timestamp"
    )
    word_timestamps: bool = Field(default=True, description="Extract word-level timestamps")
    prepend_punctuations: Optional[str] = Field(
        default="\"'([{-",
        description="Punctuations to merge with next word"
    )
    append_punctuations: Optional[str] = Field(
        default="\"'.,!?:)]}",
        description="Punctuations to merge with previous word"
    )
    max_new_tokens: Optional[int] = Field(default=None, description="Maximum number of new tokens per chunk")
    chunk_length: Optional[int] = Field(default=10, description="Length of audio segments in seconds")
    hallucination_silence_threshold: Optional[float] = Field(
        default=None,
        description="Threshold for skipping silent periods in hallucination detection"
    )
    hotwords: Optional[str] = Field(default=None, description="Hotwords/hint phrases for the model")
    language_detection_threshold: Optional[float] = Field(
        default=0.5,
        description="Threshold for language detection probability"
    )
    language_detection_segments: int = Field(
        default=1,
        gt=0,
        description="Number of segments for language detection"
    )
    use_batched_inference: bool = Field(
        default=False,
        description="Use the faster-whisper batched inference pipeline for higher throughput at the cost of accuracy"
    )
    batch_size: int = Field(default=8, gt=0, description="Batch size for processing")
    enable_offload: bool = Field(
        default=True,
        description="Offload Whisper model after transcription"
    )
    canary_generation_kwargs: Optional[str] = Field(
        default=None,
        description="Canary-Qwen raw generation keyword arguments as a JSON object"
    )
    canary_enable_thinking: bool = Field(
        default=False,
        description="Forward Canary-Qwen enable_thinking to NeMo SALM generation"
    )
    whisper_type: str = Field(
        default=WhisperImpl.FASTER_WHISPER.value,
        description="Primary transcription model family"
    )

    @staticmethod
    def normalize_lang_value(v):
        from modules.utils.constants import AUTOMATIC_DETECTION

        if isinstance(v, str) and v.strip().casefold() == AUTOMATIC_DETECTION.unwrap().casefold():
            return None
        return normalize_whisper_lang_value(v)

    @staticmethod
    def normalize_lang_choice(v):
        from modules.utils.constants import AUTOMATIC_DETECTION

        return normalize_whisper_lang_choice(v, AUTOMATIC_DETECTION.unwrap())

    @staticmethod
    def get_language_choices(available_langs: Optional[List]) -> List:
        from modules.utils.constants import AUTOMATIC_DETECTION

        choices = list(available_langs or [])
        auto = AUTOMATIC_DETECTION.unwrap()
        return [auto, *choices] if auto not in choices else choices

    @field_validator('lang')
    def validate_lang(cls, v):
        return cls.normalize_lang_value(v)

    @field_validator('suppress_tokens')
    def validate_supress_tokens(cls, v):
        import ast
        try:
            if isinstance(v, str):
                suppress_tokens = ast.literal_eval(v)
                if not isinstance(suppress_tokens, list):
                    raise ValueError("Invalid Suppress Tokens. The value must be type of List[int]")
                return suppress_tokens
            if isinstance(v, list):
                return v
        except Exception as e:
            raise ValueError(f"Invalid Suppress Tokens. The value must be type of List[int]: {e}")

    @field_validator('whisper_type')
    def validate_whisper_type(cls, v):
        if not isinstance(v, str) or not v.strip():
            return WhisperImpl.FASTER_WHISPER.value

        normalized = v.strip().lower()
        aliases = {
            "whisper": WhisperImpl.FASTER_WHISPER.value,
            "faster_whisper": WhisperImpl.FASTER_WHISPER.value,
            "faster-whisper": WhisperImpl.FASTER_WHISPER.value,
            "canary": WhisperImpl.CANARY_QWEN.value,
            "canary-qwen": WhisperImpl.CANARY_QWEN.value,
            "canary_qwen": WhisperImpl.CANARY_QWEN.value,
            "nvidia/canary-qwen-2.5b": WhisperImpl.CANARY_QWEN.value,
        }
        return aliases.get(normalized, normalized)

    @classmethod
    def advanced_input_field_names(cls) -> List[str]:
        return [
            field_name for field_name in cls.model_fields.keys()
            if field_name not in {"model_size", "lang", "is_translate", "whisper_type"}
        ]

    @staticmethod
    def supports_batch_size(whisper_type: Optional[str] = None) -> bool:
        whisper_type = WhisperImpl.FASTER_WHISPER.value if whisper_type is None else whisper_type.strip().lower()
        return whisper_type in {
            WhisperImpl.FASTER_WHISPER.value,
            WhisperImpl.INSANELY_FAST_WHISPER.value,
            WhisperImpl.CANARY_QWEN.value,
        }

    @classmethod
    def to_batch_size_input(cls,
                            defaults: Optional[Dict] = None,
                            whisper_type: Optional[str] = None):
        defaults = defaults or {}
        batch_size_value = int(defaults.get("batch_size", cls.__fields__["batch_size"].default))

        if whisper_type == WhisperImpl.CANARY_QWEN.value:
            batch_info = (
                "Number of Canary-Qwen audio chunks to send to NeMo SALM generation at once. "
                "Higher values can improve throughput on large GPUs but use more VRAM."
            )
        else:
            batch_info = (
                "When Use Batched Inference is disabled, this controls the standard faster-whisper encoder "
                "prefetch batch size on a single model instance. Higher values can improve throughput with much "
                "lower quality risk than the batched decoder path, but the gain depends on your audio and GPU. "
                "When Use Batched Inference is enabled, it controls the faster-whisper batched decoder path instead."
            )

        batch_size_input = gr.Slider(
            minimum=1,
            maximum=max(32, batch_size_value),
            step=1,
            label="Batch Size",
            value=batch_size_value,
            info=batch_info,
        )

        if not cls.supports_batch_size(whisper_type):
            batch_size_input.visible = False

        return batch_size_input

    @classmethod
    def to_start_as_subprocess_input(cls, defaults: Optional[Dict] = None):
        defaults = defaults or {}

        return gr.Checkbox(
            label="Start As Sub Process",
            value=defaults.get(
                "start_as_subprocess",
                cls.__fields__["start_as_subprocess"].default,
            ),
            info=(
                "Recommended ON. Runs the job in a dedicated subprocess so torch, VRAM, and RAM are fully released "
                "when the job finishes, and allows hard cancellation."
            ),
        )

    @classmethod
    def to_batched_inference_input(cls,
                                   defaults: Optional[Dict] = None,
                                   whisper_type: Optional[str] = None):
        whisper_type = WhisperImpl.FASTER_WHISPER.value if whisper_type is None else whisper_type.strip().lower()
        defaults = defaults or {}

        batched_inference_input = gr.Checkbox(
            label="Use Batched Inference",
            value=defaults.get("use_batched_inference", cls.__fields__["use_batched_inference"].default),
            info=(
                "Speed-first faster-whisper path. It can process more chunks in parallel, but on long-form speech "
                "it is noticeably less accurate and more prone to all-caps output, repetition, and subtitle drift. "
                "Leave disabled for best subtitle quality."
            ),
        )

        if whisper_type != WhisperImpl.FASTER_WHISPER.value:
            batched_inference_input.visible = False

        return batched_inference_input

    @classmethod
    def to_condition_on_previous_text_input(cls, defaults: Optional[Dict] = None):
        defaults = defaults or {}

        return gr.Checkbox(
            label="Condition On Previous Text",
            value=defaults.get(
                "condition_on_previous_text",
                cls.__fields__["condition_on_previous_text"].default,
            ),
            info=(
                "Use previous transcription as context for the next segment. "
                "Usually helps coherence across chunks. Important: if you see repetition, all-caps output, "
                "or subtitle drift, try disabling this. In rare cases that can significantly improve quality."
            ),
        )

    @classmethod
    def to_gradio_inputs(cls,
                         defaults: Optional[Dict] = None,
                         only_advanced: Optional[bool] = True,
                         whisper_type: Optional[str] = None,
                         available_models: Optional[List] = None,
                         available_langs: Optional[List] = None,
                         available_compute_types: Optional[List] = None,
                         compute_type: Optional[str] = None,
                         use_3col_layout: bool = False,
                         include_batch_size: bool = True,
                         include_condition_on_previous_text: bool = True):
        whisper_type = WhisperImpl.FASTER_WHISPER.value if whisper_type is None else whisper_type.strip().lower()

        inputs = []
        if not only_advanced:
            language_choices = cls.get_language_choices(available_langs)
            inputs += [
                gr.Dropdown(
                    label=_("Model"),
                    choices=available_models,
                    value=defaults.get("model_size", cls.__fields__["model_size"].default),
                ),
                gr.Dropdown(
                    label=_("Language"),
                    choices=language_choices,
                    value=cls.normalize_lang_choice(defaults.get("lang")),
                ),
                gr.Checkbox(
                    label=_("Translate to English?"),
                    value=defaults.get("is_translate", cls.__fields__["is_translate"].default),
                    info=(
                        "When enabled, Whisper outputs English text directly from the speech. "
                        "When disabled, subtitles stay in the original spoken language. "
                        "This uses Whisper's built-in translation, not DeepL or NLLB."
                    ),
                ),
            ]

        # Row 1: Beam Size, Log Probability Threshold, No Speech Threshold
        with gr.Row():
            inputs.append(gr.Number(
                label="Beam Size",
                value=defaults.get("beam_size", cls.__fields__["beam_size"].default),
                precision=0,
                info="Number of beams in beam search. Higher values are usually more accurate but slower. Range: 1-20."
            ))
            inputs.append(gr.Number(
                label="Log Probability Threshold",
                value=defaults.get("log_prob_threshold", cls.__fields__["log_prob_threshold"].default),
                info="Reject segments with average log probability below this value. Lower values are stricter."
            ))
            inputs.append(gr.Number(
                label="No Speech Threshold",
                value=defaults.get("no_speech_threshold", cls.__fields__["no_speech_threshold"].default),
                info="Probability threshold for detecting silence or no-speech. Range: 0.0-1.0."
            ))
        
        # Row 2: Compute Type, Best Of, Patience
        with gr.Row():
            inputs.append(gr.Dropdown(
                label="Compute Type",
                choices=["bfloat16", "float16", "float32", "int8"] if available_compute_types is None else available_compute_types,
                value=defaults.get("compute_type", compute_type),
                info="Precision for model computation. Use the setting that best fits your device speed, stability, and VRAM budget."
            ))
            inputs.append(gr.Number(
                label="Best Of",
                value=defaults.get("best_of", cls.__fields__["best_of"].default),
                precision=0,
                info="Number of candidate sequences to generate when sampling. Higher values can improve quality but are slower."
            ))
            inputs.append(gr.Number(
                label="Patience",
                value=defaults.get("patience", cls.__fields__["patience"].default),
                info="Beam search patience controls how long to wait for better candidates. Higher values search more thoroughly."
            ))
        
        # Row 3: Condition On Previous Text, Prompt Reset On Temperature, Initial Prompt
        with gr.Row():
            inputs.append(gr.Checkbox(
                label="Condition On Previous Text",
                value=defaults.get("condition_on_previous_text", cls.__fields__["condition_on_previous_text"].default),
                info="Use the previous transcription as context for the next segment. Disable it if the model gets stuck in repetition."
            ))
            inputs.append(cls.to_start_as_subprocess_input(defaults=defaults))
            inputs.append(gr.Slider(
                label="Prompt Reset On Temperature",
                value=defaults.get("prompt_reset_on_temperature",
                                   cls.__fields__["prompt_reset_on_temperature"].default),
                minimum=0,
                maximum=1,
                step=0.01,
                info="Reset the conditioning prompt if temperature exceeds this value. Range: 0.0-1.0."
            ))
            inputs.append(gr.Textbox(
                label="Initial Prompt",
                value=defaults.get("initial_prompt", GRADIO_NONE_STR),
                info="Text that guides transcription style or vocabulary. Useful for domain-specific terms. Leave empty for general transcription."
            ))
            inputs.append(gr.Checkbox(
                label="Repeat Initial Prompt Every Window",
                value=defaults.get(
                    "repeat_initial_prompt_every_window",
                    cls.__fields__["repeat_initial_prompt_every_window"].default
                ),
                visible=whisper_type == WhisperImpl.FASTER_WHISPER.value,
                info=(
                    "Reinject the Initial Prompt before every faster-whisper window. "
                    "Useful for forcing names, terminology, or style across long audio, "
                    "but it can over-bias the output if the prompt is too specific."
                )
            ))

        # Row 4: Temperature, Compression Ratio Threshold, Length Penalty
        with gr.Row():
            inputs.append(gr.Slider(
                label="Temperature",
                value=defaults.get("temperature", cls.__fields__["temperature"].default),
                minimum=0.0,
                step=0.01,
                maximum=1.0,
                info="Randomness in decoding. Lower values are more deterministic and accurate."
            ))
            inputs.append(gr.Number(
                label="Compression Ratio Threshold",
                value=defaults.get("compression_ratio_threshold",
                                   cls.__fields__["compression_ratio_threshold"].default),
                info="Detect repetitive or hallucinated text by gzip compression ratio. Lower values are stricter."
            ))
            inputs.append(gr.Number(
                label="Length Penalty",
                value=defaults.get("length_penalty", cls.__fields__["length_penalty"].default),
                info="Penalty for longer sequences. Values above 1.0 favor longer outputs, below 1.0 favor shorter outputs, and 1.0 is neutral."
            ))
        

        faster_whisper_inputs = []
        
        # Row 5: Repetition Penalty, No Repeat N-gram Size, Prefix
        with gr.Row():
            faster_whisper_inputs.append(gr.Number(
                label="Repetition Penalty",
                value=defaults.get("repetition_penalty", cls.__fields__["repetition_penalty"].default),
                info="Penalty applied to repeated tokens. Increase it if you see repeated phrases."
            ))
            faster_whisper_inputs.append(gr.Number(
                label="No Repeat N-gram Size",
                value=defaults.get("no_repeat_ngram_size", cls.__fields__["no_repeat_ngram_size"].default),
                precision=0,
                info="Block exact repetition of N-word phrases. Use small values such as 3-5 to reduce stuttering."
            ))
            faster_whisper_inputs.append(gr.Textbox(
                label="Prefix",
                value=defaults.get("prefix", GRADIO_NONE_STR),
                info="Text to prepend to every segment, for example a speaker name."
            ))
        
        # Row 6: Suppress Blank, Suppress Tokens, Max Initial Timestamp
        with gr.Row():
            faster_whisper_inputs.append(gr.Checkbox(
                label="Suppress Blank",
                value=defaults.get("suppress_blank", cls.__fields__["suppress_blank"].default),
                info="Suppress blank or empty outputs at the start of sampling."
            ))
            faster_whisper_inputs.append(gr.Textbox(
                label="Suppress Tokens",
                value=defaults.get("suppress_tokens", "[-1]"),
                info="Token IDs that should never be generated."
            ))
            faster_whisper_inputs.append(gr.Number(
                label="Max Initial Timestamp",
                value=defaults.get("max_initial_timestamp", cls.__fields__["max_initial_timestamp"].default),
                info="Maximum allowed initial timestamp in seconds."
            ))
        
        # Row 7: Word Timestamps, Prepend Punctuations, Append Punctuations
        with gr.Row():
            faster_whisper_inputs.append(gr.Checkbox(
                label="Word Timestamps",
                value=defaults.get("word_timestamps", cls.__fields__["word_timestamps"].default),
                info="Extract timestamps for each individual word, not just each segment."
            ))
            faster_whisper_inputs.append(gr.Textbox(
                label="Prepend Punctuations",
                value=defaults.get("prepend_punctuations", cls.__fields__["prepend_punctuations"].default),
                info="Punctuation marks to attach to the next word. Default: \"'([{-."
            ))
            faster_whisper_inputs.append(gr.Textbox(
                label="Append Punctuations",
                value=defaults.get("append_punctuations", cls.__fields__["append_punctuations"].default),
                info="Punctuation marks to attach to the previous word. Default: \"'.,!?:)]}."
            ))
        
        # Row 8: Max New Tokens, Chunk Length, Hallucination Silence Threshold
        with gr.Row():
            faster_whisper_inputs.append(gr.Number(
                label="Max New Tokens",
                value=defaults.get("max_new_tokens", GRADIO_NONE_NUMBER_MIN),
                precision=0,
                info="Maximum tokens per chunk. Leave empty for automatic behavior."
            ))
            faster_whisper_inputs.append(gr.Number(
                label="Chunk Length (s)",
                value=defaults.get("chunk_length", cls.__fields__["chunk_length"].default),
                info="Length of each audio window in seconds."
            ))
            faster_whisper_inputs.append(gr.Number(
                label="Hallucination Silence Threshold (sec)",
                value=defaults.get("hallucination_silence_threshold",
                                   GRADIO_NONE_NUMBER_MIN),
                info="Skip silent periods longer than this when detecting hallucinations."
            ))
        
        # Row 9: Hotwords, Language Detection Threshold, Language Detection Segments
        with gr.Row():
            faster_whisper_inputs.append(gr.Textbox(
                label="Hotwords",
                value=defaults.get("hotwords", cls.__fields__["hotwords"].default),
                info="Boost recognition of specific words or phrases. Leave empty for general transcription."
            ))
            faster_whisper_inputs.append(gr.Number(
                label="Language Detection Threshold",
                value=defaults.get("language_detection_threshold",
                                   GRADIO_NONE_NUMBER_MIN),
                info="Confidence threshold for language detection."
            ))
            faster_whisper_inputs.append(gr.Number(
                label="Language Detection Segments",
                value=defaults.get("language_detection_segments",
                                   cls.__fields__["language_detection_segments"].default),
                precision=0,
                info="Number of audio segments to analyze for language detection."
            ))
        

        if whisper_type == WhisperImpl.CANARY_QWEN.value:
            canary_visible_fields = {
                "repetition_penalty",
                "no_repeat_ngram_size",
                "max_new_tokens",
                "chunk_length",
            }
            faster_whisper_field_names = [
                "repetition_penalty",
                "no_repeat_ngram_size",
                "prefix",
                "suppress_blank",
                "suppress_tokens",
                "max_initial_timestamp",
                "word_timestamps",
                "prepend_punctuations",
                "append_punctuations",
                "max_new_tokens",
                "chunk_length",
                "hallucination_silence_threshold",
                "hotwords",
                "language_detection_threshold",
                "language_detection_segments",
            ]
            for field_name, input_component in zip(faster_whisper_field_names, faster_whisper_inputs):
                input_component.visible = field_name in canary_visible_fields
            faster_whisper_inputs[9].info = "Maximum generated text tokens per Canary-Qwen chunk."
            faster_whisper_inputs[10].info = "Canary-Qwen ASR window size in seconds. Values above 40 are capped."
        elif whisper_type != WhisperImpl.FASTER_WHISPER.value:
            for input_component in faster_whisper_inputs:
                input_component.visible = False

        inputs += faster_whisper_inputs

        if include_batch_size:
            # Keep these after the faster-whisper-only fields so the UI value order
            # matches WhisperParams field order during Gradio list -> model conversion.
            with gr.Row():
                inputs.append(cls.to_batched_inference_input(defaults=defaults, whisper_type=whisper_type))
                inputs.append(cls.to_batch_size_input(defaults=defaults, whisper_type=whisper_type))

        # Final row: Offload model
        with gr.Row():
            inputs.append(gr.Checkbox(
                label=_("Offload sub model when finished"),
                value=defaults.get("enable_offload", cls.__fields__["enable_offload"].default),
                info="Unload the model from VRAM after transcription."
            ))

        with gr.Row():
            inputs.append(gr.Textbox(
                label="Canary Generation Kwargs (JSON)",
                value=defaults.get("canary_generation_kwargs", GRADIO_NONE_STR),
                visible=whisper_type == WhisperImpl.CANARY_QWEN.value,
                lines=3,
                info=(
                    "Advanced Canary-Qwen/Qwen generation overrides as JSON. "
                    "Example: {\"top_p\": 0.9, \"top_k\": 50, \"do_sample\": true}."
                )
            ))
            inputs.append(gr.Checkbox(
                label="Canary Enable Thinking",
                value=defaults.get("canary_enable_thinking", cls.__fields__["canary_enable_thinking"].default),
                visible=whisper_type == WhisperImpl.CANARY_QWEN.value,
                info="Forward enable_thinking to NeMo SALM. Leave disabled for normal ASR transcription."
            ))

        if whisper_type == WhisperImpl.CANARY_QWEN.value and only_advanced:
            canary_advanced_fields = {
                "beam_size",
                "compute_type",
                "start_as_subprocess",
                "temperature",
                "length_penalty",
                "repetition_penalty",
                "no_repeat_ngram_size",
                "max_new_tokens",
                "chunk_length",
                "batch_size",
                "enable_offload",
                "canary_generation_kwargs",
                "canary_enable_thinking",
            }
            for field_name, input_component in zip(cls.advanced_input_field_names(), inputs):
                input_component.visible = field_name in canary_advanced_fields

        return [repair_component_text(component) for component in inputs]
    
    @classmethod
    def to_gradio_inputs_3col(cls,
                              defaults: Optional[Dict] = None,
                              only_advanced: Optional[bool] = True,
                              whisper_type: Optional[str] = None,
                              available_models: Optional[List] = None,
                              available_langs: Optional[List] = None,
                              available_compute_types: Optional[List] = None,
                              compute_type: Optional[str] = None):
        """Same as to_gradio_inputs but creates them in 3-column layout"""
        whisper_type = WhisperImpl.FASTER_WHISPER.value if whisper_type is None else whisper_type.strip().lower()
        
        all_inputs = []
        input_configs = []
        
        # Define all input configurations
        if not only_advanced:
            language_choices = cls.get_language_choices(available_langs)
            input_configs.extend([
                ("dropdown", "Model", available_models, defaults.get("model_size", cls.__fields__["model_size"].default)),
                ("dropdown", "Language", language_choices, cls.normalize_lang_choice(defaults.get("lang"))),
                ("checkbox", "Translate to English?", None, defaults.get("is_translate", cls.__fields__["is_translate"].default)),
            ])
        
        # Common inputs
        input_configs.extend([
            ("number", "Beam Size", 0, defaults.get("beam_size", cls.__fields__["beam_size"].default), 
             "Number of beams in beam search. Higher values are usually more accurate but slower. Range: 1-20."),
            ("number", "Log Probability Threshold", None, defaults.get("log_prob_threshold", cls.__fields__["log_prob_threshold"].default),
             "Reject segments with average log probability below this value. Lower values are stricter."),
            ("number", "No Speech Threshold", None, defaults.get("no_speech_threshold", cls.__fields__["no_speech_threshold"].default),
             "Probability threshold for detecting silence or no-speech. Range: 0.0-1.0."),
        ])
        
        # Create inputs in rows of 3
        for i in range(0, len(input_configs), 3):
            with gr.Row():
                for j in range(3):
                    if i + j < len(input_configs):
                        config = input_configs[i + j]
                        if config[0] == "number":
                            comp = gr.Number(
                                label=config[1],
                                value=config[3],
                                precision=config[2] if config[2] is not None else None,
                                info=config[4] if len(config) > 4 else ""
                            )
                        all_inputs.append(comp)
        
        # For now, fall back to original method to avoid breaking
        # This is a temporary solution
        return cls.to_gradio_inputs(defaults, only_advanced, whisper_type, available_models, 
                                    available_langs, available_compute_types, compute_type)


class TranscriptionPipelineParams(BaseModel):
    """Transcription pipeline parameters"""
    whisper: WhisperParams = Field(default_factory=WhisperParams)
    vad: VadParams = Field(default_factory=VadParams)
    diarization: DiarizationParams = Field(default_factory=DiarizationParams)
    bgm_separation: BGMSeparationParams = Field(default_factory=BGMSeparationParams)

    def to_dict(self) -> Dict:
        data = {
            "whisper": self.whisper.to_dict(),
            "vad": self.vad.to_dict(),
            "diarization": self.diarization.to_dict(),
            "bgm_separation": self.bgm_separation.to_dict()
        }
        return data

    def to_list(self) -> List:
        """
        Convert data class to the list because I have to pass the parameters as a list in the gradio.
        Related Gradio issue: https://github.com/gradio-app/gradio/issues/2471
        See more about Gradio pre-processing: https://www.gradio.app/docs/components
        """
        whisper_list = self.whisper.to_list()
        vad_list = self.vad.to_list()
        diarization_list = self.diarization.to_list()
        bgm_sep_list = self.bgm_separation.to_list()
        return whisper_list + vad_list + diarization_list + bgm_sep_list

    @staticmethod
    def from_list(pipeline_list: List) -> 'TranscriptionPipelineParams':
        """Convert list to the data class again to use it in a function."""
        data_list = deepcopy(pipeline_list)

        whisper_list = data_list[0:len(WhisperParams.__annotations__)]
        data_list = data_list[len(WhisperParams.__annotations__):]

        vad_list = data_list[0:len(VadParams.__annotations__)]
        data_list = data_list[len(VadParams.__annotations__):]

        diarization_list = data_list[0:len(DiarizationParams.__annotations__)]
        data_list = data_list[len(DiarizationParams.__annotations__):]

        bgm_sep_list = data_list[0:len(BGMSeparationParams.__annotations__)]

        return TranscriptionPipelineParams(
            whisper=WhisperParams.from_list(whisper_list),
            vad=VadParams.from_list(vad_list),
            diarization=DiarizationParams.from_list(diarization_list),
            bgm_separation=BGMSeparationParams.from_list(bgm_sep_list)
        )
