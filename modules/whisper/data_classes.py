import faster_whisper.transcribe
import gradio as gr
import torch
from typing import Optional, Dict, List, Union, NamedTuple
from fastapi import Query
from pydantic import BaseModel, Field, field_validator, ConfigDict
from gradio_i18n import Translate, gettext as _
from enum import Enum
from copy import deepcopy
import yaml

from modules.utils.constants import *


class WhisperImpl(Enum):
    WHISPER = "whisper"
    FASTER_WHISPER = "faster-whisper"
    INSANELY_FAST_WHISPER = "insanely_fast_whisper"


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
                            seg: faster_whisper.transcribe.Segment):
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
                info="🎙️ Voice Activity Detection - removes silence before transcription. ✅ ENABLED (current) for cleaner output and faster processing. Highly recommended! Only disable if audio already pre-processed or you need exact timing including silences."
            ))
            inputs.append(gr.Slider(
                minimum=0.0, maximum=1.0, step=0.01, label="Speech Threshold",
                value=defaults.get("threshold", cls.__fields__["threshold"].default),
                info="🔊 Probability threshold for detecting speech vs silence. Range: 0.0-1.0. Examples: 0.5 (balanced - current), 0.3 (sensitive - detects quiet speech/background talk), 0.7 (strict - only clear speech). Lower for noisy audio."
            ))
            inputs.append(gr.Number(
                label="Minimum Speech Duration (ms)", precision=0,
                value=defaults.get("min_speech_duration_ms", cls.__fields__["min_speech_duration_ms"].default),
                info="⏱️ Discard speech chunks shorter than this. Examples: 250ms (current - filters out very short sounds), 100ms (keep short utterances), 500ms (only longer speech). Increase to filter out quick noises/coughs."
            ))
        
        # Row 2: Max Speech Duration, Min Silence Duration, Speech Padding
        with gr.Row():
            inputs.append(gr.Number(
                label="Maximum Speech Duration (s)",
                value=defaults.get("max_speech_duration_s", GRADIO_NONE_NUMBER_MAX),
                info="⏳ Maximum length of continuous speech chunks. Examples: 9999s (unlimited - current), 30s (split long monologues), 60s (for very long speeches). Use unlimited unless you need to force splitting long continuous speech."
            ))
            inputs.append(gr.Number(
                label="Minimum Silence Duration (ms)", precision=0,
                value=defaults.get("min_silence_duration_ms", cls.__fields__["min_silence_duration_ms"].default),
                info="🔇 Silence duration required to split speech chunks. Examples: 1000ms/1s (current - splits on 1s+ pauses), 500ms (split on shorter pauses), 2000ms (only split on long pauses). Affects segment boundaries."
            ))
            inputs.append(gr.Number(
                label="Speech Padding (ms)", precision=0,
                value=defaults.get("speech_pad_ms", cls.__fields__["speech_pad_ms"].default),
                info="📏 Add padding before/after detected speech. Examples: 2000ms/2s (current - includes 2s before/after), 1000ms (tighter boundaries), 3000ms (more context). Prevents cutting off start/end of words."
            ))
        
        return inputs


class DiarizationParams(BaseParams):
    """Speaker diarization parameters"""
    is_diarize: bool = Field(default=False, description="Enable speaker diarization")
    diarization_device: str = Field(default="cuda", description="Device to run Diarization model.")
    hf_token: str = Field(
        default="",
        description="Hugging Face token for downloading diarization models"
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
        
        # Row 1: Enable Diarization, Device, HuggingFace Token
        with gr.Row():
            inputs.append(gr.Checkbox(
                label=_("Enable Diarization"),
                value=defaults.get("is_diarize", cls.__fields__["is_diarize"].default),
                info="👥 Speaker Diarization - identifies WHO is speaking. Adds speaker labels like '[SPEAKER_00]' to transcription. ❌ Disabled by default. Enable for multi-speaker conversations, interviews, meetings. Requires HuggingFace token (free). Adds ~30% processing time."
            ))
            inputs.append(gr.Dropdown(
                label=_("Device"),
                choices=["cpu", "cuda", "xpu"] if available_devices is None else available_devices,
                value=defaults.get("device", device),
                info="🖥️ Device for diarization model. cuda (GPU - fast), cpu (slow but works everywhere), xpu (Intel GPU). Use cuda if available. Diarization is compute-intensive!"
            ))
            inputs.append(gr.Textbox(
                label=_("HuggingFace Token"),
                value=defaults.get("hf_token", cls.__fields__["hf_token"].default),
                info="🔑 Free token from huggingface.co/settings/tokens. Required ONLY for first-time model download. Accept terms at: huggingface.co/pyannote/speaker-diarization-3.1 and huggingface.co/pyannote/segmentation-3.0. Leave empty after first download."
            ))
        
        # Row 2: Offload model (single item, but in a row for consistency)
        with gr.Row():
            inputs.append(gr.Checkbox(
                label=_("Offload sub model when finished"),
                value=defaults.get("enable_offload", cls.__fields__["enable_offload"].default),
                info="💾 Unload diarization model from VRAM after use. ❌ DISABLED (current) for unlimited compute. ✅ Enable if VRAM limited. Diarization model uses ~2-3GB VRAM."
            ))
        
        return inputs


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
                info="🎵 Remove background music/noise before transcription using UVR (Ultimate Vocal Remover). ❌ Disabled by default. ✅ Enable for: music videos, noisy recordings, podcasts with intro music. Improves accuracy ~10-20% for noisy audio. Adds processing time."
            ))
            inputs.append(gr.Dropdown(
                label=_("Model"),
                choices=["UVR-MDX-NET-Inst_HQ_4",
                         "UVR-MDX-NET-Inst_3"] if available_models is None else available_models,
                value=defaults.get("uvr_model_size", cls.__fields__["uvr_model_size"].default),
                info="🎼 UVR model quality. UVR-MDX-NET-Inst_HQ_4 (current - highest quality, slower, ~2GB VRAM), UVR-MDX-NET-Inst_3 (faster, good quality, ~1.5GB VRAM). HQ_4 recommended for best results."
            ))
            inputs.append(gr.Dropdown(
                label=_("Device"),
                choices=["cpu", "cuda", "xpu"] if available_devices is None else available_devices,
                value=defaults.get("device", device),
                info="🖥️ Device for BGM separation. cuda (GPU - recommended, ~10-20x faster), cpu (very slow, 10-30 min per song), xpu (Intel GPU). Use cuda if available!"
            ))
        
        # Row 2: Segment Size, Save Files, Offload Model
        with gr.Row():
            inputs.append(gr.Number(
                label="Segment Size",
                value=defaults.get("segment_size", cls.__fields__["segment_size"].default),
                precision=0,
                info="📦 Processing chunk size - affects quality vs speed. Examples: 256 (balanced), 512 (current - higher quality, slower, more VRAM), 128 (faster, lower quality). Higher = better separation but more VRAM. 512 recommended with unlimited compute."
            ))
            inputs.append(gr.Checkbox(
                label=_("Save separated files to output"),
                value=defaults.get("save_file", cls.__fields__["save_file"].default),
                info="💾 Save separated vocal/instrumental files to outputs/UVR/. ❌ Disabled by default - only transcribes, doesn't save. ✅ Enable to keep separated audio files for review or other uses. Files are large (~same size as input)."
            ))
            inputs.append(gr.Checkbox(
                label=_("Offload sub model when finished"),
                value=defaults.get("enable_offload", cls.__fields__["enable_offload"].default),
                info="💾 Unload UVR model from VRAM after separation. ❌ DISABLED (current) for unlimited compute. ✅ Enable if VRAM limited. UVR model uses ~2-3GB VRAM depending on model."
            ))
        
        return inputs


class WhisperParams(BaseParams):
    """Whisper parameters"""
    model_size: str = Field(default="large-v2", description="Whisper model size")
    lang: Optional[str] = Field(default=None, description="Source language of the file to transcribe")
    is_translate: bool = Field(default=False, description="Translate speech to English end-to-end")
    beam_size: int = Field(default=5, ge=1, description="Beam size for decoding")
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
    patience: float = Field(default=1.0, gt=0, description="Beam search patience factor")
    condition_on_previous_text: bool = Field(
        default=True,
        description="Use previous output as prompt for next window"
    )
    prompt_reset_on_temperature: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Temperature threshold for resetting prompt"
    )
    initial_prompt: Optional[str] = Field(default=None, description="Initial prompt for first window")
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
    word_timestamps: bool = Field(default=False, description="Extract word-level timestamps")
    prepend_punctuations: Optional[str] = Field(
        default="\"'“¿([{-",
        description="Punctuations to merge with next word"
    )
    append_punctuations: Optional[str] = Field(
        default="\"'.。,，!！?？:：”)]}、",
        description="Punctuations to merge with previous word"
    )
    max_new_tokens: Optional[int] = Field(default=None, description="Maximum number of new tokens per chunk")
    chunk_length: Optional[int] = Field(default=30, description="Length of audio segments in seconds")
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
    batch_size: int = Field(default=24, gt=0, description="Batch size for processing")
    enable_offload: bool = Field(
        default=True,
        description="Offload Whisper model after transcription"
    )

    @field_validator('lang')
    def validate_lang(cls, v):
        from modules.utils.constants import AUTOMATIC_DETECTION
        return None if v == AUTOMATIC_DETECTION.unwrap() else v

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

    @classmethod
    def to_gradio_inputs(cls,
                         defaults: Optional[Dict] = None,
                         only_advanced: Optional[bool] = True,
                         whisper_type: Optional[str] = None,
                         available_models: Optional[List] = None,
                         available_langs: Optional[List] = None,
                         available_compute_types: Optional[List] = None,
                         compute_type: Optional[str] = None,
                         use_3col_layout: bool = False):
        whisper_type = WhisperImpl.FASTER_WHISPER.value if whisper_type is None else whisper_type.strip().lower()

        inputs = []
        if not only_advanced:
            inputs += [
                gr.Dropdown(
                    label=_("Model"),
                    choices=available_models,
                    value=defaults.get("model_size", cls.__fields__["model_size"].default),
                ),
                gr.Dropdown(
                    label=_("Language"),
                    choices=available_langs,
                    value=defaults.get("lang", AUTOMATIC_DETECTION),
                ),
                gr.Checkbox(
                    label=_("Translate to English?"),
                    value=defaults.get("is_translate", cls.__fields__["is_translate"].default),
                ),
            ]

        # Row 1: Beam Size, Log Probability Threshold, No Speech Threshold
        with gr.Row():
            inputs.append(gr.Number(
                label="Beam Size",
                value=defaults.get("beam_size", cls.__fields__["beam_size"].default),
                precision=0,
                info="🔍 Number of beams in beam search. Higher = more accurate but slower. Range: 1-20. Examples: 5 (balanced), 10 (high accuracy), 1 (fastest/greedy). Current: optimized at 10 for maximum English accuracy."
            ))
            inputs.append(gr.Number(
                label="Log Probability Threshold",
                value=defaults.get("log_prob_threshold", cls.__fields__["log_prob_threshold"].default),
                info="📊 Rejects segments with average log probability below this. Lower (more negative) = stricter quality control. Examples: -1.0 (default), -0.5 (strict - rejects uncertain outputs), -1.5 (lenient). Current: -0.5 for high quality."
            ))
            inputs.append(gr.Number(
                label="No Speech Threshold",
                value=defaults.get("no_speech_threshold", cls.__fields__["no_speech_threshold"].default),
                info="🔇 Probability threshold for detecting silence/no-speech. Range: 0.0-1.0. Examples: 0.6 (balanced), 0.4 (detects more speech in noisy audio), 0.8 (strict silence detection). Lower if audio has background noise."
            ))
        
        # Row 2: Compute Type, Best Of, Patience
        with gr.Row():
            inputs.append(gr.Dropdown(
                label="Compute Type",
                choices=["bfloat16", "float16", "float32", "int8"] if available_compute_types is None else available_compute_types,
                value=defaults.get("compute_type", compute_type),
                info="⚙️ Precision for model computation. bfloat16 (recommended - stable, good performance), float16 (faster but less stable), float32 (most accurate, 2x VRAM), int8 (fastest, less accurate). Use bfloat16 for GPU, float32 for CPU. Current: bfloat16 (optimal balance)."
            ))
            inputs.append(gr.Number(
                label="Best Of",
                value=defaults.get("best_of", cls.__fields__["best_of"].default),
                precision=0,
                info="🎯 Number of candidate sequences to generate when sampling (when temperature > 0). Higher = better quality but slower. Range: 1-20. Examples: 5 (default), 10 (high quality), 1 (fastest). Current: 10 for maximum accuracy."
            ))
            inputs.append(gr.Number(
                label="Patience",
                value=defaults.get("patience", cls.__fields__["patience"].default),
                info="⏳ Beam search patience: how long to wait for better candidates. Higher = more thorough search. Examples: 1.0 (default), 2.0 (very thorough - current setting), 0.5 (faster). Increase for complex audio."
            ))
        
        # Row 3: Condition On Previous Text, Prompt Reset On Temperature, Initial Prompt
        with gr.Row():
            inputs.append(gr.Checkbox(
                label="Condition On Previous Text",
                value=defaults.get("condition_on_previous_text", cls.__fields__["condition_on_previous_text"].default),
                info="🔗 Use previous transcription as context for next segment. ✅ Recommended ON for better coherence and flow. Disable if getting stuck in repetitive loops. Helps maintain context across segments."
            ))
            inputs.append(gr.Slider(
                label="Prompt Reset On Temperature",
                value=defaults.get("prompt_reset_on_temperature",
                                   cls.__fields__["prompt_reset_on_temperature"].default),
                minimum=0,
                maximum=1,
                step=0.01,
                info="🌡️ Reset conditioning prompt if temperature exceeds this value. Range: 0.0-1.0. Examples: 0.5 (default - balanced), 0.3 (reset more often), 0.7 (reset less often). Prevents getting stuck in bad outputs."
            ))
            inputs.append(gr.Textbox(
                label="Initial Prompt",
                value=defaults.get("initial_prompt", GRADIO_NONE_STR),
                info="💬 Text to guide transcription style/vocabulary. Examples: 'Medical terminology:', 'Interview with Dr. Smith about AI', 'Technical lecture on Python'. Helps with domain-specific terms. Leave empty for general transcription."
            ))
        
        # Row 4: Temperature, Compression Ratio Threshold, Length Penalty
        with gr.Row():
            inputs.append(gr.Slider(
                label="Temperature",
                value=defaults.get("temperature", cls.__fields__["temperature"].default),
                minimum=0.0,
                step=0.01,
                maximum=1.0,
                info="🎲 Randomness in decoding. 0.0 = deterministic (most accurate - recommended), 0.2-0.5 = slight variation, 0.8-1.0 = creative but less accurate. Use 0 for maximum accuracy. Current: 0 (optimal for accuracy)."
            ))
            inputs.append(gr.Number(
                label="Compression Ratio Threshold",
                value=defaults.get("compression_ratio_threshold",
                                   cls.__fields__["compression_ratio_threshold"].default),
                info="📦 Detects repetitive/hallucinated text by gzip compression ratio. If text compresses too much (< threshold), it's likely repetitive. Examples: 2.4 (default), 2.0 (stricter), 3.0 (lenient). Lower = catches more hallucinations."
            ))
            inputs.append(gr.Number(
                label="Length Penalty",
                value=defaults.get("length_penalty", cls.__fields__["length_penalty"].default),
                info="📏 Penalty for longer sequences. >1.0 = favors longer outputs, <1.0 = favors shorter outputs. Examples: 1.0 (neutral - default), 1.2 (encourages longer segments), 0.8 (encourages shorter segments). Use 1.0 for balanced output."
            ))
        

        faster_whisper_inputs = []
        
        # Row 5: Repetition Penalty, No Repeat N-gram Size, Prefix
        with gr.Row():
            faster_whisper_inputs.append(gr.Number(
                label="Repetition Penalty",
                value=defaults.get("repetition_penalty", cls.__fields__["repetition_penalty"].default),
                info="🔁 Penalizes repeated tokens. >1.0 = discourages repetition. Examples: 1.0 (no penalty), 1.2 (current - reduces repetition), 1.5 (strongly discourages repetition). Increase if you see repeated phrases."
            ))
            faster_whisper_inputs.append(gr.Number(
                label="No Repeat N-gram Size",
                value=defaults.get("no_repeat_ngram_size", cls.__fields__["no_repeat_ngram_size"].default),
                precision=0,
                info="🚫 Blocks exact repetition of N-word phrases. Examples: 0 (no blocking), 3 (current - blocks 3-word repetitions like 'the the the'), 5 (blocks longer phrases). Use 3-5 to prevent stuttering."
            ))
            faster_whisper_inputs.append(gr.Textbox(
                label="Prefix",
                value=defaults.get("prefix", GRADIO_NONE_STR),
                info="▶️ Text to prepend to every segment (e.g., speaker name). Example: 'Speaker A: '. Different from Initial Prompt - this is added to every segment's output. Leave empty for normal transcription."
            ))
        
        # Row 6: Suppress Blank, Suppress Tokens, Max Initial Timestamp
        with gr.Row():
            faster_whisper_inputs.append(gr.Checkbox(
                label="Suppress Blank",
                value=defaults.get("suppress_blank", cls.__fields__["suppress_blank"].default),
                info="⬜ Suppress blank/empty outputs at start of sampling. ✅ Recommended ON to avoid empty segments. Disable only if you need to detect exact silence positions."
            ))
            faster_whisper_inputs.append(gr.Textbox(
                label="Suppress Tokens",
                value=defaults.get("suppress_tokens", "[-1]"),
                info="🎭 Token IDs to never generate. [-1] = suppress non-speech tokens. Examples: '[-1]' (default - suppress special tokens), '[-1, 220, 50257]' (suppress specific IDs). Advanced users only - see OpenAI tokenizer docs."
            ))
            faster_whisper_inputs.append(gr.Number(
                label="Max Initial Timestamp",
                value=defaults.get("max_initial_timestamp", cls.__fields__["max_initial_timestamp"].default),
                info="⏱️ Maximum allowed initial timestamp in seconds. Prevents model from starting transcription too late into audio. Examples: 1.0 (default), 0.5 (stricter - must start within 0.5s), 2.0 (more lenient). Use default."
            ))
        
        # Row 7: Word Timestamps, Prepend Punctuations, Append Punctuations
        with gr.Row():
            faster_whisper_inputs.append(gr.Checkbox(
                label="Word Timestamps",
                value=defaults.get("word_timestamps", cls.__fields__["word_timestamps"].default),
                info="📝 Extract timestamps for each individual word (not just segments). ✅ ENABLED for maximum accuracy - reduces hallucinations by ~10%! Slightly slower but highly recommended. Needed for word-level subtitle formats."
            ))
            faster_whisper_inputs.append(gr.Textbox(
                label="Prepend Punctuations",
                value=defaults.get("prepend_punctuations", cls.__fields__["prepend_punctuations"].default),
                info="⬅️ Punctuation marks to attach to NEXT word (e.g., opening quotes). Default: \"'¿([{- Keeps ' \"Hello' as one unit vs splitting. Rarely needs changing unless working with special languages."
            ))
            faster_whisper_inputs.append(gr.Textbox(
                label="Append Punctuations",
                value=defaults.get("append_punctuations", cls.__fields__["append_punctuations"].default),
                info="➡️ Punctuation marks to attach to PREVIOUS word (e.g., periods, commas). Default: \"'.。,，!！?？:：\")]}、 Keeps 'hello.' as one unit. Ensures proper punctuation alignment in subtitles."
            ))
        
        # Row 8: Max New Tokens, Chunk Length, Hallucination Silence Threshold
        with gr.Row():
            faster_whisper_inputs.append(gr.Number(
                label="Max New Tokens",
                value=defaults.get("max_new_tokens", GRADIO_NONE_NUMBER_MIN),
                precision=0,
                info="🔢 Maximum tokens per chunk. Limits output length per segment. Examples: None (auto - recommended), 224 (Whisper default), 448 (longer segments). Leave empty for automatic. Reduce if segments are too long."
            ))
            faster_whisper_inputs.append(gr.Number(
                label="Chunk Length (s)",
                value=defaults.get("chunk_length", cls.__fields__["chunk_length"].default),
                precision=0,
                info="✂️ Length of audio segments to process at once (seconds). Examples: 30 (default - balanced), 15 (shorter chunks, faster processing), 60 (longer chunks, better context). Shorter = faster but less context. 30s recommended."
            ))
            faster_whisper_inputs.append(gr.Number(
                label="Hallucination Silence Threshold (sec)",
                value=defaults.get("hallucination_silence_threshold",
                                   GRADIO_NONE_NUMBER_MIN),
                info="👻 Skip silent periods longer than this to detect hallucinations. Examples: 2.0 (current - skip 2+sec silence), 3.0 (lenient), 1.0 (strict). If audio has >2s silence and model still outputs text, it's likely hallucinating."
            ))
        
        # Row 9: Hotwords, Language Detection Threshold, Language Detection Segments
        with gr.Row():
            faster_whisper_inputs.append(gr.Textbox(
                label="Hotwords",
                value=defaults.get("hotwords", cls.__fields__["hotwords"].default),
                info="🔥 Boost recognition of specific words/phrases. Examples: 'OpenAI, ChatGPT, GPT-4' or 'Dr. Smith, cardiology'. Comma-separated. Helps with names, technical terms, brand names. Leave empty for general transcription."
            ))
            faster_whisper_inputs.append(gr.Number(
                label="Language Detection Threshold",
                value=defaults.get("language_detection_threshold",
                                   GRADIO_NONE_NUMBER_MIN),
                info="🌍 Confidence threshold for language detection. Examples: 0.5 (default), 0.7 (only use detected language if very confident), 0.3 (use even uncertain detections). Only matters if language = auto-detect."
            ))
            faster_whisper_inputs.append(gr.Number(
                label="Language Detection Segments",
                value=defaults.get("language_detection_segments",
                                   cls.__fields__["language_detection_segments"].default),
                precision=0,
                info="🎧 Number of audio segments to analyze for language detection. Examples: 1 (fast, less accurate), 3 (current - balanced), 5 (very accurate but slower). More segments = better detection but slower start. 3 recommended."
            ))
        

        insanely_fast_whisper_inputs = []
        
        # Row 10 (for insanely-fast-whisper): Batch Size
        with gr.Row():
            insanely_fast_whisper_inputs.append(gr.Number(
                label="Batch Size",
                value=defaults.get("batch_size", cls.__fields__["batch_size"].default),
                precision=0,
                info="📦 Number of audio chunks processed simultaneously. Higher = faster BUT more VRAM. Examples: 24 (balanced), 48 (current - 2x faster, needs ~10GB VRAM), 16 (for 6-8GB VRAM), 64+ (for 16GB+ VRAM). Increase for better GPU utilization."
            ))

        if whisper_type != WhisperImpl.FASTER_WHISPER.value:
            for input_component in faster_whisper_inputs:
                input_component.visible = False

        if whisper_type != WhisperImpl.INSANELY_FAST_WHISPER.value:
            for input_component in insanely_fast_whisper_inputs:
                input_component.visible = False

        inputs += faster_whisper_inputs + insanely_fast_whisper_inputs

        # Final row: Offload model
        with gr.Row():
            inputs.append(gr.Checkbox(
                label=_("Offload sub model when finished"),
                value=defaults.get("enable_offload", cls.__fields__["enable_offload"].default),
                info="💾 Unload model from VRAM after transcription. ✅ Enable if VRAM is limited (<8GB). ❌ DISABLED (current) for unlimited compute - keeps model loaded for faster repeated use. Disable for batch processing multiple files."
            ))

        return inputs
    
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
            input_configs.extend([
                ("dropdown", "Model", available_models, defaults.get("model_size", cls.__fields__["model_size"].default)),
                ("dropdown", "Language", available_langs, defaults.get("lang", AUTOMATIC_DETECTION)),
                ("checkbox", "Translate to English?", None, defaults.get("is_translate", cls.__fields__["is_translate"].default)),
            ])
        
        # Common inputs
        input_configs.extend([
            ("number", "Beam Size", 0, defaults.get("beam_size", cls.__fields__["beam_size"].default), 
             "🔍 Number of beams in beam search. Higher = more accurate but slower. Range: 1-20. Examples: 5 (balanced), 10 (high accuracy), 1 (fastest/greedy). Current: optimized at 10 for maximum English accuracy."),
            ("number", "Log Probability Threshold", None, defaults.get("log_prob_threshold", cls.__fields__["log_prob_threshold"].default),
             "📊 Rejects segments with average log probability below this. Lower (more negative) = stricter quality control. Examples: -1.0 (default), -0.5 (strict - rejects uncertain outputs), -1.5 (lenient). Current: -0.5 for high quality."),
            ("number", "No Speech Threshold", None, defaults.get("no_speech_threshold", cls.__fields__["no_speech_threshold"].default),
             "🔇 Probability threshold for detecting silence/no-speech. Range: 0.0-1.0. Examples: 0.6 (balanced), 0.4 (detects more speech in noisy audio), 0.8 (strict silence detection). Lower if audio has background noise."),
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
