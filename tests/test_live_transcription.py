import time
from pathlib import Path
import sys
import types

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if "tiktoken" not in sys.modules:
    fake_tiktoken = types.ModuleType("tiktoken")

    class _Encoding:
        def encode(self, value, **kwargs):
            return []

        def decode(self, tokens, **kwargs):
            return ""

    fake_tiktoken.Encoding = _Encoding
    fake_tiktoken.get_encoding = lambda _name=None: _Encoding()
    fake_tiktoken.encoding_for_model = lambda _name=None: _Encoding()
    sys.modules["tiktoken"] = fake_tiktoken

if "whisper" not in sys.modules:
    fake_whisper = types.ModuleType("whisper")
    fake_whisper.available_models = lambda: []
    fake_whisper.tokenizer = types.SimpleNamespace(LANGUAGES={"en": "english"})
    sys.modules["whisper"] = fake_whisper

if "ctranslate2" not in sys.modules:
    sys.modules["ctranslate2"] = types.ModuleType("ctranslate2")

if "modules.utils.torch_compat" not in sys.modules:
    fake_torch_compat = types.ModuleType("modules.utils.torch_compat")
    fake_torch_compat.enable_torchaudio_2_9_compat = lambda: None
    sys.modules["modules.utils.torch_compat"] = fake_torch_compat

if "torchaudio" not in sys.modules:
    sys.modules["torchaudio"] = types.ModuleType("torchaudio")

if "faster_whisper.vad" not in sys.modules:
    fake_faster_whisper = types.ModuleType("faster_whisper")
    fake_faster_whisper_vad = types.ModuleType("faster_whisper.vad")

    class _VadOptions:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_faster_whisper_vad.VadOptions = _VadOptions
    sys.modules["faster_whisper"] = fake_faster_whisper
    sys.modules["faster_whisper.vad"] = fake_faster_whisper_vad

if "modules.uvr.music_separator" not in sys.modules:
    fake_music_separator = types.ModuleType("modules.uvr.music_separator")

    class _MusicSeparator:
        def __init__(self, *args, **kwargs):
            self.output_dir = kwargs.get("output_dir")

    fake_music_separator.MusicSeparator = _MusicSeparator
    sys.modules["modules.uvr.music_separator"] = fake_music_separator

if "modules.diarize.diarizer" not in sys.modules:
    fake_diarizer = types.ModuleType("modules.diarize.diarizer")

    class _Diarizer:
        def __init__(self, *args, **kwargs):
            pass

    fake_diarizer.Diarizer = _Diarizer
    sys.modules["modules.diarize.diarizer"] = fake_diarizer

if "modules.vad.silero_vad" not in sys.modules:
    fake_silero_vad = types.ModuleType("modules.vad.silero_vad")

    class _SileroVAD:
        def __init__(self, *args, **kwargs):
            pass

    fake_silero_vad.SileroVAD = _SileroVAD
    sys.modules["modules.vad.silero_vad"] = fake_silero_vad

if "modules.utils.youtube_manager" not in sys.modules:
    fake_youtube_manager = types.ModuleType("modules.utils.youtube_manager")
    fake_youtube_manager.get_ytdata = lambda url: types.SimpleNamespace(title="Video")
    fake_youtube_manager.get_ytaudio = lambda _yt: ""
    sys.modules["modules.utils.youtube_manager"] = fake_youtube_manager

if "modules.utils.audio_manager" not in sys.modules:
    fake_audio_manager = types.ModuleType("modules.utils.audio_manager")
    fake_audio_manager.validate_audio = lambda audio: audio
    sys.modules["modules.utils.audio_manager"] = fake_audio_manager

if "modules.utils.files_manager" not in sys.modules:
    fake_files_manager = types.ModuleType("modules.utils.files_manager")
    fake_files_manager.get_media_files = lambda *args, **kwargs: []
    fake_files_manager.format_gradio_files = lambda files: files
    fake_files_manager.read_file = lambda file_path: Path(file_path).read_text(encoding="utf-8")
    sys.modules["modules.utils.files_manager"] = fake_files_manager

if "gradio_i18n" not in sys.modules:
    fake_gradio_i18n = types.ModuleType("gradio_i18n")
    fake_gradio_i18n_i18n = types.ModuleType("gradio_i18n.i18n")

    class _Translate:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _I18nString(str):
        def __new__(cls, value):
            obj = str.__new__(cls, value)
            obj.radd_values = []
            obj.add_values = []
            return obj

    class _TranslateContext:
        dictionary = {}

        @staticmethod
        def get_current_language(_request):
            return "en"

    fake_gradio_i18n.Translate = _Translate
    fake_gradio_i18n.gettext = lambda value: value
    fake_gradio_i18n_i18n.I18nString = _I18nString
    fake_gradio_i18n_i18n.TranslateContext = _TranslateContext
    sys.modules["gradio_i18n"] = fake_gradio_i18n
    sys.modules["gradio_i18n.i18n"] = fake_gradio_i18n_i18n

from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.whisper.data_classes import Segment, TranscriptionPipelineParams, WhisperParams


class DummyLivePipeline(BaseTranscriptionPipeline):
    def __init__(self, output_dir: Path):
        self.output_dir = str(output_dir)
        self.model = None
        self.device = "cpu"

    def transcribe(self, audio, progress=None, progress_callback=None, *whisper_params):
        return [], 0.0

    def update_model(self, model_size, compute_type, progress=None):
        return None

    def run(self, audio, progress=None, file_format="SRT", add_timestamp=True, progress_callback=None, *pipeline_params):
        segments = []
        for index in range(35):
            segment = Segment(
                start=float(index),
                end=float(index) + 0.5,
                text=f"line {index + 1}",
            )
            segments.append(segment)
            if progress_callback is not None:
                progress_callback((index + 1) / 35.0, segment)
            time.sleep(0.005)
        return segments, 1.0


def test_live_transcription_streams_recent_segment_history(monkeypatch, tmp_path):
    media_file = tmp_path / "clip.wav"
    media_file.write_bytes(b"fake")

    output_file = tmp_path / "clip.srt"
    output_file.write_text("subtitle", encoding="utf-8")

    monkeypatch.setattr(
        "modules.whisper.base_transcription_pipeline.generate_file",
        lambda **kwargs: ("subtitle", str(output_file)),
    )

    pipeline = DummyLivePipeline(output_dir=tmp_path)
    pipeline_params = TranscriptionPipelineParams(
        whisper=WhisperParams(word_timestamps=False),
    ).to_list()

    updates = list(
        pipeline.transcribe_file_with_live_output(
            [str(media_file)],
            False,
            None,
            False,
            False,
            str(tmp_path),
            ["SRT"],
            False,
            None,
            *pipeline_params,
        )
    )

    assert len(updates) > 2

    first_live_output, first_result, _ = updates[0]
    final_live_output, final_result, final_paths = updates[-1]

    assert "Processing: clip" in first_live_output
    assert first_result == ""
    assert "Done! 35 segments" in final_result
    assert final_paths == [str(output_file)]
    assert "line 35" in final_live_output
    assert "[00:00:00.000 → 00:00:00.500] line 1" not in final_live_output
    assert len(final_live_output.splitlines()) <= BaseTranscriptionPipeline.LIVE_TRANSCRIPTION_HISTORY_LINES


def test_build_output_specs_adds_plain_srt_companion_when_word_timestamps_enabled(tmp_path):
    pipeline = DummyLivePipeline(output_dir=tmp_path)

    output_specs = pipeline._build_output_specs(
        file_name="clip",
        file_formats=["SRT", "VTT"],
        writer_options={"highlight_words": True},
    )

    assert output_specs == [
        {
            "lookup_key": "srt",
            "output_format": "SRT",
            "output_file_name": "clip",
            "writer_options": {"highlight_words": True},
        },
        {
            "lookup_key": "srt_noword_timestaps",
            "output_format": "srt",
            "output_file_name": "clip_noword_timestaps",
            "writer_options": {"highlight_words": False},
        },
        {
            "lookup_key": "vtt",
            "output_format": "VTT",
            "output_file_name": "clip",
            "writer_options": {"highlight_words": True},
        },
    ]


def test_find_existing_outputs_distinguishes_main_srt_from_plain_companion(tmp_path):
    pipeline = DummyLivePipeline(output_dir=tmp_path)
    output_specs = pipeline._build_output_specs(
        file_name="clip",
        file_formats=["SRT"],
        writer_options={"highlight_words": True},
    )

    main_srt = tmp_path / "clip-123.srt"
    no_word_srt = tmp_path / "clip_noword_timestaps-123.srt"
    main_srt.write_text("main", encoding="utf-8")
    no_word_srt.write_text("plain", encoding="utf-8")

    existing_outputs = pipeline._find_existing_outputs(str(tmp_path), output_specs)

    assert existing_outputs == {
        "srt": [str(main_srt)],
        "srt_noword_timestaps": [str(no_word_srt)],
    }


def test_live_transcription_writes_plain_srt_companion_when_word_timestamps_enabled(monkeypatch, tmp_path):
    media_file = tmp_path / "clip.wav"
    media_file.write_bytes(b"fake")

    generate_calls = []

    def fake_generate_file(**kwargs):
        output_path = Path(kwargs["output_dir"]) / f"{kwargs['output_file_name']}.{kwargs['output_format'].lower()}"
        output_path.write_text(kwargs["output_file_name"], encoding="utf-8")
        generate_calls.append(kwargs)
        return kwargs["output_file_name"], str(output_path)

    monkeypatch.setattr(
        "modules.whisper.base_transcription_pipeline.generate_file",
        fake_generate_file,
    )

    pipeline = DummyLivePipeline(output_dir=tmp_path)
    pipeline_params = TranscriptionPipelineParams(
        whisper=WhisperParams(word_timestamps=True),
    ).to_list()

    updates = list(
        pipeline.transcribe_file_with_live_output(
            [str(media_file)],
            False,
            None,
            False,
            False,
            str(tmp_path),
            ["SRT"],
            False,
            None,
            *pipeline_params,
        )
    )

    _, _, final_paths = updates[-1]

    assert [call["output_file_name"] for call in generate_calls] == [
        "clip",
        "clip_noword_timestaps",
    ]
    assert [call["highlight_words"] for call in generate_calls] == [True, False]
    assert final_paths == [
        str(tmp_path / "clip.srt"),
        str(tmp_path / "clip_noword_timestaps.srt"),
    ]
