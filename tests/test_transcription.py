import requests
import pytest
import gradio as gr
import os

from modules.whisper.whisper_factory import WhisperFactory
from modules.whisper.data_classes import *
from modules.utils.subtitle_manager import read_file
from modules.utils.paths import WEBUI_DIR
from test_config import *


def srt_cue_text(path: str) -> str:
    """Text of every cue of an SRT file, speaker labels removed, as one line."""
    texts = []
    for line in read_file(path).split("\n"):
        line = line.strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        texts.append(line.split("|", 1)[-1])
    return " ".join(texts)


def assert_transcript_matches(path: str, diarization: bool):
    # The subtitle writer splits the sentence into several cues, so score the whole transcript.
    if diarization:
        assert "SPEAKER_00|" in read_file(path)
    hypothesis = srt_cue_text(path).replace(",", "").replace(".", "")
    wer = calculate_wer(TEST_ANSWER, hypothesis)
    assert wer < 0.1, f"WER is too high, it's {wer}: {hypothesis!r}"


def run_asr_pipeline(
    whisper_type: str,
    vad_filter: bool,
    bgm_separation: bool,
    diarization: bool,
):
    audio_path = TEST_FILE_PATH

    whisper_inferencer = WhisperFactory.create_whisper_inference(
        whisper_type=whisper_type,
    )
    print(
        f"""Whisper Device : {whisper_inferencer.device}\n"""
        f"""BGM Separation Device: {whisper_inferencer.music_separator.device}\n"""
        f"""Diarization Device: {whisper_inferencer.diarizer.device}"""
    )

    hparams = TranscriptionPipelineParams(
        whisper=WhisperParams(
            model_size=TEST_WHISPER_MODEL,
            compute_type=whisper_inferencer.current_compute_type,
            # Whisper's native window, as in the app's Whisper presets (the shared default of 10 s
            # suits Canary and cuts the 11 s test sentence in half).
            chunk_length=30,
        ),
        vad=VadParams(
            vad_filter=vad_filter
        ),
        bgm_separation=BGMSeparationParams(
            is_separate_bgm=bgm_separation,
            enable_offload=True
        ),
        diarization=DiarizationParams(
            is_diarize=diarization
        ),
    ).to_list()

    subtitle_str, file_paths = whisper_inferencer.transcribe_file(
        [audio_path],
        False,  # batch_mode
        None,   # input_folder_path
        None,   # include_subdirectory
        False,  # overwrite_existing
        None,   # output_dir
        "SRT",
        False,
        gr.Progress(),
        *hparams,
    )
    assert_transcript_matches(file_paths[0], diarization)

    if not is_pytube_detected_bot():
        subtitle_str, file_path = whisper_inferencer.transcribe_youtube(
            TEST_YOUTUBE_URL,
            "SRT",
            False,
            False,  # mass_transcribe_channel
            100,    # latest_video_count
            gr.Progress(),
            *hparams,
        )
        assert isinstance(subtitle_str, str) and subtitle_str
        output_paths = file_path if isinstance(file_path, list) else [file_path]
        assert all(os.path.exists(path) for path in output_paths)

    subtitle_str, file_path = whisper_inferencer.transcribe_mic(
        audio_path,
        "SRT",
        False,
        gr.Progress(),
        *hparams,
    )
    primary_output_path = file_path[0] if isinstance(file_path, list) else file_path
    assert_transcript_matches(primary_output_path, diarization)


@pytest.mark.parametrize(
    "whisper_type,vad_filter,bgm_separation,diarization",
    [
        (WhisperImpl.WHISPER.value, False, False, False),
        (WhisperImpl.FASTER_WHISPER.value, False, False, False),
        (WhisperImpl.INSANELY_FAST_WHISPER.value, False, False, False)
    ]
)
def test_transcribe(
    whisper_type: str,
    vad_filter: bool,
    bgm_separation: bool,
    diarization: bool,
):
    run_asr_pipeline(whisper_type, vad_filter, bgm_separation, diarization)


