from types import SimpleNamespace

import numpy as np
import pytest

from modules.utils.audio_manager import decode_audio
from modules.whisper.faster_whisper_inference import FasterWhisperInference

JFK = "tests/jfk.wav"
FRAMES_PER_SECOND = 100  # faster-whisper feature frames
TIME_PER_FRAME = 0.01


@pytest.mark.parametrize("seek,window_end,clip_end,expected", [
    # last window: 0.6 s of silence after the last word is not decoded on its own
    (2778, 2842, 2842, 2842),
    # a longer remaining stretch may still hold speech the window ended early on: decode it
    (2000, 2842, 2842, 2000),
    # an inner window never skips: the rest of the file comes after it
    (2778, 3000, 9000, 2778),
    # already at the end
    (2842, 2842, 2842, 2842),
])
def test_end_of_audio_tail_guard(seek, window_end, clip_end, expected):
    assert FasterWhisperInference.guard_end_of_audio_tail(seek, window_end, clip_end, TIME_PER_FRAME) == expected


def test_end_of_audio_tail_guard_can_be_disabled(monkeypatch):
    monkeypatch.setattr(FasterWhisperInference, "END_OF_AUDIO_TAIL_GUARD_SECONDS", 0)
    assert FasterWhisperInference.guard_end_of_audio_tail(2778, 2842, 2842, TIME_PER_FRAME) == 2778


def test_end_of_audio_tail_with_speech_is_still_decoded():
    speech = decode_audio(JFK)[: 3 * 16000]  # "And so my fellow Americans"
    audio = np.concatenate([np.zeros(16000, dtype=np.float32), speech])
    silence = np.zeros_like(audio)
    # the last 2 s (speech until 3.2 s): speech is decoded, silence is not
    assert FasterWhisperInference.guard_end_of_audio_tail(200, 400, 400, TIME_PER_FRAME, audio=audio) == 200
    assert FasterWhisperInference.guard_end_of_audio_tail(200, 400, 400, TIME_PER_FRAME, audio=silence) == 400


def test_tail_speech_detection_needs_enough_speech():
    audio = decode_audio(JFK)
    assert FasterWhisperInference.tail_has_speech(audio, 1.0, 3.0)
    assert not FasterWhisperInference.tail_has_speech(np.zeros(4 * 16000, dtype=np.float32), 1.0, 3.0)
    noise = (0.01 * np.random.default_rng(0).standard_normal(4 * 16000)).astype(np.float32)
    assert not FasterWhisperInference.tail_has_speech(noise, 2.0, 4.0)


def test_tail_speech_detection_ignores_the_end_of_the_last_word():
    # speech until 3 s, then silence: a tail from 2.8 s only holds the end of the last word
    audio = np.concatenate([decode_audio(JFK)[: 3 * 16000], np.zeros(int(1.5 * 16000), dtype=np.float32)])
    assert not FasterWhisperInference.tail_has_speech(audio, 2.8, 4.5)
    # a tail too short to hold speech after the ignored part is never decoded
    assert not FasterWhisperInference.tail_has_speech(decode_audio(JFK), 2.0, 2.7)


def test_decode_audio_matches_faster_whisper_bit_for_bit():
    from faster_whisper.audio import decode_audio as faster_whisper_decode_audio

    expected = faster_whisper_decode_audio(JFK, sampling_rate=16000)
    actual = decode_audio(JFK, sampling_rate=16000)
    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)


def test_batched_windows_end_at_pauses_and_keep_short_gaps(monkeypatch):
    import faster_whisper.vad as fw_vad

    sr = 16000
    speech = [
        {"start": 0, "end": 10 * sr},
        {"start": int(10.5 * sr), "end": 20 * sr},
        {"start": int(20.4 * sr), "end": 28 * sr},  # 0-28 s fits one 30 s window, pauses kept inside
        {"start": 29 * sr, "end": 40 * sr},  # starts a new window
    ]
    monkeypatch.setattr(fw_vad, "get_speech_timestamps", lambda audio, options, sampling_rate=16000: speech)

    windows, regions = FasterWhisperInference.build_vad_clip_timestamps(np.zeros(41 * sr, dtype=np.float32), 30, sr)

    assert windows == [{"start": 0.0, "end": 28.0}, {"start": 29.0, "end": 40.0}]
    assert regions == speech


def test_batched_windows_empty_when_no_speech(monkeypatch):
    import faster_whisper.vad as fw_vad

    monkeypatch.setattr(fw_vad, "get_speech_timestamps", lambda audio, options, sampling_rate=16000: [])
    assert FasterWhisperInference.build_vad_clip_timestamps(np.zeros(16000, dtype=np.float32), 30, 16000) == ([], [])


def test_uncovered_speech_tails_only_where_speech_was_left_out():
    sr = 16000
    windows = [{"start": 0.0, "end": 30.0}, {"start": 30.0, "end": 58.0}, {"start": 60.0, "end": 70.0}]
    segments = [SimpleNamespace(start=0.0, end=21.0), SimpleNamespace(start=30.5, end=57.6),
                SimpleNamespace(start=60.2, end=64.0)]
    speech = [{"start": 0, "end": 29 * sr}, {"start": 30 * sr, "end": 58 * sr}, {"start": 60 * sr, "end": 64 * sr}]

    tails = FasterWhisperInference.uncovered_speech_tails(windows, segments, speech, sr)

    # 21-30 s still held speech; 57.6-58 s is too short; 64-70 s has no detected speech
    assert tails == [{"start": 21.0, "end": 30.0}]


def test_shared_vad_model_is_cached_and_used_by_faster_whisper():
    import faster_whisper.vad as fw_vad
    from modules.vad.silero_vad import load_silero_vad_model

    model = load_silero_vad_model()
    assert model is load_silero_vad_model()
    assert fw_vad.get_vad_model() is model
    probs = model(np.zeros(512 * 4, dtype=np.float32))
    assert np.asarray(probs).reshape(-1).shape[0] == 4


@pytest.mark.parametrize("samples", [512 * 2600, 512 * 2600 + 300])
def test_speech_probability_stream_matches_faster_whisper_bit_for_bit(samples):
    from faster_whisper.utils import get_assets_path
    from faster_whisper.vad import SileroVADModel
    from modules.vad.silero_vad import SpeechProbabilityStream, load_silero_vad_model

    rng = np.random.default_rng(3)
    t = np.arange(samples) / 16000.0
    audio = (0.2 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 0.3 * t) > 0)
             + 0.01 * rng.standard_normal(samples)).astype(np.float32)
    padded = np.pad(audio, (0, (-samples) % 512))
    reference = np.asarray(SileroVADModel(f"{get_assets_path()}/silero_vad_v6.onnx")(padded.copy())).reshape(-1)
    original = audio.copy()

    stream = SpeechProbabilityStream(load_silero_vad_model(), audio)
    assert stream.wait(10).shape == reference.shape
    assert np.array_equal(stream.wait(), reference)
    assert np.array_equal(audio, original)  # faster-whisper zeroes the end of the caller's array


def test_speech_probability_stream_close_releases_waiters():
    from modules.vad.silero_vad import SpeechProbabilityStream

    class SlowModel:
        def __call__(self, audio):
            import time
            time.sleep(0.5)
            return np.zeros(audio.shape[0] // 512, dtype=np.float32)

    stream = SpeechProbabilityStream(SlowModel(), np.zeros(512 * 10, dtype=np.float32))
    stream.close()
    with pytest.raises(RuntimeError):
        stream.wait()
