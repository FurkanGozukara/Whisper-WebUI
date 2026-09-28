import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.uvr.music_separator import MusicSeparator


@pytest.mark.parametrize("sample_rate", [16000, 44100, 48000])
def test_mdx_receives_correct_frequency_and_duration_at_44100(sample_rate, tmp_path, monkeypatch):
    seconds = 1.25
    samples = np.arange(round(seconds * sample_rate)) / sample_rate
    # Different left/right signals detect accidental mono downmix as well as rate errors.
    audio = np.stack([np.sin(2 * np.pi * 440 * samples), np.sin(2 * np.pi * 880 * samples)], axis=1) * .25
    source = tmp_path / f"source-{sample_rate}.wav"
    sf.write(source, audio, sample_rate, subtype="FLOAT")
    monkeypatch.setattr(MusicSeparator, "get_device", staticmethod(lambda: "cpu"))
    separator = MusicSeparator(output_dir=str(tmp_path / "outputs"))
    captured = []

    class DummyMDX:
        sample_rate = 44100

        def __call__(self, value, sampling_rate):
            assert self.sample_rate == 44100
            assert sampling_rate == 44100
            captured.append(value.copy())
            return {"instrumental": value * .5, "vocals": value * .5}

    separator.model = DummyMDX()
    instrumental, vocals, paths = separator.separate(
        str(source), separator.default_model, device="cpu", save_file=True, progress=lambda *args, **kwargs: None)
    model_audio = captured[0]
    assert model_audio.shape == (2, round(seconds * 44100))
    assert separator.audio_info.sample_rate == 44100
    for channel, expected_hz in zip(model_audio, (440, 880)):
        peak_hz = np.argmax(np.abs(np.fft.rfft(channel))) * 44100 / len(channel)
        assert peak_hz == pytest.approx(expected_hz, abs=1)
    assert instrumental.shape == vocals.shape == (round(seconds * 44100), 2)
    for path in paths:
        output = sf.info(path)
        assert output.samplerate == 44100
        assert output.duration == pytest.approx(seconds, abs=1 / 44100)


def test_mono_asr_array_is_resampled_and_duplicated_without_shortening():
    audio = np.arange(16000 * 2, dtype=np.float32) / 32000
    converted = MusicSeparator.prepare_model_audio(audio, 16000)
    assert converted.shape == (2, 44100 * 2)
    assert np.array_equal(converted[0], converted[1])


@pytest.mark.parametrize("audio,rate", [(np.array([]), 16000), (np.ones(16000), 0), (np.ones((5, 100)), 48000)])
def test_invalid_audio_fails_before_model_loading(audio, rate):
    with pytest.raises(ValueError):
        MusicSeparator.prepare_model_audio(audio, rate)
