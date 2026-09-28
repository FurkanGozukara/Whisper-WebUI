from modules.translation.translation_base import TranslationBase
from modules.utils.subtitle_manager import get_writer


class IdentityTranslation(TranslationBase):
    def update_model(self, **kwargs):
        raise AssertionError("Unchanged language must not download or load a model")

    def translate(self, text, max_length):
        raise AssertionError("Unchanged language must retain the original words")


def test_english_to_english_preserves_words_punctuation_speakers_and_timings(tmp_path):
    source = tmp_path / "english.srt"
    source.write_text("1\n00:00:01,250 --> 00:00:03,500\nHello, world!\n\n2\n00:00:04,000 --> 00:00:06,000\nSPEAKER_00|We're checking 123 names.\n", encoding="utf-8")
    inference = IdentityTranslation(str(tmp_path / "models"), str(tmp_path / "outputs"))

    _, outputs = inference.translate_file([str(source)], "unused", "English", "English", add_timestamp=False)

    segments = get_writer("srt", str(tmp_path)).to_segments(outputs[0])
    assert [(segment.start, segment.end, segment.text) for segment in segments] == [
        (1.25, 3.5, "Hello, world!"),
        (4.0, 6.0, "SPEAKER_00|We're checking 123 names."),
    ]
