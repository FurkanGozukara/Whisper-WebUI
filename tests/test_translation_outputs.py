from pathlib import Path

import pytest

from modules.translation.deepl_api import DeepLAPI
from modules.translation.translation_base import TranslationBase
from modules.utils.subtitle_manager import generate_file, get_writer
from modules.whisper.data_classes import Segment


class IdentityTranslation(TranslationBase):
    def update_model(self, **kwargs):
        raise AssertionError("English to English must not load a model")

    def translate(self, text, max_length):
        raise AssertionError("English to English must preserve text")


@pytest.fixture(params=["nllb", "deepl"])
def translate(request, tmp_path, monkeypatch):
    output_dir = tmp_path / "translated files"
    if request.param == "nllb":
        monkeypatch.setattr(IdentityTranslation, "get_device", staticmethod(lambda: "cpu"))
        translator = IdentityTranslation(str(tmp_path / "models"), str(output_dir))
        call = lambda paths, timestamp: translator.translate_file(
            paths, "unused", "English", "English", add_timestamp=timestamp)
    else:
        translator = DeepLAPI(str(output_dir))
        # Exercise all file handling without credentials or a paid external request.
        monkeypatch.setattr(translator, "request_deepl_translate", lambda key, texts, *args:
                            [{"text": text} for text in texts])
        call = lambda paths, timestamp: translator.translate_deepl(
            "test-key", paths, "English", "English", add_timestamp=timestamp)
    return call, output_dir


def subtitle(folder, name, text):
    folder.mkdir(parents=True, exist_ok=True)
    stem, extension = name.rsplit(".", 1)
    _, path = generate_file(str(folder), stem, extension,
                            [Segment(start=1, end=3, text=text)], add_timestamp=False)
    return path


@pytest.mark.parametrize("timestamp", [False, True])
def test_all_six_same_stem_formats_remain_downloadable(translate, tmp_path, timestamp):
    call, output_dir = translate
    formats = ["srt", "vtt", "txt", "lrc", "json", "tsv"]
    sources = [subtitle(tmp_path / "input", f"jfk.{fmt}", f"English words in {fmt}.") for fmt in formats]
    status, outputs = call(sources, timestamp)
    assert len(outputs) == len(set(outputs)) == 6
    assert "6 subtitle file(s)" in status
    assert str(output_dir) in status
    for fmt, path in zip(formats, outputs):
        assert Path(path).suffix == "." + fmt
        assert get_writer(fmt, str(output_dir)).to_segments(path)[0].text == f"English words in {fmt}."


@pytest.mark.parametrize("timestamp", [False, True])
def test_duplicate_names_from_different_directories_do_not_overwrite(translate, tmp_path, timestamp):
    call, output_dir = translate
    sources = [subtitle(tmp_path / folder, name, text) for folder, name, text in [
        ("first", "clip.srt", "First input."),
        ("second", "clip.srt", "Second input."),
        ("third", "clip-2.srt", "Third input."),
        ("fourth", "CLIP.srt", "Fourth input."),
    ]]
    _, outputs = call(sources, timestamp)
    assert len(outputs) == len({Path(path).name.casefold() for path in outputs}) == 4
    assert [get_writer("srt", str(output_dir)).to_segments(path)[0].text for path in outputs] == [
        "First input.", "Second input.", "Third input.", "Fourth input."]


def test_plain_output_preserves_existing_file_and_original_input(translate, tmp_path):
    call, output_dir = translate
    existing = subtitle(output_dir, "clip.srt", "Existing output.")
    source = subtitle(tmp_path / "input", "clip.srt", "New input.")
    before = Path(existing).read_bytes()
    _, outputs = call([source, existing], False)
    assert len(set(outputs)) == 2
    assert existing not in outputs
    assert Path(existing).read_bytes() == before
    assert [get_writer("srt", str(output_dir)).to_segments(path)[0].text for path in outputs] == [
        "New input.", "Existing output."]
