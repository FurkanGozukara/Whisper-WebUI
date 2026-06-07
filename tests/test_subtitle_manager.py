import json
from pathlib import Path

from modules.utils.subtitle_manager import generate_file


def _word(start, text):
    return {
        "start": start,
        "end": start + 0.4,
        "word": text,
        "probability": 0.9,
    }


def _word_timestamp_result():
    words = [
        _word(0.0, " RACES."),
        _word(0.5, " A"),
        _word(1.0, " SHOWDOWN"),
        _word(1.5, " IS"),
        _word(2.0, " UNDERWAY"),
        _word(2.5, " IN"),
        _word(3.0, " CALIFORNIA."),
        _word(3.5, " THE"),
        _word(4.0, " LAST"),
        _word(4.5, " POLL"),
        _word(5.0, " SHOWS"),
        _word(5.5, " STEVE"),
        _word(6.0, " HILTON"),
        _word(6.5, " EMERGING"),
        _word(7.0, " AS"),
        _word(7.5, " A"),
        _word(8.0, " SERIOUS"),
        _word(8.5, " CANDIDATE"),
        _word(9.0, " FOR"),
        _word(9.5, " GOVERNOR."),
    ]
    return {
        "text": " ".join(word["word"].strip() for word in words),
        "segments": [
            {
                "start": 0.0,
                "end": 10.0,
                "text": " ".join(word["word"].strip() for word in words),
                "words": words,
            }
        ],
    }


def test_generate_file_normalizes_word_timestamps_to_sentence_srt(tmp_path):
    content, path = generate_file(
        output_dir=str(tmp_path),
        output_file_name="clip",
        output_format="srt",
        result=_word_timestamp_result(),
        add_timestamp=False,
        highlight_words=False,
        normalize_word_timestamps=True,
    )

    assert path == str(tmp_path / "clip.srt")
    assert "<u>" not in content
    assert "RACES." in content
    assert "A SHOWDOWN IS UNDERWAY IN CALIFORNIA." in content
    assert "THE LAST POLL SHOWS STEVE HILTON EMERGING AS A SERIOUS CANDIDATE FOR GOVERNOR." in content
    assert "RACES. A SHOWDOWN" not in content


def test_generate_file_normalizes_word_timestamps_to_sentence_json_without_words(tmp_path):
    content, path = generate_file(
        output_dir=str(tmp_path),
        output_file_name="clip",
        output_format="json",
        result=_word_timestamp_result(),
        add_timestamp=False,
        highlight_words=False,
        normalize_word_timestamps=True,
    )

    data = json.loads(Path(path).read_text(encoding="utf-8"))

    assert json.loads(content) == data
    assert [segment["text"] for segment in data["segments"]] == [
        "RACES.",
        "A SHOWDOWN IS UNDERWAY IN CALIFORNIA.",
        "THE LAST POLL SHOWS STEVE HILTON EMERGING AS A SERIOUS CANDIDATE FOR GOVERNOR.",
    ]
    assert all("words" not in segment for segment in data["segments"])
