"""Subtitle exports must retain English text and timings when reused for translation."""

import json

import pytest

from modules.utils.subtitle_manager import generate_file, get_writer, safe_filename
from modules.whisper.data_classes import Segment


@pytest.mark.parametrize("format_name", ["srt", "vtt", "lrc", "txt", "tsv", "json"])
def test_exported_formats_can_be_read_back_for_translation(tmp_path, format_name):
    segments = [
        Segment(start=1.125, end=3.5, text="Hello, world!"),
        Segment(start=4.0, end=7.25, text="We're checking English subtitles."),
    ]
    _, path = generate_file(str(tmp_path), "roundtrip", format_name, segments, add_timestamp=False)

    restored = get_writer(format_name, str(tmp_path)).to_segments(path)

    assert [segment.text for segment in restored] == [segment.text for segment in segments]
    if format_name != "txt":
        assert [(segment.start, segment.end) for segment in restored] == [(1.125, 3.5), (4.0, 7.25)]


@pytest.mark.parametrize("format_name", ["srt", "vtt", "lrc", "txt", "tsv", "json"])
def test_empty_translation_file_can_be_exported(tmp_path, format_name):
    _, path = generate_file(str(tmp_path), "empty", format_name, [], add_timestamp=False)
    assert get_writer(format_name, str(tmp_path)).to_segments(path) == []


def test_vtt_keeps_text_before_adjacent_timing_line(tmp_path):
    path = tmp_path / "adjacent.vtt"
    path.write_text("WEBVTT\n\n00:01.000 --> 00:02.000\nKeep this text.\n00:02.000 --> 00:03.000\nAnd this too.\n", encoding="utf-8")
    segments = get_writer("vtt", str(tmp_path)).to_segments(str(path))
    assert [segment.text for segment in segments] == ["Keep this text.", "And this too."]


def test_vtt_reads_bom_cue_ids_settings_and_inline_tags(tmp_path):
    path = tmp_path / "youtube.vtt"
    path.write_text("\ufeffWEBVTT\r\n\r\nNOTE metadata\r\nNot speech.\r\n\r\nfirst-id\r\n00:01.000 --> 00:02.000 align:start\r\nHello <00:01.200><c>world!</c>\r\n \r\nStill here.\r\n\r\nsecond-id\r\n00:03.000 --> 00:04.000\r\nGoodbye.\r\n", encoding="utf-8")
    segments = get_writer("vtt", str(tmp_path)).to_segments(str(path))
    assert [segment.text for segment in segments] == ["Hello world! Still here.", "Goodbye."]


@pytest.mark.parametrize("name", ["CON", "aux.txt", "NUL", "Lpt1", "com9.srt", "CONIN$", "CONOUT$"])
def test_output_names_are_safe_on_windows(name):
    assert safe_filename(name).split(".", 1)[0].upper() not in {
        "CON", "AUX", "NUL", "LPT1", "COM9", "CONIN$", "CONOUT$"
    }


@pytest.mark.parametrize("name", ["trailing. ", "...", "", "   "])
def test_output_names_are_nonempty_without_trailing_windows_separators(name):
    result = safe_filename(name)
    assert result and not result.endswith((".", " "))


def test_json_segment_parser_accepts_saved_dictionary(tmp_path):
    path = tmp_path / "transcript.json"
    path.write_text(json.dumps({"text": "Hello!", "segments": [{"start": 0, "end": 1, "text": "Hello!"}]}))
    segments = get_writer("json", str(tmp_path)).to_segments(str(path))
    assert [(segment.start, segment.end, segment.text) for segment in segments] == [(0, 1, "Hello!")]
