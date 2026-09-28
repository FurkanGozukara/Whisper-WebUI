from difflib import SequenceMatcher

import pytest

from english_benchmark_reference_audit import find_span, original_span


def test_verbalized_currency_maps_back_to_original_human_words():
    original = ["Revenue", "was", "$2.5", "million"]
    aligned = ["Revenue", "was", "two", "point", "five", "million", "dollars"]
    operations = SequenceMatcher(None, original, aligned, autojunk=False).get_opcodes()
    start, stop = original_span(operations, 0, len(aligned) - 1)
    assert original[start:stop] == original


def test_partial_verbalized_number_cannot_be_used_as_human_reference():
    operations = SequenceMatcher(None, ["2.5", "million"], ["two", "point", "five", "million"], autojunk=False).get_opcodes()
    with pytest.raises(RuntimeError, match="inside a verbalized"):
        original_span(operations, 2, 3)


def test_full_recording_can_keep_untimed_opening_human_word():
    rows = [{"token": "Hello", "ts": "", "endTs": "", "punctuation": ""},
            {"token": "world", "ts": "1", "endTs": "2", "punctuation": "."}]
    assert find_span(rows, "Hello world.", 0) == (0, 1)
