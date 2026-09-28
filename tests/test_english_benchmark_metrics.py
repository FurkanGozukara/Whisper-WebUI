from english_benchmark_metrics import evaluate, punctuation_scores


def test_equivalent_number_and_case_normalization():
    score = evaluate("We have twenty dollars.", "we have $20.")
    assert score["wer"] == 0


def test_word_edits_are_counted_independently_of_punctuation():
    score = evaluate("Hello, dear world!", "Hello world.")
    assert score["deletions"] == 1
    assert score["reference_words"] == 3
    assert score["punctuation_matched"] == 0
    assert score["punctuation_reference_count"] == 2
    assert score["punctuation_hypothesis_count"] == 1


def test_decimal_is_not_punctuation_and_marks_follow_aligned_words():
    result = punctuation_scores("Revenue was 1.25 million, correct?", "Revenue was 1.25 million, correct?")
    assert result["punctuation_reference_count"] == 2
    assert result["punctuation_f1"] == 1


def test_no_punctuation_claim_for_unpunctuated_dataset():
    score = evaluate("HELLO WORLD", "Hello, world!", False)
    assert score["wer"] == 0
    assert "punctuation_f1" not in score


def test_non_speech_events_are_not_scored_as_words():
    assert evaluate("Hello <inaudible> world [laughter].", "Hello world.")["wer"] == 0
