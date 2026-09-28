"""English WER and transparent, word-aligned punctuation metrics."""
from __future__ import annotations

import re
import unicodedata

import jiwer
from whisper.normalizers import EnglishTextNormalizer

NORMALIZER = EnglishTextNormalizer()
PUNCTUATION = set(".,!?;:")


def remove_events(text: str) -> str:
    return re.sub(r"<[^>]*>|\[[^\]]*\]", " ", unicodedata.normalize("NFKC", text))


def normalized(text: str) -> str:
    return NORMALIZER(remove_events(text)).strip()


def word_marks(text: str):
    words, marks = [], []
    # Decimal points stay inside numeric tokens. Apostrophes stay inside words.
    for token in re.findall(r"[\w]+(?:['’][\w]+|[.][0-9]+)*|[.,!?;:]", remove_events(text)):
        if token in PUNCTUATION:
            if marks:
                marks[-1] += token
        else:
            words.append(token.casefold().replace("’", "'"))
            marks.append("")
    return words, marks


def punctuation_scores(reference: str, hypothesis: str) -> dict:
    rw, rp = word_marks(reference)
    hw, hp = word_marks(hypothesis)
    alignment = jiwer.process_words(" ".join(rw), " ".join(hw))
    matched = 0
    # Count exact punctuation symbols at aligned word boundaries; substitutions
    # can preserve a boundary, but unrelated inserted/deleted words cannot.
    for chunk in alignment.alignments[0]:
        if chunk.type not in {"equal", "substitute"}:
            continue
        for r, h in zip(range(chunk.ref_start_idx, chunk.ref_end_idx), range(chunk.hyp_start_idx, chunk.hyp_end_idx)):
            left, right = list(rp[r]), list(hp[h])
            for symbol in left:
                if symbol in right:
                    matched += 1
                    right.remove(symbol)
    ref_count, hyp_count = sum(map(len, rp)), sum(map(len, hp))
    precision = matched / hyp_count if hyp_count else 0.0
    recall = matched / ref_count if ref_count else 0.0
    return {"punctuation_reference_count": ref_count, "punctuation_hypothesis_count": hyp_count,
            "punctuation_matched": matched, "punctuation_precision": precision, "punctuation_recall": recall,
            "punctuation_f1": 2 * matched / (ref_count + hyp_count) if ref_count + hyp_count else 1.0}


def evaluate(reference: str, hypothesis: str, punctuation_reference: bool = True) -> dict:
    ref, hyp = normalized(reference), normalized(hypothesis)
    score = jiwer.process_words(ref, hyp)
    result = {"wer": score.wer, "substitutions": score.substitutions, "deletions": score.deletions,
              "insertions": score.insertions, "hits": score.hits, "reference_words": len(ref.split()),
              "hypothesis_words": len(hyp.split()), "normalized_reference": ref, "normalized_hypothesis": hyp}
    if punctuation_reference:
        result.update(punctuation_scores(reference, hypothesis))
    return result
