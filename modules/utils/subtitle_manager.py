# Ported from https://github.com/openai/whisper/blob/main/whisper/utils.py

import json
import csv
import os
import re
import sys
import zlib
from typing import Callable, List, Optional, TextIO, Union, Dict, Tuple
from datetime import datetime

from modules.whisper.data_classes import Segment, Word
from .files_manager import read_file


def format_timestamp(
    seconds: float, always_include_hours: bool = True, decimal_marker: str = ","
) -> str:
    assert seconds is not None and seconds >= 0, "Wrong timestamp provided"

    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def time_str_to_seconds(time_str: str, decimal_marker: str = ",") -> float:
    # Both decimal markers are accepted (subtitles from other tools mix them), and a missing fraction too.
    # decimal_marker is kept for callers that still pass it.
    parts = time_str.strip().replace(",", ".").split(":")

    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    else:
        hours, minutes, seconds = 0, 0, parts[0]

    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


CUE_TIMING_RE = re.compile(r"^\s*(\S+)\s*-->\s*(\S+)")
MARKUP_TAG_RE = re.compile(r"<[^>]*>")
LRC_TIME_TAG_RE = re.compile(r"\[(\d+:\d+(?:[.,]\d+)?)\]")


def _read_subtitle_text(file_path: str) -> str:
    return read_file(file_path).lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")


CUE_BLOCK_KEYWORD_RE = re.compile(r"^(?:NOTE|STYLE|REGION)(?:\s|$)")


def _cue_timing(line: str) -> Optional[Tuple[float, float]]:
    if "-->" not in line:
        return None
    match = CUE_TIMING_RE.match(line)
    if not match:
        return None
    try:
        return time_str_to_seconds(match.group(1)), time_str_to_seconds(match.group(2))
    except ValueError:
        return None


def _parse_cue_blocks(file_path: str, strip_markup: bool = False) -> List[Segment]:
    """Cues of an SRT or WebVTT file, also from other tools (YouTube, subtitle editors).

    Accepts a byte order mark, cue numbers or ids, cue settings after the end time ("align:start"),
    NOTE/STYLE/REGION blocks, "." or "," decimals and separator lines that contain spaces. A cue's text runs
    to the next cue: YouTube captions have lines with a single space inside a cue, which ended the cue before
    and dropped its words.
    """
    lines = _read_subtitle_text(file_path).split("\n")
    segments = []
    current = None
    current_last_text_index = None
    skipping_block = False
    for index, line in enumerate(lines):
        timing = _cue_timing(line)
        if timing is not None:
            if current is not None:
                # A numbered cue, or an identifier after a blank separator, belongs to the new cue.
                # Adjacent timing lines without a separator are also common: their preceding speech
                # must not be mistaken for an identifier and silently dropped.
                previous_is_id = index > 0 and (
                    lines[index - 1].strip().isdecimal()
                    or (index > 1 and not lines[index - 2].strip())
                )
                if current_last_text_index == index - 1 and current["lines"] and previous_is_id:
                    current["lines"].pop()
                segments.append(current)
            current = {"start": timing[0], "end": timing[1], "lines": []}
            current_last_text_index = None
            skipping_block = False
            continue

        stripped = line.strip()
        if not stripped:
            if not line:
                skipping_block = False  # an empty line ends a NOTE, STYLE or REGION block
            continue
        if skipping_block:
            continue
        if (index == 0 or not lines[index - 1].strip()) and CUE_BLOCK_KEYWORD_RE.match(stripped):
            skipping_block = True
            continue
        if current is not None:
            current["lines"].append(stripped)
            current_last_text_index = index

    if current is not None:
        segments.append(current)

    result = []
    for cue in segments:
        sentence = " ".join(cue["lines"])
        if strip_markup:
            sentence = re.sub(r"\s+", " ", MARKUP_TAG_RE.sub("", sentence)).strip()
        result.append(Segment(start=cue["start"], end=cue["end"], text=sentence))
    return result


def get_start(segments: List[dict]) -> Optional[float]:
    return next(
        (w["start"] for s in segments for w in s["words"]),
        segments[0]["start"] if segments else None,
    )


def get_end(segments: List[dict]) -> Optional[float]:
    return next(
        (w["end"] for s in reversed(segments) for w in reversed(s["words"])),
        segments[-1]["end"] if segments else None,
    )


SENTENCE_END_RE = re.compile(r"[.!?。！？]+[\"')\]}」』）》]*$")
CLAUSE_END_RE = re.compile(r"[,;:，、；：]+[\"')\]}」』）》]*$")
# Scripts written without spaces between words (Chinese, Japanese, Thai, Lao, Khmer, Myanmar)
NO_SPACE_SCRIPT_RE = re.compile(
    "[฀-໿က-႟ក-៿　-ヿ㐀-䶿一-鿿豈-﫿＀-￯]"
)
ABBREVIATION_RE = re.compile(
    r"^(?:[A-Za-z]\.){2,}$|^(?:Mr|Mrs|Ms|Dr|Prof|Sen|Rep|Gov|St|No|Jr|Sr|Inc|Ltd)\.$",
    re.IGNORECASE,
)
NORMALIZED_SUBTITLE_MAX_CHARS = 92
NORMALIZED_SUBTITLE_MAX_WORDS = 24
NORMALIZED_SUBTITLE_MAX_DURATION = 8.0


def _word_text(word: dict) -> str:
    return str(word.get("word") or "")


def _join_word_text(words: List[dict]) -> str:
    raw = "".join(_word_text(word) for word in words).strip()
    # Words without separating spaces get spaces, except in scripts that are written without them:
    # "今日 は いい 天気" was written for "今日はいい天気".
    if " " not in raw and len(words) > 1 and not NO_SPACE_SCRIPT_RE.search(raw):
        raw = " ".join(_word_text(word).strip() for word in words if _word_text(word).strip())
    return re.sub(r"\s+", " ", raw).replace("-->", "->").strip()


def _is_sentence_end(word_text: str) -> bool:
    stripped = word_text.strip()
    if not stripped or ABBREVIATION_RE.match(stripped):
        return False
    return bool(SENTENCE_END_RE.search(stripped))


def _is_clause_end(word_text: str) -> bool:
    stripped = word_text.strip()
    return bool(stripped and CLAUSE_END_RE.search(stripped))


def _safe_word_time(word: dict, key: str, fallback: Optional[float]) -> Optional[float]:
    value = word.get(key)
    if value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _segment_from_words(words: List[dict], fallback_start: Optional[float] = None, fallback_end: Optional[float] = None) -> Optional[dict]:
    text = _join_word_text(words)
    if not text:
        return None

    start = _safe_word_time(words[0], "start", fallback_start)
    end = _safe_word_time(words[-1], "end", fallback_end)
    if start is None or end is None:
        return None

    return {
        "start": start,
        "end": max(start, end),
        "text": text,
    }


def normalize_result_for_segment_subtitles(result: Union[dict, List[Segment]]) -> Union[dict, List[Segment]]:
    """Convert word-timestamp results into sentence-aware segment subtitles without word-level data."""
    if isinstance(result, list) and result and isinstance(result[0], Segment):
        result = {"segments": [seg.model_dump() for seg in result]}
    if not isinstance(result, dict):
        return result

    segments = result.get("segments")
    if not isinstance(segments, list) or not any(segment.get("words") for segment in segments if isinstance(segment, dict)):
        return result

    normalized_segments: List[dict] = []
    current_words: List[dict] = []
    current_fallback_start: Optional[float] = None
    current_fallback_end: Optional[float] = None
    last_soft_break_index: Optional[int] = None
    # diarized results: a subtitle holds one speaker's words and is labelled like the segment text
    current_speaker: Optional[str] = None

    def reset_soft_break() -> None:
        nonlocal last_soft_break_index
        last_soft_break_index = None
        for index, current_word in enumerate(current_words, start=1):
            if _is_sentence_end(_word_text(current_word)) or _is_clause_end(_word_text(current_word)):
                last_soft_break_index = index

    def flush(split_at: Optional[int] = None) -> None:
        nonlocal current_words, current_fallback_start, current_fallback_end, last_soft_break_index

        if not current_words:
            return

        split_at = split_at or len(current_words)
        emitting = current_words[:split_at]
        remaining = current_words[split_at:]
        segment = _segment_from_words(emitting, current_fallback_start, current_fallback_end)
        if segment is not None:
            if current_speaker:
                segment["text"] = f"{current_speaker}|{segment['text']}"
                segment["speaker"] = current_speaker
            normalized_segments.append(segment)

        current_words = remaining
        current_fallback_start = _safe_word_time(current_words[0], "start", None) if current_words else None
        current_fallback_end = _safe_word_time(current_words[-1], "end", None) if current_words else None
        reset_soft_break()

    def append_plain_segment(segment: dict) -> None:
        text = str(segment.get("text") or "").strip().replace("-->", "->")
        if not text:
            return
        normalized_segments.append({
            "start": segment.get("start", 0.0),
            "end": segment.get("end", segment.get("start", 0.0)),
            "text": re.sub(r"\s+", " ", text),
        })

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        words = segment.get("words") or []
        if not words:
            flush()
            append_plain_segment(segment)
            continue

        speaker = segment.get("speaker")
        if speaker != current_speaker:
            flush()
            current_speaker = speaker

        for word in words:
            if not isinstance(word, dict) or not _word_text(word).strip():
                continue
            if not current_words:
                current_fallback_start = segment.get("start")
            current_words.append(word)
            current_fallback_end = segment.get("end")

            if _is_sentence_end(_word_text(word)):
                flush()
                continue

            if _is_clause_end(_word_text(word)):
                last_soft_break_index = len(current_words)

            text = _join_word_text(current_words)
            duration = (
                (_safe_word_time(current_words[-1], "end", current_fallback_end) or 0.0)
                - (_safe_word_time(current_words[0], "start", current_fallback_start) or 0.0)
            )
            too_long = (
                len(text) >= NORMALIZED_SUBTITLE_MAX_CHARS
                or len(current_words) >= NORMALIZED_SUBTITLE_MAX_WORDS
                or duration >= NORMALIZED_SUBTITLE_MAX_DURATION
            )
            if too_long:
                split_at = last_soft_break_index if last_soft_break_index and last_soft_break_index >= 4 else len(current_words)
                flush(split_at)

    flush()

    normalized = dict(result)
    normalized["segments"] = normalized_segments
    normalized["text"] = " ".join(segment["text"] for segment in normalized_segments)
    return normalized


class ResultWriter:
    extension: str

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def __call__(
        self, result: Union[dict, List[Segment]], output_file_name: str,
            options: Optional[dict] = None, **kwargs
    ):
        if isinstance(result, list) and (not result or isinstance(result[0], Segment)):
            result = {"segments": [seg.model_dump() for seg in result]}

        output_path = os.path.join(
            self.output_dir, output_file_name + "." + self.extension
        )

        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, file=f, options=options, **kwargs)

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        raise NotImplementedError

    def to_segments(self, file_path: str):
        raise NotImplementedError


class WriteTXT(ResultWriter):
    extension: str = "txt"

    def write_result(
        self, result: Union[Dict, List[Segment]], file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        for segment in result["segments"]:
            # A file without speech gives one empty placeholder segment (text None)
            if segment.get("text") is None:
                continue
            print(segment["text"].strip(), file=file, flush=True)

    def to_segments(self, file_path: str):
        segments = []

        blocks = _read_subtitle_text(file_path).splitlines()

        for block in blocks:
            segments.append(Segment(
                start=None,
                end=None,
                text=block
            ))
        return segments


class SubtitlesWriter(ResultWriter):
    always_include_hours: bool
    decimal_marker: str

    def iterate_result(
        self,
        result: dict,
        options: Optional[dict] = None,
        *,
        max_line_width: Optional[int] = None,
        max_line_count: Optional[int] = None,
        highlight_words: bool = False,
        align_lrc_words: bool = False,
        max_words_per_line: Optional[int] = None,
    ):
        options = options or {}
        max_line_width = max_line_width or options.get("max_line_width")
        max_line_count = max_line_count or options.get("max_line_count")
        highlight_words = highlight_words or options.get("highlight_words", False)
        align_lrc_words = align_lrc_words or options.get("align_lrc_words", False)
        max_words_per_line = max_words_per_line or options.get("max_words_per_line")
        preserve_segments = max_line_count is None or max_line_width is None
        max_line_width = max_line_width or 1000
        max_words_per_line = max_words_per_line or 1000

        def iterate_subtitles():
            line_len = 0
            line_count = 1
            # the next subtitle to yield (a list of word timings with whitespace)
            subtitle: List[dict] = []
            # the speaker of that subtitle; a diarized result labels its segments, and a subtitle holds one speaker
            subtitle_speaker = None
            last: float = get_start(result["segments"]) or 0.0
            for segment in result["segments"]:
                speaker = segment.get("speaker")
                chunk_index = 0
                words_count = max_words_per_line
                while chunk_index < len(segment["words"]):
                    remaining_words = len(segment["words"]) - chunk_index
                    if max_words_per_line > len(segment["words"]) - chunk_index:
                        words_count = remaining_words
                    for i, original_timing in enumerate(
                        segment["words"][chunk_index : chunk_index + words_count]
                    ):
                        timing = original_timing.copy()
                        long_pause = (
                            not preserve_segments and timing["start"] - last > 3.0
                        )
                        has_room = line_len + len(timing["word"]) <= max_line_width
                        seg_break = i == 0 and len(subtitle) > 0 and preserve_segments
                        speaker_break = len(subtitle) > 0 and speaker != subtitle_speaker
                        if (
                            line_len > 0
                            and has_room
                            and not long_pause
                            and not seg_break
                            and not speaker_break
                        ):
                            # line continuation
                            line_len += len(timing["word"])
                        else:
                            # new line
                            timing["word"] = timing["word"].strip()
                            if (
                                len(subtitle) > 0
                                and max_line_count is not None
                                and (long_pause or line_count >= max_line_count)
                                or seg_break
                                or speaker_break
                            ):
                                # subtitle break
                                yield subtitle, subtitle_speaker
                                subtitle = []
                                line_count = 1
                            elif line_len > 0:
                                # line break
                                line_count += 1
                                timing["word"] = "\n" + timing["word"]
                            line_len = len(timing["word"].strip())
                        subtitle.append(timing)
                        subtitle_speaker = speaker
                        last = timing["start"]
                    chunk_index += max_words_per_line
            if len(subtitle) > 0:
                yield subtitle, subtitle_speaker

        if len(result["segments"]) > 0 and "words" in result["segments"][0] and result["segments"][0]["words"]:
            for subtitle, speaker in iterate_subtitles():
                # the same "SPEAKER_00|" label that the segment text of a diarized result carries
                label = f"{speaker}|" if speaker else ""
                subtitle_start = self.format_timestamp(subtitle[0]["start"])
                subtitle_end = self.format_timestamp(subtitle[-1]["end"])
                subtitle_text = label + "".join([word["word"] for word in subtitle])
                if highlight_words:
                    last = subtitle_start
                    all_words = [timing["word"] for timing in subtitle]
                    for i, this_word in enumerate(subtitle):
                        start = self.format_timestamp(this_word["start"])
                        end = self.format_timestamp(this_word["end"])
                        if last != start:
                            yield last, start, subtitle_text

                        yield start, end, label + "".join(
                            [
                                re.sub(r"^(\s*)(.*)$", r"\1<u>\2</u>", word)
                                if j == i
                                else word
                                for j, word in enumerate(all_words)
                            ]
                        )
                        last = end

                # elif: with an if here, highlight mode also wrote the plain cue, a second overlapping subtitle
                elif align_lrc_words:
                    word_texts = [sub["word"] for sub in subtitle]
                    word_texts[0] = label + word_texts[0]
                    lrc_aligned_words = [
                        f"[{self.format_timestamp(sub['start'])}]{text}" for sub, text in zip(subtitle, word_texts)
                    ]
                    l_start, l_end = self.format_timestamp(subtitle[-1]['start']), self.format_timestamp(subtitle[-1]['end'])
                    lrc_aligned_words[-1] = f"[{l_start}]{word_texts[-1]}[{l_end}]"
                    lrc_aligned_words = ' '.join(lrc_aligned_words)
                    yield None, None, lrc_aligned_words

                else:
                    yield subtitle_start, subtitle_end, subtitle_text
        else:
            for segment in result["segments"]:
                if segment["text"] is None or segment.get("start") is None or segment.get("end") is None:
                    continue

                segment_start = self.format_timestamp(segment["start"])
                segment_end = self.format_timestamp(segment["end"])
                segment_text = segment["text"].strip().replace("-->", "->")
                yield segment_start, segment_end, segment_text

    def format_timestamp(self, seconds: float):
        return format_timestamp(
            seconds=seconds,
            always_include_hours=self.always_include_hours,
            decimal_marker=self.decimal_marker,
        )


class WriteVTT(SubtitlesWriter):
    extension: str = "vtt"
    always_include_hours: bool = False
    decimal_marker: str = "."

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        print("WEBVTT\n", file=file)
        for start, end, text in self.iterate_result(result, options, **kwargs):
            print(f"{start} --> {end}\n{text}\n", file=file, flush=True)

    def to_segments(self, file_path: str) -> List[Segment]:
        # Inline tags (YouTube's <00:00:01.020><c> word</c>, <u> of highlighted words) are removed
        return _parse_cue_blocks(file_path, strip_markup=True)


class WriteSRT(SubtitlesWriter):
    extension: str = "srt"
    always_include_hours: bool = True
    decimal_marker: str = ","

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        for i, (start, end, text) in enumerate(
            self.iterate_result(result, options, **kwargs), start=1
        ):
            print(f"{i}\n{start} --> {end}\n{text}\n", file=file, flush=True)

    def to_segments(self, file_path: str) -> List[Segment]:
        return _parse_cue_blocks(file_path)


class WriteLRC(SubtitlesWriter):
    extension: str = "lrc"
    always_include_hours: bool = False
    decimal_marker: str = "."

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        for i, (start, end, text) in enumerate(
            self.iterate_result(result, options, **kwargs), start=1
        ):
            if "align_lrc_words" in kwargs and kwargs["align_lrc_words"]:
                print(f"{text}\n", file=file, flush=True)
            else:
                print(f"[{start}]{text}[{end}]\n", file=file, flush=True)

    def to_segments(self, file_path: str) -> List[Segment]:
        """One segment per line: "[start]text[end]", a word-level line ("[t]word [t]word[end]", which used to
        come back as its first word repeated) or a standard LRC line without an end tag."""
        entries = []
        for line in _read_subtitle_text(file_path).split("\n"):
            tags = list(LRC_TIME_TAG_RE.finditer(line))
            if not tags:
                continue  # blank line or a metadata tag such as [ar:...]
            text = " ".join(part.strip() for part in LRC_TIME_TAG_RE.split(line)[::2] if part.strip())
            times = [time_str_to_seconds(tag.group(1)) for tag in tags]
            end = times[-1] if len(times) > 1 and line.rstrip().endswith("]") else None
            entries.append([times[0], end, text])

        segments = []
        for index, (start, end, text) in enumerate(entries):
            if end is None:
                end = entries[index + 1][0] if index + 1 < len(entries) else start
            segments.append(Segment(start=start, end=max(start, end), text=text))
        return segments


class WriteTSV(ResultWriter):
    """
    Write a transcript to a file in TSV (tab-separated values) format containing lines like:
    <start time in integer milliseconds>\t<end time in integer milliseconds>\t<transcript text>

    Using integer milliseconds as start and end times means there's no chance of interference from
    an environment setting a language encoding that causes the decimal in a floating point number
    to appear as a comma; also is faster and more efficient to parse & store, e.g., in C++.
    """

    extension: str = "tsv"

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        print("start", "end", "text", sep="\t", file=file)
        for segment in result["segments"]:
            # A file without speech gives one empty placeholder segment (no text and no times)
            if segment.get("text") is None or segment.get("start") is None or segment.get("end") is None:
                continue
            print(round(1000 * segment["start"]), file=file, end="\t")
            print(round(1000 * segment["end"]), file=file, end="\t")
            print(re.sub(r"[\t\r\n]+", " ", segment["text"].strip()), file=file, flush=True)

    def to_segments(self, file_path: str) -> List[Segment]:
        rows = csv.DictReader(_read_subtitle_text(file_path).splitlines(), delimiter="\t")
        if not rows.fieldnames:
            return []
        if not {"start", "end", "text"}.issubset(rows.fieldnames):
            raise ValueError("TSV subtitles require start, end and text columns (times in milliseconds).")
        return [
            Segment(start=float(row["start"]) / 1000, end=float(row["end"]) / 1000, text=row["text"])
            for row in rows
        ]


class WriteJSON(ResultWriter):
    extension: str = "json"

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        json.dump(result, file, indent=2, ensure_ascii=False)

    def to_segments(self, file_path: str) -> List[Segment]:
        result = json.loads(_read_subtitle_text(file_path))
        segments = result.get("segments") if isinstance(result, dict) else result
        if not isinstance(segments, list):
            raise ValueError("JSON subtitles require a segments list.")
        return [Segment(**segment) for segment in segments]


def get_writer(
    output_format: str, output_dir: str
) -> Callable[[dict, TextIO, dict], None]:
    output_format = output_format.strip().lower().replace(".", "")

    writers = {
        "txt": WriteTXT,
        "vtt": WriteVTT,
        "srt": WriteSRT,
        "tsv": WriteTSV,
        "json": WriteJSON,
        "lrc": WriteLRC
    }

    if output_format == "all":
        all_writers = [writer(output_dir) for writer in writers.values()]

        def write_all(
            result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
        ):
            for writer in all_writers:
                writer(result, file, options, **kwargs)

        return write_all

    return writers[output_format](output_dir)


def generate_file(
    output_dir: str, output_file_name: str, output_format: str, result: Union[dict, List[Segment]],
    add_timestamp: bool = True, **kwargs
) -> Tuple[str, str]:
    output_format = output_format.strip().lower().replace(".", "")
    output_format = "vtt" if output_format == "webvtt" else output_format

    if add_timestamp:
        # Use microsecond precision to ensure unique filenames even for rapid processing
        timestamp = datetime.now().strftime("%m%d%H%M%S%f")
        output_file_name += f"-{timestamp}"

    file_path = os.path.join(output_dir, f"{output_file_name}.{output_format}")
    file_writer = get_writer(output_format=output_format, output_dir=output_dir)

    if isinstance(file_writer, WriteLRC) and kwargs.get("highlight_words", False):
        kwargs["highlight_words"], kwargs["align_lrc_words"] = False, True

    normalize_word_timestamps = bool(kwargs.pop("normalize_word_timestamps", False))
    if normalize_word_timestamps:
        result = normalize_result_for_segment_subtitles(result)

    file_writer(result=result, output_file_name=output_file_name, **kwargs)
    content = read_file(file_path)
    return content, file_path


def safe_filename(name):
    INVALID_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1f]'
    MAX_FILENAME_LENGTH = 200
    safe_name = re.sub(INVALID_FILENAME_CHARS, '_', name)
    safe_name = safe_name.rstrip(" .") or "untitled"
    # Windows device names remain reserved even with an extension, including on NTFS.
    if re.match(r"^(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³]|CONIN\$|CONOUT\$)(?:\.|$)", safe_name, re.IGNORECASE):
        safe_name = "_" + safe_name
    # Truncate the filename if it exceeds the max_length (MAX_FILENAME_LENGTH)
    if len(safe_name) > MAX_FILENAME_LENGTH:
        file_extension = safe_name.split('.')[-1]
        if len(file_extension) + 1 < MAX_FILENAME_LENGTH:
            truncated_name = safe_name[:MAX_FILENAME_LENGTH - len(file_extension) - 1]
            safe_name = truncated_name + '.' + file_extension
        else:
            safe_name = safe_name[:MAX_FILENAME_LENGTH]
    return safe_name.rstrip(" .") or "untitled"
