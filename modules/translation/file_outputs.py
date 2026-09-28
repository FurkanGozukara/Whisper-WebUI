"""Unique, portable subtitle names shared by local and API translators."""

import os

from modules.utils.subtitle_manager import safe_filename


def reserve_translation_name(source_path: str, used_names: set[str]) -> tuple[str, str]:
    stem, extension = os.path.splitext(os.path.basename(source_path))
    stem = safe_filename(stem)
    extension = extension.lower()
    output_extension = ".vtt" if extension == ".webvtt" else extension
    candidate = stem
    number = 1
    # Casefold also protects batches destined for a case-insensitive Windows filesystem.
    while f"{candidate}{output_extension}".casefold() in used_names:
        number += 1
        suffix = f"-{number}"
        candidate = f"{stem[:200 - len(suffix)]}{suffix}"
    used_names.add(f"{candidate}{output_extension}".casefold())
    return candidate, extension


def existing_translation_names(output_dir: str, add_timestamp: bool) -> set[str]:
    os.makedirs(output_dir, exist_ok=True)
    # Timestamped exports already have unique microsecond suffixes. Plain exports
    # preserve prior files too, including when inputs come from the output folder.
    return set() if add_timestamp else {name.casefold() for name in os.listdir(output_dir)}
