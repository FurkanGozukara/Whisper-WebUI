from __future__ import annotations

from typing import Any


_MOJIBAKE_MARKERS = (
    "Ãƒ",
    "Ã‚",
    "Ã¢",
    "Ã°",
    "Ã…",
    "Ã¯",
    "Å“",
    "Å¸",
    "â‚¬",
    "â„¢",
    "Ã",
    "Â",
    "â",
    "ð",
    "ï¸",
    "Å",
    "œ",
    "Ÿ",
    "€",
    "™",
    "\ufffd",
)


def _looks_mojibake(value: str) -> bool:
    return any(marker in value for marker in _MOJIBAKE_MARKERS)


def _score_text(value: str) -> tuple[int, int]:
    penalty = 0
    for marker in _MOJIBAKE_MARKERS:
        penalty += value.count(marker) * 5
    penalty += sum(1 for ch in value if ord(ch) < 32 and ch not in "\n\r\t") * 10
    return penalty, len(value)


def repair_mojibake_text(value: Any) -> Any:
    if not isinstance(value, str) or not value or not _looks_mojibake(value):
        return value

    candidates = {value}
    frontier = {value}

    for _ in range(2):
        next_frontier = set()
        for item in frontier:
            for source_encoding in ("latin1", "cp1252"):
                try:
                    repaired = item.encode(source_encoding).decode("utf-8")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    continue
                if repaired not in candidates:
                    candidates.add(repaired)
                    next_frontier.add(repaired)
        frontier = next_frontier
        if not frontier:
            break

    best = min(candidates, key=_score_text)
    return best if _score_text(best) < _score_text(value) else value


def repair_mojibake_obj(value: Any) -> Any:
    if isinstance(value, str):
        return repair_mojibake_text(value)
    if isinstance(value, list):
        return [repair_mojibake_obj(item) for item in value]
    if isinstance(value, tuple):
        return tuple(repair_mojibake_obj(item) for item in value)
    if isinstance(value, dict):
        return {
            repair_mojibake_obj(key): repair_mojibake_obj(item)
            for key, item in value.items()
        }
    return value


def repair_component_text(component: Any) -> Any:
    for attribute in ("label", "info", "placeholder"):
        if hasattr(component, attribute):
            setattr(component, attribute, repair_mojibake_text(getattr(component, attribute)))

    if hasattr(component, "choices") and getattr(component, "choices") is not None:
        component.choices = repair_mojibake_obj(component.choices)

    return component


def repair_blocks_text(blocks: Any) -> Any:
    block_map = getattr(blocks, "blocks", blocks)
    if isinstance(block_map, dict):
        for component in block_map.values():
            repair_component_text(component)
    return blocks
