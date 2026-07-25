"""Conservative answer normalization for ORena FOCUS inference.

Only the deterministically parsed answer formats are normalized:

- binary
- number
- percentage
- foreign-object class
- time

Open-ended, matching, and multiple-choice answers are left unchanged apart from
surrounding whitespace because they are evaluated by an LLM judge or an
unknown regular-expression matcher.

The function does not infer the answer format from the question. Instead, it
only changes an answer when the complete generated text clearly matches one of
the strict deterministic formats after removing terminal punctuation.
"""

from __future__ import annotations

import re
from typing import Sequence

from focus.foreign_objects import FOType


_TERMINAL_PUNCTUATION = re.compile(r"[.!?;:]+\s*$")
_BINARY_PATTERN = re.compile(r"^(yes|no)$", flags=re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"^\d+$")
_PERCENTAGE_PATTERN = re.compile(r"^\d+(?:\.\d+)?\s*%?$")
_TIMESTAMP_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}$")


def extract_fo_class_names(
    fo_definitions: str,
) -> tuple[str, ...]:
    """Extract class headings from the supplied FO definitions text."""

    if not isinstance(fo_definitions, str):
        raise TypeError("fo_definitions must be a string.")

    lines = [line.rstrip() for line in fo_definitions.splitlines()]
    names: list[str] = []

    for index in range(len(lines) - 1):
        heading = lines[index].strip()
        underline = lines[index + 1].strip()

        if heading and re.fullmatch(r"-{3,}", underline):
            names.append(heading)

    return tuple(dict.fromkeys(names))


def _remove_terminal_punctuation(text: str) -> str:
    """Remove final sentence punctuation only."""

    return _TERMINAL_PUNCTUATION.sub("", text.strip())


def _split_comma_list(text: str) -> list[str]:
    """Split a comma-separated answer into non-empty stripped items."""

    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


def _normalize_timestamp_list(
    text: str,
) -> str | None:
    """Normalize valid comma-separated HH:MM:SS timestamps."""

    items = _split_comma_list(text)
    if not items:
        return None

    for item in items:
        if not _TIMESTAMP_PATTERN.fullmatch(item):
            return None

        _, minute, second = (
            int(component)
            for component in item.split(":")
        )

        if not 0 <= minute < 60 or not 0 <= second < 60:
            return None

    return ", ".join(items)


def _normalize_fo_class_list(
    text: str,
    valid_names: Sequence[str],
) -> str | None:
    """Normalize canonical comma-separated FO class names."""

    items = _split_comma_list(text)
    if not items:
        return None

    if len(items) == 1 and items[0].casefold() == "none":
        return "none"

    if any(item.casefold() == "none" for item in items):
        return None

    canonical_by_casefold = {
        name.casefold(): name
        for name in valid_names
    }

    canonical_items: list[str] = []

    for item in items:
        canonical = canonical_by_casefold.get(item.casefold())
        if canonical is None:
            return None
        canonical_items.append(canonical)

    return ", ".join(dict.fromkeys(canonical_items))


def normalize_answer(
    raw_answer: str,
    *,
    fo_class_names: Sequence[str] | None = None,
) -> str:
    """Conservatively normalize one generated ORena FOCUS answer.

    The original response is returned unchanged unless removing terminal
    punctuation produces a complete binary, number, percentage, time, or
    foreign-object-class answer.
    """

    if not isinstance(raw_answer, str):
        raise TypeError("raw_answer must be a string.")

    original = raw_answer.strip()
    if not original:
        return original

    candidate = _remove_terminal_punctuation(original)

    binary_match = _BINARY_PATTERN.fullmatch(candidate)
    if binary_match:
        return binary_match.group(1).lower()

    if _NUMBER_PATTERN.fullmatch(candidate):
        return candidate

    if _PERCENTAGE_PATTERN.fullmatch(candidate):
        return re.sub(r"\s+%", "%", candidate)

    timestamp_answer = _normalize_timestamp_list(candidate)
    if timestamp_answer is not None:
        return timestamp_answer

    valid_names = tuple(
        fo_class_names
        if fo_class_names is not None
        else FOType.names()
    )
    fo_answer = _normalize_fo_class_list(
        candidate,
        valid_names,
    )
    if fo_answer is not None:
        return fo_answer

    return original


if __name__ == "__main__":
    tests = (
        ("yes.", "yes"),
        ("NO!", "no"),
        ("2.", "2"),
        ("4.17.", "4.17"),
        ("100%.", "100%"),
        ("00:35:51.", "00:35:51"),
        (
            "00:35:51, 00:36:27.",
            "00:35:51, 00:36:27",
        ),
        ("Specimen Bag.", "Specimen Bag"),
        ("clip, silicone loop.", "Clip, Silicone Loop"),
        ("none.", "none"),
        (
            "placed in specimen bag.",
            "placed in specimen bag.",
        ),
        (
            "top/left, top/right.",
            "top/left, top/right.",
        ),
        (
            "Clip, not closed and resting on tissue.",
            "Clip, not closed and resting on tissue.",
        ),
    )

    for raw, expected in tests:
        result = normalize_answer(raw)
        print(f"{raw!r} -> {result!r}")
        assert result == expected

    print("All tests passed.")
