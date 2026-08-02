"""Prompt utilities for ORena FOCUS FRAME inference.

The current query prompt uses only information available during challenge
inference:

- ``focus.Request``;
- the batch-specific ``FO_definitions.json``; and
- optional fixed text-only demonstrations.

Reference answers, answer-format labels, capability labels, OOD labels, and
other reference-side metadata are never used to construct the prompt.

FRAME requests contain one still image. Timestamp metadata is not separately
included in the model prompt because the supplied image is the complete visual
input.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from focus import Request

import re


@dataclass(frozen=True)
class FewShotExample:
    """One optional text-only FRAME training demonstration."""

    question: str
    answer: str

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("Few-shot question must not be empty.")

        if not self.answer.strip():
            raise ValueError("Few-shot answer must not be empty.")


# Keep the default empty for zero-shot inference. If few-shot prompting is
# enabled later, add only FRAME examples here. Do not reuse SEGMENT examples
# involving temporal localization, duration, ordering, or multi-frame tracking.
DEFAULT_FEW_SHOT_EXAMPLES: tuple[FewShotExample, ...] = ()


_SHARED_INSTRUCTIONS_BEFORE_DEFINITIONS = """\
You are given one laparoscopic image from a minimally invasive surgical procedure and one question about visible foreign objects.

Answer using only:
1. visual evidence in the supplied image;
2. explicit facts and constraints stated in the question;
3. the procedure type as supporting context; and
4. the supplied foreign-object definitions.\
"""


_SHARED_INSTRUCTIONS_AFTER_DEFINITIONS = """\
Apply the foreign-object definitions exactly. A separate foreign object being held by an instrument is still a foreign object. Count a partially visible object when it is sufficiently visible to identify.

Before answering, silently identify the visible foreign-object classes and distinct physical instances, then resolve only what the question asks.

Counting rules:
- Object-instance count means the number of distinct physical foreign objects.
- Object-class count means the number of distinct foreign-object classes.
- A count for a named class includes only visible instances of that class.
- Co-occurrence is yes only when both named classes are visible.
- All objects are of the same class only when exactly one distinct class is represented among the visible foreign objects.

For questions asking for every object's position, report every visible foreign-object instance separately, including multiple instances of the same class, and follow the requested structure exactly.
For occlusion questions, name the foreign object being occluded, not the instrument or anatomical structure causing the occlusion.
For grasping questions, answer yes only when the foreign object is visibly held or clamped by an instrument, not merely touching or lying beside it.

Determine the required answer format from the wording of the question and follow it exactly:
- Binary: exactly yes or no.
- Number: one non-negative integer only.
- Foreign-object class: canonical class name(s), comma-separated, or none.
- Multiple choice: only the selected option or options exactly as written in the question.
- Open-ended: a concise direct answer that fully addresses the question.

Return only the answer, with no explanation, reasoning, prefix, or label. Do not add terminal punctuation unless the requested answer is open-ended.\
"""

def load_fo_definitions(path: str | Path) -> str:
    """Load the JSON-encoded text stored in ``FO_definitions.json``."""

    definitions_path = Path(path).expanduser().resolve()

    if not definitions_path.is_file():
        raise FileNotFoundError(
            f"FO definitions file does not exist: {definitions_path}"
        )

    try:
        with definitions_path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"FO_definitions.json is not valid JSON: {definitions_path}"
        ) from error

    if not isinstance(value, str):
        raise ValueError(
            "FO_definitions.json must contain one JSON string."
        )

    definitions = value.strip()

    if not definitions:
        raise ValueError("FO_definitions.json is empty.")

    return definitions

def format_fo_definitions(fo_definitions: str) -> str:
    """Convert the underlined FO definitions into prompt-friendly text."""

    if not isinstance(fo_definitions, str):
        raise TypeError("fo_definitions must be a string.")

    lines = fo_definitions.strip().splitlines()

    definition_heading = "Foreign Object (FO) Definition"
    classes_heading = "Foreign Object Classes"

    try:
        definition_index = lines.index(definition_heading)
        classes_index = lines.index(classes_heading)
    except ValueError as error:
        raise ValueError(
            "FO definitions do not contain the expected section headings."
        ) from error

    if definition_index >= classes_index:
        raise ValueError(
            "FO definition sections appear in an unexpected order."
        )

    if (
        definition_index + 1 >= len(lines)
        or not re.fullmatch(
            r"={3,}",
            lines[definition_index + 1].strip(),
        )
    ):
        raise ValueError(
            "The general FO definition heading has no valid underline."
        )

    if (
        classes_index + 1 >= len(lines)
        or not re.fullmatch(
            r"={3,}",
            lines[classes_index + 1].strip(),
        )
    ):
        raise ValueError(
            "The FO classes heading has no valid underline."
        )

    general_lines = lines[
        definition_index + 2 : classes_index
    ]

    while general_lines and not general_lines[0].strip():
        general_lines.pop(0)

    while general_lines and not general_lines[-1].strip():
        general_lines.pop()

    if not general_lines:
        raise ValueError("The general FO definition is empty.")

    class_lines = lines[classes_index + 2 :]
    classes: list[tuple[str, list[str]]] = []
    index = 0

    while index < len(class_lines):
        if not class_lines[index].strip():
            index += 1
            continue

        class_name = class_lines[index].strip()

        if (
            index + 1 >= len(class_lines)
            or not re.fullmatch(
                r"-{3,}",
                class_lines[index + 1].strip(),
            )
        ):
            raise ValueError(
                "Expected an underlined FO class heading, found: "
                f"{class_lines[index]!r}"
            )

        index += 2
        description_lines: list[str] = []

        while index < len(class_lines):
            current = class_lines[index]

            is_next_heading = (
                current.strip()
                and index + 1 < len(class_lines)
                and re.fullmatch(
                    r"-{3,}",
                    class_lines[index + 1].strip(),
                )
            )

            if is_next_heading:
                break

            description_lines.append(current.rstrip())
            index += 1

        while (
            description_lines
            and not description_lines[0].strip()
        ):
            description_lines.pop(0)

        while (
            description_lines
            and not description_lines[-1].strip()
        ):
            description_lines.pop()

        if not description_lines:
            raise ValueError(
                f"FO class {class_name!r} has no definition."
            )

        classes.append(
            (class_name, description_lines)
        )

    if not classes:
        raise ValueError("No FO classes were found.")

    general_definition = " ".join(
        line.strip()
        for line in general_lines
        if line.strip()
    )

    output_lines = [
        "Foreign-object (FO) definitions:",
        general_definition,
        "Foreign Object Classes:",
    ]

    for class_name, description_lines in classes:
        class_definition = " ".join(
            line.strip()
            for line in description_lines
            if line.strip()
        )
        output_lines.append(f"{class_name}: {class_definition}")

    return "\n".join(output_lines)




def format_few_shot_examples(
    examples: Sequence[FewShotExample],
) -> str:
    """Format demonstrations as repeated ``Question`` / ``Answer`` blocks."""

    blocks = [
        "\n".join(
            [
                f"Question: {example.question.strip()}",
                f"Answer: {example.answer.strip()}",
            ]
        )
        for example in examples
    ]

    return "\n\n".join(blocks)


def build_shared_prompt(
    fo_definitions: str,
    few_shot_examples: Sequence[
        FewShotExample
    ] = DEFAULT_FEW_SHOT_EXAMPLES,
) -> str:
    """Build the prompt portion reusable for every request in one batch.

    ``FO_definitions.json`` must still be read once per container run because
    the available classes and definitions may differ between batches.
    """

    definitions = fo_definitions.strip()

    if not definitions:
        raise ValueError("fo_definitions must not be empty.")

    formatted_definitions = format_fo_definitions(
        definitions
    )

    sections = [
        _SHARED_INSTRUCTIONS_BEFORE_DEFINITIONS,
        formatted_definitions,
        _SHARED_INSTRUCTIONS_AFTER_DEFINITIONS,
    ]

    if few_shot_examples:
        sections.append(
            "Examples:\n"
            + format_few_shot_examples(few_shot_examples)
        )

    return "\n\n".join(sections)


def build_request_prompt(request: Request) -> str:
    """Build the request-specific FRAME prompt."""

    question = request.question.strip()

    if not question:
        raise ValueError("request.question must not be empty.")

    procedure_type = str(request.procedure_type).strip()

    if not procedure_type:
        procedure_type = "Unknown"

    return "\n".join(
        [
            "Current request:",
            f"Procedure type: {procedure_type}",
            f"Question: {question}",
            "Answer:",
        ]
    )

def build_prompt(
    request: Request,
    shared_prompt: str,
) -> str:
    """Combine the reusable batch prompt with one FRAME request."""

    shared = shared_prompt.strip()

    if not shared:
        raise ValueError("shared_prompt must not be empty.")

    return shared + "\n\n" + build_request_prompt(request)


def main() -> None:
    """Run a local smoke test without command-line arguments."""

    project_dir = Path(__file__).resolve().parent

    definitions_path = (
        project_dir
        / "test"
        / "input"
        / "interface_1"
        / "FO_definitions.json"
    )

    definitions = load_fo_definitions(definitions_path)

    # Zero-shot by default.
    shared_prompt = build_shared_prompt(
        fo_definitions=definitions,
        few_shot_examples=(),
    )

    request = Request(
        qID="q0001",
        videoID="0029 - Heico - Sigma - 10.avi",
        start_time=1999.0,
        end_time=1999.0,
        procedure_type="Sigmoid Resection",
        question=(
            "Which combination of foreign object classes is visible "
            "in this frame? Please provide the class names or answer "
            "with none."
        ),
    )

    prompt = build_prompt(
        request=request,
        shared_prompt=shared_prompt,
    )

    print("=" * 80)
    print("ORena FOCUS FRAME prompt-utils smoke test")
    print("=" * 80)
    print(prompt)
    print("\n" + "=" * 80)
    print("Prompt length:", len(prompt), "characters")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()