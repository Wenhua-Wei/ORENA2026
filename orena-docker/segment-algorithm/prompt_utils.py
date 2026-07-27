"""Prompt utilities for ORena FOCUS SEGMENT few-shot inference.

The current query prompt uses only information available at challenge inference:
- ``focus.Request``;
- the batch-specific ``FO_definitions.json``; and
- fixed text-only demonstrations selected from training data.

The current query's answer, answer format, capability labels, OOD status, and
other reference-side metadata are never used to build the prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from focus import Request


@dataclass(frozen=True)
class FewShotExample:
    """One text-only training demonstration."""

    question: str
    answer: str

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("Few-shot question must not be empty.")
        if not self.answer.strip():
            raise ValueError("Few-shot answer must not be empty.")


DEFAULT_FEW_SHOT_EXAMPLES: tuple[FewShotExample, ...] = (
    FewShotExample(
        question=(
            "Which surgical foreign object is inserted or created (in case of a specimen) in the abdominal cavity in this video?"
        ),
        answer=(
            "Silicone loop"
        ),
    ),
    FewShotExample(
        question=(
            "This video contains one External drain. In which quadrant of the frame is the center of the External drain located for most of the time, relative to the image center? Please select one answer: top/left; top/right; bottom/left; bottom/right"
        ),
        answer="top/left",
    ),
    FewShotExample(
        question=(
            "After the first Silicone loop was inserted in the abdomen in this video, which other foreign object classes are visible in the video?"
        ),
        answer="Clip, External drain",
    ),
   FewShotExample(
       question=(
           "At which time points were clips applied to somewhere in the abdomen? Please return the time points chronologically ordered in the format hh:mm:ss separated by a comma for each individual clip instance."
       ),
       answer="01:43:51, 01:43:54",
   ),
    FewShotExample(
        question=(
            "Does a Sponge leave the field of view for at least 3 seconds and re-enter later?"
        ),
        answer="no",
    ),
    FewShotExample(
        question=(
            "After the first Clip was inserted in the abdomen in this video, which other foreign object classes are visible in the video?"
        ),
        answer="none",
    ),
    FewShotExample(
        question=(
            "Do Specimens and Specimen bags co-occur in any frame of this video?"
        ),
        answer="yes",
    ),
    FewShotExample(
        question="In %, how many of the frames of this video contain a Sponge?",
        answer="16.67",
    ),
    FewShotExample(
        question=(
            "Why is suction being used despite the sponge not being soaked? Please provide a single reason without additional explanation."
        ),
        answer="Blood pooling in the small pelvis; sponge not suitable for clearing.",
    ),

    FewShotExample(
        question=(
            "What purpose does the sponge serve in this video segment?"
        ),
        answer="Stabilization.",
    ),

    FewShotExample(
        question=(
            "How many different foreign object classes do you see in this video?"
        ),
        answer="2",
    ),

    FewShotExample(
        question=(
            "List all foreign objects that are visible in this video frame."
        ),
        answer="Specimen, Specimen bag",
    ),
)

# _SHARED_INSTRUCTIONS = """\
# You are a surgical assistant. You are given endoscopic video from a minimally \
# invasive procedure. Analyze the footage and answer the surgical question based \
# on the visual evidence.

# The supplied video is already trimmed to the requested time window. For \
# timestamp questions, return the absolute original-procedure time, not elapsed \
# clip time.

# Determine the required answer format from the wording of the question and \
# follow the corresponding rule:
# - Binary: exactly yes or no, without terminal punctuation.
# - Number: a non-negative integer only, without terminal punctuation.
# - Percentage: a non-negative number, without terminal punctuation.
# - Foreign-object class: canonical class name(s), comma-separated, or none, without terminal punctuation.
# - Time: HH:MM:SS timestamp(s), comma-separated, without terminal punctuation.
# - Multiple choice: only the selected option or options as written in the question, without terminal punctuation.
# - Open-ended: a concise direct answer that fully addresses the question.
# Return only the answer, with no explanation or prefix.\
# """

_SHARED_INSTRUCTIONS = """\
You are a surgical assistant. You are given endoscopic video from a minimally \
invasive procedure. Analyze the footage and answer the surgical question based \
on the visual evidence.

The supplied video is already trimmed to the requested time window. For \
timestamp questions, return the absolute original-procedure time, not elapsed \
clip time.

Determine the required answer format from the wording of the question and \
follow the corresponding rule:
- Binary: exactly yes or no.
- Number: a non-negative integer only.
- Percentage: a non-negative number.
- Foreign-object class: canonical class name(s), comma-separated, or none.
- Time: HH:MM:SS timestamp(s), comma-separated.
- Multiple choice: only the selected option or options as written in the question.
- Open-ended: a concise direct answer that fully addresses the question.
Return only the answer, with no explanation or prefix. Do not add terminal punctuation unless the answer format is open-ended.\
"""



def load_fo_definitions(path: str | Path) -> str:
    """Load the JSON-encoded string stored in ``FO_definitions.json``."""

    definitions_path = Path(path).expanduser().resolve()

    if not definitions_path.is_file():
        raise FileNotFoundError(
            f"FO definitions file does not exist: {definitions_path}"
        )

    with definitions_path.open("r", encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, str):
        raise ValueError(
            "FO_definitions.json must contain one JSON string."
        )

    definitions = value.strip()
    if not definitions:
        raise ValueError("FO_definitions.json is empty.")

    return definitions


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to ``HH:MM:SS`` using the floored procedure second."""

    if seconds < 0:
        raise ValueError("seconds must be non-negative.")

    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_few_shot_examples(
    examples: Sequence[FewShotExample],
) -> str:
    """Format demonstrations as repeated ``Question:`` / ``Answer:`` pairs."""

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
    few_shot_examples: Sequence[FewShotExample] = DEFAULT_FEW_SHOT_EXAMPLES,
) -> str:
    """Build the prompt portion reusable for every request in one batch.

    ``FO_definitions.json`` should still be read for every container run because
    the available foreign-object classes may differ between batches.
    """

    definitions = fo_definitions.strip()
    if not definitions:
        raise ValueError("fo_definitions must not be empty.")

    sections = [
        _SHARED_INSTRUCTIONS,
        definitions,
    ]

    if few_shot_examples:
        sections.append(
            "Examples:\n" + format_few_shot_examples(few_shot_examples)
        )

    return "\n\n".join(sections)


def build_request_prompt(request: Request) -> str:
    """Build the request-specific part of the prompt."""

    if not request.question.strip():
        raise ValueError("request.question must not be empty.")

    if request.end_time <= request.start_time:
        raise ValueError(
            "request.end_time must be greater than request.start_time."
        )

    return "\n".join(
        [
            "Current request:",
            f"Procedure type: {request.procedure_type}",
            (
                "Original procedure time window: "
                f"{seconds_to_timestamp(request.start_time)} to "
                f"{seconds_to_timestamp(request.end_time)}"
            ),
            f"Question: {request.question.strip()}",
            "Answer:",
        ]
    )


def build_prompt(
    request: Request,
    shared_prompt: str,
) -> str:
    """Combine the reusable batch prompt with one request-specific prompt."""

    shared = shared_prompt.strip()
    if not shared:
        raise ValueError("shared_prompt must not be empty.")

    return shared + "\n\n" + build_request_prompt(request)


if __name__ == "__main__":
    # smoke test. 
    project_root = Path(
        "/cs/student/projects1/aibh/2024/wenhuawe/ORENA"
    )
    definitions_path = (
        project_root
        / "few_shots_inference"
        / "FO_definitions.json"
    )

    definitions = load_fo_definitions(definitions_path)
    shared_prompt = build_shared_prompt(definitions)

    request = Request(
        qID="q001",
        videoID="example-video-01",
        start_time=132.5,
        end_time=143.5,
        procedure_type="laparoscopic cholecystectomy",
        question="Is a foreign object visible in the scene?",
    )

    print(build_prompt(request, shared_prompt))
