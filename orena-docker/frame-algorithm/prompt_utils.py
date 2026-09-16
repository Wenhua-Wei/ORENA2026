"""Prompt utilities for ORena SAVE FOCUS FRAME inference.

The current FRAME query prompt uses only information available during challenge
inference:

- ``focus.Request``;
- the batch-specific ``FO_definitions.json`` only to obtain canonical foreign-
  object class names; and
- optional fixed text-only FRAME demonstrations.

Reference answers, answer-format labels, capability labels, OOD labels, and
other reference-side metadata are never used to construct the prompt.

FRAME is a single-image task. No frame-sampling, duration, temporal-ordering,
or multi-frame tracking instructions are included here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from focus import Request
from focus.foreign_objects import FOType


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


# Zero-shot by default. If enabled later, use FRAME examples only.
DEFAULT_FEW_SHOT_EXAMPLES: tuple[FewShotExample, ...] = ()


_SHARED_INSTRUCTIONS = """\
You are a surgical assistant analysing one endoscopic image from a minimally invasive surgical procedure. There are procedures like Proctocolectomy, Rectal Resection, Sigmoid Resection, Laparoscopic Cholecystectomy, etc. as specified below under “Current request”. Answer the surgical question using the visual evidence in the provided image and the information explicitly stated in the question, paying particular attention to foreign objects. A foreign object (FO) is an object that has been fully introduced into the patient’s body cavity and is no longer connected to the external environment. Examples include {fo_class_examples}. Standard surgical instruments that remain connected to the external environment, such as graspers, scissors, trocars, staplers, suction devices, and cameras, are not considered foreign objects.
Examples of typical procedure-related actions and anatomy that may be relevant include, but are not limited to:
- Proctocolectomy: mesenteric/tissue dissection, inferior mesenteric vessel clipping, rectal staple-line management and transection, specimen bagging, bowel approximation and anastomosis-related preparation, retraction, hemostasis, and drainage.
- Rectal Resection: mesenteric/blunt dissection, vascular clipping or ligation, bowel-lesion suturing, loop-ileostomy marking, specimen-bag extraction, retraction, and drain placement.
- Sigmoid Resection: inferior mesenteric artery/vein clipping, vascular division, colon mobilization and sigmoid-mesocolon dissection, colorectal anastomosis, air-leak testing, suture reinforcement or anastomotic revision, drain placement, and hemostasis.
- Laparoscopic Cholecystectomy: Critical View of Safety preparation, cystic duct and artery clipping and division, gallbladder dissection from the liver/cystic plate, specimen bagging and retrieval, gallstone retrieval, cholangiography when applicable, retraction, and hemostasis.

Determine the required answer format strictly from the wording of the question and follow the corresponding rule, as illustrated by the examples below. Return only the answer, with no explanation or prefix.

- Binary:
Q: Do Needles and Sponges co-occur in this frame? Please answer with yes or no.
A: yes

- Number:
Q: How many Sponges appear in this frame? Please provide a number.
A: 2

- Foreign-object class:
Q: Which foreign object class is partially occluded by an instrument in this frame? Please provide the class name or answer none.
A: Sponge

- Multiple choice:
Q: Where is the center of the Specimen bag located relative to the image center in this frame? Please select one answer: top/left; top/right; bottom/left; bottom/right
A: top/right

- Open-ended:
Q: In which abdominal quadrant is the sponge located? Please provide an anatomical location.
A: left lower quadrant
"""


def load_fo_definitions(path: str | Path) -> str:
    """Load FO definitions, allowing an empty file, JSON null, or a JSON string."""

    definitions_path = Path(path).expanduser().resolve()

    if not definitions_path.is_file():
        raise FileNotFoundError(
            f"FO definitions file does not exist: {definitions_path}"
        )

    raw = definitions_path.read_text(encoding="utf-8").strip()

    # Accept a zero-byte or whitespace-only file.
    if not raw:
        return ""

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(
            "FO_definitions.json must be empty or contain one valid JSON string."
        ) from error

    # Optionally accept JSON null as empty.
    if value is None:
        return ""

    if not isinstance(value, str):
        raise ValueError(
            "FO_definitions.json must be empty or contain one JSON string."
        )

    return value.strip()


def extract_fo_class_names(
    fo_definitions: str,
) -> tuple[str, ...]:
    """Extract canonical FO class headings from the supplied definitions text.

    The submission-template definitions use reStructuredText-style headings,
    for example::

        Sponge
        ------

    Only headings underlined with hyphens are treated as class names. Section
    headings underlined with equals signs are ignored.
    """

    if not isinstance(fo_definitions, str):
        raise TypeError("fo_definitions must be a string.")

    lines = [
        line.rstrip()
        for line in fo_definitions.splitlines()
    ]

    names: list[str] = []

    for index in range(len(lines) - 1):
        heading = lines[index].strip()
        underline = lines[index + 1].strip()

        if (
            heading
            and underline
            and set(underline) == {"-"}
        ):
            names.append(heading)

    return tuple(dict.fromkeys(names))


def resolve_fo_class_names(
    fo_definitions: str,
) -> tuple[str, ...]:
    """Return class names from FO definitions with a safe package fallback."""

    extracted = extract_fo_class_names(
        fo_definitions
    )

    if extracted:
        return extracted

    return tuple(FOType.names())


def format_fo_class_examples(
    class_names: Sequence[str],
) -> str:
    """Format canonical FO class names as a compact natural-language list."""

    names = [
        str(name).strip()
        for name in class_names
        if str(name).strip()
    ]

    if not names:
        raise ValueError(
            "At least one foreign-object class name is required."
        )

    if len(names) == 1:
        return names[0]

    if len(names) == 2:
        return f"{names[0]} and {names[1]}"

    return (
        ", ".join(names[:-1])
        + f", and {names[-1]}"
    )


def format_few_shot_examples(
    examples: Sequence[FewShotExample],
) -> str:
    """Format demonstrations as repeated Question/Answer pairs."""

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

    ``FO_definitions.json`` is used only to obtain the canonical foreign-object
    class names. The full definition text is deliberately not appended to the
    prompt, matching the current SEGMENT inference design.
    """

    fo_class_names = resolve_fo_class_names(
        fo_definitions
    )

    shared_instructions = (
        _SHARED_INSTRUCTIONS.replace(
            "{fo_class_examples}",
            format_fo_class_examples(
                fo_class_names
            ),
        )
    )

    sections = [
        shared_instructions.strip()
    ]

    if few_shot_examples:
        sections.append(
            "Examples:\n"
            + format_few_shot_examples(
                few_shot_examples
            )
        )

    return "\n\n".join(sections)


def build_request_prompt(
    request: Request,
) -> str:
    """Build the request-specific FRAME prompt."""

    question = request.question.strip()

    if not question:
        raise ValueError(
            "request.question must not be empty."
        )

    procedure_type = str(
        request.procedure_type
    ).strip()

    if not procedure_type:
        procedure_type = "Unknown"

    # FRAME requests normally have start_time == end_time. The timestamp is not
    # added separately because the supplied image is the complete visual input
    # and any relevant timepoint is already stated in the question itself.
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
        raise ValueError(
            "shared_prompt must not be empty."
        )

    return (
        shared
        + "\n\n"
        + build_request_prompt(request)
    )


def main() -> None:
    """Run a local FRAME prompt smoke test."""

    project_dir = Path(__file__).resolve().parent

    definitions_path = (
        project_dir
        / "test"
        / "input"
        / "interface_1"
        / "FO_definitions.json"
    )

    definitions = load_fo_definitions(
        definitions_path
    )

    fo_class_names = resolve_fo_class_names(
        definitions
    )

    print("FO classes used in prompt:")
    print("  " + ", ".join(fo_class_names))
    print()

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
    print(
        "Prompt length:",
        len(prompt),
        "characters",
    )
    print("Smoke test passed.")


if __name__ == "__main__":
    main()