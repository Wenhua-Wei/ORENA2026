"""Prompt utilities for ORena FOCUS SEGMENT few-shot inference.

The current query prompt uses only information available at challenge inference:
- ``focus.Request``;
- the batch-specific ``FO_definitions.json`` only for canonical FO class names; and
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
from focus.foreign_objects import FOType


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

_SHARED_INSTRUCTIONS = """\
You are a surgical assistant analysing endoscopic video from a minimally invasive surgical procedure. There are procedures like Proctocolectomy, Rectal Resection, Sigmoid Resection, Laparoscopic Cholecystectomy, etc. as specified below under “Current request”. Answer the surgical question using the visual evidence in the provided frames, paying particular attention to foreign objects. A foreign object (FO) is an object that has been fully introduced into the patient’s body cavity and is no longer connected to the external environment. Each foreign object must be retrieved, intentionally left in place, or otherwise explicitly accounted for before the procedure ends. Examples include {fo_class_examples}. Standard surgical instruments that remain connected to the external environment, such as graspers, scissors, trocars, staplers, suction devices, and cameras, are not considered foreign objects.
Examples of typical procedure-related actions that may occur include, but are not limited to:
- Proctocolectomy: mesenteric/tissue dissection, inferior mesenteric vessel clipping, rectal staple-line management and transection, specimen bagging, bowel approximation and anastomosis-related preparation, retraction, hemostasis, and drainage.
- Rectal Resection: mesenteric/blunt dissection, vascular clipping or ligation, bowel-lesion suturing, loop-ileostomy marking, specimen-bag extraction, retraction, and drain placement.
- Sigmoid Resection: inferior mesenteric artery/vein clipping, vascular division, colon mobilization and sigmoid-mesocolon dissection, colorectal anastomosis, air-leak testing, suture reinforcement or anastomotic revision, drain placement, and hemostasis.
- Laparoscopic Cholecystectomy: Critical View of Safety preparation, cystic duct and artery clipping and division, gallbladder dissection from the liver/cystic plate, specimen bagging and retrieval, gallstone retrieval, cholangiography when applicable, retraction, and hemostasis.

{frame_sampling_description} For timestamp questions, return the absolute original-procedure time shown in the top-left corner of the frames, not the time elapsed since the beginning of the trimmed clip. For questions asking how long something lasts, return the elapsed duration as the end time minus the start time, in HH:MM:SS format.

Determine the required answer format strictly from the wording of the question and follow the corresponding rule, as illustrated by the examples below. Return only the answer, with no explanation or prefix.

- Binary:
Q: Does a Specimen leave the field of view for at least 3 seconds and re-enter later? Please answer with yes or no.
A: no

- Number:
Q: In total, how many distinct Sponges are visible in this video? Please provide a number.
A: 2

- Percentage:
Q: In %, how many of the frames of this video contain a Clip? Please provide the answer in the format xx%.
A: 28.33

- Foreign-object class:
Q: Which surgical foreign object is inserted or created (in case of a specimen) in the abdominal cavity in this video? Please provide a class name or answer with none.
A: Silicone loop

- Time:
Q: During the video, at what time is the 1st visible Clip inserted in the abdomen for the first time? Please provide the answer in hh:mm:ss.
A: 00:18:17

- Duration:
Q: In total, how long was the Specimen bag visible in this video? Please provide the duration in HH:MM:SS.
A: 00:00:43

- Multiple choice:
Q: This video contains one Specimen bag. When it first becomes visible, in which quadrant of the frame is its center located relative to the image center? Please select one answer: top/left; top/right; bottom/left; bottom/right
A: top/right

- Open-ended:
Q: What problem is encountered removing this specimen bag from the abdomen? Please provide a single surgical challenge encountered during specimen bag removal.
A: The bag is too big for the incision.
"""



def load_fo_definitions(path: str | Path) -> str:
    """Load FO definitions, allowing an empty file or a JSON string."""

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
            "FO_definitions.json must be empty or contain one "
            "valid JSON string."
        ) from error

    # Optionally accept JSON null as empty.
    if value is None:
        return ""

    if not isinstance(value, str):
        raise ValueError(
            "FO_definitions.json must be empty or contain one JSON string."
        )

    return value.strip()


def extract_fo_class_names(fo_definitions: str) -> tuple[str, ...]:
    """Extract foreign-object class headings from the supplied definitions text.

    The submission-template FO definitions use reStructuredText-style class headings,
    for example::

        Sponge
        ------

    Only headings underlined with hyphens are treated as class names. Section
    headings underlined with equals signs are therefore ignored automatically.
    """

    if not isinstance(fo_definitions, str):
        raise TypeError("fo_definitions must be a string.")

    lines = [line.rstrip() for line in fo_definitions.splitlines()]
    names: list[str] = []

    for index in range(len(lines) - 1):
        heading = lines[index].strip()
        underline = lines[index + 1].strip()

        if heading and underline and set(underline) == {"-"} and len(underline) >= 1:
            names.append(heading)

    return tuple(dict.fromkeys(names))


def resolve_fo_class_names(fo_definitions: str) -> tuple[str, ...]:
    """Return class names from ``FO_definitions.json`` with a safe fallback.

    During local experiments ``FO_definitions.json`` may be empty. In that case,
    use the canonical class registry from the installed ``orena-focus`` package.
    The full definition text is never inserted into the prompt.
    """

    extracted = extract_fo_class_names(fo_definitions)
    if extracted:
        return extracted

    return tuple(FOType.names())


def format_fo_class_examples(class_names: Sequence[str]) -> str:
    """Format canonical FO class names as a compact natural-language list."""

    names = [str(name).strip() for name in class_names if str(name).strip()]

    if not names:
        raise ValueError("At least one foreign-object class name is required.")
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"

    return ", ".join(names[:-1]) + f", and {names[-1]}"


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

    ``FO_definitions.json`` is used only to obtain the canonical foreign-object
    class names. Its full definition text is deliberately not appended to the
    model prompt.
    """

    fo_class_names = resolve_fo_class_names(fo_definitions)
    shared_instructions = _SHARED_INSTRUCTIONS.replace(
        "{fo_class_examples}",
        format_fo_class_examples(fo_class_names),
    )

    sections = [shared_instructions.strip()]

    if few_shot_examples:
        sections.append(
            "Examples:\n"
            + format_few_shot_examples(few_shot_examples)
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
#            (
#                "Original procedure time window: "
#                f"{seconds_to_timestamp(request.start_time)} to "
#                f"{seconds_to_timestamp(request.end_time)}"
#            ),
            f"Question: {request.question.strip()}",
            "Answer:",
        ]
    )


def build_prompt(
    request: Request,
    shared_prompt: str,
    num_frames: int,
    target_fps: float,
    max_frames: int,
) -> str:
    """Combine the reusable prompt with request-specific information."""

    shared = shared_prompt.strip()

    if not shared:
        raise ValueError("shared_prompt must not be empty.")

    duration = float(request.end_time) - float(request.start_time)

    if num_frames >= max_frames and duration > max_frames / target_fps:
        frame_sampling_description = (
            f"The input consists of {num_frames} chronologically ordered "
            f"frames uniformly selected across the {duration:g}-second "
            f"trimmed video segment after sampling at {target_fps:g} fps."
        )
    else:
        frame_sampling_description = (
            f"The input consists of {num_frames} chronologically ordered "
            f"frames sampled at {target_fps:g} fps from the "
            f"{duration:g}-second trimmed video segment."
        )

    shared = shared.replace(
        "{frame_sampling_description}",
        frame_sampling_description,
    )

    return shared + "\n\n" + build_request_prompt(request)


if __name__ == "__main__":
    project_root = Path(
        "/raid2/compass/ORENA2026/orena-docker/segment-algorithm"
    )

    definitions_path = (
        project_root
        / "test/input/interface_1"
        / "FO_definitions.json"
    )

    definitions = load_fo_definitions(definitions_path)
    fo_class_names = resolve_fo_class_names(definitions)

    print("FO classes used in prompt:")
    print("  " + ", ".join(fo_class_names))
    print()

    shared_prompt = build_shared_prompt(
        definitions,
        few_shot_examples=(),
    )

    request = Request(
        qID="q001",
        videoID="example-video-01",
        start_time=132.5,
        end_time=143.5,
        procedure_type="laparoscopic cholecystectomy",
        question="Is a foreign object visible in the scene?",
    )

    target_fps = 1.0
    max_frames = 20

    duration = request.end_time - request.start_time
    num_frames = min(
        max_frames,
        max(1, int(duration * target_fps)),
    )

    print(
        build_prompt(
            request=request,
            shared_prompt=shared_prompt,
            num_frames=num_frames,
            target_fps=target_fps,
            max_frames=max_frames,
        )
    )
