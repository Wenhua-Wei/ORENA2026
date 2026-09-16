"""Prompt utilities for ORena FOCUS PROCEDURE inference.

The prompt is deliberately built only from information available at challenge
inference time:

- ``focus.Request``;
- the batch-specific ``FO_definitions.json`` only for canonical FO class names;
- the frame-sampling description produced by PROCEDURE ``video_utils.py``; and
- optional fixed text-only demonstrations selected from training data.

The current query's reference answer, answer format, capability labels, OOD
status, clinical-relevance flag, and other reference-side metadata are never
used to build the prompt.

Compared with the SEGMENT prompt, the main difference is that PROCEDURE frames
may be selected using different temporal sampling intents (global, local anchor,
bounded interval, forward search, backward search, prefix-state, etc.). The
prompt therefore receives the actual sampling description from ``video_utils``
instead of reconstructing or assuming uniform sampling.
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


# These demonstrations are only used when inference.py explicitly enables
# few-shot prompting. The current planned PROCEDURE configuration can keep
# USE_FEW_SHOT_EXAMPLES=False, matching the working SEGMENT submission.
DEFAULT_FEW_SHOT_EXAMPLES: tuple[FewShotExample, ...] = (
    FewShotExample(
        question=(
            "Which surgical foreign object is inserted or created "
            "(in case of a specimen) in the abdominal cavity in this video?"
        ),
        answer="Silicone loop",
    ),
    FewShotExample(
        question=(
            "This video contains one External drain. In which quadrant of the "
            "frame is the center of the External drain located for most of the "
            "time, relative to the image center? Please select one answer: "
            "top/left; top/right; bottom/left; bottom/right"
        ),
        answer="top/left",
    ),
    FewShotExample(
        question=(
            "After the first Silicone loop was inserted in the abdomen in this "
            "video, which other foreign object classes are visible in the video?"
        ),
        answer="Clip, External drain",
    ),
    FewShotExample(
        question=(
            "At which time points were clips applied to somewhere in the "
            "abdomen? Please return the time points chronologically ordered in "
            "the format hh:mm:ss separated by a comma for each individual clip "
            "instance."
        ),
        answer="01:43:51, 01:43:54",
    ),
    FewShotExample(
        question=(
            "Does a Sponge leave the field of view for at least 3 seconds and "
            "re-enter later?"
        ),
        answer="no",
    ),
    FewShotExample(
        question=(
            "After the first Clip was inserted in the abdomen in this video, "
            "which other foreign object classes are visible in the video?"
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
            "Why is suction being used despite the sponge not being soaked? "
            "Please provide a single reason without additional explanation."
        ),
        answer=(
            "Blood pooling in the small pelvis; sponge not suitable for clearing."
        ),
    ),
    FewShotExample(
        question="What purpose does the sponge serve in this video segment?",
        answer="Stabilization.",
    ),
    FewShotExample(
        question=(
            "How many different foreign object classes do you see in this video?"
        ),
        answer="2",
    ),
    FewShotExample(
        question="List all foreign objects that are visible in this video frame.",
        answer="Specimen, Specimen bag",
    ),
)


_SHARED_INSTRUCTIONS = """\
You are a surgical assistant analysing endoscopic video from a minimally invasive surgical procedure. The supplied frames are chronologically ordered samples from the procedure-level video context available for the current question. There are procedures such as Proctocolectomy, Rectal Resection, Sigmoid Resection, Laparoscopic Cholecystectomy, etc., as specified below under "Current request".

Answer the surgical question using the visual evidence in the provided frames, paying particular attention to foreign objects and their history across time. A foreign object (FO) is an object that has been fully introduced into the patient's body cavity and is no longer connected to the external environment. Each foreign object must be retrieved, intentionally left in place, or otherwise explicitly accounted for before the procedure ends. Examples include {fo_class_examples}. Standard surgical instruments that remain connected to the external environment, such as graspers, scissors, trocars, staplers, suction devices, and cameras, are not considered foreign objects.

Examples of typical procedure-related actions that may occur include, but are not limited to:
- Proctocolectomy: mesenteric/tissue dissection, inferior mesenteric vessel clipping, rectal staple-line management and transection, specimen bagging, bowel approximation and anastomosis-related preparation, retraction, hemostasis, and drainage.
- Rectal Resection: mesenteric/blunt dissection, vascular clipping or ligation, bowel-lesion suturing, loop-ileostomy marking, specimen-bag extraction, retraction, and drain placement.
- Sigmoid Resection: inferior mesenteric artery/vein clipping, vascular division, colon mobilization and sigmoid-mesocolon dissection, colorectal anastomosis, air-leak testing, suture reinforcement or anastomotic revision, drain placement, and hemostasis.
- Laparoscopic Cholecystectomy: Critical View of Safety preparation, cystic duct and artery clipping and division, gallbladder dissection from the liver/cystic plate, specimen bagging and retrieval, gallstone retrieval, cholangiography when applicable, retraction, and hemostasis.

Frame sampling for this question: {frame_sampling_description}
For timestamp questions, return the absolute original-procedure time shown in the top-left corner of the frames. For questions asking how long something lasts, return elapsed duration as end time minus start time in HH:MM:SS format.

Determine the required answer format strictly from the wording of the question and follow the corresponding rule, as illustrated by the examples below. Return only the answer, with no explanation or prefix.

- Binary:
Q: For the surgical foreign object visible at 00:09:17: Does a retrieval of this object exist? Please answer with yes or no.
A: yes

- Number:
Q: What is the maximum number of Specimens appearing at once in a single frame? Please provide a number.
A: 1

- Percentage:
Q: In %, how many of the frames of this video contain a Clip? Please provide the answer in the format xx%.
A: 28.33

- Foreign-object class:
Q: Which foreign object classes appear in this video? Please provide the class name(s) or answer with none.
A: Clip

- Time:
Q: During the video, at what time is the 1st visible Clip inserted in the abdomen for the first time? Please provide the answer in hh:mm:ss.
A: 00:40:39

- Duration:
Q: For how long do you see at least one Clip in this video? Please provide the answer in hh:mm:ss.
A: 00:02:39

- Multiple choice:
Q: This video contains one Silicone loop. In which quadrant of the frame is the center of the Silicone loop located for most of the time, relative to the image center? Please select one answer: top/left; top/right; bottom/left; bottom/right
A: bottom/right

- Open-ended:
Q: Before applying the clip, which CVS precondition is not met regarding the hepatocystic triangle? Please provide a single CVS precondition status statement about the hepatocystic triangle (one short clause), without explanation.
A: The hepatocystic triangle is not fully cleared of fat and fibrous tissue.
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


def extract_fo_class_names(fo_definitions: str) -> tuple[str, ...]:
    """Extract foreign-object class headings from supplied definitions text.

    The submission-template FO definitions use reStructuredText-style class
    headings, for example::

        Sponge
        ------

    Only headings underlined with hyphens are treated as class names. Section
    headings underlined with equals signs are ignored automatically.
    """

    if not isinstance(fo_definitions, str):
        raise TypeError("fo_definitions must be a string.")

    lines = [line.rstrip() for line in fo_definitions.splitlines()]
    names: list[str] = []

    for index in range(len(lines) - 1):
        heading = lines[index].strip()
        underline = lines[index + 1].strip()

        if (
            heading
            and underline
            and set(underline) == {"-"}
            and len(underline) >= 1
        ):
            names.append(heading)

    return tuple(dict.fromkeys(names))


def resolve_fo_class_names(fo_definitions: str) -> tuple[str, ...]:
    """Return class names from ``FO_definitions.json`` with a safe fallback.

    During local experiments ``FO_definitions.json`` may be empty. In that case,
    use the canonical class registry from the installed ``orena-focus`` package.
    The full definitions text is deliberately not appended to the prompt.
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

    ``FO_definitions.json`` is used only to obtain canonical foreign-object
    class names. Its full definition text is deliberately not appended.

    The ``{frame_sampling_description}`` placeholder remains unresolved here
    because the sampling strategy is request-specific in the PROCEDURE track.
    ``build_prompt`` replaces it after ``video_utils.load_clip_frames`` has
    selected the actual frames.
    """

    fo_class_names = resolve_fo_class_names(fo_definitions)

    shared_instructions = _SHARED_INSTRUCTIONS.replace(
        "{fo_class_examples}",
        format_fo_class_examples(fo_class_names),
    )

    sections = [shared_instructions.strip()]

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

    # Keep the request layout close to the working SEGMENT prompt because the
    # current DoRA adapter was fine-tuned on SEGMENT data. The frames themselves
    # carry absolute timestamp labels, so the original time window does not need
    # to be repeated here.
    return "\n".join(
        [
            "Current request:",
            f"Procedure type: {request.procedure_type}",
            f"Question: {request.question.strip()}",
            "Answer:",
        ]
    )


def build_prompt(
    request: Request,
    shared_prompt: str,
    sampling_description: str,
) -> str:
    """Combine reusable instructions with request-specific sampling information.

    Parameters
    ----------
    request:
        Current challenge request.
    shared_prompt:
        Output of :func:`build_shared_prompt`.
    sampling_description:
        The exact description returned as
        ``ClipFrames.sampling_description`` by PROCEDURE ``video_utils.py``.
        This makes ``video_utils`` the single source of truth for how the visual
        evidence was selected and avoids falsely claiming that all PROCEDURE
        frames were sampled uniformly.
    """

    shared = shared_prompt.strip()
    if not shared:
        raise ValueError("shared_prompt must not be empty.")

    sampling = str(sampling_description).strip()
    if not sampling:
        raise ValueError("sampling_description must not be empty.")

    placeholder = "{frame_sampling_description}"
    placeholder_count = shared.count(placeholder)

    if placeholder_count != 1:
        raise RuntimeError(
            "Expected shared_prompt to contain exactly one "
            f"{placeholder!r} placeholder, found {placeholder_count}."
        )

    shared = shared.replace(placeholder, sampling)

    return shared + "\n\n" + build_request_prompt(request)


def _run_smoke_test() -> None:
    """Run lightweight prompt-building checks without loading a model."""

    # Avoid requiring a real FO_definitions.json for this standalone check.
    fo_definitions = ""
    fo_class_names = resolve_fo_class_names(fo_definitions)

    print("FO classes used in prompt:")
    print("  " + ", ".join(fo_class_names))
    print()

    shared_prompt = build_shared_prompt(
        fo_definitions,
        few_shot_examples=(),
    )

    request = Request(
        qID="q001",
        videoID="example-video-01",
        start_time=0.0,
        end_time=5400.0,
        procedure_type="Laparoscopic Cholecystectomy",
        question=(
            "For the surgical foreign object visible at 00:46:31: "
            "Does a retrieval of this object exist? Please answer with yes or no."
        ),
    )

    sampling_description = (
        "64 chronologically ordered frames combining dense sampling around "
        "00:46:31 with broad coverage from that point to the end of the supplied "
        "context."
    )

    prompt = build_prompt(
        request=request,
        shared_prompt=shared_prompt,
        sampling_description=sampling_description,
    )

    if "{frame_sampling_description}" in prompt:
        raise RuntimeError("Sampling placeholder was not replaced.")

    if sampling_description not in prompt:
        raise RuntimeError("Sampling description is missing from the prompt.")

    if request.question not in prompt:
        raise RuntimeError("Request question is missing from the prompt.")

    print(prompt)
    print("\nPrompt-utils smoke test passed.")


if __name__ == "__main__":
    _run_smoke_test()