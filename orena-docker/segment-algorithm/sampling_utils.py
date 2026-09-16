#!/usr/bin/env python3
"""
Strict, precision-first sampling-intent router for ORena SAVE FOCUS SEGMENT.

Purpose
-------
Classify each request into exactly one of:

    GLOBAL
        Keep the existing full-segment uniform sampling policy.

    INTERVAL
        Use interval-specific sampling only for the single verified released
        template:
            "What types of foreign objects are seen between T1 and T2?
             Please provide the class name(s) or answer none."

    FORWARD
        Use forward-specific sampling only for either of the two verified
        released template families:
            1) "There is one <object> in the frame at T. When is it retrieved
                from the surgical site? Please provide the answer in hh:mm:ss."
            2) "Does the <object>, last visible just before T, re-appear later
                in the video? Please answer with yes or no."

Everything else falls back to GLOBAL.

Design principle
----------------
This router deliberately optimizes precision rather than recall. A false
negative is acceptable because it falls back to the already-tested GLOBAL
sampling policy. A false positive INTERVAL/FORWARD is considered dangerous,
so matching is intentionally strict:

- full-question template matching, not loose keyword matching;
- only whitespace and case are normalized;
- timestamps are validated against the supplied request window;
- any malformed/ambiguous input falls back to GLOBAL;
- only information available at challenge inference is used:
  request.question, request.start_time, request.end_time.

This module decides *where* to sample. It does not decode video frames.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal


SamplingMode = Literal["GLOBAL", "INTERVAL", "FORWARD"]


# -----------------------------------------------------------------------------
# Verified released-template vocabularies
# -----------------------------------------------------------------------------

# Objects observed in the verified "When is it retrieved..." released family.
_RETRIEVAL_OBJECTS = (
    "specimen bag",
    "silicone loop",
    "specimen",
    "sponge",
    "needle",
)

# Objects observed in the verified "re-appear later..." released family.
_REAPPEAR_OBJECTS = (
    "external drain",
    "specimen bag",
    "silicone loop",
    "gallstone",
    "specimen",
    "sponge",
    "needle",
)


def _alternation(values: tuple[str, ...]) -> str:
    """Build a regex alternation, longest strings first."""
    return "|".join(
        re.escape(value)
        for value in sorted(values, key=len, reverse=True)
    )


_RETRIEVAL_OBJECT_RE = _alternation(_RETRIEVAL_OBJECTS)
_REAPPEAR_OBJECT_RE = _alternation(_REAPPEAR_OBJECTS)

_OBJECT_SLOT_RE = r"[a-z][a-z0-9 /_-]{0,60}?"

# Deliberately strict HH:MM:SS. The released questions use this form.
_TIME_RE = r"\d{2}:[0-5]\d:[0-5]\d"


# -----------------------------------------------------------------------------
# Strict whitelist patterns
# -----------------------------------------------------------------------------

_INTERVAL_RE = re.compile(
    rf"^what types of foreign objects are seen between "
    rf"(?P<t1>{_TIME_RE}) and (?P<t2>{_TIME_RE})\? "
    rf"please provide the class name\(s\) or answer none\.$"
)

_FORWARD_RETRIEVAL_RE = re.compile(
    rf"^there is one (?P<object>{_OBJECT_SLOT_RE}) "
    rf"in the frame at (?P<t>{_TIME_RE})\. "
    rf"when is it retrieved from the surgical site\? "
    rf"please provide the answer in hh:mm:ss\.$"
)

_FORWARD_REAPPEAR_RE = re.compile(
    rf"^does the (?P<object>{_OBJECT_SLOT_RE}), "
    rf"last visible just before (?P<t>{_TIME_RE}), "
    rf"re-appear later in the video\? "
    rf"please answer with yes or no\.$"
)


# -----------------------------------------------------------------------------
# Sampling-plan object
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SamplingPlan:
    """Question-derived plan consumed later by the video sampler."""

    mode: SamplingMode

    # FORWARD only.
    anchor_abs_s: float | None = None

    # INTERVAL only.
    interval_abs_s: tuple[float, float] | None = None

    # Diagnostic metadata only.
    template_id: str = "global_fallback"
    reason: str = "Question did not match a strict specialized template."
    object_name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON/log-friendly representation."""
        value = asdict(self)
        if self.interval_abs_s is not None:
            value["interval_abs_s"] = list(self.interval_abs_s)
        return value


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def normalize_question(question: str) -> str:
    """
    Apply only safe normalization for strict matching.

    We intentionally:
    - collapse whitespace;
    - strip leading/trailing whitespace;
    - case-fold.

    We intentionally DO NOT:
    - remove punctuation;
    - rewrite hyphens;
    - rewrite wording/synonyms;
    - normalize arbitrary Unicode punctuation.

    If an unseen/OOD question is worded differently, GLOBAL is the desired
    conservative fallback.
    """
    if not isinstance(question, str):
        return ""

    return " ".join(question.strip().split()).casefold()


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert strict HH:MM:SS to absolute procedure seconds."""
    match = re.fullmatch(
        r"(?P<hour>\d{2}):(?P<minute>[0-5]\d):(?P<second>[0-5]\d)",
        timestamp,
    )

    if match is None:
        raise ValueError(f"Invalid strict timestamp: {timestamp!r}")

    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = int(match.group("second"))

    return float(hour * 3600 + minute * 60 + second)


def _valid_request_window(
    request_start_time: float,
    request_end_time: float,
) -> bool:
    try:
        start = float(request_start_time)
        end = float(request_end_time)
    except (TypeError, ValueError):
        return False

    return start >= 0.0 and end > start


def _global(reason: str) -> SamplingPlan:
    """Return the fail-safe GLOBAL plan."""
    return SamplingPlan(
        mode="GLOBAL",
        template_id="global_fallback",
        reason=reason,
    )


# -----------------------------------------------------------------------------
# Public classification API
# -----------------------------------------------------------------------------

def classify_question_sampling(
    question: str,
    *,
    request_start_time: float,
    request_end_time: float,
) -> SamplingPlan:
    """
    Classify one SEGMENT question using the strict whitelist.

    Specialized routing happens only if BOTH conditions hold:
      1. the complete normalized question matches a verified template; and
      2. the extracted timestamp(s) pass request-window validation.

    Otherwise the result is GLOBAL.
    """

    if not _valid_request_window(request_start_time, request_end_time):
        return _global("Invalid request time window.")

    start = float(request_start_time)
    end = float(request_end_time)
    q = normalize_question(question)

    if not q:
        return _global("Question is empty or invalid.")

    # -------------------------------------------------------------------------
    # 1) INTERVAL
    # -------------------------------------------------------------------------
    match = _INTERVAL_RE.fullmatch(q)

    if match is not None:
        try:
            t1 = timestamp_to_seconds(match.group("t1"))
            t2 = timestamp_to_seconds(match.group("t2"))
        except ValueError:
            return _global("INTERVAL template matched but timestamp parsing failed.")

        # Precision-first safety gate:
        # - a real interval must have positive duration;
        # - both endpoints must belong to the supplied request window.
        #
        # t2 == request_end_time is accepted because it is a valid semantic
        # boundary; video_utils can select the last available frame <= t2.
        if not (start <= t1 < t2 <= end):
            return _global(
                "INTERVAL template matched but interval is invalid or outside "
                "the supplied request window."
            )

        return SamplingPlan(
            mode="INTERVAL",
            interval_abs_s=(t1, t2),
            template_id="interval_fo_seen_between",
            reason=(
                "Exact verified bounded-interval template matched and both "
                "timestamps passed request-window validation."
            ),
        )

    # -------------------------------------------------------------------------
    # 2) FORWARD — retrieval after a known visible anchor
    # -------------------------------------------------------------------------
    match = _FORWARD_RETRIEVAL_RE.fullmatch(q)

    if match is not None:
        try:
            anchor = timestamp_to_seconds(match.group("t"))
        except ValueError:
            return _global(
                "FORWARD retrieval template matched but timestamp parsing failed."
            )

        # Anchor must be a real time inside the supplied clip. We use end as an
        # exclusive bound because there must be video after the anchor to search.
        if not (start <= anchor < end):
            return _global(
                "FORWARD retrieval template matched but anchor is outside "
                "the supplied request window."
            )

        return SamplingPlan(
            mode="FORWARD",
            anchor_abs_s=anchor,
            template_id="forward_retrieval_after_visible_anchor",
            reason=(
                "Exact verified retrieval-after-visible-anchor template matched "
                "and the anchor passed request-window validation."
            ),
            object_name=match.group("object"),
        )

    # -------------------------------------------------------------------------
    # 3) FORWARD — reappearance later than a known disappearance boundary
    # -------------------------------------------------------------------------
    match = _FORWARD_REAPPEAR_RE.fullmatch(q)

    if match is not None:
        try:
            anchor = timestamp_to_seconds(match.group("t"))
        except ValueError:
            return _global(
                "FORWARD reappearance template matched but timestamp parsing failed."
            )

        if not (start <= anchor < end):
            return _global(
                "FORWARD reappearance template matched but anchor is outside "
                "the supplied request window."
            )

        return SamplingPlan(
            mode="FORWARD",
            anchor_abs_s=anchor,
            template_id="forward_reappearance_after_last_visible",
            reason=(
                "Exact verified later-reappearance template matched and the "
                "anchor passed request-window validation."
            ),
            object_name=match.group("object"),
        )

    # -------------------------------------------------------------------------
    # 4) Fail-safe fallback
    # -------------------------------------------------------------------------
    return _global("No strict specialized template matched.")


def classify_sampling_plan(request: Any) -> SamplingPlan:
    """
    Convenience wrapper for focus.Request-like objects.

    Required attributes:
      request.question
      request.start_time
      request.end_time

    No answer, capability, OOD, split, or reference-side metadata is used.
    """
    try:
        question = request.question
        start = request.start_time
        end = request.end_time
    except AttributeError:
        return _global(
            "Request object is missing question/start_time/end_time."
        )

    return classify_question_sampling(
        question,
        request_start_time=start,
        request_end_time=end,
    )


# -----------------------------------------------------------------------------
# Standalone safety tests
# -----------------------------------------------------------------------------

def _run_self_test() -> None:
    """Run lightweight tests without importing orena-focus."""

    @dataclass
    class DummyRequest:
        question: str
        start_time: float
        end_time: float

    cases = [
        # Positive INTERVAL.
        (
            DummyRequest(
                question=(
                    "What types of foreign objects are seen between "
                    "00:07:46 and 00:08:47? Please provide the class "
                    "name(s) or answer none."
                ),
                start_time=7 * 60,
                end_time=10 * 60,
            ),
            "INTERVAL",
        ),

        # Positive FORWARD retrieval.
        (
            DummyRequest(
                question=(
                    "There is one Sponge in the frame at 00:09:19. "
                    "When is it retrieved from the surgical site? "
                    "Please provide the answer in hh:mm:ss."
                ),
                start_time=8 * 60,
                end_time=14 * 60,
            ),
            "FORWARD",
        ),

        # Positive FORWARD reappearance.
        (
            DummyRequest(
                question=(
                    "Does the Sponge, last visible just before 00:09:20, "
                    "re-appear later in the video? Please answer with yes or no."
                ),
                start_time=8 * 60,
                end_time=10 * 60,
            ),
            "FORWARD",
        ),

        # Two timestamps but NOT an interval.
        (
            DummyRequest(
                question=(
                    "Does the Sponge at 00:09:19 also appear at 00:08:43? "
                    "Please answer with yes or no."
                ),
                start_time=8 * 60,
                end_time=14 * 60,
            ),
            "GLOBAL",
        ),

        # Retrieval at the anchor, NOT a forward search.
        (
            DummyRequest(
                question=(
                    "Is the Sponge visible at 00:10:00 being retrieved at "
                    "that moment? Please answer with yes or no."
                ),
                start_time=9 * 60,
                end_time=12 * 60,
            ),
            "GLOBAL",
        ),

        # Conceptually forward but unseen wording: deliberately GLOBAL.
        (
            DummyRequest(
                question=(
                    "At 00:09:19 a Sponge is visible. When is it subsequently "
                    "removed from the abdomen?"
                ),
                start_time=8 * 60,
                end_time=14 * 60,
            ),
            "GLOBAL",
        ),

        # Conceptually interval but unseen wording: deliberately GLOBAL.
        (
            DummyRequest(
                question=(
                    "Which foreign objects can be seen from 00:07:46 to "
                    "00:08:47?"
                ),
                start_time=7 * 60,
                end_time=10 * 60,
            ),
            "GLOBAL",
        ),

        # Dangerous backward wording: must never become FORWARD.
        (
            DummyRequest(
                question=(
                    "At what time was the Sponge removed before 00:20:00?"
                ),
                start_time=15 * 60,
                end_time=21 * 60,
            ),
            "GLOBAL",
        ),

        # Exact INTERVAL wording but invalid/out-of-window timestamps.
        (
            DummyRequest(
                question=(
                    "What types of foreign objects are seen between "
                    "00:01:00 and 00:02:00? Please provide the class "
                    "name(s) or answer none."
                ),
                start_time=10 * 60,
                end_time=15 * 60,
            ),
            "GLOBAL",
        ),

        # Exact FORWARD wording but anchor at the request end:
        # there is no subsequent video to search.
        (
            DummyRequest(
                question=(
                    "There is one Sponge in the frame at 00:10:00. "
                    "When is it retrieved from the surgical site? "
                    "Please provide the answer in hh:mm:ss."
                ),
                start_time=9 * 60,
                end_time=10 * 60,
            ),
            "GLOBAL",
        ),
                # Positive FORWARD retrieval with an unseen object name.
        (
            DummyRequest(
                question=(
                    "There is one Absorbable Hemostatic Agent in the frame at "
                    "00:09:19. When is it retrieved from the surgical site? "
                    "Please provide the answer in hh:mm:ss."
                ),
                start_time=8 * 60,
                end_time=14 * 60,
            ),
            "FORWARD",
        ),
    ]

    for index, (request, expected_mode) in enumerate(cases, start=1):
        plan = classify_sampling_plan(request)

        if plan.mode != expected_mode:
            raise AssertionError(
                f"Self-test {index} failed: expected {expected_mode}, "
                f"got {plan.mode}. plan={plan}"
            )

    print(f"sampling_utils.py self-test passed: {len(cases)} cases.")


if __name__ == "__main__":
    _run_self_test()