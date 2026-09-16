"""Video utilities for ORena FOCUS PROCEDURE Docker inference.

The PROCEDURE track supplies one already-trimmed procedure-level MP4 per qID.
The MP4 must therefore be decoded from its own beginning. ``request.start_time``
and ``request.end_time`` refer to the original, untrimmed procedure timeline.

The seven sampling intents are:

- GLOBAL: broad coverage of the full supplied video context.
- LOCAL_ANCHOR: dense sampling around one explicit timestamp.
- TWO_ANCHOR: dense sampling around two or more explicit timestamps.
- INTERVAL: broad coverage restricted to an explicitly bounded interval.
- FORWARD_SEARCH: local context at an anchor plus broad coverage after it.
- BACKWARD_SEARCH: broad coverage before an anchor plus local context at it.
- PREFIX_STATE: broad coverage from clip start through an anchor, with local
  context at the anchor (for cumulative state such as "how many remain by T").

The high-confidence text rules were designed against the released HeiCo-FOCUS
and LapChole-FOCUS PROCEDURE question families. They deliberately optimize for
precision rather than recall: an unfamiliar/OOD phrasing falls back to GLOBAL
instead of risking a confidently wrong temporal restriction.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence

# Import torch before decord to avoid Decord/CUDA initialisation problems.
import torch  # noqa: F401
import decord
import numpy as np
from PIL import Image


# -----------------------------------------------------------------------------
# Temporal-intent parsing
# -----------------------------------------------------------------------------

_TIMESTAMP_RE = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")
_TIME_TOKEN = "<time>"


class TemporalIntent(str, Enum):
    """High-confidence temporal sampling modes for PROCEDURE questions."""

    GLOBAL = "global"
    LOCAL_ANCHOR = "local_anchor"
    TWO_ANCHOR = "two_anchor"
    INTERVAL = "interval"
    FORWARD_SEARCH = "forward_search"
    BACKWARD_SEARCH = "backward_search"
    PREFIX_STATE = "prefix_state"


@dataclass(frozen=True)
class TemporalQuery:
    """Parsed temporal information extracted from one question."""

    intent: TemporalIntent
    absolute_anchors: tuple[float, ...]
    normalized_question: str
    reason: str


@dataclass
class ClipFrames:
    """Frames sampled from one already-trimmed PROCEDURE clip."""

    images: list[Image.Image]
    frame_indices: list[int]
    timestamps: list[float]
    source_fps: float
    clip_duration: float
    temporal_intent: TemporalIntent = TemporalIntent.GLOBAL
    absolute_anchor_times: list[float] = field(default_factory=list)
    sampling_description: str = ""

    @property
    def num_frames(self) -> int:
        return len(self.images)


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert one HH:MM:SS timestamp to seconds."""

    parts = timestamp.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unexpected timestamp format: {timestamp!r}")

    hours, minutes, seconds = (int(part) for part in parts)
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"Invalid HH:MM:SS timestamp: {timestamp!r}")

    return float(hours * 3600 + minutes * 60 + seconds)


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS using the floored second."""

    if seconds < 0:
        raise ValueError("seconds must be non-negative.")

    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _normalize_question_for_matching(question: str) -> str:
    """Normalize superficial wording variants without changing semantics."""

    text = question.casefold()
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # Match both "timepoint", "time point", and "time-point" later as one form.
    text = re.sub(r"\btime[\s-]*point\b", "time point", text)

    # Match reappear / re-appear / re appear as one form.
    text = re.sub(r"\bre[\s-]*appear\b", "reappear", text)

    text = re.sub(r"\s+", " ", text).strip()
    return _TIMESTAMP_RE.sub(_TIME_TOKEN, text)


def _unique_preserving_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


_T = re.escape(_TIME_TOKEN)

# Strong bounded-range cues. These run before the generic multi-anchor rule.
_INTERVAL_PATTERNS = (
    re.compile(rf"between\s+{_T}\s+and\s+{_T}"),
    re.compile(rf"from\s+{_T}\s*(?:-|to|until|through(?:\s+to)?)\s*{_T}"),
)

# Cumulative state up to an anchor. These must run before LOCAL_ANCHOR.
_PREFIX_STATE_PATTERNS = (
    re.compile(r"considering\s+all\s+prior"),
    re.compile(
        rf"by\s+(?:frame\s+)?{_T}.*\b(?:remain|remains|remaining|left|present)\b"
    ),
)

# Known object/state at T; unknown evidence is expected later than T.
_FORWARD_PATTERNS = (
    re.compile(r"\breappear\b.*\b(?:later|afterwards|subsequently)\b"),
    re.compile(r"\bdoes\s+(?:a\s+)?retrieval\b.*\bexist\b"),
    re.compile(r"\bwhen\s+(?:is|was)\s+it\s+retrieved\b"),
    re.compile(rf"\bwhen\s+is\b.*\bseen\s+in\s+frame\s+{_T}.*\blast\s+seen\b"),
)

# Known object/state at T; unknown origin event is expected before T.
_BACKWARD_PATTERNS = (
    re.compile(
        rf"\bwhen\s+was\b.*\b(?:visible|seen)\b.*"
        rf"(?:frame\s+at|frame|at)\s+{_T}.*"
        r"\bfirst\s+(?:inserted|created)\b"
    ),
)

# A released question family contains "first occurs" but also directly supplies
# the exact timestamp at which the requested spatial property must be read. It is
# therefore a local query rather than a backward search.
_LOCAL_SPECIAL_PATTERNS = (
    re.compile(rf"\bfirst\s+occurs\b.*at\s+time\s+point\s+{_T}"),
)

# Generic direct-at-anchor cues. These are checked only after all stronger
# temporal relations and after unresolved directional cues are excluded.
_LOCAL_ANCHOR_PATTERNS = (
    re.compile(rf"\bat\s+(?:time\s+point\s+)?{_T}"),
    re.compile(rf"\bin\s+frame\s+{_T}"),
)

# If a timestamp question contains one of these unresolved temporal words but no
# high-confidence rule above matched, fall back to GLOBAL. This prevents a novel
# OOD phrasing from being incorrectly forced into LOCAL_ANCHOR.
_UNRESOLVED_DIRECTIONAL_CUES = re.compile(
    r"\b(?:before|after|later|earlier|first|last|prior|previous(?:ly)?|"
    r"subsequent(?:ly)?|until|remain(?:s|ing)?|retriev\w*|reappear)\b"
)


def determine_temporal_intent(question: str) -> TemporalQuery:
    """Classify one question into a conservative temporal sampling intent.

    The classifier intentionally uses only information available at challenge
    inference time: the natural-language question itself. It does not use
    answer formats, capability labels, OOD labels, or reference answers.
    """

    if not isinstance(question, str):
        raise TypeError("question must be a string.")
    if not question.strip():
        raise ValueError("question must not be empty.")

    raw_timestamps = _TIMESTAMP_RE.findall(question)
    unique_timestamps = _unique_preserving_order(raw_timestamps)
    anchors = tuple(timestamp_to_seconds(value) for value in unique_timestamps)
    normalized = _normalize_question_for_matching(question)

    if not anchors:
        return TemporalQuery(
            intent=TemporalIntent.GLOBAL,
            absolute_anchors=(),
            normalized_question=normalized,
            reason="no explicit HH:MM:SS timestamp",
        )

    if any(pattern.search(normalized) for pattern in _INTERVAL_PATTERNS):
        return TemporalQuery(
            intent=TemporalIntent.INTERVAL,
            absolute_anchors=anchors,
            normalized_question=normalized,
            reason="explicit bounded interval",
        )

    # After interval detection, two or more distinct explicit timestamps are a
    # high-confidence comparison/multi-anchor case in the released PROCEDURE QA
    # families. Repeated mentions of the same timestamp have already been deduped.
    if len(anchors) >= 2:
        return TemporalQuery(
            intent=TemporalIntent.TWO_ANCHOR,
            absolute_anchors=anchors,
            normalized_question=normalized,
            reason="two or more distinct explicit timestamps",
        )

    if any(pattern.search(normalized) for pattern in _PREFIX_STATE_PATTERNS):
        return TemporalQuery(
            intent=TemporalIntent.PREFIX_STATE,
            absolute_anchors=anchors,
            normalized_question=normalized,
            reason="cumulative state up to an explicit timestamp",
        )

    if any(pattern.search(normalized) for pattern in _FORWARD_PATTERNS):
        return TemporalQuery(
            intent=TemporalIntent.FORWARD_SEARCH,
            absolute_anchors=anchors,
            normalized_question=normalized,
            reason="explicit anchor with later/retrieval search",
        )

    if any(pattern.search(normalized) for pattern in _BACKWARD_PATTERNS):
        return TemporalQuery(
            intent=TemporalIntent.BACKWARD_SEARCH,
            absolute_anchors=anchors,
            normalized_question=normalized,
            reason="explicit anchor with earlier insertion/creation search",
        )

    if any(pattern.search(normalized) for pattern in _LOCAL_SPECIAL_PATTERNS):
        return TemporalQuery(
            intent=TemporalIntent.LOCAL_ANCHOR,
            absolute_anchors=anchors,
            normalized_question=normalized,
            reason="explicit timestamp already identifies the requested local event",
        )

    if _UNRESOLVED_DIRECTIONAL_CUES.search(normalized):
        return TemporalQuery(
            intent=TemporalIntent.GLOBAL,
            absolute_anchors=anchors,
            normalized_question=normalized,
            reason="timestamp present but temporal wording is not confidently recognized",
        )

    if any(pattern.search(normalized) for pattern in _LOCAL_ANCHOR_PATTERNS):
        return TemporalQuery(
            intent=TemporalIntent.LOCAL_ANCHOR,
            absolute_anchors=anchors,
            normalized_question=normalized,
            reason="direct visual query at one explicit timestamp",
        )

    return TemporalQuery(
        intent=TemporalIntent.GLOBAL,
        absolute_anchors=anchors,
        normalized_question=normalized,
        reason="no high-confidence temporal relation matched",
    )


# -----------------------------------------------------------------------------
# Frame-index sampling helpers
# -----------------------------------------------------------------------------


def _uniformly_cap_indices(
    frame_indices: np.ndarray,
    max_frames: int | None,
) -> np.ndarray:
    """Uniformly retain no more than ``max_frames`` ordered indices."""

    frame_indices = np.asarray(frame_indices, dtype=np.int64)

    if max_frames is None or len(frame_indices) <= max_frames:
        return frame_indices

    if max_frames <= 0:
        raise ValueError("max_frames must be positive or None.")

    positions = np.linspace(0, len(frame_indices) - 1, num=max_frames)
    positions = np.rint(positions).astype(np.int64)
    return frame_indices[positions]


def _cap_indices_preserving_required(
    frame_indices: np.ndarray,
    max_frames: int,
    required_indices: Sequence[int],
) -> np.ndarray:
    """Cap indices while guaranteeing that specified valid indices survive."""

    indices = np.unique(np.asarray(frame_indices, dtype=np.int64))
    required = np.unique(np.asarray(required_indices, dtype=np.int64))
    required = required[np.isin(required, indices)]

    if len(indices) <= max_frames:
        return indices

    if max_frames <= 0:
        raise ValueError("max_frames must be positive.")

    if len(required) >= max_frames:
        return _uniformly_cap_indices(required, max_frames)

    optional = indices[~np.isin(indices, required)]
    remaining = max_frames - len(required)
    optional = _uniformly_cap_indices(optional, remaining)

    return np.unique(np.concatenate([required, optional]))


def _nearest_frame_index(
    relative_seconds: float,
    *,
    source_fps: float,
    total_frames: int,
) -> int:
    index = int(round(float(relative_seconds) * source_fps))
    return int(np.clip(index, 0, total_frames - 1))


def _sample_region_indices(
    *,
    start_seconds: float,
    end_seconds: float,
    source_fps: float,
    total_frames: int,
    target_fps: float,
    max_frames: int | None,
) -> np.ndarray:
    """Sample an inclusive relative-time region at target FPS, then cap it."""

    if target_fps <= 0:
        raise ValueError("target_fps must be positive.")

    last_frame_time = (total_frames - 1) / source_fps
    start = float(np.clip(start_seconds, 0.0, last_frame_time))
    end = float(np.clip(end_seconds, 0.0, last_frame_time))

    if end < start:
        start, end = end, start

    if np.isclose(start, end):
        return np.asarray(
            [
                _nearest_frame_index(
                    start,
                    source_fps=source_fps,
                    total_frames=total_frames,
                )
            ],
            dtype=np.int64,
        )

    step = 1.0 / target_fps
    count = int(np.floor((end - start) / step)) + 1
    times = start + np.arange(count, dtype=np.float64) * step

    # Include the region end so bounded intervals and procedure endpoints retain
    # explicit boundary coverage even when the target-FPS grid does not land on it.
    if len(times) == 0 or times[-1] < end - 1e-9:
        times = np.concatenate([times, np.asarray([end], dtype=np.float64)])

    indices = np.rint(times * source_fps).astype(np.int64)
    indices = np.clip(indices, 0, total_frames - 1)
    indices = np.unique(indices)
    return _uniformly_cap_indices(indices, max_frames)


def _sample_anchor_window_indices(
    *,
    anchor_seconds: float,
    source_fps: float,
    total_frames: int,
    radius_seconds: float,
    local_fps: float,
    max_frames: int,
) -> np.ndarray:
    """Sample densely around one relative-time anchor and preserve the anchor."""

    if radius_seconds < 0:
        raise ValueError("radius_seconds must be non-negative.")
    if local_fps <= 0:
        raise ValueError("local_fps must be positive.")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive.")

    last_frame_time = (total_frames - 1) / source_fps
    anchor = float(np.clip(anchor_seconds, 0.0, last_frame_time))
    start = max(0.0, anchor - radius_seconds)
    end = min(last_frame_time, anchor + radius_seconds)

    indices = _sample_region_indices(
        start_seconds=start,
        end_seconds=end,
        source_fps=source_fps,
        total_frames=total_frames,
        target_fps=local_fps,
        max_frames=None,
    )

    anchor_index = _nearest_frame_index(
        anchor,
        source_fps=source_fps,
        total_frames=total_frames,
    )
    indices = np.unique(np.concatenate([indices, np.asarray([anchor_index])]))

    return _cap_indices_preserving_required(
        indices,
        max_frames=max_frames,
        required_indices=[anchor_index],
    )


def _single_anchor_budget(max_frames: int) -> int:
    """Reserve exactly 5 frames for dense local evidence."""

    return min(5, max_frames)


def _combine_indices(*groups: np.ndarray) -> np.ndarray:
    non_empty = [np.asarray(group, dtype=np.int64) for group in groups if len(group)]
    if not non_empty:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(non_empty))


def _fill_from_region(
    selected: np.ndarray,
    *,
    start_seconds: float,
    end_seconds: float,
    source_fps: float,
    total_frames: int,
    target_fps: float,
    max_frames: int,
) -> np.ndarray:
    """Fill unused budget with additional broad region samples."""

    selected = np.unique(np.asarray(selected, dtype=np.int64))
    if len(selected) >= max_frames:
        return _uniformly_cap_indices(selected, max_frames)

    candidates = _sample_region_indices(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        source_fps=source_fps,
        total_frames=total_frames,
        target_fps=target_fps,
        max_frames=None,
    )
    remaining_candidates = candidates[~np.isin(candidates, selected)]
    needed = max_frames - len(selected)
    additions = _uniformly_cap_indices(remaining_candidates, needed)
    return _combine_indices(selected, additions)


def _relative_anchors_in_clip(
    absolute_anchors: Sequence[float],
    *,
    absolute_start_time: float,
    source_fps: float,
    total_frames: int,
) -> tuple[list[float], list[float]]:
    """Convert absolute procedure anchors to valid clip-relative seconds.

    Returns
    -------
    relative, absolute
        Matched lists containing only anchors that lie within the supplied clip.
    """

    last_frame_time = (total_frames - 1) / source_fps
    tolerance = max(1.0 / source_fps, 0.25)

    relative: list[float] = []
    kept_absolute: list[float] = []

    for absolute_anchor in absolute_anchors:
        rel = float(absolute_anchor) - float(absolute_start_time)
        if rel < -tolerance or rel > last_frame_time + tolerance:
            continue

        rel = float(np.clip(rel, 0.0, last_frame_time))
        relative.append(rel)
        kept_absolute.append(float(absolute_anchor))

    return relative, kept_absolute


def _select_indices_for_intent(
    *,
    temporal_query: TemporalQuery,
    absolute_start_time: float,
    source_fps: float,
    total_frames: int,
    target_fps: float,
    max_frames: int,
    local_fps: float,
    local_radius_seconds: float,
) -> tuple[np.ndarray, TemporalIntent, list[float], str]:
    """Turn a temporal intent into concrete frame indices."""

    last_frame_time = (total_frames - 1) / source_fps
    relative_anchors, absolute_anchors = _relative_anchors_in_clip(
        temporal_query.absolute_anchors,
        absolute_start_time=absolute_start_time,
        source_fps=source_fps,
        total_frames=total_frames,
    )

    intent = temporal_query.intent

    # If a specialized rule was identified but its explicit timestamp lies
    # outside the actual supplied clip, do not guess: use safe global coverage.
    if intent is not TemporalIntent.GLOBAL and not relative_anchors:
        intent = TemporalIntent.GLOBAL

    if intent is TemporalIntent.GLOBAL:
        indices = _sample_region_indices(
            start_seconds=0.0,
            end_seconds=last_frame_time,
            source_fps=source_fps,
            total_frames=total_frames,
            target_fps=target_fps,
            max_frames=max_frames,
        )
        description = (
            f"{len(indices)} chronologically ordered frames sampled for broad "
            "coverage across the supplied procedure-level video context."
        )
        return indices, intent, absolute_anchors, description

    if intent is TemporalIntent.INTERVAL:
        if len(relative_anchors) < 2:
            # Defensive fallback; the parser normally cannot produce this state.
            return _select_indices_for_intent(
                temporal_query=TemporalQuery(
                    TemporalIntent.GLOBAL,
                    tuple(absolute_anchors),
                    temporal_query.normalized_question,
                    "interval missing a second valid anchor",
                ),
                absolute_start_time=absolute_start_time,
                source_fps=source_fps,
                total_frames=total_frames,
                target_fps=target_fps,
                max_frames=max_frames,
                local_fps=local_fps,
                local_radius_seconds=local_radius_seconds,
            )

        start, end = sorted(relative_anchors[:2])
        indices = _sample_region_indices(
            start_seconds=start,
            end_seconds=end,
            source_fps=source_fps,
            total_frames=total_frames,
            target_fps=target_fps,
            max_frames=max_frames,
        )
        description = (
            f"{len(indices)} chronologically ordered frames sampled across the "
            f"explicit interval {seconds_to_timestamp(absolute_anchors[0])} to "
            f"{seconds_to_timestamp(absolute_anchors[1])}."
        )
        return indices, intent, absolute_anchors, description

    if intent is TemporalIntent.TWO_ANCHOR:
        n_anchors = len(relative_anchors)
        per_anchor_budget = min(5, max_frames)
        groups = [
            _sample_anchor_window_indices(
                anchor_seconds=anchor,
                source_fps=source_fps,
                total_frames=total_frames,
                radius_seconds=local_radius_seconds,
                local_fps=local_fps,
                max_frames=per_anchor_budget,
            )
            for anchor in relative_anchors
        ]
        indices = _combine_indices(*groups)
        indices = _uniformly_cap_indices(indices, max_frames)
        anchor_text = ", ".join(seconds_to_timestamp(value) for value in absolute_anchors)
        description = (
            f"{len(indices)} chronologically ordered frames sampled densely "
            f"around the explicit comparison time points {anchor_text}."
        )
        return indices, intent, absolute_anchors, description

    # The remaining specialized modes use exactly one anchor in the released
    # PROCEDURE templates. If future OOD text supplies more, use the first.
    anchor = relative_anchors[0]
    absolute_anchor = absolute_anchors[0]
    anchor_budget = _single_anchor_budget(max_frames)
    local_indices = _sample_anchor_window_indices(
        anchor_seconds=anchor,
        source_fps=source_fps,
        total_frames=total_frames,
        radius_seconds=local_radius_seconds,
        local_fps=local_fps,
        max_frames=max(1, anchor_budget),
    )

    if intent is TemporalIntent.LOCAL_ANCHOR:
        local_budget = min(max_frames, 5)
        indices = _sample_anchor_window_indices(
            anchor_seconds=anchor,
            source_fps=source_fps,
            total_frames=total_frames,
            radius_seconds=local_radius_seconds,
            local_fps=local_fps,
            max_frames=local_budget,
        )
        description = (
            f"{len(indices)} chronologically ordered frames sampled densely "
            f"around the explicit time point {seconds_to_timestamp(absolute_anchor)}."
        )
        return indices, intent, absolute_anchors, description

    broad_budget = max(1, max_frames - len(local_indices))

    if intent is TemporalIntent.FORWARD_SEARCH:
        broad = _sample_region_indices(
            start_seconds=anchor,
            end_seconds=last_frame_time,
            source_fps=source_fps,
            total_frames=total_frames,
            target_fps=target_fps,
            max_frames=broad_budget,
        )
        indices = _combine_indices(local_indices, broad)
        indices = _fill_from_region(
            indices,
            start_seconds=anchor,
            end_seconds=last_frame_time,
            source_fps=source_fps,
            total_frames=total_frames,
            target_fps=target_fps,
            max_frames=max_frames,
        )
        description = (
            f"{len(indices)} chronologically ordered frames combining dense "
            f"sampling around {seconds_to_timestamp(absolute_anchor)} with broad "
            "coverage from that point to the end of the supplied context."
        )
        return indices, intent, absolute_anchors, description

    if intent is TemporalIntent.BACKWARD_SEARCH:
        broad = _sample_region_indices(
            start_seconds=0.0,
            end_seconds=anchor,
            source_fps=source_fps,
            total_frames=total_frames,
            target_fps=target_fps,
            max_frames=broad_budget,
        )
        indices = _combine_indices(local_indices, broad)
        indices = _fill_from_region(
            indices,
            start_seconds=0.0,
            end_seconds=anchor,
            source_fps=source_fps,
            total_frames=total_frames,
            target_fps=target_fps,
            max_frames=max_frames,
        )
        description = (
            f"{len(indices)} chronologically ordered frames combining broad "
            f"coverage from the start through {seconds_to_timestamp(absolute_anchor)} "
            "with dense sampling around that time point."
        )
        return indices, intent, absolute_anchors, description

    if intent is TemporalIntent.PREFIX_STATE:
        broad = _sample_region_indices(
            start_seconds=0.0,
            end_seconds=anchor,
            source_fps=source_fps,
            total_frames=total_frames,
            target_fps=target_fps,
            max_frames=broad_budget,
        )
        indices = _combine_indices(local_indices, broad)
        indices = _fill_from_region(
            indices,
            start_seconds=0.0,
            end_seconds=anchor,
            source_fps=source_fps,
            total_frames=total_frames,
            target_fps=target_fps,
            max_frames=max_frames,
        )
        description = (
            f"{len(indices)} chronologically ordered frames covering the supplied "
            f"procedure history up to {seconds_to_timestamp(absolute_anchor)}, "
            "with denser sampling near that time point."
        )
        return indices, intent, absolute_anchors, description

    raise RuntimeError(f"Unhandled temporal intent: {intent}")


# -----------------------------------------------------------------------------
# Public video loading API
# -----------------------------------------------------------------------------


def load_clip_frames(
    video_path: str | Path,
    *,
    question: str | None = None,
    absolute_start_time: float = 0.0,
    target_fps: float = 1.0,
    max_frames: int = 64,
    local_fps: float = 2.0,
    local_radius_seconds: float = 3.0,
    num_threads: int = 1,
) -> ClipFrames:
    """Sample one already-trimmed PROCEDURE clip using question-conditioned logic.

    Parameters
    ----------
    video_path:
        The qID-specific MP4 supplied by the challenge. It is already trimmed to
        the current request's video context, so decoding always starts at local
        time 0.
    question:
        Current ``focus.Request.question``. If omitted, the sampler uses GLOBAL.
    absolute_start_time:
        ``focus.Request.start_time`` in seconds on the original procedure
        timeline. Explicit HH:MM:SS anchors extracted from ``question`` are
        converted to clip-relative times by subtracting this value.
    target_fps:
        Candidate density for broad/global coverage before the frame cap is
        applied. ``1.0`` preserves the SEGMENT pipeline's broad sampling density.
    max_frames:
        Maximum number of frames returned for non-local queries. The value should
        be benchmarked with the final InternVL model (e.g. 32/48/64).
    local_fps:
        Dense sampling rate around explicit temporal anchors.
    local_radius_seconds:
        Radius on each side of an explicit anchor for dense local sampling.
    num_threads:
        Decord CPU decoder threads.
    """

    path = Path(video_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")
    if target_fps <= 0:
        raise ValueError("target_fps must be positive.")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive.")
    if local_fps <= 0:
        raise ValueError("local_fps must be positive.")
    if local_radius_seconds < 0:
        raise ValueError("local_radius_seconds must be non-negative.")
    if absolute_start_time < 0:
        raise ValueError("absolute_start_time must be non-negative.")

    video_reader = decord.VideoReader(
        str(path),
        ctx=decord.cpu(0),
        num_threads=max(1, int(num_threads)),
    )

    total_frames = len(video_reader)
    if total_frames == 0:
        raise RuntimeError(f"Video contains no decodable frames: {path}")

    source_fps = float(video_reader.get_avg_fps())
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise RuntimeError(
            f"Could not determine a valid FPS for {path}: {source_fps}"
        )

    clip_duration = total_frames / source_fps

    temporal_query = (
        determine_temporal_intent(question)
        if question is not None and question.strip()
        else TemporalQuery(
            intent=TemporalIntent.GLOBAL,
            absolute_anchors=(),
            normalized_question="",
            reason="no question supplied to sampler",
        )
    )

    frame_indices, effective_intent, kept_anchors, sampling_description = (
        _select_indices_for_intent(
            temporal_query=temporal_query,
            absolute_start_time=float(absolute_start_time),
            source_fps=source_fps,
            total_frames=total_frames,
            target_fps=target_fps,
            max_frames=max_frames,
            local_fps=local_fps,
            local_radius_seconds=local_radius_seconds,
        )
    )

    frame_indices = np.unique(np.asarray(frame_indices, dtype=np.int64))
    frame_indices = np.sort(frame_indices)

    if len(frame_indices) == 0:
        raise RuntimeError(f"No valid frames selected from {path.name}.")
    if len(frame_indices) > max_frames:
        raise RuntimeError(
            f"Internal sampler error: selected {len(frame_indices)} frames "
            f"with max_frames={max_frames}."
        )

    # One batched read is substantially cheaper than repeated seeks/reads.
    decoded = video_reader.get_batch(frame_indices.tolist()).asnumpy()
    del video_reader

    images = [Image.fromarray(frame).convert("RGB") for frame in decoded]
    relative_timestamps = [
        float(frame_index / source_fps) for frame_index in frame_indices
    ]

    return ClipFrames(
        images=images,
        frame_indices=[int(frame_index) for frame_index in frame_indices],
        timestamps=relative_timestamps,
        source_fps=source_fps,
        clip_duration=clip_duration,
        temporal_intent=effective_intent,
        absolute_anchor_times=list(kept_anchors),
        sampling_description=sampling_description,
    )


# -----------------------------------------------------------------------------
# Debug helpers / smoke tests
# -----------------------------------------------------------------------------


def save_debug_frames(
    clip: ClipFrames,
    output_dir: str | Path,
    *,
    absolute_start_time: float | None = None,
    prefix: str = "frame",
) -> None:
    """Save selected frames for local debugging."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for order, (image, frame_index, relative_time) in enumerate(
        zip(
            clip.images,
            clip.frame_indices,
            clip.timestamps,
            strict=True,
        )
    ):
        label_time = relative_time
        if absolute_start_time is not None:
            label_time += absolute_start_time

        timestamp_text = seconds_to_timestamp(label_time).replace(":", "-")
        filename = (
            f"{prefix}_{order:03d}_{timestamp_text}_"
            f"frame{frame_index:07d}.jpg"
        )
        image.save(output_path / filename, quality=95)


def _validate_clip_frames(
    clip: ClipFrames,
    *,
    max_frames: int,
) -> None:
    """Validate internal consistency of sampled PROCEDURE frames."""

    if clip.num_frames == 0:
        raise RuntimeError("No frames were returned.")
    if clip.num_frames > max_frames:
        raise RuntimeError(
            f"Returned {clip.num_frames} frames, which exceeds max_frames={max_frames}."
        )

    lengths = {
        len(clip.images),
        len(clip.frame_indices),
        len(clip.timestamps),
    }
    if len(lengths) != 1:
        raise RuntimeError(
            "images, frame_indices, and timestamps have different lengths."
        )

    if not np.isfinite(clip.source_fps) or clip.source_fps <= 0:
        raise RuntimeError(f"Invalid source FPS: {clip.source_fps}")
    if not np.isfinite(clip.clip_duration) or clip.clip_duration <= 0:
        raise RuntimeError(f"Invalid clip duration: {clip.clip_duration}")

    if clip.frame_indices != sorted(clip.frame_indices):
        raise RuntimeError("Frame indices are not chronologically ordered.")
    if len(set(clip.frame_indices)) != len(clip.frame_indices):
        raise RuntimeError("Duplicate frame indices were returned.")
    if clip.timestamps != sorted(clip.timestamps):
        raise RuntimeError("Frame timestamps are not chronologically ordered.")

    for image in clip.images:
        if image.mode != "RGB":
            raise RuntimeError(f"Expected RGB image, received mode={image.mode!r}.")
        if image.width <= 0 or image.height <= 0:
            raise RuntimeError(f"Invalid image dimensions: {image.size}")

    for frame_index, timestamp in zip(
        clip.frame_indices,
        clip.timestamps,
        strict=True,
    ):
        expected_timestamp = frame_index / clip.source_fps
        if not np.isclose(timestamp, expected_timestamp, rtol=0.0, atol=1e-9):
            raise RuntimeError(
                "Timestamp does not correspond to its frame index: "
                f"frame={frame_index}, timestamp={timestamp}, "
                f"expected={expected_timestamp}."
            )

        if timestamp < 0 or timestamp >= clip.clip_duration:
            raise RuntimeError(f"Timestamp outside clip: {timestamp:.6f}")


def _run_temporal_intent_smoke_tests() -> None:
    """Check representative released PROCEDURE question families."""

    tests = (
        (
            "What types of foreign objects are seen between 00:11:00 and "
            "01:00:01? Please provide the class name(s) or answer none.",
            TemporalIntent.INTERVAL,
        ),
        (
            "Does the Sponge at 00:13:07 also appear at 00:10:58? "
            "Please answer with yes or no.",
            TemporalIntent.TWO_ANCHOR,
        ),
        (
            "For the surgical foreign object visible at 04:46:31: "
            "Does a retrieval of this object exist? Please answer with yes or no.",
            TemporalIntent.FORWARD_SEARCH,
        ),
        (
            "Does the Silicone loop, last visible just before 04:47:38, "
            "re-appear later in the video? Please answer with yes or no.",
            TemporalIntent.FORWARD_SEARCH,
        ),
        (
            "When was the Sponge visible in the frame at 00:35:20 first "
            "inserted in the abdomen? Please provide an answer in the format "
            "hh:mm:ss.",
            TemporalIntent.BACKWARD_SEARCH,
        ),
        (
            "By frame 00:49:00, how many sponges remain in the abdomen? "
            "Please provide a single integer.",
            TemporalIntent.PREFIX_STATE,
        ),
        (
            "How many Clips are present in the surgical site (visible and not visible) at time point 04:02:00, considering all prior insertions or creations (in case of a specimen) and removals? Please provide a number.",
            TemporalIntent.PREFIX_STATE,
        ),        
        (
            "At timepoint 02:13:00 please provide all relative central "
            "positions of foreign objects present in the frame.",
            TemporalIntent.LOCAL_ANCHOR,
        ),
        (
            "At time point 02:13:00 please provide all relative central "
            "positions of foreign objects present in the frame.",
            TemporalIntent.LOCAL_ANCHOR,
        ),
        (
            "At what time was the first Sponge visible in the video?",
            TemporalIntent.GLOBAL,
        ),
        (
            "After the first Clip was inserted, which other foreign objects "
            "are visible in the video?",
            TemporalIntent.GLOBAL,
        ),
    )

    for question, expected in tests:
        parsed = determine_temporal_intent(question)
        if parsed.intent is not expected:
            raise RuntimeError(
                "Temporal-intent smoke test failed:\n"
                f"  question={question!r}\n"
                f"  expected={expected.value}\n"
                f"  received={parsed.intent.value}\n"
                f"  reason={parsed.reason}"
            )

    print("Temporal-intent smoke tests passed.")


def main() -> None:
    """Run intent tests and optionally sample one PROCEDURE video."""

    _run_temporal_intent_smoke_tests()

    video_env = os.environ.get("VIDEO_UTILS_TEST_VIDEO")
    if not video_env:
        print(
            "No VIDEO_UTILS_TEST_VIDEO supplied; skipping video decode test.\n"
            "Set it to a qID-specific PROCEDURE MP4 to test frame selection."
        )
        return

    video_path = Path(video_env).expanduser().resolve()
    question = os.environ.get(
        "VIDEO_UTILS_TEST_QUESTION",
        "For the surgical foreign object visible at 00:10:00: "
        "Does a retrieval of this object exist? Please answer with yes or no.",
    )
    absolute_start_time = float(
        os.environ.get("VIDEO_UTILS_TEST_START_TIME", "0")
    )
    max_frames = int(os.environ.get("VIDEO_UTILS_TEST_MAX_FRAMES", "64"))
    output_dir = Path(
        os.environ.get(
            "VIDEO_UTILS_TEST_OUTPUT",
            "/tmp/procedure_video_utils_test_frames",
        )
    ).expanduser().resolve()

    parsed = determine_temporal_intent(question)
    print("\nQuestion:", question)
    print("Intent:", parsed.intent.value)
    print("Reason:", parsed.reason)
    print(
        "Absolute anchors:",
        ", ".join(seconds_to_timestamp(value) for value in parsed.absolute_anchors)
        or "none",
    )

    clip = load_clip_frames(
        video_path=video_path,
        question=question,
        absolute_start_time=absolute_start_time,
        target_fps=1.0,
        max_frames=max_frames,
        local_fps=2.0,
        local_radius_seconds=3.0,
        num_threads=1,
    )
    _validate_clip_frames(clip, max_frames=max_frames)

    print("\nSampling:", clip.sampling_description)
    print(f"Source FPS: {clip.source_fps:.6f}")
    print(f"Clip duration: {clip.clip_duration:.3f} s")
    print(f"Selected frames: {clip.num_frames}")

    for order, (index, relative_time) in enumerate(
        zip(clip.frame_indices, clip.timestamps, strict=True)
    ):
        absolute_time = absolute_start_time + relative_time
        print(
            f"  {order:03d}: frame={index:07d}, "
            f"relative={relative_time:9.3f}, "
            f"absolute={seconds_to_timestamp(absolute_time)}"
        )

    save_debug_frames(
        clip,
        output_dir,
        absolute_start_time=absolute_start_time,
    )
    print(f"\nSaved {clip.num_frames} debug frames to {output_dir}")


if __name__ == "__main__":
    main()