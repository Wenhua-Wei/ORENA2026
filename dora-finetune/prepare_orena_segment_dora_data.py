"""
Prepare ORena SAVE FOCUS SEGMENT frames for InternVL3.5-8B DoRA fine-tuning.

This script ONLY prepares:
    1) cached timestamp-overlay JPEG frames
    2) frame/data manifests

It deliberately does NOT:
    - construct prompts
    - construct InternVL conversation/SFT JSONL
    - create intermediate per-question MP4 clips
    - rebalance/oversample training samples

A later script should read the manifests produced here and build the exact
training conversations/prompts.

Released splits used
--------------------
Training:
    - HeiCo SEGMENT train
    - LapChole SEGMENT train

Validation:
    - HeiCo SEGMENT test
    - LapChole SEGMENT test

No extra validation split is created.

Current Seymour source-video paths
----------------------------------
HeiCo:
    /SAN/medic/Surgical_LLM_Agent/orena2026/data/heico/videos

LapChole:
    /SAN/medic/Surgical_LLM_Agent/orena2026/data/lapchole/videos

Frame policy
------------
For each VQA segment [start_time, end_time):

1. Build 1-fps candidate times relative to the segment:
       0, 1, 2, ... seconds
   while the time is strictly before end_time.

2. If there are more than 32 candidate frames, uniformly retain 32 using:
       np.linspace(0, N - 1, num=32)
       np.rint(...)
   This matches the endpoint-preserving cap rule in the current Docker
   video_utils.py, so the first and last 1-fps candidates are preserved.

3. Read those frames directly from the original full-procedure video.
   No intermediate 5-fps H.264 MP4 is generated.

4. Resize before overlay:
       height <= 576
       preserve aspect ratio
       no upscaling
       even dimensions

5. Burn the absolute original-procedure HH:MM:SS timestamp using:
       origin      = (20, 50)
       font        = cv2.FONT_HERSHEY_SIMPLEX
       scale       = 1.5
       color       = white
       thickness   = 1
       line type   = cv2.LINE_AA

6. Save as JPEG quality 95.

7. Deduplicate identical cached images across VQAs that request the same source
   frame with the same visible timestamp. Deduplication only saves storage.
   Each VQA still keeps its own complete ordered list of frame paths.

Why no 5-fps/keyframe settings here?
------------------------------------
The final training input is a set of decoded still images. MP4 codec, GOP and
keyframe structure are not visible to InternVL after decoding. We therefore
reproduce the temporal sampling and visual appearance directly without writing
thousands of intermediate clips.

Outputs
-------
OUTPUT_ROOT/
    frames/
        heico/...
        lapchole/...
    train_manifest.jsonl
    val_heico_manifest.jsonl
    val_lapchole_manifest.jsonl
    val_all_manifest.jsonl
    manifest.csv
    summary.json

Requirements
------------
    pip install "orena-focus==0.3.4" decord opencv-python-headless numpy

PyTorch is imported before decord because that matches the working Docker
environment and avoids Decord/CUDA initialization problems seen there.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch  # noqa: F401  # intentionally before decord
import cv2
import decord
import numpy as np

from focus import DatasetSplit, FocusDataset, Track


# =============================================================================
# CONFIGURATION
# =============================================================================

ORENA_ROOT = Path("/SAN/medic/Surgical_LLM_Agent/orena2026")

HEICO_VIDEO_DIR = Path(
    "/SAN/medic/Surgical_LLM_Agent/orena2026/data/heico/videos"
)
LAPCHOLE_VIDEO_DIR = Path(
    "/SAN/medic/Surgical_LLM_Agent/orena2026/data/lapchole/videos"
)

# IMPORTANT:
# train/test are annotation splits only. On Seymour, both splits use the SAME
# physical source-video folder for each dataset.
SOURCE_VIDEO_DIRS: dict[str, Path] = {
    "heico": HEICO_VIDEO_DIR,
    "lapchole": LAPCHOLE_VIDEO_DIR,
}

DEFAULT_OUTPUT_ROOT = Path(
    "/SAN/medic/Cholec/orena_data_tmp/dora_8b_max32"
)

TARGET_FPS = 1.0
MAX_FRAMES = 32
MAX_HEIGHT = 576

TEXT_ORG = (20, 50)
TEXT_FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1.5
TEXT_COLOR_BGR = (255, 255, 255)
TEXT_THICKNESS = 1
TEXT_LINE_TYPE = cv2.LINE_AA

JPEG_QUALITY = 95

DECORD_THREADS = 1
DECODE_BATCH_SIZE = 128

EXPECTED_COUNTS: dict[tuple[str, str], int] = {
    ("heico", "train"): 8000,
    ("lapchole", "train"): 5746,
    ("heico", "test"): 4000,
    ("lapchole", "test"): 2254,
}

# =============================================================================
# SMOKE-TEST CONFIGURATION
# =============================================================================
# Smoke mode is only for checking the complete fine-tuning pipeline quickly.
#
# Training:
#   2 HeiCo train source videos
#   2 LapChole train source videos
#   = 4 train source videos total
#
# Test/validation:
#   1 HeiCo test source video
#   0 LapChole test source videos
#   = 1 test source video total
#
# Within each split, source video IDs are sorted and the first N are selected,
# making the subset deterministic and reproducible.
SMOKE_VIDEO_LIMITS: dict[tuple[str, str], int] = {
    ("heico", "train"): 2,
    ("lapchole", "train"): 2,
    ("heico", "test"): 1,
    ("lapchole", "test"): 0,
}

SMOKE_OUTPUT_ROOT = Path(
    "/SAN/medic/Surgical_LLM_Agent/orena2026/data/segment/"
    "dora_8b_max32_smoke_4train_1test"
)

LOG_LEVEL = logging.INFO


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class FramePlan:
    source_index: int
    relative_time_seconds: float
    absolute_time_seconds: float
    timestamp: str
    path: str  # path relative to OUTPUT_ROOT


@dataclass
class SamplePlan:
    dataset: str
    split: str
    sample_id: str
    qid: str
    request: Any
    reference: Any
    source_path: Path
    source_fps: float
    source_total_frames: int
    candidate_count: int
    frames: list[FramePlan]


# =============================================================================
# CLI / LOGGING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ORena SEGMENT frame cache for InternVL DoRA."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Load all annotations, resolve videos, calculate exact frame plans "
            "and write manifests, but do not decode/write JPEGs."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate cached JPEGs even if they already exist.",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help=(
            "Do not fail if the installed orena-focus release contains a "
            "different number of VQAs than the currently verified release."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Prepare only 4 train source videos "
            "(2 HeiCo + 2 LapChole) and 1 HeiCo test source video."
        ),
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    os.replace(tmp, path)


def atomic_write_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")
    os.replace(tmp, path)


def seconds_to_hhmmss(seconds: float) -> str:
    if seconds < 0:
        raise ValueError(f"seconds must be non-negative, got {seconds}")

    total_seconds = int(math.floor(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def safe_slug(text: str, max_length: int = 60) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return (text or "video")[:max_length]


def stable_video_cache_dir(
    dataset: str,
    video_id: str,
) -> Path:
    """
    Cache directory intentionally does NOT include train/test split.

    This means the same physical source frame can be shared even if the same
    source video were ever referenced by multiple annotation splits.
    """
    digest = hashlib.sha1(
        f"{dataset}|{video_id}".encode("utf-8")
    ).hexdigest()[:12]

    stem = safe_slug(Path(video_id).stem)

    return Path("frames") / dataset / f"{stem}_{digest}"


def split_to_enum(split: str) -> DatasetSplit:
    if split == "train":
        return DatasetSplit.TRAIN
    if split == "test":
        return DatasetSplit.TEST
    raise ValueError(f"Unsupported split: {split}")


def reference_answer(reference: Any) -> str:
    answer = str(reference.answer).strip()
    if not answer:
        raise ValueError(
            f"Empty reference answer for qID={getattr(reference, 'qID', '?')}"
        )
    return answer


def answer_format_name(reference: Any) -> str:
    value = getattr(reference, "_format", None)

    if value is None:
        fmt = getattr(reference, "format", None)
        value = getattr(fmt, "type", None)

    return str(value or "")


def capability_code(capability: Any) -> str:
    code = getattr(capability, "code", None)
    if code:
        return str(code)

    raw = str(getattr(capability, "value", capability))

    mapping = {
        "object_identification": "1a",
        "instance_matching": "1b",
        "object_attributes": "1c",
        "spatial_localization_camera": "1d",
        "spatial_localization_situs": "1e",
        "temporal_localization": "2a",
        "duration_estimation": "2b",
        "object_aggregation": "3a",
        "event_aggregation": "3b",
        "fo_interaction_recognition": "4a",
        "fo_usage_purpose": "4b",
        "temporal_ordering": "4c",
        "functional_reasoning": "5a",
        "causal_consequence_reasoning": "5b",
        "multi_step_reasoning": "5c",
    }

    return mapping.get(raw, raw)


def secondary_capabilities(reference: Any) -> list[str]:
    return [
        capability_code(item)
        for item in getattr(reference, "secondaries", ())
    ]


# =============================================================================
# VIDEO LOOKUP
# =============================================================================

def build_video_index(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Source video directory does not exist: {directory}"
        )

    files = [path for path in directory.iterdir() if path.is_file()]

    if not files:
        raise RuntimeError(
            f"No files found in source video directory: {directory}"
        )

    index: dict[str, Path] = {}

    for path in files:
        index[path.name] = path
        index[f"__lower__:{path.name.lower()}"] = path

    return index


def resolve_video(
    *,
    video_id: str,
    directory: Path,
    index: dict[str, Path],
) -> Path:
    """
    Resolve FocusDataset request.videoID to the actual Seymour filename.

    Resolution order:
      1. exact filename
      2. case-insensitive exact filename
      3. unique four-digit numeric prefix

    The prefix fallback is useful if metadata capitalization/descriptive text
    differs slightly while the procedure index is the same.
    """
    path = index.get(video_id)
    if path is not None:
        return path

    path = index.get(f"__lower__:{video_id.lower()}")
    if path is not None:
        return path

    match = re.match(r"^(\d{4})\b", video_id)

    if match:
        prefix = match.group(1)

        candidates = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.name.startswith(prefix)
        )

        if len(candidates) == 1:
            logging.warning(
                "Resolved videoID %r by numeric prefix -> %r",
                video_id,
                candidates[0].name,
            )
            return candidates[0]

        if len(candidates) > 1:
            raise RuntimeError(
                f"Ambiguous videoID {video_id!r}. Prefix {prefix!r} matches: "
                f"{[path.name for path in candidates]}"
            )

    raise FileNotFoundError(
        f"Could not resolve videoID {video_id!r} in {directory}"
    )


def inspect_video(path: Path) -> tuple[float, int, float]:
    """
    Lightweight metadata probe used during planning.

    Use OpenCV here instead of Decord so --plan-only does not spend minutes
    indexing every full surgical procedure. Decord is still used later for
    the actual selected-frame extraction.

    Returns:
        fps, total_frames, duration_seconds
    """
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()

    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(
            f"Could not determine valid FPS for {path}: {fps}"
        )

    if total_frames <= 0:
        raise RuntimeError(
            f"Could not determine frame count for {path}: {total_frames}"
        )

    duration = total_frames / fps

    if not np.isfinite(duration) or duration <= 0:
        raise RuntimeError(
            f"Could not determine duration for {path}: {duration}"
        )

    return fps, total_frames, duration


# =============================================================================
# FRAME SAMPLING
# =============================================================================

def one_fps_candidate_times(
    duration_seconds: float,
) -> np.ndarray:
    """
    Return 0, 1, 2, ... while relative time is strictly < duration.

    Examples:
      duration=10.0 -> [0,...,9]
      duration=10.7 -> [0,...,10]
      duration=0.4  -> [0]
    """
    if duration_seconds <= 0:
        raise ValueError(
            f"Segment duration must be positive, got {duration_seconds}"
        )

    step = 1.0 / TARGET_FPS

    # ceil(duration / step), with tiny epsilon to keep exact boundaries stable.
    count = max(
        1,
        int(math.ceil(duration_seconds / step - 1e-9)),
    )

    times = np.arange(count, dtype=np.float64) * step

    # Defensive filter.
    times = times[times < duration_seconds + 1e-9]

    if len(times) == 0:
        return np.array([0.0], dtype=np.float64)

    return times


def uniformly_cap_times(
    candidate_times: np.ndarray,
) -> np.ndarray:
    if len(candidate_times) <= MAX_FRAMES:
        selected = candidate_times.copy()
    else:
        positions = np.linspace(
            0,
            len(candidate_times) - 1,
            num=MAX_FRAMES,
        )
        positions = np.rint(positions).astype(np.int64)
        selected = candidate_times[positions]

    if not np.isclose(selected[0], candidate_times[0]):
        raise RuntimeError("First candidate frame was lost.")

    if not np.isclose(selected[-1], candidate_times[-1]):
        raise RuntimeError("Last candidate frame was lost.")

    if len(selected) > 1 and not np.all(np.diff(selected) > 0):
        raise RuntimeError(
            f"Selected frame times are not strictly increasing: {selected}"
        )

    return selected


def absolute_time_to_source_index(
    *,
    absolute_time_seconds: float,
    source_fps: float,
    total_frames: int,
) -> int:
    """
    Match the current Docker convention:
        np.rint(time * source_fps)
    """
    index = int(np.rint(absolute_time_seconds * source_fps))
    return int(np.clip(index, 0, total_frames - 1))


def make_frame_plan(
    *,
    dataset: str,
    request: Any,
    source_fps: float,
    source_total_frames: int,
) -> tuple[list[FramePlan], int]:
    start = float(request.start_time)
    end = float(request.end_time)
    duration = end - start

    if duration <= 0:
        raise ValueError(
            f"{request.qID}: invalid duration {duration:.6f}s"
        )

    if duration > 300.001:
        raise ValueError(
            f"{request.qID}: segment duration {duration:.3f}s exceeds 5 min"
        )

    candidates = one_fps_candidate_times(duration)
    selected = uniformly_cap_times(candidates)

    cache_dir = stable_video_cache_dir(
        dataset=dataset,
        video_id=str(request.videoID),
    )

    frames: list[FramePlan] = []

    for relative_time in selected:
        absolute_time = start + float(relative_time)

        source_index = absolute_time_to_source_index(
            absolute_time_seconds=absolute_time,
            source_fps=source_fps,
            total_frames=source_total_frames,
        )

        timestamp = seconds_to_hhmmss(absolute_time)

        # Timestamp is part of the filename because the same source frame could
        # theoretically be requested with a different visible clock label.
        filename = (
            f"frame_{source_index:09d}_"
            f"{timestamp.replace(':', '-')}.jpg"
        )

        frames.append(
            FramePlan(
                source_index=source_index,
                relative_time_seconds=float(relative_time),
                absolute_time_seconds=absolute_time,
                timestamp=timestamp,
                path=str(cache_dir / filename),
            )
        )

    expected = min(len(candidates), MAX_FRAMES)

    if len(frames) != expected:
        raise RuntimeError(
            f"{request.qID}: selected {len(frames)} frames, "
            f"expected {expected}"
        )

    return frames, len(candidates)


# =============================================================================
# LOAD ANNOTATIONS + PLAN
# =============================================================================

def load_all_plans(
    *,
    allow_count_mismatch: bool,
    smoke_test: bool,
) -> list[SamplePlan]:
    plans: list[SamplePlan] = []

    jobs = (
        ("heico", "train"),
        ("lapchole", "train"),
        ("heico", "test"),
        ("lapchole", "test"),
    )

    video_indices = {
        dataset: build_video_index(directory)
        for dataset, directory in SOURCE_VIDEO_DIRS.items()
    }

    # Same source video only needs FPS/frame-count probing once.
    video_info_cache: dict[
        tuple[str, str],
        tuple[Path, float, int, float],
    ] = {}

    for dataset_name, split in jobs:
        source_dir = SOURCE_VIDEO_DIRS[dataset_name]

        logging.info(
            "Loading FocusDataset(dataset=%s, split=%s, track=segment)...",
            dataset_name,
            split,
        )

        dataset = FocusDataset(
            dataset=dataset_name,
            split=split_to_enum(split),
            track=Track.SEGMENT,
        )

        actual_count = len(dataset)
        expected_count = EXPECTED_COUNTS[(dataset_name, split)]

        logging.info(
            "%s/%s: %d VQAs",
            dataset_name,
            split,
            actual_count,
        )

        if (
            not allow_count_mismatch
            and actual_count != expected_count
        ):
            raise RuntimeError(
                f"{dataset_name}/{split} has {actual_count} VQAs, "
                f"expected {expected_count}. "
                "If this is an intentional dataset revision, rerun with "
                "--allow-count-mismatch."
            )

        selected_video_ids: set[str] | None = None

        if smoke_test:
            video_limit = SMOKE_VIDEO_LIMITS[(dataset_name, split)]

            available_video_ids = sorted(
                {
                    str(request.videoID)
                    for request, _ in dataset
                }
            )

            if video_limit > len(available_video_ids):
                raise RuntimeError(
                    f"Smoke mode requests {video_limit} videos for "
                    f"{dataset_name}/{split}, but only "
                    f"{len(available_video_ids)} are available."
                )

            selected_video_ids = set(
                available_video_ids[:video_limit]
            )

            logging.info(
                "  smoke %s/%s: selected %d source video(s): %s",
                dataset_name,
                split,
                len(selected_video_ids),
                sorted(selected_video_ids),
            )

        selected_sample_count = 0

        for request, reference in dataset:
            video_id = str(request.videoID)

            if (
                selected_video_ids is not None
                and video_id not in selected_video_ids
            ):
                continue

            selected_sample_count += 1
            ordinal = selected_sample_count
            video_key = (dataset_name, video_id)

            if video_key not in video_info_cache:
                source_path = resolve_video(
                    video_id=video_id,
                    directory=source_dir,
                    index=video_indices[dataset_name],
                )

                fps, total_frames, source_duration = inspect_video(source_path)

                start_time = float(request.start_time)
                end_time = float(request.end_time)

                # Requests should lie within the original full-procedure video.
                if start_time < 0:
                    raise RuntimeError(
                        f"{request.qID}: negative start_time={start_time}"
                    )

                if end_time > source_duration + 1.0:
                    raise RuntimeError(
                        f"{request.qID}: end_time={end_time} "
                        f"exceeds source duration={source_duration:.3f}s "
                        f"for {source_path}"
                    )

                video_info_cache[video_key] = (
                    source_path,
                    fps,
                    total_frames,
                    source_duration,
                )

            (
                source_path,
                fps,
                total_frames,
                source_duration,
            ) = video_info_cache[video_key]

            frames, candidate_count = make_frame_plan(
                dataset=dataset_name,
                request=request,
                source_fps=fps,
                source_total_frames=total_frames,
            )

            sample_id = (
                f"{dataset_name}_{split}_{request.qID}"
            )

            plans.append(
                SamplePlan(
                    dataset=dataset_name,
                    split=split,
                    sample_id=sample_id,
                    qid=str(request.qID),
                    request=request,
                    reference=reference,
                    source_path=source_path,
                    source_fps=fps,
                    source_total_frames=total_frames,
                    candidate_count=candidate_count,
                    frames=frames,
                )
            )

            if ordinal % 1000 == 0:
                if smoke_test:
                    logging.info(
                        "  planned %d selected %s/%s samples",
                        ordinal,
                        dataset_name,
                        split,
                    )
                else:
                    logging.info(
                        "  planned %d/%d %s/%s samples",
                        ordinal,
                        actual_count,
                        dataset_name,
                        split,
                    )

    return plans


def validate_smoke_selection(
    plans: Sequence[SamplePlan],
) -> None:
    """Verify exactly 4 unique train videos and 1 unique test video."""
    train_videos = {
        (plan.dataset, str(plan.request.videoID))
        for plan in plans
        if plan.split == "train"
    }

    test_videos = {
        (plan.dataset, str(plan.request.videoID))
        for plan in plans
        if plan.split == "test"
    }

    if len(train_videos) != 4:
        raise RuntimeError(
            "Smoke selection must contain exactly 4 unique train videos; "
            f"got {len(train_videos)}: {sorted(train_videos)}"
        )

    if len(test_videos) != 1:
        raise RuntimeError(
            "Smoke selection must contain exactly 1 unique test video; "
            f"got {len(test_videos)}: {sorted(test_videos)}"
        )

    if not any(
        plan.dataset == "heico" and plan.split == "test"
        for plan in plans
    ):
        raise RuntimeError(
            "Smoke selection must contain one HeiCo test video."
        )

    if any(
        plan.dataset == "lapchole" and plan.split == "test"
        for plan in plans
    ):
        raise RuntimeError(
            "Smoke selection should not contain LapChole test samples."
        )

    logging.info(
        "Smoke selection validated: 4 train videos + 1 test video."
    )
    logging.info("  train videos: %s", sorted(train_videos))
    logging.info("  test video: %s", sorted(test_videos))


# =============================================================================
# IMAGE PREPROCESSING + CACHE
# =============================================================================

def platform_like_dimensions(
    width: int,
    height: int,
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid frame resolution: {width}x{height}"
        )

    target_height = min(height, MAX_HEIGHT)
    target_height = max(
        2,
        target_height - (target_height % 2),
    )

    scale = target_height / float(height)

    target_width = max(
        2,
        int(round((width * scale) / 2.0)) * 2,
    )

    return target_width, target_height


def prepare_one_image(
    *,
    rgb_frame: np.ndarray,
    timestamp: str,
) -> np.ndarray:
    """
    Resize first, then draw timestamp.

    Returns BGR image for OpenCV JPEG encoding.
    """
    if (
        rgb_frame.ndim != 3
        or rgb_frame.shape[2] != 3
    ):
        raise ValueError(
            f"Expected RGB HxWx3, got {rgb_frame.shape}"
        )

    height, width = rgb_frame.shape[:2]

    target_width, target_height = platform_like_dimensions(
        width,
        height,
    )

    if (
        target_width != width
        or target_height != height
    ):
        rgb_frame = cv2.resize(
            rgb_frame,
            (target_width, target_height),
            interpolation=cv2.INTER_LANCZOS4,
        )

    bgr = cv2.cvtColor(
        rgb_frame,
        cv2.COLOR_RGB2BGR,
    )

    cv2.putText(
        bgr,
        timestamp,
        TEXT_ORG,
        TEXT_FONT_FACE,
        TEXT_SCALE,
        TEXT_COLOR_BGR,
        thickness=TEXT_THICKNESS,
        lineType=TEXT_LINE_TYPE,
    )

    return bgr


def cache_file_exists(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size > 0
    )


def atomic_write_jpeg(
    path: Path,
    image_bgr: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ok, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            JPEG_QUALITY,
        ],
    )

    if not ok:
        raise RuntimeError(
            f"JPEG encoding failed: {path}"
        )

    tmp = path.with_name(
        f".{path.name}.tmp"
    )

    with tmp.open("wb") as file:
        file.write(encoded.tobytes())

    os.replace(tmp, path)


def collect_cache_jobs(
    plans: Sequence[SamplePlan],
) -> dict[
    tuple[str, str],
    dict[str, Any],
]:
    """
    Group by physical source video.

    jobs[source_index][timestamp] -> relative output path
    """
    groups: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for plan in plans:
        key = (
            plan.dataset,
            str(plan.request.videoID),
        )

        if key not in groups:
            groups[key] = {
                "source_path": plan.source_path,
                "jobs": defaultdict(dict),
            }

        jobs = groups[key]["jobs"]

        for frame in plan.frames:
            existing = jobs[
                frame.source_index
            ].get(frame.timestamp)

            if (
                existing is not None
                and existing != frame.path
            ):
                raise RuntimeError(
                    "Cache-path collision:\n"
                    f"existing={existing}\n"
                    f"new={frame.path}"
                )

            jobs[
                frame.source_index
            ][frame.timestamp] = frame.path

    return groups


def decode_video_cache(
    *,
    output_root: Path,
    source_path: Path,
    jobs: dict[int, dict[str, str]],
    overwrite: bool,
) -> tuple[int, int]:
    """
    Decode only source frames for which at least one cached JPEG is missing.

    Returns:
        written_count, reused_count
    """
    required_indices: list[int] = []
    reused = 0

    for source_index, timestamp_paths in jobs.items():
        need_decode = False

        for relative_path in timestamp_paths.values():
            output_path = output_root / relative_path

            if (
                overwrite
                or not cache_file_exists(output_path)
            ):
                need_decode = True
            else:
                reused += 1

        if need_decode:
            required_indices.append(source_index)

    if not required_indices:
        return 0, reused

    required_indices = sorted(
        set(required_indices)
    )

    vr = decord.VideoReader(
        str(source_path),
        ctx=decord.cpu(0),
        num_threads=DECORD_THREADS,
    )

    written = 0

    for batch_start in range(
        0,
        len(required_indices),
        DECODE_BATCH_SIZE,
    ):
        indices = required_indices[
            batch_start:
            batch_start + DECODE_BATCH_SIZE
        ]

        rgb_batch = vr.get_batch(
            indices
        ).asnumpy()

        for source_index, rgb_frame in zip(
            indices,
            rgb_batch,
            strict=True,
        ):
            for (
                timestamp,
                relative_path,
            ) in jobs[source_index].items():
                output_path = (
                    output_root / relative_path
                )

                if (
                    not overwrite
                    and cache_file_exists(output_path)
                ):
                    continue

                image_bgr = prepare_one_image(
                    rgb_frame=rgb_frame,
                    timestamp=timestamp,
                )

                atomic_write_jpeg(
                    output_path,
                    image_bgr,
                )

                written += 1

    del vr

    return written, reused


def build_cache(
    *,
    plans: Sequence[SamplePlan],
    output_root: Path,
    overwrite: bool,
) -> tuple[int, int]:
    groups = collect_cache_jobs(plans)

    unique_cache_paths = {
        frame.path
        for plan in plans
        for frame in plan.frames
    }

    total_unique = len(unique_cache_paths)

    logging.info(
        "Frame cache: %d source videos, %d unique JPEGs",
        len(groups),
        total_unique,
    )

    started = time.monotonic()

    written_total = 0
    reused_total = 0

    for video_number, group in enumerate(
        groups.values(),
        start=1,
    ):
        written, reused = decode_video_cache(
            output_root=output_root,
            source_path=group["source_path"],
            jobs=group["jobs"],
            overwrite=overwrite,
        )

        written_total += written
        reused_total += reused

        elapsed = time.monotonic() - started

        completed = min(
            total_unique,
            written_total + reused_total,
        )

        rate = (
            completed / elapsed
            if elapsed > 0
            else 0.0
        )

        remaining = max(
            0,
            total_unique - completed,
        )

        eta_seconds = (
            remaining / rate
            if rate > 0
            else float("inf")
        )

        if (
            video_number == 1
            or video_number % 5 == 0
            or video_number == len(groups)
        ):
            eta_text = (
                f"{eta_seconds / 60:.1f} min"
                if np.isfinite(eta_seconds)
                else "unknown"
            )

            logging.info(
                "[%d/%d videos] "
                "written=%d reused=%d "
                "rate=%.2f images/s ETA=%s",
                video_number,
                len(groups),
                written_total,
                reused_total,
                rate,
                eta_text,
            )

    return written_total, reused_total


# =============================================================================
# MANIFESTS
# =============================================================================

def sample_record(
    plan: SamplePlan,
) -> dict[str, Any]:
    primary = capability_code(
        getattr(
            plan.reference,
            "primary",
            "",
        )
    )

    request = plan.request

    return {
        "sample_id": plan.sample_id,
        "qID": plan.qid,
        "dataset": plan.dataset,
        "split": plan.split,
        "videoID": str(request.videoID),
        "source_video": str(plan.source_path),
        "procedure_type": str(
            request.procedure_type
        ),
        "start_time": float(
            request.start_time
        ),
        "end_time": float(
            request.end_time
        ),
        "duration_seconds": (
            float(request.end_time)
            - float(request.start_time)
        ),
        "question": str(
            request.question
        ),
        "answer": reference_answer(
            plan.reference
        ),
        "answer_format": answer_format_name(
            plan.reference
        ),
        "primary_capability": primary,
        "capability_group": (
            primary[:1]
            if primary
            else ""
        ),
        "secondary_capabilities": (
            secondary_capabilities(
                plan.reference
            )
        ),
        "ood": bool(
            getattr(
                plan.reference,
                "ood",
                False,
            )
        ),
        "clinical": bool(
            getattr(
                plan.reference,
                "clinical",
                False,
            )
        ),
        "source_fps": plan.source_fps,
        "source_total_frames": (
            plan.source_total_frames
        ),
        "candidate_count_1fps": (
            plan.candidate_count
        ),
        "num_frames": len(
            plan.frames
        ),
        "frames": [
            asdict(frame)
            for frame in plan.frames
        ],
    }


def write_manifests(
    *,
    plans: Sequence[SamplePlan],
    output_root: Path,
) -> None:
    train = [
        plan
        for plan in plans
        if plan.split == "train"
    ]

    val_heico = [
        plan
        for plan in plans
        if (
            plan.dataset == "heico"
            and plan.split == "test"
        )
    ]

    val_lapchole = [
        plan
        for plan in plans
        if (
            plan.dataset == "lapchole"
            and plan.split == "test"
        )
    ]

    val_all = (
        val_heico
        + val_lapchole
    )

    atomic_write_jsonl(
        output_root
        / "train_manifest.jsonl",
        (
            sample_record(plan)
            for plan in train
        ),
    )

    atomic_write_jsonl(
        output_root
        / "val_heico_manifest.jsonl",
        (
            sample_record(plan)
            for plan in val_heico
        ),
    )

    atomic_write_jsonl(
        output_root
        / "val_lapchole_manifest.jsonl",
        (
            sample_record(plan)
            for plan in val_lapchole
        ),
    )

    atomic_write_jsonl(
        output_root
        / "val_all_manifest.jsonl",
        (
            sample_record(plan)
            for plan in val_all
        ),
    )

    write_csv_manifest(
        plans=plans,
        output_root=output_root,
    )

    logging.info(
        "Manifest counts: "
        "train=%d, val_heico=%d, "
        "val_lapchole=%d, val_all=%d",
        len(train),
        len(val_heico),
        len(val_lapchole),
        len(val_all),
    )


def write_csv_manifest(
    *,
    plans: Sequence[SamplePlan],
    output_root: Path,
) -> None:
    path = output_root / "manifest.csv"
    tmp = path.with_name(
        f".{path.name}.tmp"
    )

    fieldnames = [
        "sample_id",
        "qID",
        "dataset",
        "split",
        "videoID",
        "source_video",
        "procedure_type",
        "start_time",
        "end_time",
        "duration_seconds",
        "question",
        "answer",
        "answer_format",
        "primary_capability",
        "capability_group",
        "secondary_capabilities",
        "ood",
        "clinical",
        "source_fps",
        "source_total_frames",
        "candidate_count_1fps",
        "num_frames",
        "frame_paths",
        "frame_timestamps",
        "frame_relative_times",
        "frame_absolute_times",
        "source_frame_indices",
    ]

    with tmp.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for plan in plans:
            record = sample_record(plan)
            frames = record.pop("frames")

            writer.writerow(
                {
                    **record,
                    "secondary_capabilities": json.dumps(
                        record[
                            "secondary_capabilities"
                        ],
                        ensure_ascii=False,
                    ),
                    "frame_paths": json.dumps(
                        [
                            frame["path"]
                            for frame in frames
                        ],
                        ensure_ascii=False,
                    ),
                    "frame_timestamps": json.dumps(
                        [
                            frame["timestamp"]
                            for frame in frames
                        ],
                        ensure_ascii=False,
                    ),
                    "frame_relative_times": json.dumps(
                        [
                            frame[
                                "relative_time_seconds"
                            ]
                            for frame in frames
                        ]
                    ),
                    "frame_absolute_times": json.dumps(
                        [
                            frame[
                                "absolute_time_seconds"
                            ]
                            for frame in frames
                        ]
                    ),
                    "source_frame_indices": json.dumps(
                        [
                            frame[
                                "source_index"
                            ]
                            for frame in frames
                        ]
                    ),
                }
            )

    os.replace(tmp, path)


# =============================================================================
# VALIDATION + SUMMARY
# =============================================================================

def validate_plans(
    plans: Sequence[SamplePlan],
) -> None:
    sample_ids = [
        plan.sample_id
        for plan in plans
    ]

    if len(sample_ids) != len(
        set(sample_ids)
    ):
        duplicate_ids = [
            sample_id
            for (
                sample_id,
                count,
            ) in Counter(
                sample_ids
            ).items()
            if count > 1
        ]

        raise RuntimeError(
            "Duplicate sample IDs: "
            f"{duplicate_ids[:20]}"
        )

    errors: list[str] = []

    for plan in plans:
        if not plan.frames:
            errors.append(
                f"{plan.sample_id}: no frames"
            )
            continue

        if len(plan.frames) > MAX_FRAMES:
            errors.append(
                f"{plan.sample_id}: "
                f"{len(plan.frames)} > "
                f"MAX_FRAMES={MAX_FRAMES}"
            )

        # First candidate must always be segment-relative t=0.
        if not np.isclose(
            plan.frames[0]
            .relative_time_seconds,
            0.0,
        ):
            errors.append(
                f"{plan.sample_id}: "
                "first frame is not t=0"
            )

        relative_times = np.array(
            [
                frame.relative_time_seconds
                for frame in plan.frames
            ],
            dtype=np.float64,
        )

        if (
            len(relative_times) > 1
            and not np.all(
                np.diff(relative_times) > 0
            )
        ):
            errors.append(
                f"{plan.sample_id}: "
                "frame times not chronological"
            )

        duration = (
            float(plan.request.end_time)
            - float(plan.request.start_time)
        )

        candidates = (
            one_fps_candidate_times(
                duration
            )
        )

        selected = (
            uniformly_cap_times(
                candidates
            )
        )

        if plan.candidate_count != len(
            candidates
        ):
            errors.append(
                f"{plan.sample_id}: "
                "candidate-count mismatch"
            )

        if not np.isclose(
            plan.frames[-1]
            .relative_time_seconds,
            selected[-1],
        ):
            errors.append(
                f"{plan.sample_id}: "
                "last candidate not preserved"
            )

        if len(errors) >= 100:
            break

    if errors:
        raise RuntimeError(
            "Plan validation failed:\n  - "
            + "\n  - ".join(errors)
        )


def validate_cache_files(
    *,
    plans: Sequence[SamplePlan],
    output_root: Path,
) -> None:
    unique_paths = {
        frame.path
        for plan in plans
        for frame in plan.frames
    }

    missing: list[str] = []

    for relative_path in unique_paths:
        path = (
            output_root
            / relative_path
        )

        if not cache_file_exists(path):
            missing.append(str(path))

            if len(missing) >= 100:
                break

    if missing:
        raise RuntimeError(
            "Missing/empty cached JPEGs:\n  - "
            + "\n  - ".join(missing)
        )


def make_summary(
    *,
    plans: Sequence[SamplePlan],
    output_root: Path,
    plan_only: bool,
    written: int,
    reused: int,
) -> dict[str, Any]:
    split_counts = Counter(
        (
            plan.dataset,
            plan.split,
        )
        for plan in plans
    )

    group_counts = Counter(
        capability_code(
            getattr(
                plan.reference,
                "primary",
                "",
            )
        )[:1]
        for plan in plans
    )

    leaf_counts = Counter(
        capability_code(
            getattr(
                plan.reference,
                "primary",
                "",
            )
        )
        for plan in plans
    )

    total_frame_references = sum(
        len(plan.frames)
        for plan in plans
    )

    unique_frame_paths = {
        frame.path
        for plan in plans
        for frame in plan.frames
    }

    frame_counts = [
        len(plan.frames)
        for plan in plans
    ]

    durations = [
        (
            float(plan.request.end_time)
            - float(plan.request.start_time)
        )
        for plan in plans
    ]

    return {
        "purpose": (
            "Frame cache and metadata only. "
            "No prompt construction."
        ),
        "source_video_directories": {
            "heico": str(
                HEICO_VIDEO_DIR
            ),
            "lapchole": str(
                LAPCHOLE_VIDEO_DIR
            ),
        },
        "configuration": {
            "target_fps": TARGET_FPS,
            "max_frames": MAX_FRAMES,
            "max_height": MAX_HEIGHT,
            "jpeg_quality": JPEG_QUALITY,
            "intermediate_mp4_created": False,
            "orena_focus_used_for": (
                "Loading released VQA "
                "annotations only"
            ),
            "timestamp_overlay": {
                "absolute_procedure_time": True,
                "format": "HH:MM:SS",
                "origin": list(
                    TEXT_ORG
                ),
                "scale": TEXT_SCALE,
                "thickness": (
                    TEXT_THICKNESS
                ),
            },
        },
        "counts": {
            "total_vqas": len(plans),
            "by_dataset_split": {
                f"{dataset}/{split}": count
                for (
                    dataset,
                    split,
                ), count in sorted(
                    split_counts.items()
                )
            },
            "total_frame_references": (
                total_frame_references
            ),
            "unique_cached_jpegs": len(
                unique_frame_paths
            ),
            "deduplicated_references": (
                total_frame_references
                - len(unique_frame_paths)
            ),
            "written_this_run": written,
            "reused_this_run": reused,
            "capability_groups": dict(
                sorted(
                    group_counts.items()
                )
            ),
            "capability_leaves": dict(
                sorted(
                    leaf_counts.items()
                )
            ),
        },
        "frames_per_vqa": {
            "min": min(frame_counts),
            "max": max(frame_counts),
            "mean": (
                sum(frame_counts)
                / len(frame_counts)
            ),
            "median": float(
                np.median(frame_counts)
            ),
        },
        "duration_seconds": {
            "min": min(durations),
            "max": max(durations),
            "mean": (
                sum(durations)
                / len(durations)
            ),
            "median": float(
                np.median(durations)
            ),
        },
        "plan_only": plan_only,
        "output_root": str(
            output_root
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    args = parse_args()
    configure_logging()

    requested_output_root = args.output_root

    if (
        args.smoke_test
        and requested_output_root == DEFAULT_OUTPUT_ROOT
    ):
        requested_output_root = SMOKE_OUTPUT_ROOT

    output_root = (
        requested_output_root
        .expanduser()
        .resolve()
    )

    started = time.monotonic()

    logging.info(
        "=== ORena SEGMENT DoRA frame preparation ==="
    )
    logging.info(
        "HeiCo videos: %s",
        HEICO_VIDEO_DIR,
    )
    logging.info(
        "LapChole videos: %s",
        LAPCHOLE_VIDEO_DIR,
    )
    logging.info(
        "Output: %s",
        output_root,
    )
    logging.info(
        "Sampling: %.1f fps, max %d frames",
        TARGET_FPS,
        MAX_FRAMES,
    )
    logging.info(
        "No prompt construction. No intermediate MP4s."
    )

    if args.smoke_test:
        logging.info(
            "SMOKE MODE: 4 train source videos "
            "(2 HeiCo + 2 LapChole) + 1 HeiCo test source video."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_root
        / "frames"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    plans = load_all_plans(
        allow_count_mismatch=(
            args.allow_count_mismatch
        ),
        smoke_test=args.smoke_test,
    )

    validate_plans(plans)

    if args.smoke_test:
        validate_smoke_selection(plans)

    total_frame_refs = sum(
        len(plan.frames)
        for plan in plans
    )

    unique_frame_paths = {
        frame.path
        for plan in plans
        for frame in plan.frames
    }

    logging.info(
        "Planned %d VQAs",
        len(plans),
    )
    logging.info(
        "Total per-VQA frame references: %d",
        total_frame_refs,
    )
    logging.info(
        "Unique cached JPEGs after "
        "deduplication: %d",
        len(unique_frame_paths),
    )
    logging.info(
        "Duplicate frame references saved: %d",
        (
            total_frame_refs
            - len(unique_frame_paths)
        ),
    )

    # Write the exact frame plan before the long extraction stage.
    write_manifests(
        plans=plans,
        output_root=output_root,
    )

    atomic_write_json(
        output_root / "summary.json",
        make_summary(
            plans=plans,
            output_root=output_root,
            plan_only=args.plan_only,
            written=0,
            reused=0,
        ),
    )

    written = 0
    reused = 0

    if args.plan_only:
        logging.info(
            "--plan-only: "
            "skipping JPEG extraction"
        )
    else:
        written, reused = build_cache(
            plans=plans,
            output_root=output_root,
            overwrite=args.overwrite,
        )

        validate_cache_files(
            plans=plans,
            output_root=output_root,
        )

    atomic_write_json(
        output_root / "summary.json",
        make_summary(
            plans=plans,
            output_root=output_root,
            plan_only=args.plan_only,
            written=written,
            reused=reused,
        ),
    )

    elapsed = (
        time.monotonic()
        - started
    )

    logging.info(
        "=== complete in %.1f min ===",
        elapsed / 60.0,
    )

    logging.info(
        "Next step is a SEPARATE script "
        "that builds InternVL SFT JSONL/prompts "
        "from these manifests."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
