"""Video utilities for local ORena FOCUS SEGMENT experiments.

This module reads a time window directly from a full surgical video, samples it
at an approximate target FPS, and uniformly caps the result to a manageable
number of frames for a VLM.

The returned frames are RGB PIL images in chronological order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import decord
import numpy as np
from PIL import Image


@dataclass
class SegmentFrames:
    """Frames sampled from one time window of a full procedure video."""

    images: list[Image.Image]
    frame_indices: list[int]
    timestamps: list[float]
    source_fps: float
    video_duration: float
    requested_start_time: float
    requested_end_time: float

    @property
    def num_frames(self) -> int:
        """Number of returned frames."""
        return len(self.images)


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert ``HH:MM:SS`` or ``HH:MM:SS.sss`` to seconds."""

    parts = timestamp.strip().split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Timestamp must have format HH:MM:SS, got {timestamp!r}."
        )

    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])

    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"Invalid timestamp: {timestamp!r}.")

    return hours * 3600 + minutes * 60 + seconds


def seconds_to_timestamp(seconds: float) -> str:
    """Convert non-negative seconds to ``HH:MM:SS``."""

    if seconds < 0:
        raise ValueError("seconds must be non-negative.")

    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _uniformly_cap_indices(
    frame_indices: np.ndarray,
    max_frames: int | None,
) -> np.ndarray:
    """Uniformly reduce sorted frame indices to at most ``max_frames``."""

    if max_frames is None or len(frame_indices) <= max_frames:
        return frame_indices

    if max_frames <= 0:
        raise ValueError("max_frames must be positive or None.")

    positions = np.linspace(
        0,
        len(frame_indices) - 1,
        num=max_frames,
    )
    positions = np.rint(positions).astype(np.int64)
    return frame_indices[positions]


def load_segment_frames(
    video_path: str | Path,
    start_time: float,
    end_time: float,
    target_fps: float = 1.0,
    max_frames: int | None = 8,
    num_threads: int = 1,
) -> SegmentFrames:
    """Load sampled frames from a window of a full surgical video.

    The function first forms a candidate sequence sampled at approximately
    ``target_fps`` over ``[start_time, end_time]``. If that sequence is longer
    than ``max_frames``, it uniformly selects ``max_frames`` candidates across
    the complete segment.

    Only the final selected frames are decoded.
    """

    path = Path(video_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")

    if start_time < 0:
        raise ValueError("start_time must be non-negative.")

    if end_time <= start_time:
        raise ValueError(
            f"end_time must be greater than start_time, got "
            f"{start_time:.3f} to {end_time:.3f}."
        )

    if target_fps <= 0:
        raise ValueError("target_fps must be positive.")

    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive or None.")

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

    # Approximate timestamp of the final decodable frame.
    video_duration = (total_frames - 1) / source_fps

    if start_time > video_duration:
        raise ValueError(
            f"start_time {start_time:.3f}s is beyond the video duration "
            f"{video_duration:.3f}s for {path.name}."
        )

    # Dataset timestamps can occasionally point slightly beyond the final
    # decodable frame, so clamp the requested end to the actual video.
    effective_end_time = min(end_time, video_duration)

    step_seconds = 1.0 / target_fps
    candidate_count = int(
        np.floor((effective_end_time - start_time) / step_seconds)
    ) + 1

    candidate_times = (
        start_time
        + np.arange(candidate_count, dtype=np.float64) * step_seconds
    )

    # Convert requested times to nearest source frame indices.
    frame_indices = np.rint(candidate_times * source_fps).astype(np.int64)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)

    # np.unique preserves sorted order and removes duplicates when the target
    # FPS is higher than the source FPS.
    frame_indices = np.unique(frame_indices)
    frame_indices = _uniformly_cap_indices(frame_indices, max_frames)

    if len(frame_indices) == 0:
        raise RuntimeError(
            f"No valid frames selected from {path.name} for "
            f"{start_time:.3f}s to {end_time:.3f}s."
        )

    decoded = video_reader.get_batch(frame_indices.tolist()).asnumpy()
    del video_reader

    images = [Image.fromarray(frame) for frame in decoded]
    actual_timestamps = [
        float(frame_index / source_fps)
        for frame_index in frame_indices
    ]

    return SegmentFrames(
        images=images,
        frame_indices=[int(index) for index in frame_indices],
        timestamps=actual_timestamps,
        source_fps=source_fps,
        video_duration=video_duration,
        requested_start_time=float(start_time),
        requested_end_time=float(end_time),
    )


def save_debug_frames(
    segment: SegmentFrames,
    output_dir: str | Path,
    prefix: str = "frame",
) -> None:
    """Save sampled frames as JPEGs for visual inspection."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for order, (image, frame_index, timestamp) in enumerate(
        zip(
            segment.images,
            segment.frame_indices,
            segment.timestamps,
            strict=True,
        )
    ):
        timestamp_text = seconds_to_timestamp(timestamp).replace(":", "-")
        filename = (
            f"{prefix}_{order:02d}_"
            f"{timestamp_text}_"
            f"sourceframe{frame_index:07d}.jpg"
        )
        image.save(output_path / filename, quality=95)


if __name__ == "__main__":
    # Standalone smoke test. Edit these constants directly; no command-line
    # arguments are required.
    project_root = Path(
        "/cs/student/projects1/aibh/2024/wenhuawe/ORENA"
    )
    test_video = (
        project_root
        / "heico-focus-vqa"
        / "train"
        / "0009 - Heico - Prokto - 10.avi"
    )
    output_dir = project_root / "internvl_fewshot" / "video_utils_test_frames"
    print('HI')
    # Known HeiCo SEGMENT example:
    # timestamp_start = 00:09:15
    # timestamp_end   = 00:14:14
    segment = load_segment_frames(
        video_path=test_video,
        start_time=timestamp_to_seconds("00:09:15"),
        end_time=timestamp_to_seconds("00:14:14"),
        target_fps=1.0,
        max_frames=8,
    )

    print(f"Video: {test_video}")
    print(f"Source FPS: {segment.source_fps:.3f}")
    print(f"Video duration: {seconds_to_timestamp(segment.video_duration)}")
    print(f"Returned frames: {segment.num_frames}")

    for index, (frame_index, timestamp) in enumerate(
        zip(segment.frame_indices, segment.timestamps, strict=True)
    ):
        print(
            f"  {index:02d}: "
            f"source frame {frame_index:7d}, "
            f"time {seconds_to_timestamp(timestamp)}"
        )

    save_debug_frames(segment, output_dir)
    print(f"Saved debug frames to: {output_dir}")
