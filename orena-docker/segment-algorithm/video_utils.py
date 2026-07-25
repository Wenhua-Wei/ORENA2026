"""Video utilities for ORena FOCUS SEGMENT Docker inference.

Each input video is already trimmed to the question's requested window.
Frames are therefore sampled from the beginning of the supplied clip.

Returned timestamps are relative to the beginning of the supplied clip.
The caller must add request.start_time to obtain original-procedure times.
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path

# Import torch before decord to avoid Decord/CUDA initialisation problems.
import torch  # noqa: F401
import decord
import numpy as np
from PIL import Image


@dataclass
class ClipFrames:
    """Frames sampled from one already-trimmed question clip."""

    images: list[Image.Image]
    frame_indices: list[int]
    timestamps: list[float]
    source_fps: float
    clip_duration: float

    @property
    def num_frames(self) -> int:
        return len(self.images)


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS using the floored second."""

    if seconds < 0:
        raise ValueError("seconds must be non-negative.")

    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _uniformly_cap_indices(
    frame_indices: np.ndarray,
    max_frames: int | None,
) -> np.ndarray:
    """Uniformly retain no more than max_frames indices."""

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


def load_clip_frames(
    video_path: str | Path,
    *,
    target_fps: float = 1.0,
    max_frames: int | None = 20,
    num_threads: int = 1,
) -> ClipFrames:
    """Sample an already-trimmed SEGMENT clip.

    The supplied clip is decoded from its own beginning. The returned
    timestamps are clip-relative seconds.
    """

    path = Path(video_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")

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
        raise RuntimeError(
            f"Video contains no decodable frames: {path}"
        )

    source_fps = float(video_reader.get_avg_fps())

    if not np.isfinite(source_fps) or source_fps <= 0:
        raise RuntimeError(
            f"Could not determine a valid FPS for {path}: "
            f"{source_fps}"
        )

    last_frame_time = (total_frames - 1) / source_fps
    clip_duration = total_frames / source_fps
    step_seconds = 1.0 / target_fps

    candidate_count = (
        int(np.floor(last_frame_time / step_seconds)) + 1
    )

    candidate_times = (
        np.arange(candidate_count, dtype=np.float64)
        * step_seconds
    )

    frame_indices = np.rint(
        candidate_times * source_fps
    ).astype(np.int64)

    frame_indices = np.clip(
        frame_indices,
        0,
        total_frames - 1,
    )

    frame_indices = np.unique(frame_indices)
    frame_indices = _uniformly_cap_indices(
        frame_indices,
        max_frames,
    )

    if len(frame_indices) == 0:
        raise RuntimeError(
            f"No valid frames selected from {path.name}."
        )

    decoded = video_reader.get_batch(
        frame_indices.tolist()
    ).asnumpy()

    del video_reader

    images = [
        Image.fromarray(frame).convert("RGB")
        for frame in decoded
    ]

    relative_timestamps = [
        float(frame_index / source_fps)
        for frame_index in frame_indices
    ]

    return ClipFrames(
        images=images,
        frame_indices=[
            int(frame_index)
            for frame_index in frame_indices
        ],
        timestamps=relative_timestamps,
        source_fps=source_fps,
        clip_duration=clip_duration,
    )


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

        timestamp_text = seconds_to_timestamp(
            label_time
        ).replace(":", "-")

        filename = (
            f"{prefix}_{order:02d}_"
            f"{timestamp_text}_"
            f"frame{frame_index:07d}.jpg"
        )

        image.save(
            output_path / filename,
            quality=95,
        )

def _validate_clip_frames(
    clip: ClipFrames,
    *,
    max_frames: int | None,
) -> None:
    """Validate the internal consistency of sampled clip frames."""

    if clip.num_frames == 0:
        raise RuntimeError("No frames were returned.")

    if max_frames is not None and clip.num_frames > max_frames:
        raise RuntimeError(
            f"Returned {clip.num_frames} frames, "
            f"which exceeds max_frames={max_frames}."
        )

    lengths = {
        len(clip.images),
        len(clip.frame_indices),
        len(clip.timestamps),
    }

    if len(lengths) != 1:
        raise RuntimeError(
            "images, frame_indices, and timestamps have "
            "different lengths."
        )

    if not np.isfinite(clip.source_fps) or clip.source_fps <= 0:
        raise RuntimeError(
            f"Invalid source FPS: {clip.source_fps}"
        )

    if not np.isfinite(clip.clip_duration) or clip.clip_duration <= 0:
        raise RuntimeError(
            f"Invalid clip duration: {clip.clip_duration}"
        )

    if clip.frame_indices != sorted(clip.frame_indices):
        raise RuntimeError(
            "Frame indices are not chronologically ordered."
        )

    if len(set(clip.frame_indices)) != len(clip.frame_indices):
        raise RuntimeError("Duplicate frame indices were returned.")

    if clip.timestamps != sorted(clip.timestamps):
        raise RuntimeError(
            "Frame timestamps are not chronologically ordered."
        )

    for image in clip.images:
        if image.mode != "RGB":
            raise RuntimeError(
                f"Expected RGB image, received mode={image.mode!r}."
            )

        if image.width <= 0 or image.height <= 0:
            raise RuntimeError(
                f"Invalid image dimensions: {image.size}"
            )

    for frame_index, timestamp in zip(
        clip.frame_indices,
        clip.timestamps,
        strict=True,
    ):
        expected_timestamp = frame_index / clip.source_fps

        if not np.isclose(
            timestamp,
            expected_timestamp,
            rtol=0.0,
            atol=1e-9,
        ):
            raise RuntimeError(
                "Timestamp does not correspond to its frame index: "
                f"frame={frame_index}, "
                f"timestamp={timestamp}, "
                f"expected={expected_timestamp}."
            )

        if timestamp < 0 or timestamp >= clip.clip_duration:
            raise RuntimeError(
                f"Timestamp outside clip: {timestamp:.6f}"
            )


def main() -> None:
    """Run a standalone smoke test on one trimmed SEGMENT clip."""

    default_video_path = Path(
        "/input/plain/q001.mp4"
    )

    video_path = Path(
        os.environ.get(
            "VIDEO_UTILS_TEST_VIDEO",
            str(default_video_path),
        )
    ).expanduser().resolve()

    output_dir = Path(
        os.environ.get(
            "VIDEO_UTILS_TEST_OUTPUT",
            "/tmp/video_utils_test_frames",
        )
    ).expanduser().resolve()

    absolute_start_time = float(
        os.environ.get(
            "VIDEO_UTILS_TEST_START_TIME",
            "132.5",
        )
    )

    target_fps = float(
        os.environ.get(
            "VIDEO_UTILS_TEST_FPS",
            "1.0",
        )
    )

    max_frames = int(
        os.environ.get(
            "VIDEO_UTILS_TEST_MAX_FRAMES",
            "20",
        )
    )

    print("=" * 80)
    print("ORena FOCUS video-utils smoke test")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Target FPS: {target_fps}")
    print(f"Maximum frames: {max_frames}")

    clip = load_clip_frames(
        video_path=video_path,
        target_fps=target_fps,
        max_frames=max_frames,
        num_threads=1,
    )

    _validate_clip_frames(
        clip,
        max_frames=max_frames,
    )

    print("\nDecoded clip")
    print(f"  Source FPS: {clip.source_fps:.6f}")
    print(f"  Clip duration: {clip.clip_duration:.3f} s")
    print(f"  Selected frames: {clip.num_frames}")
    print(
        "  First relative timestamp: "
        f"{clip.timestamps[0]:.3f} s"
    )
    print(
        "  Last relative timestamp: "
        f"{clip.timestamps[-1]:.3f} s"
    )

    print("\nSelected frames:")

    for order, (frame_index, relative_time, image) in enumerate(
        zip(
            clip.frame_indices,
            clip.timestamps,
            clip.images,
            strict=True,
        )
    ):
        absolute_time = absolute_start_time + relative_time

        print(
            f"  {order:02d}: "
            f"frame={frame_index:04d}, "
            f"relative={relative_time:7.3f} s, "
            f"absolute={seconds_to_timestamp(absolute_time)}, "
            f"size={image.size}"
        )

    save_debug_frames(
        clip=clip,
        output_dir=output_dir,
        absolute_start_time=absolute_start_time,
    )

    saved_files = sorted(output_dir.glob("*.jpg"))

    if len(saved_files) != clip.num_frames:
        raise RuntimeError(
            f"Expected {clip.num_frames} saved JPEGs, "
            f"found {len(saved_files)} in {output_dir}."
        )

    print(f"\nSaved {len(saved_files)} debug frames to:")
    print(f"  {output_dir}")

    # Run a second test specifically exercising the uniform frame cap.
    capped_clip = load_clip_frames(
        video_path=video_path,
        target_fps=5.0,
        max_frames=5,
        num_threads=1,
    )

    _validate_clip_frames(
        capped_clip,
        max_frames=5,
    )

    if capped_clip.num_frames != min(
        5,
        round(capped_clip.clip_duration * capped_clip.source_fps),
    ):
        raise RuntimeError(
            "Unexpected result from capped sampling test: "
            f"{capped_clip.num_frames} frames."
        )

    print("\nUniform-cap test passed.")
    print(f"  Selected indices: {capped_clip.frame_indices}")
    print(
        "  Relative timestamps: "
        + ", ".join(
            f"{timestamp:.3f}"
            for timestamp in capped_clip.timestamps
        )
    )

    print("\nAll video-utils smoke tests passed.")


if __name__ == "__main__":
    main()