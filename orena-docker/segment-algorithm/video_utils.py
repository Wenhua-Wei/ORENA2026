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
from sampling_utils import SamplingPlan



@dataclass
class ClipFrames:
    """Frames sampled from one already-trimmed question clip."""

    images: list[Image.Image]
    frame_indices: list[int]
    timestamps: list[float]
    source_fps: float
    clip_duration: float

    # Actual sampling mode used after all safety checks.
    sampling_mode: str = "GLOBAL"

    # Keep None for GLOBAL so prompt_utils can reproduce the old
    # GLOBAL wording exactly.
    sampling_description: str | None = None

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

def _nearest_candidate_index(
    candidate_indices: np.ndarray,
    *,
    target_relative_time: float,
    source_fps: float,
) -> int:
    """Return the candidate frame nearest one clip-relative time."""

    candidate_times = (
        candidate_indices.astype(np.float64)
        / source_fps
    )

    position = int(
        np.argmin(
            np.abs(
                candidate_times
                - target_relative_time
            )
        )
    )

    return int(candidate_indices[position])


def _select_interval_indices(
    candidate_indices: np.ndarray,
    *,
    source_fps: float,
    request_start_time: float,
    interval_abs_s: tuple[float, float],
    max_frames: int | None,
) -> np.ndarray | None:
    """
    Select frames only inside an explicit interval.

    Return None on any problem so the caller can safely retain GLOBAL.
    """

    t1_abs = float(interval_abs_s[0])
    t2_abs = float(interval_abs_s[1])

    if (
        not np.isfinite(t1_abs)
        or not np.isfinite(t2_abs)
        or not t1_abs < t2_abs
    ):
        return None

    relative_times = (
        candidate_indices.astype(np.float64)
        / source_fps
    )

    absolute_times = (
        float(request_start_time)
        + relative_times
    )

    mask = (
        (absolute_times >= t1_abs)
        & (absolute_times <= t2_abs)
    )

    interval_indices = candidate_indices[mask]

    if len(interval_indices) == 0:
        return None

    return _uniformly_cap_indices(
        interval_indices,
        max_frames,
    )


def _select_forward_indices(
    candidate_indices: np.ndarray,
    *,
    source_fps: float,
    request_start_time: float,
    anchor_abs_s: float,
    max_frames: int | None,
) -> np.ndarray | None:
    """
    FORWARD policy:

        4 local frames around the anchor:
            T-1, T, T+1, T+2

        remaining budget:
            uniformly distributed after T+2 to clip end

    Return None on any problem so the caller safely retains GLOBAL.
    """

    anchor_relative = (
        float(anchor_abs_s)
        - float(request_start_time)
    )

    if not np.isfinite(anchor_relative):
        return None

    candidate_times = (
        candidate_indices.astype(np.float64)
        / source_fps
    )

    if len(candidate_times) == 0:
        return None

    first_time = float(candidate_times[0])
    last_time = float(candidate_times[-1])

    if not (
        first_time
        <= anchor_relative
        <= last_time
    ):
        return None

    # ---------------------------------------------------------
    # Local context around the anchor.
    # ---------------------------------------------------------
    local_targets = (
        anchor_relative - 1.0,
        anchor_relative,
        anchor_relative + 1.0,
        anchor_relative + 2.0,
    )

    local_indices: list[int] = []

    for target in local_targets:
        if first_time <= target <= last_time:
            local_indices.append(
                _nearest_candidate_index(
                    candidate_indices,
                    target_relative_time=target,
                    source_fps=source_fps,
                )
            )

    local = np.array(
        sorted(set(local_indices)),
        dtype=np.int64,
    )

    # ---------------------------------------------------------
    # Broad coverage after the local anchor region.
    # ---------------------------------------------------------
    forward_mask = (
        candidate_times
        > anchor_relative + 2.0
    )

    forward_pool = candidate_indices[
        forward_mask
    ]

    if max_frames is None:
        broad = forward_pool

    else:
        remaining = (
            max_frames
            - len(local)
        )

        if remaining <= 0:
            broad = np.empty(
                0,
                dtype=np.int64,
            )

        else:
            broad = _uniformly_cap_indices(
                forward_pool,
                remaining,
            )

    selected = np.unique(
        np.concatenate(
            [local, broad]
        ).astype(np.int64)
    )

    if len(selected) == 0:
        return None

    if (
        max_frames is not None
        and len(selected) > max_frames
    ):
        selected = _uniformly_cap_indices(
            selected,
            max_frames,
        )

    return selected

def load_clip_frames(
    video_path: str | Path,
    *,
    target_fps: float = 1.0,
    max_frames: int | None = 20,
    num_threads: int = 1,
    sampling_plan: SamplingPlan | None = None,
    request_start_time: float | None = None,
) -> ClipFrames:
    """Sample an already-trimmed SEGMENT clip.

    GLOBAL:
        Preserve the original full-clip uniform sampling exactly.

    INTERVAL:
        Sample only inside the explicit interval from the question.

    FORWARD:
        Use local context around the anchor plus broad temporal
        coverage after the anchor.

    Returned timestamps are always clip-relative seconds.
    """

    path = Path(
        video_path
    ).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Video does not exist: {path}"
        )

    if target_fps <= 0:
        raise ValueError(
            "target_fps must be positive."
        )

    if (
        max_frames is not None
        and max_frames <= 0
    ):
        raise ValueError(
            "max_frames must be positive or None."
        )

    video_reader = decord.VideoReader(
        str(path),
        ctx=decord.cpu(0),
        num_threads=max(
            1,
            int(num_threads),
        ),
    )

    total_frames = len(video_reader)

    if total_frames == 0:
        raise RuntimeError(
            f"Video contains no decodable frames: "
            f"{path}"
        )

    source_fps = float(
        video_reader.get_avg_fps()
    )

    if (
        not np.isfinite(source_fps)
        or source_fps <= 0
    ):
        raise RuntimeError(
            f"Could not determine a valid FPS "
            f"for {path}: {source_fps}"
        )

    last_frame_time = (
        (total_frames - 1)
        / source_fps
    )

    clip_duration = (
        total_frames
        / source_fps
    )

    step_seconds = (
        1.0
        / target_fps
    )

    # ---------------------------------------------------------
    # Construct EXACTLY the same candidate grid as before.
    # ---------------------------------------------------------

    candidate_count = (
        int(
            np.floor(
                last_frame_time
                / step_seconds
            )
        )
        + 1
    )

    candidate_times = (
        np.arange(
            candidate_count,
            dtype=np.float64,
        )
        * step_seconds
    )

    candidate_indices = np.rint(
        candidate_times
        * source_fps
    ).astype(np.int64)

    candidate_indices = np.clip(
        candidate_indices,
        0,
        total_frames - 1,
    )

    candidate_indices = np.unique(
        candidate_indices
    )

    # =========================================================
    # FAIL-SAFE DESIGN:
    #
    # First calculate GLOBAL exactly as the old code did.
    # A specialized strategy is allowed to replace it only
    # after all its own checks succeed.
    # =========================================================

    frame_indices = (
        _uniformly_cap_indices(
            candidate_indices,
            max_frames,
        )
    )

    actual_mode = "GLOBAL"
    sampling_description = None

    # ---------------------------------------------------------
    # INTERVAL
    # ---------------------------------------------------------

    if (
        sampling_plan is not None
        and sampling_plan.mode
        == "INTERVAL"
        and sampling_plan.interval_abs_s
        is not None
        and request_start_time
        is not None
    ):

        specialized = (
            _select_interval_indices(
                candidate_indices,
                source_fps=source_fps,
                request_start_time=float(
                    request_start_time
                ),
                interval_abs_s=(
                    sampling_plan.interval_abs_s
                ),
                max_frames=max_frames,
            )
        )

        # Only override GLOBAL if specialized sampling succeeded.
        if specialized is not None:
            frame_indices = specialized
            actual_mode = "INTERVAL"

            t1, t2 = (
                sampling_plan.interval_abs_s
            )

            sampling_description = (
                f"The input consists of "
                f"{len(frame_indices)} "
                f"chronologically ordered frames "
                f"sampled from the explicit interval "
                f"{seconds_to_timestamp(t1)} to "
                f"{seconds_to_timestamp(t2)} "
                f"specified in the question. "
                f"When the interval contains more "
                f"candidate frames than the frame "
                f"budget, they are uniformly selected "
                f"within that interval."
            )

    # ---------------------------------------------------------
    # FORWARD
    # ---------------------------------------------------------

    elif (
        sampling_plan is not None
        and sampling_plan.mode
        == "FORWARD"
        and sampling_plan.anchor_abs_s
        is not None
        and request_start_time
        is not None
    ):

        specialized = (
            _select_forward_indices(
                candidate_indices,
                source_fps=source_fps,
                request_start_time=float(
                    request_start_time
                ),
                anchor_abs_s=(
                    sampling_plan.anchor_abs_s
                ),
                max_frames=max_frames,
            )
        )

        # Again: only override GLOBAL after success.
        if specialized is not None:
            frame_indices = specialized
            actual_mode = "FORWARD"

            sampling_description = (
                f"The input consists of "
                f"{len(frame_indices)} "
                f"chronologically ordered frames "
                f"selected for a forward temporal "
                f"search from the question's anchor "
                f"at "
                f"{seconds_to_timestamp(sampling_plan.anchor_abs_s)}. "
                f"The frames include local context "
                f"around the anchor and broad coverage "
                f"after it toward the end of the "
                f"supplied video segment. "
                f"Sampling is not uniform across the "
                f"full segment."
            )

    # ---------------------------------------------------------
    # Decode selected frames.
    # ---------------------------------------------------------

    if len(frame_indices) == 0:
        raise RuntimeError(
            f"No valid frames selected from "
            f"{path.name}."
        )

    decoded = video_reader.get_batch(
        frame_indices.tolist()
    ).asnumpy()

    del video_reader

    images = [
        Image.fromarray(frame).convert(
            "RGB"
        )
        for frame in decoded
    ]

    relative_timestamps = [
        float(
            frame_index
            / source_fps
        )
        for frame_index
        in frame_indices
    ]

    return ClipFrames(
        images=images,
        frame_indices=[
            int(frame_index)
            for frame_index
            in frame_indices
        ],
        timestamps=relative_timestamps,
        source_fps=source_fps,
        clip_duration=clip_duration,
        sampling_mode=actual_mode,
        sampling_description=(
            sampling_description
        ),
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