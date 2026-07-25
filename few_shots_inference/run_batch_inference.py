"""Batch inference for local HeiCo-FOCUS SEGMENT experiments.

The script:
- selects a small balanced subset from the four downloaded training videos;
- builds the shared prompt once;
- loads InternVL3.5-4B once;
- runs all selected questions;
- saves answer.json and diagnostics.json after every question; and
- resumes by skipping qIDs already present in answer.json.

Reference answers and answer formats are used only for local selection and
diagnostics. They are never added to the model prompt.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import gc
import json
import random
import time

import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from focus import DatasetSplit, FocusDataset, Response, Track, save_items

from answer_utils import (
    extract_fo_class_names,
    normalize_answer,
)

from model_utils import InternVLInferenceEngine
from prompt_utils import (
    build_prompt,
    build_shared_prompt,
    load_fo_definitions,
    seconds_to_timestamp,
)
from video_utils import load_segment_frames, save_debug_frames


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(
    "/cs/student/projects1/aibh/2024/wenhuawe/ORENA"
)

VIDEO_DIR = PROJECT_ROOT / "heico-focus-vqa" / "train"
FO_DEFINITIONS_PATH = (
    PROJECT_ROOT
    / "few_shots_inference"
    / "FO_definitions.json"
)
RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "internvl3_5_4b_zero_shot_batch"
)

MODEL_ID = "OpenGVLab/InternVL3_5-4B-Instruct"

SELECTED_VIDEOS = {
    "0002 - Heico - Prokto - 3.avi",
    "0009 - Heico - Prokto - 10.avi",
    "0013 - Heico - Rektum - 4.avi",
    "0019 - Heico - Rektum - 10.avi",
}

# Leave empty for automatic balanced selection.
# Example: TARGET_QIDS = ("37569", "2682529")
TARGET_QIDS: tuple[str, ...] = ()

ANSWER_FORMAT_ORDER = (
    "binary",
    "number",
    "percentage",
    "fo_class",
    "time",
    "multiple_choice",
    "open_ended",
)
REQUESTS_PER_FORMAT = 3
SELECTION_SEED = 0
MAX_REQUESTS: int | None = None

USE_FEW_SHOT_EXAMPLES = True

TARGET_FPS = 1.0
MAX_FRAMES = 20
NUM_DECODE_THREADS = 1
SAVE_DEBUG_FRAMES = False

DEVICE = "cuda:0"
DTYPE = torch.float16
INPUT_SIZE = 448
MAX_TILES_PER_FRAME = 1
MAX_NEW_TOKENS = 64


# =============================================================================
# Helpers
# =============================================================================


def get_reference_format(reference: Any) -> str:
    """Return reference._format as a plain string."""

    value = reference._format
    return str(value.value if hasattr(value, "value") else value)


def write_json(data: Any, path: Path) -> None:
    """Write readable JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    temporary.replace(path)


def save_responses(
    responses: Sequence[Response],
    path: Path,
) -> None:
    """Save responses atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_items(list(responses), temporary)
    temporary.replace(path)


def load_existing_responses(path: Path) -> list[Response]:
    """Load an existing answer.json for resume support."""

    if not path.is_file():
        return []

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")

    return [
        Response(
            qID=str(item["qID"]),
            content=str(item["content"]),
            latency=float(item.get("latency", 0.0)),
        )
        for item in data
    ]


def load_existing_diagnostics(path: Path) -> dict[str, Any]:
    """Load diagnostics from an interrupted run."""

    if not path.is_file():
        return {"items": {}}

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    data.setdefault("items", {})
    return data


def eligible_pairs(
    dataset: FocusDataset,
) -> list[tuple[Any, Any]]:
    """Return samples whose videos are selected and available locally."""

    pairs = [
        (request, reference)
        for request, reference in dataset
        if request.videoID in SELECTED_VIDEOS
        and (VIDEO_DIR / request.videoID).is_file()
    ]

    if not pairs:
        raise RuntimeError(
            "No requests were found for the selected local videos."
        )

    return pairs


def select_requests(
    dataset: FocusDataset,
) -> list[tuple[Any, Any]]:
    """Select explicit qIDs or a balanced subset by answer format."""

    pairs = eligible_pairs(dataset)

    if TARGET_QIDS:
        by_qid = {
            str(request.qID): (request, reference)
            for request, reference in pairs
        }

        missing = [qid for qid in TARGET_QIDS if qid not in by_qid]
        if missing:
            raise ValueError(f"TARGET_QIDS not found: {missing}")

        selected = [by_qid[qid] for qid in TARGET_QIDS]

    else:
        grouped: dict[str, list[tuple[Any, Any]]] = defaultdict(list)

        for request, reference in pairs:
            grouped[get_reference_format(reference)].append(
                (request, reference)
            )

        rng = random.Random(SELECTION_SEED)
        selected = []

        for answer_format in ANSWER_FORMAT_ORDER:
            candidates = list(grouped.get(answer_format, ()))
            rng.shuffle(candidates)
            selected.extend(candidates[:REQUESTS_PER_FORMAT])

    if MAX_REQUESTS is not None:
        if MAX_REQUESTS <= 0:
            raise ValueError("MAX_REQUESTS must be positive or None")
        selected = selected[:MAX_REQUESTS]

    selected.sort(
        key=lambda pair: (
            pair[0].videoID,
            float(pair[0].start_time),
            str(pair[0].qID),
        )
    )

    return selected


def build_shared_inference_prompt(
) -> tuple[str, tuple[str, ...]]:
    """Build the shared prompt and load canonical FO class names."""

    definitions = load_fo_definitions(
        FO_DEFINITIONS_PATH
    )
    fo_class_names = extract_fo_class_names(
        definitions
    )

    if USE_FEW_SHOT_EXAMPLES:
        shared_prompt = build_shared_prompt(
            fo_definitions=definitions,
        )
    else:
        shared_prompt = build_shared_prompt(
            fo_definitions=definitions,
            few_shot_examples=(),
        )

    return shared_prompt, fo_class_names

def save_progress(
    responses: Sequence[Response],
    diagnostics: dict[str, Any],
    answer_path: Path,
    diagnostics_path: Path,
) -> None:
    """Save answers and diagnostics after each question."""

    latencies = [float(response.latency) for response in responses]
    items = diagnostics.get("items", {})

    diagnostics["progress"] = {
        "successful_answers": len(responses),
        "failed_questions": sum(
            isinstance(item, dict)
            and item.get("status") == "failed"
            for item in items.values()
        ),
        "mean_decode_plus_predict_seconds": (
            sum(latencies) / len(latencies)
            if latencies
            else None
        ),
    }

    save_responses(responses, answer_path)
    write_json(diagnostics, diagnostics_path)


def print_gpu_memory(label: str) -> None:
    """Print current PyTorch CUDA allocator usage."""

    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)

    print(
        f"  GPU memory {label}: "
        f"allocated={allocated:.2f} GB, "
        f"reserved={reserved:.2f} GB",
        flush=True,
    )


def print_gpu_memory(label: str) -> None:
    """Print current PyTorch CUDA allocator usage."""

    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)

    print(
        f"  GPU memory {label}: "
        f"allocated={allocated:.2f} GB, "
        f"reserved={reserved:.2f} GB",
        flush=True,
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    answer_path = RESULT_DIR / "answer.json"
    diagnostics_path = RESULT_DIR / "diagnostics.json"

    print("Loading HeiCo-FOCUS SEGMENT annotations...", flush=True)
    dataset = FocusDataset(
        dataset="heico",
        split=DatasetSplit.TRAIN,
        track=Track.SEGMENT,
    )

    selected_pairs = select_requests(dataset)
    format_counts = Counter(
        get_reference_format(reference)
        for _, reference in selected_pairs
    )

    print(
        f"Selected {len(selected_pairs)} request(s).",
        flush=True,
    )
    print(
        "Formats: "
        + ", ".join(
            f"{name}={count}"
            for name, count in sorted(format_counts.items())
        ),
        flush=True,
    )

    responses = load_existing_responses(answer_path)
    completed_qids = {
        str(response.qID)
        for response in responses
    }

    diagnostics = load_existing_diagnostics(diagnostics_path)
    diagnostics["experiment"] = {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "use_few_shot_examples": USE_FEW_SHOT_EXAMPLES,
        "target_fps": TARGET_FPS,
        "max_frames": MAX_FRAMES,
        "input_size": INPUT_SIZE,
        "max_tiles_per_frame": MAX_TILES_PER_FRAME,
        "max_new_tokens": MAX_NEW_TOKENS,
        "requests_per_format": REQUESTS_PER_FORMAT,
        "selection_seed": SELECTION_SEED,
        "selected_videos": sorted(SELECTED_VIDEOS),
        "selected_qids": [
            str(request.qID)
            for request, _ in selected_pairs
        ],
        "selected_format_counts": dict(sorted(format_counts.items())),
    }

    if completed_qids:
        print(
            f"Resuming with {len(completed_qids)} completed qID(s).",
            flush=True,
        )

    shared_prompt, fo_class_names = (
        build_shared_inference_prompt()
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    print(
        f"CUDA device: {torch.cuda.get_device_name(0)}",
        flush=True,
    )

    engine = InternVLInferenceEngine(
        model_id=MODEL_ID,
        device=DEVICE,
        dtype=DTYPE,
        input_size=INPUT_SIZE,
        max_tiles_per_frame=MAX_TILES_PER_FRAME,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    load_start = time.perf_counter()
    engine.load()
    diagnostics["model_load_seconds"] = (
        time.perf_counter() - load_start
    )

    try:
        for index, (request, reference) in enumerate(
            selected_pairs,
            start=1,
        ):
            qid = str(request.qID)

            if qid in completed_qids:
                print(
                    f"[{index}/{len(selected_pairs)}] "
                    f"Skipping completed qID={qid}",
                    flush=True,
                )
                continue

            video_path = VIDEO_DIR / request.videoID

            print(
                f"\n[{index}/{len(selected_pairs)}] qID={qid}",
                flush=True,
            )
            print(f"  video: {request.videoID}", flush=True)
            print(
                "  window: "
                f"{seconds_to_timestamp(request.start_time)}-"
                f"{seconds_to_timestamp(request.end_time)}",
                flush=True,
            )
            print(f"  question: {request.question}", flush=True)

            segment = None
            prediction = None

            try:
                decode_start = time.perf_counter()

                segment = load_segment_frames(
                    video_path=video_path,
                    start_time=request.start_time,
                    end_time=request.end_time,
                    target_fps=TARGET_FPS,
                    max_frames=MAX_FRAMES,
                    num_threads=NUM_DECODE_THREADS,
                )

                decode_seconds = (
                    time.perf_counter() - decode_start
                )

                frame_labels = [
                    seconds_to_timestamp(timestamp)
                    for timestamp in segment.timestamps
                ]

                if SAVE_DEBUG_FRAMES:
                    save_debug_frames(
                        segment=segment,
                        output_dir=(
                            RESULT_DIR
                            / "debug_frames"
                            / f"qid_{qid}"
                        ),
                    )

                prompt = build_prompt(
                    request=request,
                    shared_prompt=shared_prompt,
                )

                print_gpu_memory("before prediction")

                predict_start = time.perf_counter()

                prediction = engine.predict(
                    images=segment.images,
                    prompt=prompt,
                    frame_labels=frame_labels,
                )

                normalized_answer = normalize_answer(
                    prediction.answer,
                    fo_class_names=fo_class_names,
                )

                predict_seconds = (
                    time.perf_counter() - predict_start
                )
                total_seconds = decode_seconds + predict_seconds

                responses.append(
                    Response(
                        qID=qid,
                        content=normalized_answer,
                        latency=total_seconds,
                    )
                )
                completed_qids.add(qid)

                diagnostics["items"][qid] = {
                    "status": "succeeded",
                    "request": {
                        "qID": qid,
                        "videoID": request.videoID,
                        "video_path": str(video_path),
                        "start_time_seconds": request.start_time,
                        "end_time_seconds": request.end_time,
                        "start_time": seconds_to_timestamp(
                            request.start_time
                        ),
                        "end_time": seconds_to_timestamp(
                            request.end_time
                        ),
                        "procedure_type": request.procedure_type,
                        "question": request.question,
                    },
                    "reference": {
                        "answer": reference.answer,
                        "answer_format": get_reference_format(
                            reference
                        ),
                        "primary_capability": reference.primary.value,
                        "secondary_capabilities": [
                            capability.value
                            for capability in reference.secondaries
                        ],
                    },
                    "prediction": {
                        "raw_answer": prediction.answer,
                        "normalized_answer": normalized_answer,
                        "num_frames": prediction.num_frames,
                        "num_patches": prediction.num_patches,
                        "frame_indices": segment.frame_indices,
                        "frame_timestamps_seconds": segment.timestamps,
                        "frame_timestamps": frame_labels,
                        "source_fps": segment.source_fps,
                    },
                    "timing_seconds": {
                        "video_decode": decode_seconds,
                        "model_generation_timer": (
                            prediction.inference_seconds
                        ),
                        "full_predict_call": predict_seconds,
                        "decode_plus_predict": total_seconds,
                    },
                    "peak_gpu_memory_gb": (
                        prediction.peak_gpu_memory_gb
                    ),
                    "prompt": prompt,
                }

                print(
                    f"  raw prediction: {prediction.answer}",
                    flush=True,
                )
                print(
                    f"  normalized prediction: {normalized_answer}",
                    flush=True,
                )
                print(
                    f"  reference: {reference.answer}",
                    flush=True,
                )
                print(
                    f"  decode + predict: {total_seconds:.2f} s",
                    flush=True,
                )

            except Exception as error:
                diagnostics["items"][qid] = {
                    "status": "failed",
                    "error": repr(error),
                    "request": {
                        "qID": qid,
                        "videoID": request.videoID,
                        "start_time_seconds": request.start_time,
                        "end_time_seconds": request.end_time,
                        "question": request.question,
                    },
                    "reference": {
                        "answer": reference.answer,
                        "answer_format": get_reference_format(
                            reference
                        ),
                    },
                }

                print(
                    f"  Failed: {error!r}",
                    flush=True,
                )

            finally:
                if prediction is not None:
                    del prediction

                if segment is not None:
                    del segment

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print_gpu_memory("after cleanup")

            save_progress(
                responses=responses,
                diagnostics=diagnostics,
                answer_path=answer_path,
                diagnostics_path=diagnostics_path,
            )

    except KeyboardInterrupt:
        print(
            "\nInterrupted. Progress has already been saved.",
            flush=True,
        )
        raise

    finally:
        engine.unload()

    print("\nBatch inference complete.", flush=True)
    print(f"Answers: {answer_path}", flush=True)
    print(f"Diagnostics: {diagnostics_path}", flush=True)


if __name__ == "__main__":
    main()
