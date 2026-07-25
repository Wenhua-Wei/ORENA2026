"""Run one real HeiCo-FOCUS SEGMENT question with InternVL3.5-4B.

This is a local one-question smoke test, not the final Docker entrypoint.

Pipeline
--------
1. Load the public HeiCo SEGMENT training annotations with ``orena-focus``.
2. Select one fixed VQA request from one of the four downloaded videos.
3. Build the shared few-shot prompt once.
4. Decode and sample the request's full-video time window.
5. Load InternVL3.5-4B once.
6. Run inference with absolute procedure timestamps attached to the frames.
7. Save ``answer.json`` and ``diagnostics.json``.

No command-line arguments are used. Edit the configuration constants below.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from focus import DatasetSplit, FocusDataset, Response, Track, save_items

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
    / "internvl3_5_4b_fewshot_smoke"
)

MODEL_ID = (
    "/cs/student/projects1/aibh/2024/wenhuawe/ORENA/"
    "models/InternVL3_5-4B-Instruct-a3fd3158"
)




# Known request:
# video: 0009 - Heico - Prokto - 10.avi
# window: 00:09:15-00:14:14
# reference answer: 00:10:12
TARGET_QID = "2682529"

SELECTED_VIDEOS = {
    "0002 - Heico - Prokto - 3.avi",
    "0009 - Heico - Prokto - 10.avi",
    "0013 - Heico - Rektum - 4.avi",
    "0019 - Heico - Rektum - 10.avi",
}

TARGET_FPS = 1.0
MAX_FRAMES = 20

DEVICE = "cuda:0"
DTYPE = torch.float16
INPUT_SIZE = 448
MAX_TILES_PER_FRAME = 1
MAX_NEW_TOKENS = 64


def find_target_sample(
    dataset: FocusDataset,
    target_qid: str,
) -> tuple[Any, Any]:
    """Find one request/reference pair and verify its local video exists."""

    eligible_pairs = [
        (request, reference)
        for request, reference in dataset
        if request.videoID in SELECTED_VIDEOS
    ]

    if not eligible_pairs:
        raise RuntimeError(
            "No dataset requests reference the four downloaded videos."
        )

    for request, reference in eligible_pairs:
        if request.qID == target_qid:
            video_path = VIDEO_DIR / request.videoID
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"The target video file is missing: {video_path}"
                )
            return request, reference

    available_ids = [request.qID for request, _ in eligible_pairs[:20]]
    raise ValueError(
        f"TARGET_QID={target_qid!r} was not found among requests for the "
        f"downloaded videos. First available IDs include: {available_ids}"
    )


def write_json(data: Any, path: Path) -> None:
    """Write readable UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading HeiCo-FOCUS SEGMENT annotations...", flush=True)
    dataset = FocusDataset(
        dataset="heico",
        split=DatasetSplit.TRAIN,
        track=Track.SEGMENT,
    )

    request, reference = find_target_sample(
        dataset=dataset,
        target_qid=TARGET_QID,
    )
    video_path = VIDEO_DIR / request.videoID

    print("request.start_time:", request.start_time)
    print("type:", type(request.start_time))

    print("request.end_time:", request.end_time)
    print("type:", type(request.end_time))
    print("video path:", video_path)


    print("\nSelected request", flush=True)
    print(f"  qID: {request.qID}", flush=True)
    print(f"  video: {request.videoID}", flush=True)
    print(
        "  time window: "
        f"{seconds_to_timestamp(request.start_time)} to "
        f"{seconds_to_timestamp(request.end_time)}",
        flush=True,
    )
    print(f"  question: {request.question}", flush=True)
    print(f"  reference format: {reference._format}", flush=True)
    print(f"  reference answer: {reference.answer}", flush=True)

    # Fixed few-shot examples are already defined in prompt_utils.py.
    fo_definitions = load_fo_definitions(FO_DEFINITIONS_PATH)
#    shared_prompt = build_shared_prompt(
#        fo_definitions=fo_definitions,
#    )

    shared_prompt = build_shared_prompt(
        fo_definitions=fo_definitions,
        few_shot_examples=(),
    )
    prompt = build_prompt(
        request=request,
        shared_prompt=shared_prompt,
    )

    # For this one-question local test, decode before loading the model so the
    # GPU is not occupied while Decord indexes the long AVI.
    print("\nOpening the full video and sampling frames...", flush=True)
    decode_start = time.perf_counter()

    segment = load_segment_frames(
        video_path=video_path,
        start_time=request.start_time,
        end_time=request.end_time,
        target_fps=TARGET_FPS,
        max_frames=MAX_FRAMES,
        num_threads=1,
    )

    debug_frames_dir = (
        RESULT_DIR
        / "debug_frames"
        / f"qid_{request.qID}"
    )

    save_debug_frames(
        segment=segment,
        output_dir=debug_frames_dir,
    )

    print(
        f"Saved sampled frames to: {debug_frames_dir}",
        flush=True,
    )


    video_decode_seconds = time.perf_counter() - decode_start

    frame_labels = [
        seconds_to_timestamp(timestamp)
        for timestamp in segment.timestamps
    ]

    print(
        f"Decoded {segment.num_frames} frames in "
        f"{video_decode_seconds:.2f} s.",
        flush=True,
    )
    print(
        "Frame timestamps: " + ", ".join(frame_labels),
        flush=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. This smoke test expects a CUDA GPU."
        )

    print(
        f"\nCUDA device: {torch.cuda.get_device_name(0)}",
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

    model_load_start = time.perf_counter()
    engine.load()
    model_load_seconds = time.perf_counter() - model_load_start

    print(
        f"Model setup time: {model_load_seconds:.2f} s.",
        flush=True,
    )
    print("\nRunning InternVL inference...", flush=True)

    # This outer timer includes image transforms, GPU transfer, and generation.
    predict_start = time.perf_counter()

    prediction = engine.predict(
        images=segment.images,
        prompt=prompt,
        frame_labels=frame_labels,
    )

    predict_call_seconds = time.perf_counter() - predict_start
    total_question_seconds = (
        video_decode_seconds + predict_call_seconds
    )

    print("\nPrediction", flush=True)
    print(f"  raw answer: {prediction.answer}", flush=True)
    print(f"  reference: {reference.answer}", flush=True)
    print(
        f"  model generation timer: "
        f"{prediction.inference_seconds:.2f} s",
        flush=True,
    )
    print(
        f"  full predict() call: {predict_call_seconds:.2f} s",
        flush=True,
    )
    print(
        f"  decode + predict(): {total_question_seconds:.2f} s",
        flush=True,
    )
    print(f"  frames: {prediction.num_frames}", flush=True)
    print(f"  visual patches: {prediction.num_patches}", flush=True)

    if prediction.peak_gpu_memory_gb is not None:
        print(
            f"  peak allocated GPU memory: "
            f"{prediction.peak_gpu_memory_gb:.2f} GB",
            flush=True,
        )

    response = Response(
        qID=request.qID,
        content=prediction.answer,
        latency=total_question_seconds,
    )

    answer_path = RESULT_DIR / "answer.json"
    diagnostics_path = RESULT_DIR / "diagnostics.json"

    save_items([response], answer_path)

    diagnostics = {
        "experiment": {
            "model_id": MODEL_ID,
            "device": DEVICE,
            "dtype": str(DTYPE),
            "target_fps": TARGET_FPS,
            "max_frames": MAX_FRAMES,
            "input_size": INPUT_SIZE,
            "max_tiles_per_frame": MAX_TILES_PER_FRAME,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "request": {
            "qID": request.qID,
            "videoID": request.videoID,
            "video_path": str(video_path),
            "start_time_seconds": request.start_time,
            "end_time_seconds": request.end_time,
            "start_time": seconds_to_timestamp(request.start_time),
            "end_time": seconds_to_timestamp(request.end_time),
            "procedure_type": request.procedure_type,
            "question": request.question,
        },
        "reference": {
            "answer": reference.answer,
            "answer_format": reference._format,
            "primary_capability": reference.primary.value,
            "secondary_capabilities": [
                capability.value
                for capability in reference.secondaries
            ],
        },
        "prediction": {
            "raw_answer": prediction.answer,
            "num_frames": prediction.num_frames,
            "num_patches": prediction.num_patches,
            "frame_indices": segment.frame_indices,
            "frame_timestamps_seconds": segment.timestamps,
            "frame_timestamps": frame_labels,
            "source_fps": segment.source_fps,
        },
        "timing_seconds": {
            "video_decode": video_decode_seconds,
            "model_load": model_load_seconds,
            "model_generation_timer": prediction.inference_seconds,
            "full_predict_call": predict_call_seconds,
            "decode_plus_predict": total_question_seconds,
        },
        "peak_gpu_memory_gb": prediction.peak_gpu_memory_gb,
        "prompt": prompt,
    }

    write_json(diagnostics, diagnostics_path)

    print("\nSaved outputs", flush=True)
    print(f"  {answer_path}", flush=True)
    print(f"  {diagnostics_path}", flush=True)


if __name__ == "__main__":
    main()
