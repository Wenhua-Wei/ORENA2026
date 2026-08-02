#!/usr/bin/env python3
"""Run local SEGMENT inference on the prepared balanced ID-only evaluation set.

This script has no command-line arguments. Edit the hard-coded configuration
near the top when paths or model settings change.

It uses the exact inference modules from:

    /cs/student/projects1/aibh/2024/wenhuawe/ORENA/
        orena-docker/segment-algorithm/

and evaluates the 250 prepared cases in:

    /cs/student/projects1/aibh/2024/wenhuawe/ORENA/
        data/segment/prepared_balanced_50/

The model is loaded once for the complete local run. Successful questions are
checkpointed and skipped on later runs; failed questions are retried by default.

Outputs
-------
predictions_balanced_50_8b/
├── responses.json
├── inference_log.csv
├── failed_questions.json
├── run_config.json
└── prediction_summary.json

Important
---------
This script is intended for local accuracy evaluation. The challenge platform
loads the model once per 20-question job and applies a pooled latency budget.
Because this script loads the model only once for all 250 questions, its wall
clock timing should not be interpreted as an exact platform-runtime simulation.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# These must be set before importing Transformers through model_utils.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import torch
from focus import Request, Response, load_requests, load_responses, save_items


# =============================================================================
# Hard-coded paths
# =============================================================================

ORENA_ROOT = Path(
    "/raid2/compass/ORENA2026"
)

PREPARED_DIR = (
    ORENA_ROOT / "data/segment"
)

ALGORITHM_DIR = (
    ORENA_ROOT / "orena-docker/segment-algorithm"
)

MODEL_PATH = (
    ALGORITHM_DIR
    / "resources"
    / "InternVL3_5-4B-Instruct"
)

OUTPUT_DIR = (
    ORENA_ROOT
    / "predict"
    / "segment"
    / "predictions_4B_32-SMV2"
)

REQUESTS_PATH = PREPARED_DIR / "requests.json"
SELECTED_QIDS_PATH = PREPARED_DIR / "selected_qids.json"
FO_DEFINITIONS_PATH = PREPARED_DIR / "FO_definitions.json"
PLAIN_DIR = PREPARED_DIR / "plain"

RESPONSES_PATH = OUTPUT_DIR / "responses.json"
INFERENCE_LOG_PATH = OUTPUT_DIR / "inference_log.csv"
FAILED_QUESTIONS_PATH = OUTPUT_DIR / "failed_questions.json"
RUN_CONFIG_PATH = OUTPUT_DIR / "run_config.json"
SUMMARY_PATH = OUTPUT_DIR / "prediction_summary.json"


# =============================================================================
# Inference configuration — keep aligned with segment-algorithm/inference.py
# =============================================================================

EXPECTED_TOTAL_QUESTIONS = 250

DEVICE = "cuda:0"
DTYPE = torch.float16

USE_FEW_SHOT_EXAMPLES = False

TARGET_FPS = 1.0
MAX_FRAMES = 32
NUM_DECODE_THREADS = 1

INPUT_SIZE = 448
MAX_TILES_PER_FRAME = 1
MAX_NEW_TOKENS = 64

SAVE_EVERY = 10
RETRY_FAILED_ON_RESUME = True


# =============================================================================
# Import the exact submitted algorithm modules
# =============================================================================

ALGORITHM_DIR = ALGORITHM_DIR.expanduser().resolve()

if str(ALGORITHM_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_DIR))

from answer_utils import extract_fo_class_names, normalize_answer
from model_utils import InternVLInferenceEngine
from prompt_utils import (
    build_prompt,
    build_shared_prompt,
    load_fo_definitions,
    seconds_to_timestamp,
)
from video_utils import load_clip_frames


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

LOG = logging.getLogger("run_segment_predictions")

LOG_FIELDS = [
    "qID",
    "request_index",
    "status",
    "raw_answer",
    "normalized_answer",
    "latency_s",
    "model_inference_s",
    "source_fps",
    "clip_duration_s",
    "num_frames",
    "num_patches",
    "first_relative_timestamp_s",
    "last_relative_timestamp_s",
    "first_absolute_timestamp",
    "last_absolute_timestamp",
    "peak_gpu_memory_gb",
    "error_type",
    "error_message",
    "completed_at_utc",
]


# =============================================================================
# Generic helpers
# =============================================================================


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def temporary_json_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.tmp{path.suffix}")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_json_path(path)

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    os.replace(temporary, path)


def save_focus_items_atomic(
    items: Sequence[Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_json_path(path)
    save_items(items, temporary)
    os.replace(temporary, path)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file is missing: {path}"
        )


def require_directory(path: Path) -> None:
    if not path.is_dir():
        raise NotADirectoryError(
            f"Required directory is missing: {path}"
        )


def clip_path_for(request: Request) -> Path:
    return PLAIN_DIR / f"{request.qID}.mp4"


def validate_requests(requests: Sequence[Request]) -> None:
    qids = [str(request.qID).strip() for request in requests]

    if not qids:
        raise ValueError("requests.json contains no requests.")

    if any(not qid for qid in qids):
        raise ValueError(
            "Every request must have a non-empty qID."
        )

    if len(qids) != len(set(qids)):
        raise ValueError(
            "requests.json contains duplicate qIDs."
        )


def read_selected_qids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, list):
        raise ValueError(
            "selected_qids.json must contain one JSON list."
        )

    qids = [str(item).strip() for item in value]

    if any(not qid for qid in qids):
        raise ValueError(
            "selected_qids.json contains an empty qID."
        )

    if len(qids) != len(set(qids)):
        raise ValueError(
            "selected_qids.json contains duplicate qIDs."
        )

    return qids


def validate_inputs(
    requests: Sequence[Request],
    selected_qids: Sequence[str],
) -> None:
    for path in (
        REQUESTS_PATH,
        SELECTED_QIDS_PATH,
        FO_DEFINITIONS_PATH,
    ):
        require_file(path)

    for path in (
        PREPARED_DIR,
        PLAIN_DIR,
        ALGORITHM_DIR,
        MODEL_PATH,
    ):
        require_directory(path)

    request_qids = [str(request.qID) for request in requests]

    if len(requests) != EXPECTED_TOTAL_QUESTIONS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_QUESTIONS} requests, "
            f"but found {len(requests)}."
        )

    if len(selected_qids) != EXPECTED_TOTAL_QUESTIONS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_QUESTIONS} selected qIDs, "
            f"but found {len(selected_qids)}."
        )

    if request_qids != list(selected_qids):
        if set(request_qids) != set(selected_qids):
            missing_from_requests = (
                set(selected_qids).difference(request_qids)
            )
            missing_from_selected = (
                set(request_qids).difference(selected_qids)
            )
            raise ValueError(
                "requests.json and selected_qids.json do not "
                "contain the same qID set. "
                f"Missing from requests: "
                f"{sorted(missing_from_requests)[:20]}; "
                f"missing from selected_qids: "
                f"{sorted(missing_from_selected)[:20]}."
            )

        raise ValueError(
            "requests.json and selected_qids.json contain the "
            "same qIDs but in different orders."
        )

    missing_clips = [
        str(clip_path_for(request))
        for request in requests
        if not clip_path_for(request).is_file()
    ]

    if missing_clips:
        raise FileNotFoundError(
            f"{len(missing_clips)} prepared clip(s) are missing. "
            "First examples:\n"
            + "\n".join(
                f"  - {path}"
                for path in missing_clips[:20]
            )
        )


def make_absolute_frame_labels(
    request: Request,
    relative_timestamps: Sequence[float],
) -> list[str]:
    return [
        seconds_to_timestamp(
            float(request.start_time)
            + float(relative_timestamp)
        )
        for relative_timestamp in relative_timestamps
    ]


# =============================================================================
# Resume state
# =============================================================================


def load_existing_responses() -> dict[str, Response]:
    if not RESPONSES_PATH.is_file():
        return {}

    responses_by_qid: dict[str, Response] = {}

    for response in load_responses(RESPONSES_PATH):
        qid = str(response.qID)

        if qid in responses_by_qid:
            raise ValueError(
                "Existing responses.json contains duplicate "
                f"qID={qid}."
            )

        responses_by_qid[qid] = response

    LOG.info(
        "Loaded %d existing response(s).",
        len(responses_by_qid),
    )
    return responses_by_qid


def load_existing_log() -> dict[str, dict[str, str]]:
    if not INFERENCE_LOG_PATH.is_file():
        return {}

    rows_by_qid: dict[str, dict[str, str]] = {}

    with INFERENCE_LOG_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        missing_fields = set(LOG_FIELDS).difference(
            reader.fieldnames or []
        )

        if missing_fields:
            raise ValueError(
                "Existing inference_log.csv is missing columns: "
                + ", ".join(sorted(missing_fields))
            )

        for row in reader:
            qid = str(row["qID"])

            if qid in rows_by_qid:
                raise ValueError(
                    "inference_log.csv contains duplicate "
                    f"qID={qid}."
                )

            rows_by_qid[qid] = {
                field: str(row.get(field, ""))
                for field in LOG_FIELDS
            }

    LOG.info(
        "Loaded %d existing inference log row(s).",
        len(rows_by_qid),
    )
    return rows_by_qid


def completed_qids(
    responses_by_qid: dict[str, Response],
    log_by_qid: dict[str, dict[str, str]],
) -> set[str]:
    completed: set[str] = set()

    for qid, response in responses_by_qid.items():
        row = log_by_qid.get(qid)

        if row is None:
            # A non-empty response may have been saved immediately before
            # an interruption prevented the CSV checkpoint.
            if str(response.content).strip():
                completed.add(qid)
            continue

        status = row.get("status", "").strip().lower()

        if status == "success":
            completed.add(qid)
        elif (
            status == "failed"
            and not RETRY_FAILED_ON_RESUME
        ):
            completed.add(qid)

    return completed


# =============================================================================
# Output/checkpoint writing
# =============================================================================


def ordered_responses(
    requests: Sequence[Request],
    responses_by_qid: dict[str, Response],
) -> list[Response]:
    return [
        responses_by_qid[str(request.qID)]
        for request in requests
        if str(request.qID) in responses_by_qid
    ]


def ordered_log_rows(
    requests: Sequence[Request],
    log_by_qid: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    return [
        log_by_qid[str(request.qID)]
        for request in requests
        if str(request.qID) in log_by_qid
    ]


def write_log_atomic(
    requests: Sequence[Request],
    log_by_qid: dict[str, dict[str, str]],
) -> None:
    INFERENCE_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = INFERENCE_LOG_PATH.with_name(
        f"{INFERENCE_LOG_PATH.stem}.tmp"
        f"{INFERENCE_LOG_PATH.suffix}"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=LOG_FIELDS,
        )
        writer.writeheader()
        writer.writerows(
            ordered_log_rows(requests, log_by_qid)
        )

    os.replace(temporary, INFERENCE_LOG_PATH)


def failed_rows(
    requests: Sequence[Request],
    log_by_qid: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []

    for request in requests:
        qid = str(request.qID)
        row = log_by_qid.get(qid)

        if (
            row is not None
            and row.get("status", "").strip().lower()
            == "failed"
        ):
            failures.append(
                {
                    "qID": qid,
                    "error_type": row.get(
                        "error_type",
                        "",
                    ),
                    "error_message": row.get(
                        "error_message",
                        "",
                    ),
                    "latency_s": row.get(
                        "latency_s",
                        "",
                    ),
                    "completed_at_utc": row.get(
                        "completed_at_utc",
                        "",
                    ),
                }
            )

    return failures


def checkpoint(
    requests: Sequence[Request],
    responses_by_qid: dict[str, Response],
    log_by_qid: dict[str, dict[str, str]],
) -> None:
    save_focus_items_atomic(
        ordered_responses(
            requests,
            responses_by_qid,
        ),
        RESPONSES_PATH,
    )
    write_log_atomic(requests, log_by_qid)
    write_json_atomic(
        FAILED_QUESTIONS_PATH,
        failed_rows(requests, log_by_qid),
    )


def write_run_config(total_requests: int) -> None:
    gpu_name = None
    total_vram_gb = None

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024**3)
        )

    write_json_atomic(
        RUN_CONFIG_PATH,
        {
            "created_at_utc": utc_now(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "pytorch": torch.__version__,
            "pytorch_cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": gpu_name,
            "total_vram_gb": total_vram_gb,
            "prepared_dir": str(PREPARED_DIR),
            "algorithm_dir": str(ALGORITHM_DIR),
            "output_dir": str(OUTPUT_DIR),
            "model_path": str(MODEL_PATH),
            "total_requests": total_requests,
            "device": DEVICE,
            "dtype": str(DTYPE),
            "use_few_shot_examples": (
                USE_FEW_SHOT_EXAMPLES
            ),
            "target_fps": TARGET_FPS,
            "max_frames": MAX_FRAMES,
            "num_decode_threads": (
                NUM_DECODE_THREADS
            ),
            "input_size": INPUT_SIZE,
            "max_tiles_per_frame": (
                MAX_TILES_PER_FRAME
            ),
            "max_new_tokens": MAX_NEW_TOKENS,
            "save_every": SAVE_EVERY,
            "retry_failed_on_resume": (
                RETRY_FAILED_ON_RESUME
            ),
            "timing_note": (
                "Model loaded once for all local questions; "
                "not an exact simulation of platform "
                "20-question jobs."
            ),
        },
    )


def write_summary(
    *,
    all_requests: Sequence[Request],
    responses_by_qid: dict[str, Response],
    log_by_qid: dict[str, dict[str, str]],
    process_start: float,
    model_load_seconds: float | None,
) -> None:
    rows = [
        log_by_qid[str(request.qID)]
        for request in all_requests
        if str(request.qID) in log_by_qid
    ]

    success_rows = [
        row
        for row in rows
        if row.get("status", "").strip().lower()
        == "success"
    ]

    failure_rows = [
        row
        for row in rows
        if row.get("status", "").strip().lower()
        == "failed"
    ]

    latencies = [
        float(row["latency_s"])
        for row in success_rows
        if row.get("latency_s", "").strip()
    ]

    model_latencies = [
        float(row["model_inference_s"])
        for row in success_rows
        if row.get("model_inference_s", "").strip()
    ]

    clip_durations = [
        float(row["clip_duration_s"])
        for row in success_rows
        if row.get("clip_duration_s", "").strip()
    ]

    frame_counts = [
        int(row["num_frames"])
        for row in success_rows
        if row.get("num_frames", "").strip()
    ]

    write_json_atomic(
        SUMMARY_PATH,
        {
            "updated_at_utc": utc_now(),
            "elapsed_wall_clock_s": (
                time.monotonic() - process_start
            ),
            "model_load_seconds": model_load_seconds,
            "prepared_requests": len(all_requests),
            "responses_saved_total": len(
                responses_by_qid
            ),
            "successes": len(success_rows),
            "failures": len(failure_rows),
            "unprocessed": (
                len(all_requests)
                - len(success_rows)
                - len(failure_rows)
            ),
            "mean_success_latency_s": (
                sum(latencies) / len(latencies)
                if latencies
                else None
            ),
            "mean_model_inference_s": (
                sum(model_latencies)
                / len(model_latencies)
                if model_latencies
                else None
            ),
            "mean_clip_duration_s": (
                sum(clip_durations)
                / len(clip_durations)
                if clip_durations
                else None
            ),
            "mean_sampled_frames": (
                sum(frame_counts) / len(frame_counts)
                if frame_counts
                else None
            ),
            "timing_note": (
                "Local model loading is amortized across all "
                "250 questions and is not directly comparable "
                "with platform batch timing."
            ),
        },
    )


# =============================================================================
# Main inference
# =============================================================================


def run() -> int:
    process_start = time.monotonic()
    model_load_seconds: float | None = None
    engine: InternVLInferenceEngine | None = None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    LOG.info(
        "=== ORena FOCUS SEGMENT local prediction run start ==="
    )
    LOG.info("Prepared directory: %s", PREPARED_DIR)
    LOG.info("Algorithm directory: %s", ALGORITHM_DIR)
    LOG.info("Model path: %s", MODEL_PATH)
    LOG.info("Output directory: %s", OUTPUT_DIR)
    LOG.info("PyTorch: %s", torch.__version__)
    LOG.info(
        "PyTorch CUDA runtime: %s",
        torch.version.cuda,
    )
    LOG.info(
        "CUDA available: %s",
        torch.cuda.is_available(),
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this script on a GPU "
            "machine."
        )

    LOG.info(
        "GPU: %s",
        torch.cuda.get_device_name(0),
    )
    LOG.info(
        "Total GPU memory: %.2f GB",
        torch.cuda.get_device_properties(0).total_memory
        / (1024**3),
    )

    requests = list(load_requests(REQUESTS_PATH))
    validate_requests(requests)

    selected_qids = read_selected_qids(
        SELECTED_QIDS_PATH
    )

    validate_inputs(requests, selected_qids)
    write_run_config(total_requests=len(requests))

    request_index_by_qid = {
        str(request.qID): index
        for index, request in enumerate(
            requests,
            start=1,
        )
    }

    LOG.info(
        "Loaded %d prepared request(s).",
        len(requests),
    )

    fo_definitions = load_fo_definitions(
        FO_DEFINITIONS_PATH
    )
    fo_class_names = extract_fo_class_names(
        fo_definitions
    )

    if not fo_class_names:
        raise ValueError(
            "No canonical FO classes were extracted from "
            "FO_definitions.json."
        )

    if USE_FEW_SHOT_EXAMPLES:
        shared_prompt = build_shared_prompt(
            fo_definitions=fo_definitions,
        )
    else:
        shared_prompt = build_shared_prompt(
            fo_definitions=fo_definitions,
            few_shot_examples=(),
        )

    LOG.info(
        "Loaded %d FO classes. Shared prompt length: %d.",
        len(fo_class_names),
        len(shared_prompt),
    )

    responses_by_qid = load_existing_responses()
    log_by_qid = load_existing_log()

    known_qids = {
        str(request.qID)
        for request in requests
    }

    unknown_responses = set(
        responses_by_qid
    ).difference(known_qids)

    unknown_logs = set(
        log_by_qid
    ).difference(known_qids)

    if unknown_responses:
        raise ValueError(
            "Existing responses contain unknown qIDs: "
            + ", ".join(
                sorted(unknown_responses)[:20]
            )
        )

    if unknown_logs:
        raise ValueError(
            "Existing logs contain unknown qIDs: "
            + ", ".join(sorted(unknown_logs)[:20])
        )

    done_qids = completed_qids(
        responses_by_qid,
        log_by_qid,
    )

    pending_requests = [
        request
        for request in requests
        if str(request.qID) not in done_qids
    ]

    LOG.info(
        "Resume state: %d complete, %d pending.",
        len(requests) - len(pending_requests),
        len(pending_requests),
    )

    if not pending_requests:
        checkpoint(
            requests,
            responses_by_qid,
            log_by_qid,
        )
        write_summary(
            all_requests=requests,
            responses_by_qid=responses_by_qid,
            log_by_qid=log_by_qid,
            process_start=process_start,
            model_load_seconds=None,
        )
        LOG.info("Nothing to process.")
        return 0

    engine = InternVLInferenceEngine(
        model_path=MODEL_PATH,
        device=DEVICE,
        dtype=DTYPE,
        input_size=INPUT_SIZE,
        max_tiles_per_frame=MAX_TILES_PER_FRAME,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    try:
        load_start = time.monotonic()
        engine.load()
        model_load_seconds = (
            time.monotonic() - load_start
        )

        LOG.info(
            "InternVL loaded in %.2f seconds.",
            model_load_seconds,
        )

        newly_processed = 0

        for request in pending_requests:
            qid = str(request.qID)
            request_index = request_index_by_qid[qid]
            question_start = time.monotonic()

            clip = None
            prediction = None

            LOG.info(
                "[%d/%d] qID=%s, source=%s, "
                "window=[%.3f, %.3f]",
                request_index,
                len(requests),
                qid,
                request.videoID,
                float(request.start_time),
                float(request.end_time),
            )

            try:
                clip = load_clip_frames(
                    video_path=clip_path_for(request),
                    target_fps=TARGET_FPS,
                    max_frames=MAX_FRAMES,
                    num_threads=NUM_DECODE_THREADS,
                )

                frame_labels = make_absolute_frame_labels(
                    request=request,
                    relative_timestamps=clip.timestamps,
                )

                prompt = build_prompt(
                    request=request,
                    shared_prompt=shared_prompt,
                )

                prediction = engine.predict(
                    images=clip.images,
                    prompt=prompt,
                    frame_labels=frame_labels,
                )

                answer = normalize_answer(
                    prediction.answer,
                    fo_class_names=fo_class_names,
                )

                latency = (
                    time.monotonic() - question_start
                )

                responses_by_qid[qid] = Response(
                    qID=qid,
                    content=answer,
                    latency=latency,
                )

                first_relative = clip.timestamps[0]
                last_relative = clip.timestamps[-1]

                log_by_qid[qid] = {
                    "qID": qid,
                    "request_index": str(
                        request_index
                    ),
                    "status": "success",
                    "raw_answer": prediction.answer,
                    "normalized_answer": answer,
                    "latency_s": f"{latency:.6f}",
                    "model_inference_s": (
                        f"{prediction.inference_seconds:.6f}"
                    ),
                    "source_fps": (
                        f"{clip.source_fps:.6f}"
                    ),
                    "clip_duration_s": (
                        f"{clip.clip_duration:.6f}"
                    ),
                    "num_frames": str(
                        prediction.num_frames
                    ),
                    "num_patches": str(
                        prediction.num_patches
                    ),
                    "first_relative_timestamp_s": (
                        f"{first_relative:.6f}"
                    ),
                    "last_relative_timestamp_s": (
                        f"{last_relative:.6f}"
                    ),
                    "first_absolute_timestamp": (
                        frame_labels[0]
                    ),
                    "last_absolute_timestamp": (
                        frame_labels[-1]
                    ),
                    "peak_gpu_memory_gb": (
                        ""
                        if prediction.peak_gpu_memory_gb
                        is None
                        else (
                            f"{prediction.peak_gpu_memory_gb:.6f}"
                        )
                    ),
                    "error_type": "",
                    "error_message": "",
                    "completed_at_utc": utc_now(),
                }

                LOG.info(
                    "[%d/%d] qID=%s answered in "
                    "%.2f s: %r",
                    request_index,
                    len(requests),
                    qid,
                    latency,
                    answer,
                )
                LOG.info(
                    "  frames=%d, patches=%d, "
                    "source_fps=%.3f, duration=%.3f s",
                    prediction.num_frames,
                    prediction.num_patches,
                    clip.source_fps,
                    clip.clip_duration,
                )

            except Exception as error:
                latency = (
                    time.monotonic() - question_start
                )

                # Match the submitted algorithm's failure behavior.
                responses_by_qid[qid] = Response(
                    qID=qid,
                    content="",
                    latency=latency,
                )

                log_by_qid[qid] = {
                    "qID": qid,
                    "request_index": str(
                        request_index
                    ),
                    "status": "failed",
                    "raw_answer": "",
                    "normalized_answer": "",
                    "latency_s": f"{latency:.6f}",
                    "model_inference_s": "",
                    "source_fps": "",
                    "clip_duration_s": "",
                    "num_frames": "",
                    "num_patches": "",
                    "first_relative_timestamp_s": "",
                    "last_relative_timestamp_s": "",
                    "first_absolute_timestamp": "",
                    "last_absolute_timestamp": "",
                    "peak_gpu_memory_gb": "",
                    "error_type": (
                        type(error).__name__
                    ),
                    "error_message": str(error),
                    "completed_at_utc": utc_now(),
                }

                LOG.exception(
                    "[%d/%d] qID=%s failed after "
                    "%.2f s.",
                    request_index,
                    len(requests),
                    qid,
                    latency,
                )

            finally:
                if prediction is not None:
                    del prediction
                if clip is not None:
                    del clip

                # Keep the CUDA allocator cache for later questions.
                # Do not call torch.cuda.empty_cache() here.

            newly_processed += 1

            if newly_processed % SAVE_EVERY == 0:
                checkpoint(
                    requests,
                    responses_by_qid,
                    log_by_qid,
                )
                write_summary(
                    all_requests=requests,
                    responses_by_qid=responses_by_qid,
                    log_by_qid=log_by_qid,
                    process_start=process_start,
                    model_load_seconds=(
                        model_load_seconds
                    ),
                )
                LOG.info(
                    "Checkpointed after %d new "
                    "question(s).",
                    newly_processed,
                )

        checkpoint(
            requests,
            responses_by_qid,
            log_by_qid,
        )
        write_summary(
            all_requests=requests,
            responses_by_qid=responses_by_qid,
            log_by_qid=log_by_qid,
            process_start=process_start,
            model_load_seconds=model_load_seconds,
        )

    finally:
        if engine is not None:
            engine.unload()

    missing_responses = {
        str(request.qID)
        for request in requests
    }.difference(responses_by_qid)

    if missing_responses:
        raise RuntimeError(
            f"{len(missing_responses)} response(s) are "
            "missing after inference: "
            + ", ".join(
                sorted(missing_responses)[:20]
            )
        )

    LOG.info(
        "Wrote %d response(s) to %s.",
        len(responses_by_qid),
        RESPONSES_PATH,
    )
    LOG.info(
        "=== prediction run complete in %.2f seconds ===",
        time.monotonic() - process_start,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
