#!/usr/bin/env python3
"""Evaluate local ORena FOCUS SEGMENT predictions on the balanced ID-only set.

This script has no command-line arguments. Edit the hard-coded paths and judge
configuration near the top when the directory layout or judge model changes.

It evaluates exactly the 250 qIDs listed in:

    data/segment/prepared_balanced_50/selected_qids.json

using:

    data/segment/prepared_balanced_50/requests.json
    data/segment/prepared_balanced_50/references.json
    inference/segment/predictions_4b/responses.json

Scoring
-------
- The official ``orena-focus`` Evaluator performs answer-format parsing,
  deterministic comparison, adversarial-response checks, tolerance-aware
  time/percentage scoring, and local LLM judging.
- ``multiple_choice``, ``open_ended``, and ``matching`` questions are routed
  to the official local TransformersJudge.
- No per-response latency threshold is applied. The challenge platform uses a
  pooled batch budget measured from whole jobs, while this local prediction
  run loaded the model once for all 250 questions.
- The local headline score is the unweighted mean of the five populated ID
  capability buckets:

      object recognition
      temporal grounding
      aggregation
      event understanding
      complex reasoning

This is an ID-only local proxy, not the official 10-bucket platform score.

Outputs
-------
predictions_4b/evaluation/
├── results.csv
├── summary.csv
├── detailed_results.csv
├── bucket_scores.csv
├── answer_format_scores.csv
├── leaf_capability_scores.csv
├── dataset_scores.csv
├── procedure_scores.csv
├── invalid_format_responses.csv
├── incorrect_responses.csv
└── local_evaluation_report.json
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import re
import socket
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch

import focus
from focus import (
    Evaluator,
    Reference,
    Request,
    Response,
    load_references,
    load_requests,
    load_responses,
)
from focus.data.formats import JUDGE_FORMATS
from focus.evaluation.judges import TransformersJudge


# =============================================================================
# Hard-coded paths
# =============================================================================

ORENA_ROOT = Path(
    "/SAN/medic/Surgical_LLM_Agent/orena2026"
)

PREPARED_DIR = (
    ORENA_ROOT / "data/segment"
)

PREDICTIONS_DIR = (
    ORENA_ROOT
    / "segment/dora-ft/predictions/"
    / "full-batch4-4NFT-32mF-64mNT-ep1-1000eval"
)

EVALUATION_DIR = PREDICTIONS_DIR / "evaluation"

REQUESTS_PATH = PREPARED_DIR / "prepared_200_per_capability/request.json"
REFERENCES_PATH = PREPARED_DIR / "prepared_200_per_capability/references.json"
SELECTED_QIDS_PATH = PREPARED_DIR / "prepared_200_per_capability/selected_qids.json"

RESPONSES_PATH = PREDICTIONS_DIR / "responses.json"
INFERENCE_LOG_PATH = PREDICTIONS_DIR / "inference_log.csv"
PREDICTION_SUMMARY_PATH = (
    PREDICTIONS_DIR / "prediction_summary.json"
)
RUN_CONFIG_PATH = PREDICTIONS_DIR / "run_config.json"

DETAILED_RESULTS_PATH = EVALUATION_DIR / "detailed_results.csv"
BUCKET_SCORES_PATH = EVALUATION_DIR / "bucket_scores.csv"
ANSWER_FORMAT_SCORES_PATH = (
    EVALUATION_DIR / "answer_format_scores.csv"
)
LEAF_CAPABILITY_SCORES_PATH = (
    EVALUATION_DIR / "leaf_capability_scores.csv"
)
DATASET_SCORES_PATH = EVALUATION_DIR / "dataset_scores.csv"
PROCEDURE_SCORES_PATH = EVALUATION_DIR / "procedure_scores.csv"
INVALID_FORMAT_PATH = (
    EVALUATION_DIR / "invalid_format_responses.csv"
)
INCORRECT_RESPONSES_PATH = (
    EVALUATION_DIR / "incorrect_responses.csv"
)
REPORT_PATH = EVALUATION_DIR / "local_evaluation_report.json"


# =============================================================================
# Evaluation configuration
# =============================================================================

# EXPECTED_SELECTED_QUESTIONS = 250
# EXPECTED_CASES_PER_ID_BUCKET = 50
EXPECTED_SELECTED_QUESTIONS = 1000
EXPECTED_CASES_PER_ID_BUCKET = 200



# Use the same local judge family as the existing FRAME evaluator.
# Edit this if the model is stored under a different local name/path.
JUDGE_MODEL = "Qwen/Qwen3-4B"
JUDGE_DEVICE = "cuda:0"
JUDGE_MAX_NEW_TOKENS = 8

NUM_WORKERS = 1
N_BOOT = 1000
BOOTSTRAP_SEED = 42

EXPECTED_ID_GROUPS = (
    "object_recognition",
    "temporal_grounding",
    "aggregation",
    "event_understanding",
    "complex_reasoning",
)


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

LOG = logging.getLogger("eval_segment_predictions")


# =============================================================================
# Generic helpers
# =============================================================================


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.stem}.tmp{path.suffix}"
    )

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    os.replace(temporary, path)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file is missing: {path}"
        )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ensure_unique(
    items: Sequence[Any],
    *,
    label: str,
) -> None:
    qids = [str(item.qID) for item in items]
    duplicates = [
        qid
        for qid, count in Counter(qids).items()
        if count > 1
    ]

    if duplicates:
        raise ValueError(
            f"{label} contains duplicate qIDs: "
            + ", ".join(sorted(duplicates)[:20])
        )


def map_by_qid(
    items: Sequence[Any],
    *,
    label: str,
) -> dict[str, Any]:
    ensure_unique(items, label=label)
    return {
        str(item.qID): item
        for item in items
    }


def read_selected_qids(path: Path) -> list[str]:
    value = read_json(path)

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

    if len(qids) != EXPECTED_SELECTED_QUESTIONS:
        raise ValueError(
            f"Expected {EXPECTED_SELECTED_QUESTIONS} selected qIDs, "
            f"but found {len(qids)}."
        )

    return qids


def select_in_qid_order(
    qids: Sequence[str],
    mapping: dict[str, Any],
    *,
    label: str,
) -> list[Any]:
    missing = [
        qid
        for qid in qids
        if qid not in mapping
    ]

    if missing:
        raise ValueError(
            f"{label} is missing {len(missing)} selected qID(s): "
            + ", ".join(missing[:20])
        )

    return [mapping[qid] for qid in qids]


def infer_dataset(qid: str) -> str:
    if qid.startswith("heico_"):
        return "heico"
    if qid.startswith("lapchole_"):
        return "lapchole"
    return "unknown"


def normalize_token(value: object) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def canonical_capability_group(
    reference: Reference,
) -> str:
    """Map the official Capability group enum to platform metric names."""

    group = reference.primary.group
    candidates = [
        getattr(group, "value", ""),
        getattr(group, "name", ""),
        str(group),
    ]
    tokens = [
        normalize_token(value)
        for value in candidates
        if value
    ]

    for token in tokens:
        if "object_recognition" in token:
            return "object_recognition"

        if "temporal_grounding" in token:
            return "temporal_grounding"

        if (
            token == "aggregation"
            or token.endswith("_aggregation")
        ):
            return "aggregation"

        if (
            "event_understanding" in token
            or "event_and_procedural_understanding" in token
        ):
            return "event_understanding"

        if "complex_reasoning" in token:
            return "complex_reasoning"

    raise ValueError(
        "Could not map capability group. "
        f"primary={reference.primary!r}, "
        f"group={group!r}, tokens={tokens!r}"
    )


def capability_code(reference: Reference) -> str:
    return str(
        reference.primary.code
        or reference.primary.value
    )


def secondary_codes(reference: Reference) -> str:
    return ";".join(
        str(
            capability.code
            or capability.value
        )
        for capability in reference.secondaries
    )


def load_inference_log(
    path: Path,
) -> dict[str, dict[str, str]]:
    rows_by_qid: dict[str, dict[str, str]] = {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(
                "inference_log.csv has no header."
            )

        required_fields = {
            "qID",
            "status",
            "raw_answer",
            "normalized_answer",
            "latency_s",
            "model_inference_s",
            "source_fps",
            "clip_duration_s",
            "num_frames",
            "num_patches",
            "peak_gpu_memory_gb",
        }
        missing_fields = required_fields.difference(
            reader.fieldnames
        )

        if missing_fields:
            raise ValueError(
                "inference_log.csv is missing columns: "
                + ", ".join(sorted(missing_fields))
            )

        for row in reader:
            qid = str(row.get("qID", "")).strip()

            if not qid:
                continue

            if qid in rows_by_qid:
                raise ValueError(
                    "inference_log.csv contains duplicate "
                    f"qID={qid}."
                )

            rows_by_qid[qid] = {
                str(key): (
                    ""
                    if value is None
                    else str(value)
                )
                for key, value in row.items()
            }

    return rows_by_qid


def format_is_valid(
    reference: Reference,
    response: Response,
) -> tuple[bool, str]:
    try:
        reference.format.read(response.content)
    except ValueError as error:
        return False, str(error)

    return True, ""


def scored_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    return (
        frame.groupby(
            group_columns,
            dropna=False,
            sort=True,
        )
        .agg(
            correct=("correctness", "sum"),
            count=("correctness", "size"),
            accuracy=("correctness", "mean"),
        )
        .reset_index()
    )


def float_or_none(value: object) -> float | None:
    text = str(value).strip()

    if not text or text.lower() == "nan":
        return None

    try:
        return float(text)
    except ValueError:
        return None


def int_or_none(value: object) -> int | None:
    parsed = float_or_none(value)

    if parsed is None:
        return None

    return int(parsed)


# =============================================================================
# Detailed report construction
# =============================================================================


def build_detailed_results(
    *,
    selected_qids: Sequence[str],
    requests_by_qid: dict[str, Request],
    references_by_qid: dict[str, Reference],
    responses_by_qid: dict[str, Response],
    official_results: pd.DataFrame,
    inference_log_by_qid: dict[str, dict[str, str]],
) -> pd.DataFrame:
    if "qID" not in official_results.columns:
        raise ValueError(
            "Official results do not contain a qID column."
        )

    official_by_qid = official_results.set_index("qID")
    rows: list[dict[str, Any]] = []

    for selected_index, qid in enumerate(
        selected_qids,
        start=1,
    ):
        request = requests_by_qid[qid]
        reference = references_by_qid[qid]
        response = responses_by_qid[qid]

        if qid not in official_by_qid.index:
            raise ValueError(
                f"Official results are missing qID={qid}."
            )

        official = official_by_qid.loc[qid]
        log_row = inference_log_by_qid.get(qid, {})

        valid_format, format_error = format_is_valid(
            reference,
            response,
        )

        timed_out = bool(
            official["timed_out"]
        ) if "timed_out" in official.index else False

        rows.append(
            {
                "selected_index": selected_index,
                "qID": qid,
                "dataset": infer_dataset(qid),
                "videoID": request.videoID,
                "procedure_type": request.procedure_type,
                "start_time_s": request.start_time,
                "end_time_s": request.end_time,
                "question": request.question,
                "reference_answer": reference.answer,
                "candidate_answer": response.content,
                "raw_model_answer": log_row.get(
                    "raw_answer",
                    "",
                ),
                "answer_format": reference._format,
                "scoring_method": (
                    "llm_judge"
                    if reference.format.type
                    in JUDGE_FORMATS
                    else "deterministic"
                ),
                "primary_capability": capability_code(
                    reference
                ),
                "primary_capability_value": (
                    reference.primary.value
                ),
                "capability_group": (
                    canonical_capability_group(
                        reference
                    )
                ),
                "secondary_capabilities": (
                    secondary_codes(reference)
                ),
                "ood": bool(reference.ood),
                "clinical": bool(reference.clinical),
                "format_valid": valid_format,
                "format_error": format_error,
                "latency_s": float(response.latency),
                "timed_out": timed_out,
                "correctness": bool(
                    official["correctness"]
                ),
                "inference_status": log_row.get(
                    "status",
                    "",
                ),
                "model_inference_s": float_or_none(
                    log_row.get(
                        "model_inference_s",
                        "",
                    )
                ),
                "source_fps": float_or_none(
                    log_row.get(
                        "source_fps",
                        "",
                    )
                ),
                "clip_duration_s": float_or_none(
                    log_row.get(
                        "clip_duration_s",
                        "",
                    )
                ),
                "num_frames": int_or_none(
                    log_row.get(
                        "num_frames",
                        "",
                    )
                ),
                "num_patches": int_or_none(
                    log_row.get(
                        "num_patches",
                        "",
                    )
                ),
                "peak_gpu_memory_gb": float_or_none(
                    log_row.get(
                        "peak_gpu_memory_gb",
                        "",
                    )
                ),
                "first_absolute_timestamp": (
                    log_row.get(
                        "first_absolute_timestamp",
                        "",
                    )
                ),
                "last_absolute_timestamp": (
                    log_row.get(
                        "last_absolute_timestamp",
                        "",
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def build_bucket_scores(
    detailed_results: pd.DataFrame,
) -> pd.DataFrame:
    buckets = scored_summary(
        detailed_results,
        ["capability_group", "ood"],
    )

    buckets["distribution"] = buckets["ood"].map(
        {
            False: "id",
            True: "ood",
        }
    )

    buckets["bucket_name"] = buckets.apply(
        lambda row: (
            f"{row['capability_group']}_"
            f"{row['distribution']}"
        ),
        axis=1,
    )

    return buckets[
        [
            "bucket_name",
            "capability_group",
            "distribution",
            "ood",
            "correct",
            "count",
            "accuracy",
        ]
    ]


def validate_balanced_id_buckets(
    bucket_scores: pd.DataFrame,
) -> pd.DataFrame:
    if bucket_scores["ood"].astype(bool).any():
        raise ValueError(
            "This local evaluator is configured for ID-only "
            "data, but OOD buckets were observed."
        )

    id_buckets = bucket_scores[
        bucket_scores["ood"] == False  # noqa: E712
    ].copy()

    observed_groups = set(
        id_buckets["capability_group"].tolist()
    )
    expected_groups = set(EXPECTED_ID_GROUPS)

    missing_groups = expected_groups.difference(
        observed_groups
    )
    unexpected_groups = observed_groups.difference(
        expected_groups
    )

    if missing_groups:
        raise ValueError(
            "Missing required SEGMENT ID capability "
            "group(s): "
            + ", ".join(sorted(missing_groups))
        )

    if unexpected_groups:
        raise ValueError(
            "Unexpected SEGMENT capability group(s): "
            + ", ".join(sorted(unexpected_groups))
        )

    counts_by_group = {
        str(row.capability_group): int(row.count)
        for row in id_buckets.itertuples(index=False)
    }

    incorrect_counts = {
        group: count
        for group, count in counts_by_group.items()
        if count != EXPECTED_CASES_PER_ID_BUCKET
    }

    if incorrect_counts:
        raise ValueError(
            "The local set is not balanced at "
            f"{EXPECTED_CASES_PER_ID_BUCKET} cases per ID "
            f"bucket: {incorrect_counts}"
        )

    order = {
        group: index
        for index, group in enumerate(
            EXPECTED_ID_GROUPS
        )
    }

    id_buckets["_order"] = id_buckets[
        "capability_group"
    ].map(order)

    return (
        id_buckets.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def build_local_report(
    *,
    detailed_results: pd.DataFrame,
    id_bucket_scores: pd.DataFrame,
    evaluation_seconds: float,
    selected_qids: Sequence[str],
    prediction_summary: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    accuracy_by_group = {
        str(row.capability_group): float(row.accuracy)
        for row in id_bucket_scores.itertuples(
            index=False
        )
    }

    count_by_group = {
        str(row.capability_group): int(row.count)
        for row in id_bucket_scores.itertuples(
            index=False
        )
    }

    correct_by_group = {
        str(row.capability_group): int(row.correct)
        for row in id_bucket_scores.itertuples(
            index=False
        )
    }

    local_id_score = float(
        id_bucket_scores["accuracy"].mean()
    )

    raw_accuracy = float(
        detailed_results["correctness"].mean()
    )

    # Because this prepared set has equal bucket sizes, the two values should
    # be mathematically identical apart from floating-point precision.
    if abs(raw_accuracy - local_id_score) > 1e-12:
        raise RuntimeError(
            "Raw accuracy and five-bucket mean differ despite "
            "equal bucket sizes. "
            f"raw={raw_accuracy}, bucket_mean={local_id_score}"
        )

    judged_mask = (
        detailed_results["scoring_method"]
        == "llm_judge"
    )
    deterministic_mask = (
        detailed_results["scoring_method"]
        == "deterministic"
    )

    report = {
        "created_at_utc": utc_now(),
        "orena_focus_version": getattr(
            focus,
            "__version__",
            "unknown",
        ),
        "evaluation_seconds": evaluation_seconds,
        "track": "segment",
        "split": "test",
        "id_only": True,
        "balanced_cases_per_bucket": (
            EXPECTED_CASES_PER_ID_BUCKET
        ),
        "selected_questions": len(selected_qids),
        "correct_questions": int(
            detailed_results["correctness"].sum()
        ),
        "incorrect_questions": int(
            (~detailed_results["correctness"]).sum()
        ),
        "raw_accuracy": raw_accuracy,
        "local_pre_evaluation_score_id_only": (
            local_id_score
        ),
        "accuracy_object_recognition_id": (
            accuracy_by_group[
                "object_recognition"
            ]
        ),
        "accuracy_temporal_grounding_id": (
            accuracy_by_group[
                "temporal_grounding"
            ]
        ),
        "accuracy_aggregation_id": (
            accuracy_by_group[
                "aggregation"
            ]
        ),
        "accuracy_event_understanding_id": (
            accuracy_by_group[
                "event_understanding"
            ]
        ),
        "accuracy_complex_reasoning_id": (
            accuracy_by_group[
                "complex_reasoning"
            ]
        ),
        "bucket_counts": count_by_group,
        "bucket_correct": correct_by_group,
        "judge_model": JUDGE_MODEL,
        "judge_device": JUDGE_DEVICE,
        "judge_questions": int(judged_mask.sum()),
        "judge_accuracy": (
            float(
                detailed_results.loc[
                    judged_mask,
                    "correctness",
                ].mean()
            )
            if judged_mask.any()
            else None
        ),
        "deterministic_questions": int(
            deterministic_mask.sum()
        ),
        "deterministic_accuracy": (
            float(
                detailed_results.loc[
                    deterministic_mask,
                    "correctness",
                ].mean()
            )
            if deterministic_mask.any()
            else None
        ),
        "invalid_format_responses": int(
            (~detailed_results["format_valid"]).sum()
        ),
        "empty_responses": int(
            detailed_results["candidate_answer"]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        ),
        "timed_out_by_official_local_evaluator": int(
            detailed_results["timed_out"].sum()
        ),
        "datasets": {
            str(dataset): int(count)
            for dataset, count in (
                detailed_results["dataset"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "answer_formats": {
            str(answer_format): int(count)
            for answer_format, count in (
                detailed_results["answer_format"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "prediction_runtime": {
            "elapsed_wall_clock_s": (
                prediction_summary.get(
                    "elapsed_wall_clock_s"
                )
            ),
            "model_load_seconds": (
                prediction_summary.get(
                    "model_load_seconds"
                )
            ),
            "mean_success_latency_s": (
                prediction_summary.get(
                    "mean_success_latency_s"
                )
            ),
            "mean_model_inference_s": (
                prediction_summary.get(
                    "mean_model_inference_s"
                )
            ),
            "mean_clip_duration_s": (
                prediction_summary.get(
                    "mean_clip_duration_s"
                )
            ),
            "mean_sampled_frames": (
                prediction_summary.get(
                    "mean_sampled_frames"
                )
            ),
            "timing_note": (
                prediction_summary.get(
                    "timing_note"
                )
            ),
        },
        "inference_configuration": {
            key: run_config.get(key)
            for key in (
                "model_path",
                "gpu_name",
                "total_vram_gb",
                "pytorch",
                "pytorch_cuda_runtime",
                "device",
                "dtype",
                "target_fps",
                "max_frames",
                "num_decode_threads",
                "input_size",
                "max_tiles_per_frame",
                "max_new_tokens",
                "use_few_shot_examples",
            )
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "paths": {
            "prepared_dir": str(PREPARED_DIR),
            "predictions_dir": str(PREDICTIONS_DIR),
            "evaluation_dir": str(EVALUATION_DIR),
        },
        "interpretation_note": (
            "This is a five-bucket ID-only local proxy. "
            "The official SEGMENT pre-evaluation score averages "
            "ten ID/OOD buckets and may use different undisclosed "
            "judge models."
        ),
    }

    return report


# =============================================================================
# Main
# =============================================================================


def run() -> int:
    process_start = time.monotonic()

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG.info(
        "=== ORena FOCUS SEGMENT local evaluation start ==="
    )
    LOG.info("Prepared directory: %s", PREPARED_DIR)
    LOG.info("Predictions directory: %s", PREDICTIONS_DIR)
    LOG.info("Evaluation directory: %s", EVALUATION_DIR)
    LOG.info(
        "orena-focus version: %s",
        getattr(focus, "__version__", "unknown"),
    )

    for path in (
        REQUESTS_PATH,
        REFERENCES_PATH,
        SELECTED_QIDS_PATH,
        RESPONSES_PATH,
        INFERENCE_LOG_PATH,
        PREDICTION_SUMMARY_PATH,
        RUN_CONFIG_PATH,
    ):
        require_file(path)

    all_requests = list(
        load_requests(REQUESTS_PATH)
    )
    all_references = list(
        load_references(REFERENCES_PATH)
    )
    all_responses = list(
        load_responses(RESPONSES_PATH)
    )

    requests_by_qid = map_by_qid(
        all_requests,
        label="requests.json",
    )
    references_by_qid = map_by_qid(
        all_references,
        label="references.json",
    )
    responses_by_qid = map_by_qid(
        all_responses,
        label="responses.json",
    )

    selected_qids = read_selected_qids(
        SELECTED_QIDS_PATH
    )

    selected_requests = select_in_qid_order(
        selected_qids,
        requests_by_qid,
        label="requests.json",
    )
    selected_references = select_in_qid_order(
        selected_qids,
        references_by_qid,
        label="references.json",
    )
    selected_responses = select_in_qid_order(
        selected_qids,
        responses_by_qid,
        label="responses.json",
    )

    LOG.info(
        "Loaded %d requests, %d references, "
        "%d responses.",
        len(all_requests),
        len(all_references),
        len(all_responses),
    )
    LOG.info(
        "Selected evaluation set: %d questions.",
        len(selected_qids),
    )

    selected_id_set = set(selected_qids)
    selected_request_ids = {
        str(item.qID)
        for item in selected_requests
    }
    selected_reference_ids = {
        str(item.qID)
        for item in selected_references
    }
    selected_response_ids = {
        str(item.qID)
        for item in selected_responses
    }

    if not (
        selected_request_ids
        == selected_reference_ids
        == selected_response_ids
        == selected_id_set
    ):
        raise RuntimeError(
            "Selected request/reference/response qID sets "
            "do not match."
        )

    ood_references = [
        str(reference.qID)
        for reference in selected_references
        if reference.ood
    ]

    if ood_references:
        raise ValueError(
            "This evaluator is ID-only, but selected "
            "references contain ood=True: "
            + ", ".join(ood_references[:20])
        )

    inference_log_by_qid = load_inference_log(
        INFERENCE_LOG_PATH
    )

    missing_log_rows = selected_id_set.difference(
        inference_log_by_qid
    )

    if missing_log_rows:
        raise ValueError(
            "inference_log.csv is missing selected qIDs: "
            + ", ".join(
                sorted(missing_log_rows)[:20]
            )
        )

    non_success_rows = [
        qid
        for qid in selected_qids
        if inference_log_by_qid[qid]
        .get("status", "")
        .strip()
        .lower()
        != "success"
    ]

    if non_success_rows:
        LOG.warning(
            "%d selected inference log row(s) are not "
            "marked success. They will still be evaluated: %s",
            len(non_success_rows),
            ", ".join(non_success_rows[:20]),
        )

    prediction_summary = read_json(
        PREDICTION_SUMMARY_PATH
    )
    run_config = read_json(
        RUN_CONFIG_PATH
    )

    judged_count = sum(
        reference.format.type in JUDGE_FORMATS
        for reference in selected_references
    )

    LOG.info(
        "Judged-format questions: %d; "
        "deterministic-format questions: %d.",
        judged_count,
        len(selected_references) - judged_count,
    )

    judges = []

    if judged_count:
        if not torch.cuda.is_available():
            LOG.warning(
                "CUDA is unavailable. The judge will be "
                "requested on %s and may fail or be very slow.",
                JUDGE_DEVICE,
            )

        LOG.info(
            "Loading local judge %s on %s.",
            JUDGE_MODEL,
            JUDGE_DEVICE,
        )

        judges.append(
            TransformersJudge(
                model_name=JUDGE_MODEL,
                device=JUDGE_DEVICE,
                max_new_tokens=JUDGE_MAX_NEW_TOKENS,
            )
        )

    evaluator = Evaluator(
        judges=judges,
        num_workers=NUM_WORKERS,
        n_boot=N_BOOT,
        seed=BOOTSTRAP_SEED,
    )

    LOG.info(
        "Running official evaluator with max_latency=None. "
        "No standalone per-response latency penalty will "
        "be applied."
    )

    official_results, official_summary = evaluator.run(
        requests=selected_requests,
        references=selected_references,
        responses=selected_responses,
        output_dir=EVALUATION_DIR,
        max_latency=None,
        track=None,
    )

    # Save explicitly as well, even though Evaluator(output_dir=...) normally
    # writes these files. This makes the expected filenames deterministic.
    official_results.to_csv(
        EVALUATION_DIR / "results.csv",
        index=False,
    )
    official_summary.to_csv(
        EVALUATION_DIR / "summary.csv",
        index=False,
    )

    detailed_results = build_detailed_results(
        selected_qids=selected_qids,
        requests_by_qid=requests_by_qid,
        references_by_qid=references_by_qid,
        responses_by_qid=responses_by_qid,
        official_results=official_results,
        inference_log_by_qid=inference_log_by_qid,
    )

    bucket_scores = build_bucket_scores(
        detailed_results
    )
    id_bucket_scores = validate_balanced_id_buckets(
        bucket_scores
    )

    answer_format_scores = scored_summary(
        detailed_results,
        ["answer_format", "scoring_method"],
    )
    leaf_capability_scores = scored_summary(
        detailed_results,
        [
            "primary_capability",
            "primary_capability_value",
            "capability_group",
        ],
    )
    dataset_scores = scored_summary(
        detailed_results,
        ["dataset"],
    )
    procedure_scores = scored_summary(
        detailed_results,
        ["procedure_type"],
    )

    detailed_results.to_csv(
        DETAILED_RESULTS_PATH,
        index=False,
    )
    bucket_scores.to_csv(
        BUCKET_SCORES_PATH,
        index=False,
    )
    answer_format_scores.to_csv(
        ANSWER_FORMAT_SCORES_PATH,
        index=False,
    )
    leaf_capability_scores.to_csv(
        LEAF_CAPABILITY_SCORES_PATH,
        index=False,
    )
    dataset_scores.to_csv(
        DATASET_SCORES_PATH,
        index=False,
    )
    procedure_scores.to_csv(
        PROCEDURE_SCORES_PATH,
        index=False,
    )

    detailed_results[
        ~detailed_results["format_valid"]
    ].to_csv(
        INVALID_FORMAT_PATH,
        index=False,
    )
    detailed_results[
        ~detailed_results["correctness"]
    ].to_csv(
        INCORRECT_RESPONSES_PATH,
        index=False,
    )

    evaluation_seconds = (
        time.monotonic() - process_start
    )

    report = build_local_report(
        detailed_results=detailed_results,
        id_bucket_scores=id_bucket_scores,
        evaluation_seconds=evaluation_seconds,
        selected_qids=selected_qids,
        prediction_summary=prediction_summary,
        run_config=run_config,
    )

    write_json_atomic(REPORT_PATH, report)

    print("\n" + "=" * 92)
    print(
        "ORena FOCUS SEGMENT local balanced ID-only evaluation"
    )
    print("=" * 92)
    print(
        "Selected questions: "
        f"{report['selected_questions']}"
    )
    print(
        "Correct questions: "
        f"{report['correct_questions']}"
    )
    print(
        "Raw accuracy: "
        f"{report['raw_accuracy']:.6f}"
    )
    print(
        "Object recognition ID accuracy: "
        f"{report['accuracy_object_recognition_id']:.6f}"
    )
    print(
        "Temporal grounding ID accuracy: "
        f"{report['accuracy_temporal_grounding_id']:.6f}"
    )
    print(
        "Aggregation ID accuracy: "
        f"{report['accuracy_aggregation_id']:.6f}"
    )
    print(
        "Event understanding ID accuracy: "
        f"{report['accuracy_event_understanding_id']:.6f}"
    )
    print(
        "Complex reasoning ID accuracy: "
        f"{report['accuracy_complex_reasoning_id']:.6f}"
    )
    print(
        "Local pre-evaluation score (ID-only): "
        f"{report['local_pre_evaluation_score_id_only']:.6f}"
    )
    print(
        "Invalid-format responses: "
        f"{report['invalid_format_responses']}"
    )
    print(
        "Judge-evaluated questions: "
        f"{report['judge_questions']}"
    )
    print(
        "Deterministically evaluated questions: "
        f"{report['deterministic_questions']}"
    )
    print(f"Reports: {EVALUATION_DIR}")
    print(
        "Evaluation duration: "
        f"{evaluation_seconds:.2f} seconds"
    )

    LOG.info(
        "=== local SEGMENT evaluation complete in "
        "%.2f seconds ===",
        evaluation_seconds,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(run())

