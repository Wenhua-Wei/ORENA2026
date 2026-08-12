#!/usr/bin/env python3
"""
Build simplified ORena SEGMENT multimodal SFT JSONL files from cached-frame manifests.

Permanent output schema for both smoke testing and the full dataset:

    sample_id
    qID
    dataset
    split
    images
    frame_labels
    num_frames
    user_text
    assistant_text
    metadata

Training contract:
    images          -> ordered visual inputs
    user_text       -> exact Docker-aligned user/query text, exactly once
    assistant_text  -> supervised target
    metadata        -> sampling / analysis / evaluation only

This script does not decode videos and does not preprocess JPEGs into tensors.
It reuses the current Docker prompt_utils.py directly.

The later InternVL training loader must use InternVL's multimodal visual-token
expansion. Do not plain-tokenize literal <image> strings with a normal tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


# =============================================================================
# CURRENT DOCKER PROMPT CODE
# =============================================================================

SEGMENT_ALGORITHM_DIR = Path(
    "/SAN/medic/Surgical_LLM_Agent/orena2026/"
    "orena-docker/segment-algorithm"
)

if not SEGMENT_ALGORITHM_DIR.is_dir():
    raise FileNotFoundError(
        f"SEGMENT algorithm directory does not exist: {SEGMENT_ALGORITHM_DIR}"
    )

if str(SEGMENT_ALGORITHM_DIR) not in sys.path:
    sys.path.insert(0, str(SEGMENT_ALGORITHM_DIR))

from prompt_utils import (  # noqa: E402
    build_prompt,
    build_shared_prompt,
    load_fo_definitions,
    resolve_fo_class_names,
    seconds_to_timestamp,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_DATA_ROOT = Path(
    "/SAN/medic/Surgical_LLM_Agent/orena2026/data/segment/"
    "dora_8b_max32_smoke_4train_1test"
)

TARGET_FPS = 1.0
MAX_FRAMES = 32
USE_FEW_SHOT_EXAMPLES = False

INPUT_MANIFESTS = {
    "train": "train_manifest.jsonl",
    "val_heico": "val_heico_manifest.jsonl",
    "val_lapchole": "val_lapchole_manifest.jsonl",
    "val_all": "val_all_manifest.jsonl",
}

OUTPUT_JSONLS = {
    "train": "train_sft.jsonl",
    "val_heico": "val_heico_sft.jsonl",
    "val_lapchole": "val_lapchole_sft.jsonl",
    "val_all": "val_all_sft.jsonl",
}

SUMMARY_FILENAME = "sft_summary.json"
LOG_LEVEL = logging.INFO


# =============================================================================
# MINIMAL REQUEST VIEW
# =============================================================================

@dataclass(frozen=True)
class PromptRequest:
    qID: str
    videoID: str
    start_time: float
    end_time: float
    procedure_type: str
    question: str


# =============================================================================
# CLI / LOGGING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build simplified ORena SEGMENT multimodal SFT JSONL files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Frame/manifest root. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory. Default: same as --data-root.",
    )
    parser.add_argument(
        "--fo-definitions",
        type=Path,
        default=None,
        help=(
            "Optional FO_definitions.json. If omitted, first try "
            "<data-root>/FO_definitions.json; otherwise use the canonical "
            "FOType registry fallback."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing SFT JSONL and summary outputs.",
    )
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Skip checking every referenced JPEG exists and is non-empty.",
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
# IO
# =============================================================================

def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def atomic_write_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
            count += 1
    os.replace(tmp, path)
    return count


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} line {line_number}: {error}"
                ) from error
            if not isinstance(obj, dict):
                raise TypeError(
                    f"{path} line {line_number} is not a JSON object."
                )
            records.append(obj)
    return records


def require_output_available(
    paths: Sequence[Path],
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output(s). Use --overwrite:\n  - "
            + "\n  - ".join(str(path) for path in existing)
        )


# =============================================================================
# PROMPT ALIGNMENT
# =============================================================================

def load_training_fo_definitions(
    *,
    data_root: Path,
    explicit_path: Path | None,
) -> tuple[str, str]:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        return load_fo_definitions(path), str(path)

    candidate = data_root / "FO_definitions.json"
    if candidate.is_file():
        return load_fo_definitions(candidate), str(candidate)

    return "", "canonical FOType registry fallback (empty definitions string)"


def make_frame_labels(record: dict[str, Any]) -> list[str]:
    start_time = float(record["start_time"])
    labels: list[str] = []

    for index, frame in enumerate(record["frames"]):
        relative_time = float(frame["relative_time_seconds"])
        label = seconds_to_timestamp(start_time + relative_time)
        stored = str(frame["timestamp"])

        if label != stored:
            raise RuntimeError(
                f"{record['sample_id']}: frame {index} timestamp mismatch: "
                f"recomputed={label!r}, manifest={stored!r}"
            )

        labels.append(label)

    return labels


def make_video_prefix(frame_labels: Sequence[str]) -> str:
    return "".join(
        f"{label}: <image>\n"
        for label in frame_labels
    )


def build_exact_user_text(
    *,
    request: PromptRequest,
    shared_prompt: str,
    frame_labels: Sequence[str],
    num_frames: int,
) -> str:
    prompt = build_prompt(
        request=request,
        shared_prompt=shared_prompt,
        num_frames=num_frames,
        target_fps=TARGET_FPS,
        max_frames=MAX_FRAMES,
    )
    return make_video_prefix(frame_labels) + prompt.strip()


# =============================================================================
# VALIDATION
# =============================================================================

def validate_manifest_record(
    *,
    record: dict[str, Any],
    data_root: Path,
    check_images: bool,
) -> None:
    required = (
        "sample_id",
        "qID",
        "dataset",
        "split",
        "videoID",
        "procedure_type",
        "start_time",
        "end_time",
        "question",
        "answer",
        "candidate_count_1fps",
        "num_frames",
        "frames",
    )

    missing = [key for key in required if key not in record]
    if missing:
        raise KeyError(
            f"{record.get('sample_id', '<unknown>')}: missing keys {missing}"
        )

    sample_id = str(record["sample_id"])
    frames = record["frames"]

    if not isinstance(frames, list):
        raise TypeError(f"{sample_id}: frames must be a list.")

    num_frames = int(record["num_frames"])
    candidate_count = int(record["candidate_count_1fps"])
    expected = min(candidate_count, MAX_FRAMES)

    if num_frames != len(frames):
        raise RuntimeError(
            f"{sample_id}: num_frames={num_frames}, "
            f"but frames has {len(frames)} entries."
        )

    if num_frames != expected:
        raise RuntimeError(
            f"{sample_id}: num_frames={num_frames}, expected {expected}."
        )

    if not (1 <= num_frames <= MAX_FRAMES):
        raise RuntimeError(
            f"{sample_id}: invalid num_frames={num_frames}."
        )

    start_time = float(record["start_time"])
    end_time = float(record["end_time"])

    if not math.isfinite(start_time) or not math.isfinite(end_time):
        raise RuntimeError(f"{sample_id}: start/end time must be finite.")

    if end_time <= start_time:
        raise RuntimeError(
            f"{sample_id}: end_time must be greater than start_time."
        )

    relative_times = [
        float(frame["relative_time_seconds"])
        for frame in frames
    ]

    if not math.isclose(relative_times[0], 0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"{sample_id}: first frame is not segment-relative t=0."
        )

    if any(
        current <= previous
        for previous, current in zip(relative_times, relative_times[1:])
    ):
        raise RuntimeError(
            f"{sample_id}: frame times are not strictly chronological."
        )

    relative_paths = [
        str(frame["path"])
        for frame in frames
    ]

    if len(relative_paths) != len(set(relative_paths)):
        raise RuntimeError(
            f"{sample_id}: duplicate frame path inside one VQA."
        )

    if check_images:
        for relative_path in relative_paths:
            image_path = data_root / relative_path
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"{sample_id}: missing frame: {image_path}"
                )
            if image_path.stat().st_size <= 0:
                raise RuntimeError(
                    f"{sample_id}: empty frame file: {image_path}"
                )


# =============================================================================
# MANIFEST -> SIMPLIFIED SFT RECORD
# =============================================================================

def convert_record(
    *,
    record: dict[str, Any],
    data_root: Path,
    shared_prompt: str,
    check_images: bool,
) -> dict[str, Any]:
    validate_manifest_record(
        record=record,
        data_root=data_root,
        check_images=check_images,
    )

    request = PromptRequest(
        qID=str(record["qID"]),
        videoID=str(record["videoID"]),
        start_time=float(record["start_time"]),
        end_time=float(record["end_time"]),
        procedure_type=str(record["procedure_type"]),
        question=str(record["question"]),
    )

    num_frames = int(record["num_frames"])
    frame_labels = make_frame_labels(record)

    user_text = build_exact_user_text(
        request=request,
        shared_prompt=shared_prompt,
        frame_labels=frame_labels,
        num_frames=num_frames,
    )

    images = [
        str((data_root / str(frame["path"])).resolve())
        for frame in record["frames"]
    ]

    assistant_text = str(record["answer"]).strip()
    if not assistant_text:
        raise RuntimeError(
            f"{record['sample_id']}: empty reference answer."
        )

    if len(images) != num_frames:
        raise RuntimeError(
            f"{record['sample_id']}: images={len(images)}, "
            f"num_frames={num_frames}."
        )

    placeholder_count = user_text.count("<image>")
    if placeholder_count != num_frames:
        raise RuntimeError(
            f"{record['sample_id']}: <image> count={placeholder_count}, "
            f"num_frames={num_frames}."
        )

    metadata = {
        "answer_format": str(record.get("answer_format", "")),
        "primary_capability": str(record.get("primary_capability", "")),
        "capability_group": str(record.get("capability_group", "")),
        "secondary_capabilities": record.get("secondary_capabilities", []),
        "ood": bool(record.get("ood", False)),
        "clinical": bool(record.get("clinical", False)),
        "candidate_count_1fps": int(record["candidate_count_1fps"]),
    }

    return {
        "sample_id": str(record["sample_id"]),
        "qID": request.qID,
        "dataset": str(record["dataset"]),
        "split": str(record["split"]),
        "images": images,
        "frame_labels": frame_labels,
        "num_frames": num_frames,
        "user_text": user_text,
        "assistant_text": assistant_text,
        "metadata": metadata,
    }


# =============================================================================
# BUILD SPLIT
# =============================================================================

def build_one_split(
    *,
    split_name: str,
    manifest_path: Path,
    output_path: Path,
    data_root: Path,
    shared_prompt: str,
    check_images: bool,
) -> dict[str, Any]:
    records = load_jsonl(manifest_path)

    logging.info(
        "%s: loaded %d manifest record(s) from %s",
        split_name,
        len(records),
        manifest_path,
    )

    converted: list[dict[str, Any]] = []
    total_frames = 0
    min_frames: int | None = None
    max_frames: int | None = None

    for index, record in enumerate(records, start=1):
        item = convert_record(
            record=record,
            data_root=data_root,
            shared_prompt=shared_prompt,
            check_images=check_images,
        )

        converted.append(item)
        n = int(item["num_frames"])
        total_frames += n
        min_frames = n if min_frames is None else min(min_frames, n)
        max_frames = n if max_frames is None else max(max_frames, n)

        if index % 1000 == 0:
            logging.info(
                "  %s: converted %d/%d",
                split_name,
                index,
                len(records),
            )

    written = atomic_write_jsonl(output_path, converted)

    if written != len(records):
        raise RuntimeError(
            f"{split_name}: wrote {written} records, expected {len(records)}."
        )

    return {
        "manifest": str(manifest_path),
        "output": str(output_path),
        "samples": len(records),
        "total_frame_references": total_frames,
        "frames_per_sample": {
            "min": min_frames,
            "max": max_frames,
            "mean": (
                total_frames / len(records)
                if records
                else None
            ),
        },
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    args = parse_args()
    configure_logging()

    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Data root does not exist: {data_root}"
        )

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else data_root
    )
    output_root.mkdir(parents=True, exist_ok=True)

    output_paths = [
        output_root / filename
        for filename in OUTPUT_JSONLS.values()
    ]
    summary_path = output_root / SUMMARY_FILENAME

    require_output_available(
        [*output_paths, summary_path],
        overwrite=args.overwrite,
    )

    logging.info(
        "=== ORena SEGMENT simplified DoRA SFT JSONL build ==="
    )
    logging.info("Data root: %s", data_root)
    logging.info("Output root: %s", output_root)
    logging.info(
        "Prompt settings: target_fps=%.1f, max_frames=%d, few_shot=%s",
        TARGET_FPS,
        MAX_FRAMES,
        USE_FEW_SHOT_EXAMPLES,
    )

    fo_definitions, fo_source = load_training_fo_definitions(
        data_root=data_root,
        explicit_path=args.fo_definitions,
    )

    fo_class_names = resolve_fo_class_names(fo_definitions)

    if USE_FEW_SHOT_EXAMPLES:
        shared_prompt = build_shared_prompt(
            fo_definitions=fo_definitions,
        )
    else:
        shared_prompt = build_shared_prompt(
            fo_definitions=fo_definitions,
            few_shot_examples=(),
        )

    shared_prompt_sha256 = hashlib.sha256(
        shared_prompt.encode("utf-8")
    ).hexdigest()

    logging.info("FO class source: %s", fo_source)
    logging.info(
        "Canonical FO classes (%d): %s",
        len(fo_class_names),
        ", ".join(fo_class_names),
    )
    logging.info(
        "Shared prompt: %d chars, sha256=%s",
        len(shared_prompt),
        shared_prompt_sha256,
    )

    split_summaries: dict[str, Any] = {}

    for split_name in (
        "train",
        "val_heico",
        "val_lapchole",
        "val_all",
    ):
        manifest_path = data_root / INPUT_MANIFESTS[split_name]
        output_path = output_root / OUTPUT_JSONLS[split_name]

        split_summaries[split_name] = build_one_split(
            split_name=split_name,
            manifest_path=manifest_path,
            output_path=output_path,
            data_root=data_root,
            shared_prompt=shared_prompt,
            check_images=not args.skip_image_check,
        )

        logging.info(
            "%s: wrote %d sample(s) -> %s",
            split_name,
            split_summaries[split_name]["samples"],
            output_path,
        )

    if split_summaries["train"]["samples"] <= 0:
        raise RuntimeError("Training SFT JSONL is empty.")

    if split_summaries["val_all"]["samples"] <= 0:
        raise RuntimeError("Combined validation SFT JSONL is empty.")

    summary = {
        "purpose": (
            "Simplified permanent multimodal SFT schema aligned with the "
            "current ORena Docker InternVL inference prompt."
        ),
        "schema_fields": [
            "sample_id",
            "qID",
            "dataset",
            "split",
            "images",
            "frame_labels",
            "num_frames",
            "user_text",
            "assistant_text",
            "metadata",
        ],
        "training_contract": {
            "visual_input": "images",
            "model_user_text": "user_text",
            "supervised_target": "assistant_text",
            "analysis_only": "metadata",
        },
        "data_root": str(data_root),
        "output_root": str(output_root),
        "prompt_alignment": {
            "few_shot_examples": USE_FEW_SHOT_EXAMPLES,
            "target_fps": TARGET_FPS,
            "max_frames": MAX_FRAMES,
            "frame_prefix_format": "HH:MM:SS: <image>\\n",
            "fo_definitions_source": fo_source,
            "fo_class_names": list(fo_class_names),
            "shared_prompt_chars": len(shared_prompt),
            "shared_prompt_sha256": shared_prompt_sha256,
        },
        "image_check_performed": not args.skip_image_check,
        "splits": split_summaries,
    }

    atomic_write_json(summary_path, summary)

    logging.info("Summary written to %s", summary_path)
    logging.info(
        "=== simplified SFT JSONL build complete ==="
    )
    logging.info(
        "Next step: InternVL3.5-8B DoRA training loader/collator. "
        "Use only images + user_text + assistant_text for model training; "
        "metadata is analysis-only."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())