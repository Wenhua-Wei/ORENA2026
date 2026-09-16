#!/usr/bin/env python3
"""
Run ORena SAVE FOCUS SEGMENT inference on the same balanced 250-case local set
using a fine-tuned InternVL3.5-8B checkpoint consisting of:

    <checkpoint>/dora_adapter/
    <checkpoint>/tokenizer/          (optional; falls back to base tokenizer)
    <checkpoint>/training_state.json (metadata; not required for inference)

The reference-aligned training keeps InternVL's mlp1 projector frozen, so no
mlp1.pt is expected or loaded. Inference uses the original mlp1 from the base
InternVL3.5-8B checkpoint.

The script mirrors the zero-shot local prediction pipeline:
- same prepared_50_per_capability/request.json and selected_qids.json
- same overlayed/<qID>.mp4 inputs
- same segment-algorithm prompt_utils.py / video_utils.py / model_utils.py /
  answer_utils.py
- 1 fps, max 32 frames
- 448x448, one visual tile per frame
- deterministic generation
- same output schema expected by eval_segment_predictions.py

This is a local accuracy run, not an exact platform runtime simulation. The base
model is loaded once for all 250 questions, then the saved DoRA adapter is
attached once. The frozen mlp1 projector is retained from the base model.

Edit the HARD-CODED PATHS / INFERENCE CONFIGURATION sections below before
running a different checkpoint. No command-line arguments are used.
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

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from peft import PeftModel
from transformers import AutoTokenizer
from focus import Request, Response, load_requests, load_responses, save_items

# =============================================================================
# HARD-CODED PATHS
# =============================================================================

ORENA_ROOT = Path("/SAN/medic/Surgical_LLM_Agent/orena2026")

ALGORITHM_DIR = (
    ORENA_ROOT
    / "orena-docker"
    / "segment-algorithm"
)

PREPARED_DIR = (
    ORENA_ROOT
    / "data"
    / "segment"
    / "prepared_200_per_capability"
)

MODEL_PATH = (
    ALGORITHM_DIR
    / "resources"
    / "InternVL3_5-8B-Instruct"
)

# Change this when evaluating a different fine-tuned checkpoint.
CHECKPOINT_DIR = (
    ORENA_ROOT
    / "segment"
    / "dora-ft"
    / "outputs"
    / "full-b1ga6-22ep-2Sep"
    / "checkpoint-epoch-2"
)

# Change this together with CHECKPOINT_DIR so different checkpoints do not
# overwrite each other's predictions.
OUTPUT_DIR = (
    ORENA_ROOT
    / "segment"
    / "dora-ft"
    / "predictions"
    / "full-b1ga6-2Sep-ep2-1000eval"
)


# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================

#EXPECTED_TOTAL_QUESTIONS = 250

EXPECTED_TOTAL_QUESTIONS = 1000


DEVICE = "cuda:0"
PRECISION = "fp16"

USE_FEW_SHOT_EXAMPLES = False
USE_SYSTEM_TURN = False

TARGET_FPS = 1.0
MAX_FRAMES = 32
NUM_DECODE_THREADS = 1

INPUT_SIZE = 448
MAX_TILES_PER_FRAME = 1
MAX_NEW_TOKENS = 64

SAVE_EVERY = 10
RESUME = True

if not ALGORITHM_DIR.is_dir():
    raise FileNotFoundError(f"SEGMENT algorithm directory does not exist: {ALGORITHM_DIR}")
if str(ALGORITHM_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_DIR))

from answer_utils import normalize_answer  # noqa: E402
from model_utils import InternVLInferenceEngine  # noqa: E402
from prompt_utils import (  # noqa: E402
    build_prompt,
    build_shared_prompt,
    load_fo_definitions,
    resolve_fo_class_names,
    seconds_to_timestamp,
)
from video_utils import load_clip_frames  # noqa: E402

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("run_segment_dora_checkpoint_predictions")

LOG_FIELDS = [
    "qID", "request_index", "status", "raw_answer", "normalized_answer",
    "latency_s", "model_inference_s", "source_fps", "clip_duration_s",
    "num_frames", "num_patches", "first_relative_timestamp_s",
    "last_relative_timestamp_s", "first_absolute_timestamp",
    "last_absolute_timestamp", "peak_gpu_memory_gb", "error_type",
    "error_message", "completed_at_utc",
]

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def temp_json(path: Path) -> Path:
    return path.with_name(f"{path.stem}.tmp{path.suffix}")

def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_json(path)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)

def save_focus_items_atomic(items: Sequence[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_json(path)
    save_items(items, tmp)
    os.replace(tmp, path)

def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")

def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise NotADirectoryError(f"Required directory is missing: {path}")

def choose_dtype(name: str) -> tuple[torch.dtype, str]:
    if name == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested but unsupported by this GPU")
        return torch.bfloat16, "bf16"
    if name == "fp16":
        return torch.float16, "fp16"
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bf16"
    return torch.float16, "fp16"

def read_selected_qids(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("selected_qids.json must contain one JSON list")
    qids = [str(x).strip() for x in value]
    if any(not x for x in qids):
        raise ValueError("selected_qids.json contains an empty qID")
    if len(qids) != len(set(qids)):
        raise ValueError("selected_qids.json contains duplicate qIDs")
    if len(qids) != EXPECTED_TOTAL_QUESTIONS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_QUESTIONS} selected qIDs, got {len(qids)}")
    return qids

def validate_requests(requests: Sequence[Request], selected_qids: Sequence[str]) -> None:
    qids = [str(r.qID).strip() for r in requests]
    if len(qids) != EXPECTED_TOTAL_QUESTIONS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_QUESTIONS} requests, got {len(qids)}")
    if len(qids) != len(set(qids)):
        raise ValueError("request.json contains duplicate qIDs")
    if qids != list(selected_qids):
        if set(qids) != set(selected_qids):
            raise ValueError("request.json and selected_qids.json have different qID sets")
        raise ValueError("request.json and selected_qids.json have the same qIDs in different orders")

def make_absolute_frame_labels(request: Request, relative_timestamps: Sequence[float]) -> list[str]:
    return [
        seconds_to_timestamp(float(request.start_time) + float(t))
        for t in relative_timestamps
    ]

def validate_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    cfg_path = adapter_dir / "adapter_config.json"
    require_file(cfg_path)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not bool(cfg.get("use_dora", False)):
        raise RuntimeError("adapter_config.json does not have use_dora=true")
    targets = set(cfg.get("target_modules", []))
    required = {"q_proj", "k_proj", "v_proj", "o_proj"}
    if not required.issubset(targets):
        raise RuntimeError(f"DoRA target mismatch: {sorted(targets)}")
    LOG.info("Adapter config PASSED: use_dora=True targets=%s", sorted(targets))
    if cfg.get("base_model_name_or_path"):
        LOG.info("Adapter metadata base path: %s", cfg["base_model_name_or_path"])
        LOG.info("Ignoring that hint and attaching to the explicitly loaded InternVL language_model.")
    return cfg

def attach_checkpoint(
    engine: InternVLInferenceEngine,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Attach the DoRA adapter saved by the reference-aligned trainer.

    mlp1 is deliberately NOT loaded from the checkpoint because it was frozen
    throughout training. The base InternVL3.5-8B model already contains the
    correct unchanged mlp1 projector.
    """
    adapter_dir = checkpoint_dir / "dora_adapter"
    tokenizer_dir = checkpoint_dir / "tokenizer"
    training_state_path = checkpoint_dir / "training_state.json"

    require_dir(adapter_dir)
    adapter_cfg = validate_adapter_config(adapter_dir)

    LOG.info("Attaching saved DoRA adapter to language_model...")
    engine.model.language_model = PeftModel.from_pretrained(
        engine.model.language_model,
        str(adapter_dir),
        is_trainable=False,
    )
    engine.model.language_model = engine.model.language_model.to(engine.device)

    if not getattr(engine.model.language_model, "peft_config", None):
        raise RuntimeError("Reloaded PEFT model has no peft_config")

    LOG.info(
        "DoRA adapter reload PASSED: adapters=%s",
        sorted(engine.model.language_model.peft_config.keys()),
    )

    # Reference-aligned training froze mlp1, so the correct inference behavior is
    # to keep the original projector already present in the base InternVL model.
    LOG.info(
        "Using frozen base-model mlp1 projector; no mlp1 checkpoint is loaded."
    )

    tokenizer_source = tokenizer_dir if tokenizer_dir.is_dir() else engine.model_path
    LOG.info("Loading tokenizer from %s", tokenizer_source)
    engine.tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source),
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )

    training_state: dict[str, Any] | None = None
    if training_state_path.is_file():
        training_state = json.loads(training_state_path.read_text(encoding="utf-8"))
        LOG.info(
            "Checkpoint metadata: epoch=%s global_step=%s val_loss=%s",
            training_state.get("epoch"),
            training_state.get("global_step"),
            training_state.get("val_loss"),
        )

        saved_template = training_state.get("model_template")
        current_template = str(getattr(engine.model, "template", ""))
        if saved_template and saved_template != current_template:
            raise RuntimeError(
                "Checkpoint/base-model template mismatch: "
                f"checkpoint={saved_template!r}, base_model={current_template!r}"
            )

        saved_num_image_token = training_state.get("num_image_token")
        if (
            saved_num_image_token is not None
            and int(saved_num_image_token) != int(engine.model.num_image_token)
        ):
            raise RuntimeError(
                "Checkpoint/base-model num_image_token mismatch: "
                f"checkpoint={saved_num_image_token}, "
                f"base_model={engine.model.num_image_token}"
            )

    for parameter in engine.model.parameters():
        parameter.requires_grad = False

    engine.model.eval()
    engine.model.vision_model.eval()
    engine.model.mlp1.eval()
    engine.model.language_model.eval()

    trainable = sum(
        parameter.numel()
        for parameter in engine.model.parameters()
        if parameter.requires_grad
    )
    if trainable != 0:
        raise RuntimeError(
            f"Inference model still has {trainable} trainable parameters"
        )

    LOG.info("Inference freeze PASSED: trainable parameters = 0")

    return {
        "adapter_dir": str(adapter_dir),
        "tokenizer_source": str(tokenizer_source),
        "adapter_config": adapter_cfg,
        "mlp1_source": "frozen base InternVL3.5-8B model",
        "training_state_path": (
            str(training_state_path) if training_state_path.is_file() else None
        ),
        "training_state": training_state,
    }

def load_existing_responses(path: Path) -> dict[str, Response]:
    if not path.is_file():
        return {}
    out: dict[str, Response] = {}
    for r in load_responses(path):
        qid = str(r.qID)
        if qid in out:
            raise ValueError(f"Duplicate qID in existing responses: {qid}")
        out[qid] = r
    return out

def load_existing_log(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(LOG_FIELDS).difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Existing inference_log.csv missing columns: " + ", ".join(sorted(missing)))
        for row in reader:
            qid = str(row.get("qID", "")).strip()
            if not qid:
                continue
            if qid in out:
                raise ValueError(f"Duplicate qID in existing inference log: {qid}")
            out[qid] = {field: str(row.get(field, "")) for field in LOG_FIELDS}
    return out

def completed_qids(
    responses_by_qid: dict[str, Response],
    log_by_qid: dict[str, dict[str, str]],
) -> set[str]:
    done: set[str] = set()
    for qid, response in responses_by_qid.items():
        row = log_by_qid.get(qid)
        if row is None:
            if str(response.content).strip():
                done.add(qid)
            continue
        if row.get("status", "").strip().lower() == "success":
            done.add(qid)
    return done

def ordered_responses(requests: Sequence[Request], mapping: dict[str, Response]) -> list[Response]:
    return [mapping[str(r.qID)] for r in requests if str(r.qID) in mapping]

def ordered_log_rows(
    requests: Sequence[Request],
    mapping: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    return [mapping[str(r.qID)] for r in requests if str(r.qID) in mapping]

def write_log_atomic(
    path: Path,
    requests: Sequence[Request],
    mapping: dict[str, dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(ordered_log_rows(requests, mapping))
    os.replace(tmp, path)

def failed_rows(
    requests: Sequence[Request],
    log_by_qid: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    out = []
    for r in requests:
        qid = str(r.qID)
        row = log_by_qid.get(qid)
        if row and row.get("status", "").strip().lower() == "failed":
            out.append({
                "qID": qid,
                "error_type": row.get("error_type", ""),
                "error_message": row.get("error_message", ""),
                "latency_s": row.get("latency_s", ""),
                "completed_at_utc": row.get("completed_at_utc", ""),
            })
    return out

def checkpoint_outputs(
    *,
    requests: Sequence[Request],
    responses_by_qid: dict[str, Response],
    log_by_qid: dict[str, dict[str, str]],
    responses_path: Path,
    inference_log_path: Path,
    failed_questions_path: Path,
) -> None:
    save_focus_items_atomic(ordered_responses(requests, responses_by_qid), responses_path)
    write_log_atomic(inference_log_path, requests, log_by_qid)
    write_json_atomic(failed_questions_path, failed_rows(requests, log_by_qid))

def write_prediction_summary(
    *,
    path: Path,
    requests: Sequence[Request],
    responses_by_qid: dict[str, Response],
    log_by_qid: dict[str, dict[str, str]],
    process_start: float,
    model_load_seconds: float | None,
    checkpoint_load_seconds: float | None,
) -> None:
    rows = [log_by_qid[str(r.qID)] for r in requests if str(r.qID) in log_by_qid]
    success_rows = [x for x in rows if x.get("status", "").strip().lower() == "success"]
    failure_rows = [x for x in rows if x.get("status", "").strip().lower() == "failed"]

    def floats(field: str) -> list[float]:
        return [float(x[field]) for x in success_rows if x.get(field, "").strip()]

    def ints(field: str) -> list[int]:
        return [int(x[field]) for x in success_rows if x.get(field, "").strip()]

    latencies = floats("latency_s")
    model_latencies = floats("model_inference_s")
    clip_durations = floats("clip_duration_s")
    frame_counts = ints("num_frames")

    write_json_atomic(path, {
        "updated_at_utc": utc_now(),
        "elapsed_wall_clock_s": time.monotonic() - process_start,
        "model_load_seconds": model_load_seconds,
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "prepared_requests": len(requests),
        "responses_saved_total": len(responses_by_qid),
        "successes": len(success_rows),
        "failures": len(failure_rows),
        "unprocessed": len(requests) - len(success_rows) - len(failure_rows),
        "mean_success_latency_s": sum(latencies) / len(latencies) if latencies else None,
        "mean_model_inference_s": (
            sum(model_latencies) / len(model_latencies) if model_latencies else None
        ),
        "mean_clip_duration_s": (
            sum(clip_durations) / len(clip_durations) if clip_durations else None
        ),
        "mean_sampled_frames": (
            sum(frame_counts) / len(frame_counts) if frame_counts else None
        ),
        "timing_note": (
            "Base model and fine-tuned checkpoint loaded once for all 250 local "
            "questions; not an exact platform runtime simulation."
        ),
    })

def run() -> int:
    process_start = time.monotonic()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if MAX_NEW_TOKENS <= 0:
        raise ValueError("MAX_NEW_TOKENS must be positive")

    if SAVE_EVERY <= 0:
        raise ValueError("SAVE_EVERY must be positive")

    prepared_dir = PREPARED_DIR.expanduser().resolve()
    model_path = MODEL_PATH.expanduser().resolve()
    checkpoint_dir = CHECKPOINT_DIR.expanduser().resolve()
    output_dir = OUTPUT_DIR.expanduser().resolve()

    require_dir(prepared_dir)
    require_dir(model_path)
    require_dir(checkpoint_dir)

    requests_path = prepared_dir / "request.json"
    selected_qids_path = prepared_dir / "selected_qids.json"
    fo_definitions_path = prepared_dir / "FO_definitions.json"
    overlayed_dir = prepared_dir / "overlayed"

    for p in (requests_path, selected_qids_path, fo_definitions_path):
        require_file(p)
    require_dir(overlayed_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / "responses.json"
    inference_log_path = output_dir / "inference_log.csv"
    failed_questions_path = output_dir / "failed_questions.json"
    run_config_path = output_dir / "run_config.json"
    summary_path = output_dir / "prediction_summary.json"

    if not RESUME:
        for p in (
            responses_path,
            inference_log_path,
            failed_questions_path,
            run_config_path,
            summary_path,
        ):
            if p.exists():
                p.unlink()

    dtype, precision = choose_dtype(PRECISION)

    LOG.info("=== ORena SEGMENT fine-tuned checkpoint prediction start ===")
    LOG.info("GPU: %s", torch.cuda.get_device_name(0))
    LOG.info("Precision: %s", precision)
    LOG.info("Prepared set: %s", prepared_dir)
    LOG.info("Base model: %s", model_path)
    LOG.info("Checkpoint: %s", checkpoint_dir)
    LOG.info("Output: %s", output_dir)
    LOG.info(
        "Settings: %.1f fps, max_frames=%d, input_size=%d, "
        "max_tiles=%d, max_new_tokens=%d, use_system_turn=%s",
        TARGET_FPS,
        MAX_FRAMES,
        INPUT_SIZE,
        MAX_TILES_PER_FRAME,
        MAX_NEW_TOKENS,
        USE_SYSTEM_TURN,
    )

    requests = list(load_requests(requests_path))
    selected_qids = read_selected_qids(selected_qids_path)
    validate_requests(requests, selected_qids)

    missing_clips = [
        str(overlayed_dir / f"{r.qID}.mp4")
        for r in requests
        if not (overlayed_dir / f"{r.qID}.mp4").is_file()
    ]
    if missing_clips:
        raise FileNotFoundError(
            f"{len(missing_clips)} overlayed clips missing:\n  - "
            + "\n  - ".join(missing_clips[:20])
        )

    fo_definitions = load_fo_definitions(fo_definitions_path)
    fo_class_names = resolve_fo_class_names(fo_definitions)
    shared_prompt = (
        build_shared_prompt(fo_definitions=fo_definitions)
        if USE_FEW_SHOT_EXAMPLES
        else build_shared_prompt(fo_definitions=fo_definitions, few_shot_examples=())
    )

    responses_by_qid = load_existing_responses(responses_path) if RESUME else {}
    log_by_qid = load_existing_log(inference_log_path) if RESUME else {}

    known_qids = {str(r.qID) for r in requests}
    if set(responses_by_qid).difference(known_qids):
        raise ValueError("Existing responses contain unknown qIDs")
    if set(log_by_qid).difference(known_qids):
        raise ValueError("Existing logs contain unknown qIDs")

    done_qids = completed_qids(responses_by_qid, log_by_qid)
    pending_requests = [r for r in requests if str(r.qID) not in done_qids]
    LOG.info("Resume state: %d complete, %d pending",
             len(requests) - len(pending_requests), len(pending_requests))

    request_index_by_qid = {
        str(r.qID): i for i, r in enumerate(requests, start=1)
    }

    engine = InternVLInferenceEngine(
        model_path=model_path,
        device=DEVICE,
        dtype=dtype,
        input_size=INPUT_SIZE,
        max_tiles_per_frame=MAX_TILES_PER_FRAME,
        max_new_tokens=MAX_NEW_TOKENS,
        use_system_turn=USE_SYSTEM_TURN,
    )

    model_load_seconds = None
    checkpoint_load_seconds = None

    try:
        t0 = time.monotonic()
        engine.load()
        model_load_seconds = time.monotonic() - t0
        LOG.info("Base InternVL loaded in %.2fs", model_load_seconds)

        t0 = time.monotonic()
        checkpoint_info = attach_checkpoint(engine, checkpoint_dir)
        checkpoint_load_seconds = time.monotonic() - t0
        LOG.info("Fine-tuned checkpoint attached in %.2fs", checkpoint_load_seconds)

        write_json_atomic(run_config_path, {
            "created_at_utc": utc_now(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "pytorch": torch.__version__,
            "pytorch_cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0),
            "total_vram_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "model_path": str(model_path),
            "checkpoint_dir": str(checkpoint_dir),
            "prepared_dir": str(prepared_dir),
            "output_dir": str(output_dir),
            "device": DEVICE,
            "dtype": str(dtype),
            "precision": precision,
            "target_fps": TARGET_FPS,
            "max_frames": MAX_FRAMES,
            "num_decode_threads": NUM_DECODE_THREADS,
            "input_size": INPUT_SIZE,
            "max_tiles_per_frame": MAX_TILES_PER_FRAME,
            "max_new_tokens": MAX_NEW_TOKENS,
            "use_few_shot_examples": USE_FEW_SHOT_EXAMPLES,
            "use_system_turn": USE_SYSTEM_TURN,
            "checkpoint": checkpoint_info,
            "model_system_message": getattr(engine.model, "system_message", None),
        })

        newly_processed = 0

        for request in pending_requests:
            qid = str(request.qID)
            request_index = request_index_by_qid[qid]
            question_start = time.monotonic()
            clip = None
            prediction = None

            LOG.info(
                "[%d/%d] qID=%s source=%s window=[%.3f, %.3f]",
                request_index, len(requests), qid, request.videoID,
                float(request.start_time), float(request.end_time),
            )

            try:
                clip = load_clip_frames(
                    video_path=overlayed_dir / f"{qid}.mp4",
                    target_fps=TARGET_FPS,
                    max_frames=MAX_FRAMES,
                    num_threads=NUM_DECODE_THREADS,
                )

                frame_labels = make_absolute_frame_labels(request, clip.timestamps)

                prompt = build_prompt(
                    request=request,
                    shared_prompt=shared_prompt,
                    num_frames=clip.num_frames,
                    target_fps=TARGET_FPS,
                    max_frames=MAX_FRAMES,
                )

                prediction = engine.predict(
                    images=clip.images,
                    prompt=prompt,
                    frame_labels=frame_labels,
                    max_new_tokens=MAX_NEW_TOKENS,
                )

                answer = normalize_answer(
                    prediction.answer,
                    fo_class_names=fo_class_names,
                )
                latency = time.monotonic() - question_start

                responses_by_qid[qid] = Response(
                    qID=qid, content=answer, latency=latency
                )

                log_by_qid[qid] = {
                    "qID": qid,
                    "request_index": str(request_index),
                    "status": "success",
                    "raw_answer": prediction.answer,
                    "normalized_answer": answer,
                    "latency_s": f"{latency:.6f}",
                    "model_inference_s": f"{prediction.inference_seconds:.6f}",
                    "source_fps": f"{clip.source_fps:.6f}",
                    "clip_duration_s": f"{clip.clip_duration:.6f}",
                    "num_frames": str(prediction.num_frames),
                    "num_patches": str(prediction.num_patches),
                    "first_relative_timestamp_s": f"{clip.timestamps[0]:.6f}",
                    "last_relative_timestamp_s": f"{clip.timestamps[-1]:.6f}",
                    "first_absolute_timestamp": frame_labels[0],
                    "last_absolute_timestamp": frame_labels[-1],
                    "peak_gpu_memory_gb": (
                        "" if prediction.peak_gpu_memory_gb is None
                        else f"{prediction.peak_gpu_memory_gb:.6f}"
                    ),
                    "error_type": "",
                    "error_message": "",
                    "completed_at_utc": utc_now(),
                }

                LOG.info(
                    "[%d/%d] qID=%s answered in %.2fs: %r",
                    request_index, len(requests), qid, latency, answer,
                )

            except Exception as error:
                latency = time.monotonic() - question_start
                responses_by_qid[qid] = Response(qID=qid, content="", latency=latency)
                log_by_qid[qid] = {
                    "qID": qid,
                    "request_index": str(request_index),
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
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "completed_at_utc": utc_now(),
                }
                LOG.exception("[%d/%d] qID=%s failed after %.2fs",
                              request_index, len(requests), qid, latency)

            finally:
                if prediction is not None:
                    del prediction
                if clip is not None:
                    del clip

            newly_processed += 1

            if newly_processed % SAVE_EVERY == 0:
                checkpoint_outputs(
                    requests=requests,
                    responses_by_qid=responses_by_qid,
                    log_by_qid=log_by_qid,
                    responses_path=responses_path,
                    inference_log_path=inference_log_path,
                    failed_questions_path=failed_questions_path,
                )
                write_prediction_summary(
                    path=summary_path,
                    requests=requests,
                    responses_by_qid=responses_by_qid,
                    log_by_qid=log_by_qid,
                    process_start=process_start,
                    model_load_seconds=model_load_seconds,
                    checkpoint_load_seconds=checkpoint_load_seconds,
                )
                LOG.info("Checkpointed after %d new questions", newly_processed)

        checkpoint_outputs(
            requests=requests,
            responses_by_qid=responses_by_qid,
            log_by_qid=log_by_qid,
            responses_path=responses_path,
            inference_log_path=inference_log_path,
            failed_questions_path=failed_questions_path,
        )
        write_prediction_summary(
            path=summary_path,
            requests=requests,
            responses_by_qid=responses_by_qid,
            log_by_qid=log_by_qid,
            process_start=process_start,
            model_load_seconds=model_load_seconds,
            checkpoint_load_seconds=checkpoint_load_seconds,
        )

    finally:
        engine.unload()

    missing = known_qids.difference(responses_by_qid)
    if missing:
        raise RuntimeError(
            f"{len(missing)} responses missing after inference: "
            + ", ".join(sorted(missing)[:20])
        )

    LOG.info("Wrote %d responses to %s", len(responses_by_qid), responses_path)
    LOG.info("=== complete in %.2fs ===", time.monotonic() - process_start)
    return 0

if __name__ == "__main__":
    raise SystemExit(run())
