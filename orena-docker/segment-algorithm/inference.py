"""ORena SAVE FOCUS — SEGMENT Track — InternVL3.5 inference.

Docker/platform mode reads a batch of focus.Request objects from
/input/request.json and writes /output/answer.json.

Direct local mode (``python inference.py``) automatically reads the committed
sample batch from ``test/input/interface_1`` and writes to
``test/output/interface_1``. Paths can also be overridden with
FOCUS_INPUT_PATH and FOCUS_OUTPUT_PATH.

Each request has its own already-trimmed video clip. This implementation uses
the overlayed <qID>.mp4 so the original-procedure HH:MM:SS clock is visible.

The model, tokenizer, foreign-object class names, and shared prompt are loaded
once per batch. One focus.Response is written for every request.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import torch

from focus import Request, Response, load_requests, save_items

from answer_utils import normalize_answer
from model_utils import InternVLInferenceEngine
from prompt_utils import (
    build_prompt,
    build_shared_prompt,
    load_fo_definitions,
    resolve_fo_class_names,
    seconds_to_timestamp,
)
from video_utils import load_clip_frames


# =============================================================================
# Paths supplied by the challenge platform
# =============================================================================

APP_PATH = Path(__file__).resolve().parent
#print(APP_PATH)


def resolve_io_paths() -> tuple[Path, Path, str]:
    """Resolve input/output paths for Docker or direct local execution.

    Priority:
    1. FOCUS_INPUT_PATH + FOCUS_OUTPUT_PATH environment variables, if both set.
    2. Docker/platform mode when running inside a container.
    3. Direct ``python inference.py`` mode using the committed local test batch.
    """

    env_input = os.environ.get("FOCUS_INPUT_PATH")
#    print("env_input: ", env_input)
    env_output = os.environ.get("FOCUS_OUTPUT_PATH")
#    print("env_input: ", env_output)

    if env_input or env_output:
        if not env_input or not env_output:
            raise ValueError(
                "Set both FOCUS_INPUT_PATH and FOCUS_OUTPUT_PATH, or neither."
            )

        return (
            Path(env_input).expanduser().resolve(),
            Path(env_output).expanduser().resolve(),
            "environment",
        )

    # Challenge execution and do_test_run.sh both run inside Docker.
    if Path("/.dockerenv").exists() or Path("/input/request.json").is_file():
        return Path("/input"), Path("/output"), "docker"

    # Direct host execution:
    #   cd segment-algorithm
    #   python inference.py
    return (
        APP_PATH / "test" / "input" / "interface_1",
        APP_PATH / "test" / "output" / "interface_1",
        "local",
    )


INPUT_PATH, OUTPUT_PATH, EXECUTION_MODE = resolve_io_paths()

REQUESTS_PATH = INPUT_PATH / "request.json"
FO_DEFINITIONS_PATH = INPUT_PATH / "FO_definitions.json"

USE_OVERLAY = True
VIDEO_DIR = INPUT_PATH / ("overlayed" if USE_OVERLAY else "plain")

MODEL_PATH = (
    APP_PATH
    / "resources"
    / "InternVL3_5-8B-Instruct"
)


# =============================================================================
# Inference configuration
# =============================================================================

#USE_FEW_SHOT_EXAMPLES = True
USE_FEW_SHOT_EXAMPLES = False

TARGET_FPS = 1.0
MAX_FRAMES = 32
NUM_DECODE_THREADS = 1

DEVICE = "cuda:0"
DTYPE = torch.float16
INPUT_SIZE = 448
MAX_TILES_PER_FRAME = 1
MAX_NEW_TOKENS = 128


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def clip_path_for(request: Request) -> Path:
    """Return the already-trimmed clip belonging to one request."""

    return VIDEO_DIR / f"{request.qID}.mp4"


def build_shared_inference_prompt(
    fo_definitions: str,
) -> tuple[str, tuple[str, ...]]:
    """Build the shared prompt and canonical FO-class list once per batch."""

    fo_class_names = resolve_fo_class_names(
        fo_definitions
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

    return shared_prompt, fo_class_names


def ordered_responses(
    requests: Sequence[Request],
    responses_by_qid: dict[str, Response],
) -> list[Response]:
    """Return responses in the same order as request.json."""

    return [
        responses_by_qid[str(request.qID)]
        for request in requests
    ]


def save_responses(
    requests: Sequence[Request],
    responses_by_qid: dict[str, Response],
    output_path: Path,
) -> None:
    """Write the complete response list to answer.json.

    The response map is pre-populated with empty answers, so every saved file
    contains one response for every qID.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_items(
        ordered_responses(requests, responses_by_qid),
        output_path,
    )


def make_absolute_frame_labels(
    request: Request,
    relative_timestamps: Sequence[float],
) -> list[str]:
    """Convert clip-relative frame times to original-procedure timestamps."""

    return [
        seconds_to_timestamp(
            float(request.start_time) + float(relative_timestamp)
        )
        for relative_timestamp in relative_timestamps
    ]


def run_cpu_interface_smoke_test(
    requests: Sequence[Request],
    responses_by_qid: dict[str, Response],
    output_path: Path,
) -> int:
    """Write structurally valid empty responses when CUDA is unavailable.

    This mode exists so the Docker interface can still be tested on an Intel Mac.
    It does not test InternVL inference and must not be interpreted as a model test.
    The official SEGMENT platform supplies an H100 GPU, so this branch should not
    run during challenge evaluation.
    """

    log.warning(
        "CUDA is unavailable. Running interface-only CPU smoke mode; "
        "InternVL will not be loaded and all answers will be empty."
    )

    save_responses(
        requests=requests,
        responses_by_qid=responses_by_qid,
        output_path=output_path,
    )

    log.info(
        "Wrote %d placeholder response(s) to %s.",
        len(requests),
        output_path,
    )
    return 0


# =============================================================================
# Main
# =============================================================================


def run() -> int:
    process_start = time.monotonic()
    output_path = OUTPUT_PATH / "answer.json"

    log.info("=== ORena SAVE FOCUS SEGMENT inference start ===")
    log.info("Execution mode: %s", EXECUTION_MODE)
    log.info("Input path: %s", INPUT_PATH)
    log.info("Output path: %s", OUTPUT_PATH)
    log.info("PyTorch: %s", torch.__version__)
    log.info("PyTorch CUDA runtime: %s", torch.version.cuda)
    log.info("CUDA available: %s", torch.cuda.is_available())

    # -------------------------------------------------------------------------
    # Load batch inputs
    # -------------------------------------------------------------------------

    if not REQUESTS_PATH.is_file():
        log.error("Missing request file: %s", REQUESTS_PATH)
        return 1

    if not FO_DEFINITIONS_PATH.is_file():
        log.error(
            "Missing foreign-object definitions: %s",
            FO_DEFINITIONS_PATH,
        )
        return 1

    requests = list(load_requests(REQUESTS_PATH))

    if not requests:
        log.error("request.json contains no requests.")
        return 1

    log.info("Loaded %d request(s).", len(requests))
    log.info("Video directory: %s", VIDEO_DIR)

    fo_definitions = load_fo_definitions(FO_DEFINITIONS_PATH)
    shared_prompt, fo_class_names = build_shared_inference_prompt(
        fo_definitions
    )

    log.info(
        "Loaded %d canonical foreign-object class name(s).",
        len(fo_class_names),
    )
    log.info("Shared prompt length: %d characters.", len(shared_prompt))

    # Pre-populate every qID with an empty answer. This guarantees that the
    # output remains structurally complete if a later question fails.
    responses_by_qid = {
        str(request.qID): Response(
            qID=str(request.qID),
            content="",
            latency=0.0,
        )
        for request in requests
    }

    save_responses(
        requests=requests,
        responses_by_qid=responses_by_qid,
        output_path=output_path,
    )

    # Docker Desktop on an Intel Mac cannot expose an NVIDIA CUDA GPU.
    if not torch.cuda.is_available():
        return run_cpu_interface_smoke_test(
            requests=requests,
            responses_by_qid=responses_by_qid,
            output_path=output_path,
        )

    log.info("GPU: %s", torch.cuda.get_device_name(0))
    total_vram_gb = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024**3)
    )
    log.info("Total GPU memory: %.2f GB", total_vram_gb)

    # -------------------------------------------------------------------------
    # Load InternVL exactly once for the complete batch
    # -------------------------------------------------------------------------

    engine = InternVLInferenceEngine(
        model_path=MODEL_PATH,
        device=DEVICE,
        dtype=DTYPE,
        input_size=INPUT_SIZE,
        max_tiles_per_frame=MAX_TILES_PER_FRAME,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    model_load_start = time.monotonic()

    try:
        engine.load()
    except Exception:
        # All qIDs already have empty placeholder responses in answer.json.
        log.exception("InternVL model loading failed.")
        return 0

    log.info(
        "Model loaded in %.2f seconds.",
        time.monotonic() - model_load_start,
    )

    # -------------------------------------------------------------------------
    # Process each independent (clip, question) pair
    # -------------------------------------------------------------------------

    try:
        for index, request in enumerate(requests, start=1):
            qid = str(request.qID)
            question_start = time.monotonic()

            log.info(
                "[%d/%d] qID=%s, videoID=%s, "
                "original window=[%.3f, %.3f] seconds",
                index,
                len(requests),
                qid,
                request.videoID,
                float(request.start_time),
                float(request.end_time),
            )

            clip = None
            prediction = None

            try:
                video_path = clip_path_for(request)

                clip = load_clip_frames(
                    video_path=video_path,
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
                    num_frames=clip.num_frames,
                    target_fps=TARGET_FPS,
                    max_frames=MAX_FRAMES,
                )

#                print("\n" + "=" * 100)
#                print(f"PROMPT FOR qID={qid}")
#                print("=" * 100)
#                print(prompt)
#                print("=" * 100 + "\n", flush=True)





                prediction = engine.predict(
                    images=clip.images,
                    prompt=prompt,
                    frame_labels=frame_labels,
                )

                answer = normalize_answer(
                    prediction.answer,
                    fo_class_names=fo_class_names,
                )

                latency = time.monotonic() - question_start

                responses_by_qid[qid] = Response(
                    qID=qid,
                    content=answer,
                    latency=latency,
                )

                log.info(
                    "[%d/%d] qID=%s answered in %.2f s: %r",
                    index,
                    len(requests),
                    qid,
                    latency,
                    answer,
                )
                log.info(
                    "  frames=%d, patches=%d, source_fps=%.3f, "
                    "clip_duration=%.3f s",
                    prediction.num_frames,
                    prediction.num_patches,
                    clip.source_fps,
                    clip.clip_duration,
                )

                if prediction.peak_gpu_memory_gb is not None:
                    log.info(
                        "  peak allocated GPU memory: %.2f GB",
                        prediction.peak_gpu_memory_gb,
                    )

            except Exception:
                latency = time.monotonic() - question_start

                # Keep the pre-populated empty content, but record the measured
                # latency for this failed question.
                responses_by_qid[qid] = Response(
                    qID=qid,
                    content="",
                    latency=latency,
                )

                log.exception(
                    "[%d/%d] qID=%s failed after %.2f s; "
                    "emitting an empty answer.",
                    index,
                    len(requests),
                    qid,
                    latency,
                )

            finally:
                # Do not call torch.cuda.empty_cache() after every question:
                # retaining the CUDA allocator cache is faster for the batch.
                if prediction is not None:
                    del prediction
                if clip is not None:
                    del clip

                # Save after every question. answer.json remains complete because
                # unanswered qIDs retain their empty placeholder responses.
                save_responses(
                    requests=requests,
                    responses_by_qid=responses_by_qid,
                    output_path=output_path,
                )

    finally:
        engine.unload()

    total_seconds = time.monotonic() - process_start

    log.info(
        "Wrote %d response(s) to %s.",
        len(requests),
        output_path,
    )
    log.info(
        "=== inference complete in %.2f seconds ===",
        total_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
