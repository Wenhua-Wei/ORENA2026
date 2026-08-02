"""ORena SAVE FOCUS — FRAME Track — InternVL3.5 inference.

One container run receives a batch of ``focus.Request`` objects in
``/input/request.json``. Each request has one still image at
``/input/frames/<qID>.png``.

The model, tokenizer, foreign-object definitions, and shared zero-shot prompt
are loaded once per batch. One ``focus.Response`` is written for every request.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import torch

from focus import Request, Response, load_requests, save_items

from answer_utils import extract_fo_class_names, normalize_answer
from image_utils import load_frame
from model_utils import InternVLInferenceEngine
from prompt_utils import (
    build_prompt,
    build_shared_prompt,
    load_fo_definitions,
)


# =============================================================================
# Paths
# =============================================================================

APP_PATH = Path(__file__).resolve().parent

DOCKER_INPUT_PATH = Path("/input")
DOCKER_OUTPUT_PATH = Path("/output")

LOCAL_INPUT_PATH = (
    APP_PATH
    / "test"
    / "input"
    / "interface_1"
)

LOCAL_OUTPUT_PATH = (
    APP_PATH
    / "test"
    / "output"
    / "interface_1"
)

# Inside Docker, /input/request.json is mounted by the platform.
# Otherwise, use the committed local test batch.
if (DOCKER_INPUT_PATH / "request.json").is_file():
    INPUT_PATH = DOCKER_INPUT_PATH
    OUTPUT_PATH = DOCKER_OUTPUT_PATH
    EXECUTION_MODE = "docker"
else:
    INPUT_PATH = LOCAL_INPUT_PATH
    OUTPUT_PATH = LOCAL_OUTPUT_PATH
    EXECUTION_MODE = "local"

REQUESTS_PATH = INPUT_PATH / "request.json"
FO_DEFINITIONS_PATH = INPUT_PATH / "FO_definitions.json"
FRAME_DIR = INPUT_PATH / "frames"

MODEL_PATH = (
    APP_PATH
    / "resources"
    / "InternVL3_5-8B-Instruct"
)


# =============================================================================
# Inference configuration
# =============================================================================

USE_FEW_SHOT_EXAMPLES = False

DEVICE = "cuda:0"
DTYPE = torch.float16

INPUT_SIZE = 448
MAX_TILES_PER_IMAGE = 4
USE_THUMBNAIL = True
MAX_NEW_TOKENS = 64

# Useful during development. Disable for the final submission if profiling
# shows that CUDA synchronization and peak-memory collection add meaningful
# latency.
COLLECT_GPU_DIAGNOSTICS = False


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


def frame_path_for(request: Request) -> Path:
    """Return the PNG frame belonging to one request."""

    return FRAME_DIR / f"{request.qID}.png"


def build_shared_inference_prompt(
    fo_definitions: str,
) -> tuple[str, tuple[str, ...]]:
    """Build the shared prompt and canonical FO-class list once per batch."""

    fo_class_names = extract_fo_class_names(
        fo_definitions
    )

    if not fo_class_names:
        raise ValueError(
            "No foreign-object class names could be extracted from "
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

    return shared_prompt, fo_class_names


def validate_requests(
    requests: Sequence[Request],
) -> None:
    """Check that the batch contains unique, non-empty qIDs."""

    qids = [
        str(request.qID).strip()
        for request in requests
    ]

    if any(not qid for qid in qids):
        raise ValueError(
            "Every request must contain a non-empty qID."
        )

    if len(qids) != len(set(qids)):
        raise ValueError(
            "request.json contains duplicate qIDs."
        )


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
    """Write the complete ordered response list to answer.json."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_items(
        ordered_responses(
            requests,
            responses_by_qid,
        ),
        output_path,
    )


def run_cpu_interface_smoke_test(
    requests: Sequence[Request],
    responses_by_qid: dict[str, Response],
    output_path: Path,
) -> int:
    """Write valid placeholder responses when CUDA is unavailable.

    This mode verifies the Docker request/response interface only. It does not
    load InternVL and must not be interpreted as a model test. The official
    FRAME platform provides an NVIDIA L40S GPU.
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
    """Run FRAME inference for one complete request batch."""

    process_start = time.monotonic()
    output_path = OUTPUT_PATH / "answer.json"

    log.info(
        "=== ORena SAVE FOCUS FRAME inference start ==="
    )
    log.info("PyTorch: %s", torch.__version__)
    log.info(
        "PyTorch CUDA runtime: %s",
        torch.version.cuda,
    )
    log.info(
        "CUDA available: %s",
        torch.cuda.is_available(),
    )

    log.info("Execution mode: %s", EXECUTION_MODE)
    log.info("Input path: %s", INPUT_PATH)
    log.info("Output path: %s", OUTPUT_PATH)

    if not REQUESTS_PATH.is_file():
        log.error(
            "Missing request file: %s",
            REQUESTS_PATH,
        )
        return 1

    if not FO_DEFINITIONS_PATH.is_file():
        log.error(
            "Missing foreign-object definitions: %s",
            FO_DEFINITIONS_PATH,
        )
        return 1

    if not FRAME_DIR.is_dir():
        log.error(
            "Missing FRAME image directory: %s",
            FRAME_DIR,
        )
        return 1

    try:
        requests = list(
            load_requests(REQUESTS_PATH)
        )
        validate_requests(requests)
    except Exception:
        log.exception(
            "Failed to load or validate request.json."
        )
        return 1

    if not requests:
        log.error(
            "request.json contains no requests."
        )
        return 1

    log.info(
        "Loaded %d request(s).",
        len(requests),
    )
    log.info(
        "Frame directory: %s",
        FRAME_DIR,
    )

    try:
        fo_definitions = load_fo_definitions(
            FO_DEFINITIONS_PATH
        )

        shared_prompt, fo_class_names = (
            build_shared_inference_prompt(
                fo_definitions
            )
        )
    except Exception:
        log.exception(
            "Failed to load the foreign-object definitions "
            "or build the shared prompt."
        )
        return 1

    log.info(
        "Loaded %d canonical foreign-object class name(s).",
        len(fo_class_names),
    )
    log.info(
        "Shared prompt length: %d characters.",
        len(shared_prompt),
    )

    responses_by_qid = {
        str(request.qID): Response(
            qID=str(request.qID),
            content="",
            latency=0.0,
        )
        for request in requests
    }

    try:
        save_responses(
            requests=requests,
            responses_by_qid=responses_by_qid,
            output_path=output_path,
        )
    except Exception:
        log.exception(
            "Failed to write the initial answer.json."
        )
        return 1

    if not torch.cuda.is_available():
        return run_cpu_interface_smoke_test(
            requests=requests,
            responses_by_qid=responses_by_qid,
            output_path=output_path,
        )

    log.info(
        "GPU: %s",
        torch.cuda.get_device_name(0),
    )

    total_vram_gb = (
        torch.cuda.get_device_properties(
            0
        ).total_memory
        / (1024**3)
    )

    log.info(
        "Total GPU memory: %.2f GB",
        total_vram_gb,
    )
    log.info(
        "Model path: %s",
        MODEL_PATH,
    )
    log.info(
        "FRAME configuration: input_size=%d, "
        "max_tiles_per_image=%d, thumbnail=%s, "
        "max_new_tokens=%d",
        INPUT_SIZE,
        MAX_TILES_PER_IMAGE,
        USE_THUMBNAIL,
        MAX_NEW_TOKENS,
    )

    engine = InternVLInferenceEngine(
        model_path=MODEL_PATH,
        device=DEVICE,
        dtype=DTYPE,
        input_size=INPUT_SIZE,
        max_tiles_per_image=MAX_TILES_PER_IMAGE,
        use_thumbnail=USE_THUMBNAIL,
        max_new_tokens=MAX_NEW_TOKENS,
        collect_gpu_diagnostics=(
            COLLECT_GPU_DIAGNOSTICS
        ),
    )

    model_load_start = time.monotonic()

    try:
        engine.load()
    except Exception:
        log.exception(
            "InternVL model loading failed."
        )
        return 0

    log.info(
        "Model loaded in %.2f seconds.",
        time.monotonic() - model_load_start,
    )

    try:
        for index, request in enumerate(
            requests,
            start=1,
        ):
            qid = str(request.qID)
            question_start = time.monotonic()

            log.info(
                "[%d/%d] qID=%s, videoID=%s, "
                "frame_time=%.3f seconds",
                index,
                len(requests),
                qid,
                request.videoID,
                float(request.start_time),
            )

            image = None
            prediction = None

            try:
                image_path = frame_path_for(
                    request
                )

                image = load_frame(
                    image_path
                )

                native_size = image.size

                prompt = build_prompt(
                    request=request,
                    shared_prompt=shared_prompt,
                )

                prediction = engine.predict(
                    image=image,
                    prompt=prompt,
                )

                answer = normalize_answer(
                    prediction.answer,
                    fo_class_names=fo_class_names,
                )

                latency = (
                    time.monotonic()
                    - question_start
                )

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

                if answer != prediction.answer:
                    log.info(
                        "  raw model answer: %r",
                        prediction.answer,
                    )

                log.info(
                    "  native_size=%s, images=%d, "
                    "patches=%d, model_inference=%.2f s",
                    native_size,
                    prediction.num_images,
                    prediction.num_patches,
                    prediction.inference_seconds,
                )

                if (
                    prediction.peak_gpu_memory_gb
                    is not None
                ):
                    log.info(
                        "  peak allocated GPU memory: %.2f GB",
                        prediction.peak_gpu_memory_gb,
                    )

            except Exception:
                latency = (
                    time.monotonic()
                    - question_start
                )

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
                if prediction is not None:
                    del prediction

                if image is not None:
                    del image

                try:
                    save_responses(
                        requests=requests,
                        responses_by_qid=responses_by_qid,
                        output_path=output_path,
                    )
                except Exception:
                    log.exception(
                        "Failed to update answer.json after qID=%s.",
                        qid,
                    )

    finally:
        engine.unload()

    total_seconds = (
        time.monotonic()
        - process_start
    )

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