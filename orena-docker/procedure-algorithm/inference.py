"""ORena SAVE FOCUS — PROCEDURE Track — InternVL3.5 inference.

Docker/platform mode reads a batch of ``focus.Request`` objects from
``/input/request.json`` and writes ``/output/answer.json``.

Direct local mode (``python inference.py``) automatically reads the committed
sample batch from ``test/input/interface_1`` and writes to
``test/output/interface_1``. Paths can also be overridden with
``FOCUS_INPUT_PATH`` and ``FOCUS_OUTPUT_PATH``.

Each request has its own already-trimmed PROCEDURE video context. The platform
timestamps in ``request.start_time`` / ``request.end_time`` refer to the
original procedure timeline; decoding of the supplied qID MP4 always begins at
local video time 0.

This PROCEDURE version uses question-conditioned frame sampling from
``video_utils.py``:

- GLOBAL
- LOCAL_ANCHOR
- TWO_ANCHOR
- INTERVAL
- FORWARD_SEARCH
- BACKWARD_SEARCH
- PREFIX_STATE

The model, tokenizer, DoRA adapter, foreign-object class names, and shared
prompt are loaded once per batch. One ``focus.Response`` is written for every
request, even if an individual question fails.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Sequence

# -----------------------------------------------------------------------------
# Offline / runtime environment
# -----------------------------------------------------------------------------

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import torch
from focus import Request, Response, load_requests, save_items
from peft import PeftModel

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
# Paths
# =============================================================================

APP_PATH = Path(__file__).resolve().parent


def resolve_io_paths() -> tuple[Path, Path, str]:
    """Resolve input/output paths for Docker or direct local execution.

    Priority:
    1. ``FOCUS_INPUT_PATH`` + ``FOCUS_OUTPUT_PATH`` when both are set.
    2. Docker/platform mode when ``/input`` is mounted.
    3. Direct host execution using ``test/input/interface_1``.
    """

    env_input = os.environ.get("FOCUS_INPUT_PATH")
    env_output = os.environ.get("FOCUS_OUTPUT_PATH")

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

    if Path("/.dockerenv").exists() or Path("/input/request.json").is_file():
        return Path("/input"), Path("/output"), "docker"

    return (
        APP_PATH / "test" / "input" / "interface_1",
        APP_PATH / "test" / "output" / "interface_1",
        "local",
    )


INPUT_PATH, OUTPUT_PATH, EXECUTION_MODE = resolve_io_paths()

REQUESTS_PATH = INPUT_PATH / "request.json"
FO_DEFINITIONS_PATH = INPUT_PATH / "FO_definitions.json"

# The current PROCEDURE Docker deliberately uses the timestamp-overlayed videos.
USE_OVERLAY = True
VIDEO_DIR = INPUT_PATH / ("overlayed" if USE_OVERLAY else "plain")

MODEL_PATH = (
    APP_PATH
    / "resources"
    / "InternVL3_5-8B-Instruct"
)

DORA_ADAPTER_PATH = (
    APP_PATH
    / "resources"
    / "checkpoint-epoch-5"
    / "dora_adapter"
)


# =============================================================================
# Inference configuration
# =============================================================================

# Keep prompting close to the working SEGMENT/DoRA setup.
USE_FEW_SHOT_EXAMPLES = False
USE_SYSTEM_TURN = False

# Broad PROCEDURE sampling density before capping. The actual returned frames
# depend on the temporal intent selected by video_utils.py.
TARGET_FPS = 1.0

# Start conservatively for the first Docker smoke test. Benchmark 32/48/64 later.
MAX_FRAMES = 52

# Dense sampling around explicit temporal anchors.
LOCAL_FPS = 2.0
LOCAL_RADIUS_SECONDS = 1.0

NUM_DECODE_THREADS = 1

DEVICE = "cuda:0"
DTYPE = torch.float16
INPUT_SIZE = 448
MAX_TILES_PER_FRAME = 1
MAX_NEW_TOKENS = 64


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
    """Return the already-trimmed qID MP4 belonging to one request."""

    return VIDEO_DIR / f"{request.qID}.mp4"


def build_shared_inference_prompt(
    fo_definitions: str,
) -> tuple[str, tuple[str, ...]]:
    """Build the shared prompt and canonical FO-class list once per batch."""

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

    return shared_prompt, fo_class_names


def ordered_responses(
    requests: Sequence[Request],
    responses_by_qid: dict[str, Response],
) -> list[Response]:
    """Return responses in exactly the same order as ``request.json``."""

    return [
        responses_by_qid[str(request.qID)]
        for request in requests
    ]


def save_responses(
    requests: Sequence[Request],
    responses_by_qid: dict[str, Response],
    output_path: Path,
) -> None:
    """Write the complete response list to ``answer.json``.

    ``responses_by_qid`` is pre-populated with empty answers so every write is
    structurally complete even if the process later fails on one question.
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
    """Convert clip-relative frame times to original-procedure HH:MM:SS labels."""

    return [
        seconds_to_timestamp(
            float(request.start_time) + float(relative_timestamp)
        )
        for relative_timestamp in relative_timestamps
    ]


def format_anchor_times(anchor_times: Sequence[float]) -> str:
    """Format absolute temporal anchors for logging."""

    if not anchor_times:
        return "none"

    return ", ".join(
        seconds_to_timestamp(float(value))
        for value in anchor_times
    )


def attach_dora_adapter(
    engine: InternVLInferenceEngine,
    adapter_path: Path,
) -> None:
    """Attach the fine-tuned DoRA adapter to InternVL's language model."""

    if not engine.is_loaded or engine.model is None:
        raise RuntimeError("Base InternVL model must be loaded before DoRA.")

    if not adapter_path.is_dir():
        raise FileNotFoundError(
            f"DoRA adapter directory does not exist: {adapter_path}"
        )

    adapter_config = adapter_path / "adapter_config.json"
    if not adapter_config.is_file():
        raise FileNotFoundError(
            f"Missing DoRA adapter config: {adapter_config}"
        )

    if not hasattr(engine.model, "language_model"):
        raise RuntimeError(
            "Loaded InternVL model does not expose model.language_model."
        )

    log.info("Loading DoRA adapter from: %s", adapter_path)

    engine.model.language_model = PeftModel.from_pretrained(
        engine.model.language_model,
        str(adapter_path),
        is_trainable=False,
    )

    engine.model.language_model = (
        engine.model.language_model
        .to(engine.device)
        .eval()
    )

    # Inference only.
    for parameter in engine.model.parameters():
        parameter.requires_grad = False

    engine.model.eval()

    if hasattr(engine.model, "vision_model"):
        engine.model.vision_model.eval()

    if hasattr(engine.model, "mlp1"):
        engine.model.mlp1.eval()

    engine.model.language_model.eval()

    log.info("DoRA adapter attached successfully.")


def run_cpu_interface_smoke_test(
    requests: Sequence[Request],
    responses_by_qid: dict[str, Response],
    output_path: Path,
) -> int:
    """Write structurally valid empty responses when CUDA is unavailable.

    This exists only for interface testing on machines without an NVIDIA GPU.
    It does not run InternVL inference and must not be interpreted as a model
    smoke test.
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

    log.info("=== ORena SAVE FOCUS PROCEDURE inference start ===")
    log.info("Execution mode: %s", EXECUTION_MODE)
    log.info("Input path: %s", INPUT_PATH)
    log.info("Output path: %s", OUTPUT_PATH)
    log.info("Video directory: %s", VIDEO_DIR)
    log.info("Use overlayed videos: %s", USE_OVERLAY)
    log.info("Use system turn: %s", USE_SYSTEM_TURN)
    log.info("Base model path: %s", MODEL_PATH)
    log.info("DoRA adapter path: %s", DORA_ADAPTER_PATH)
    log.info(
        "Sampling config: target_fps=%.2f, max_frames=%d, "
        "local_fps=%.2f, local_radius=%.1fs",
        TARGET_FPS,
        MAX_FRAMES,
        LOCAL_FPS,
        LOCAL_RADIUS_SECONDS,
    )
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

    if not VIDEO_DIR.is_dir():
        log.error("Missing video directory: %s", VIDEO_DIR)
        return 1

    try:
        requests = list(load_requests(REQUESTS_PATH))
    except Exception:
        log.exception("Failed to load request.json.")
        return 1

    if not requests:
        log.error("request.json contains no requests.")
        return 1

    qids = [str(request.qID) for request in requests]
    if len(qids) != len(set(qids)):
        log.error("request.json contains duplicate qIDs.")
        return 1

    missing_videos = [
        clip_path_for(request)
        for request in requests
        if not clip_path_for(request).is_file()
    ]

    if missing_videos:
        log.error(
            "Missing %d qID video file(s). First missing paths: %s",
            len(missing_videos),
            ", ".join(str(path) for path in missing_videos[:10]),
        )
        return 1

    log.info("Loaded %d request(s).", len(requests))

    try:
        fo_definitions = load_fo_definitions(FO_DEFINITIONS_PATH)
        shared_prompt, fo_class_names = build_shared_inference_prompt(
            fo_definitions
        )
    except Exception:
        log.exception("Failed to prepare FO definitions/shared prompt.")
        return 1

    log.info(
        "Loaded %d canonical foreign-object class name(s).",
        len(fo_class_names),
    )
    log.info("Shared prompt length: %d characters.", len(shared_prompt))

    # Pre-populate every qID with an empty answer. This guarantees that
    # answer.json stays structurally complete if a later question fails.
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
        use_system_turn=USE_SYSTEM_TURN,
    )

    model_load_start = time.monotonic()

    try:
        engine.load()

        log.info(
            "Base InternVL3.5-8B loaded in %.2f seconds.",
            time.monotonic() - model_load_start,
        )

        adapter_load_start = time.monotonic()

        attach_dora_adapter(
            engine=engine,
            adapter_path=DORA_ADAPTER_PATH,
        )

        log.info(
            "DoRA adapter loaded in %.2f seconds.",
            time.monotonic() - adapter_load_start,
        )

    except Exception:
        # answer.json has already been written with one empty response per qID.
        log.exception("InternVL3.5-8B / DoRA model loading failed.")
        return 0

    # -------------------------------------------------------------------------
    # Process each independent (procedure context, question) pair
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
            log.info("  question=%s", request.question)

            clip = None
            prediction = None

            try:
                video_path = clip_path_for(request)

                # IMPORTANT:
                # - the supplied MP4 is already trimmed to this request context;
                # - explicit HH:MM:SS times in the question are on the ORIGINAL
                #   procedure timeline;
                # - video_utils converts those absolute anchors to local MP4 time
                #   by subtracting request.start_time.
                clip = load_clip_frames(
                    video_path=video_path,
                    question=request.question,
                    absolute_start_time=float(request.start_time),
                    target_fps=TARGET_FPS,
                    max_frames=MAX_FRAMES,
                    local_fps=LOCAL_FPS,
                    local_radius_seconds=LOCAL_RADIUS_SECONDS,
                    num_threads=NUM_DECODE_THREADS,
                )

                frame_labels = make_absolute_frame_labels(
                    request=request,
                    relative_timestamps=clip.timestamps,
                )

                # video_utils is the single source of truth for how the current
                # frames were sampled. Pass its exact description to the prompt.
                prompt = build_prompt(
                    request=request,
                    shared_prompt=shared_prompt,
                    sampling_description=clip.sampling_description,
                )

                log.info(
                    "  temporal_intent=%s, anchors=%s",
                    clip.temporal_intent.value,
                    format_anchor_times(clip.absolute_anchor_times),
                )
                log.info(
                    "  sampling=%s",
                    clip.sampling_description,
                )
                log.info(
                    "  selected_frames=%d, source_fps=%.3f, "
                    "clip_duration=%.3f s",
                    clip.num_frames,
                    clip.source_fps,
                    clip.clip_duration,
                )

                if frame_labels:
                    log.info(
                        "  frame-label range=%s -> %s",
                        frame_labels[0],
                        frame_labels[-1],
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
                    "  frames=%d, patches=%d, model_generation=%.2f s",
                    prediction.num_frames,
                    prediction.num_patches,
                    prediction.inference_seconds,
                )

                if prediction.peak_gpu_memory_gb is not None:
                    log.info(
                        "  peak allocated GPU memory: %.2f GB",
                        prediction.peak_gpu_memory_gb,
                    )

            except Exception:
                latency = time.monotonic() - question_start

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
                # Do not call torch.cuda.empty_cache() after every question.
                # Retaining the CUDA allocator cache is faster for the batch.
                if prediction is not None:
                    del prediction
                if clip is not None:
                    del clip

                # Keep answer.json valid and complete after every question.
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
        "=== PROCEDURE inference complete in %.2f seconds ===",
        total_seconds,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
