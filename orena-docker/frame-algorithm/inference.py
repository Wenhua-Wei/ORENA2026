"""ORena SAVE FOCUS — FRAME Track — InternVL3.5 + DoRA inference.

Docker/platform mode reads a batch of ``focus.Request`` objects from
``/input/request.json`` and writes ``/output/answer.json``.

Direct local mode (``python inference.py``) automatically reads the committed
sample batch from ``test/input/interface_1`` and writes to
``test/output/interface_1``. Paths can also be overridden with
``FOCUS_INPUT_PATH`` and ``FOCUS_OUTPUT_PATH``.

Each FRAME request has one still image at ``frames/<qID>.png``. The supplied
image corresponds to ``request.start_time`` in the original procedure. This
implementation labels the visual input as, for example:

    Image at 00:09:39: <image>

so questions that explicitly mention a timepoint are unambiguously tied to the
single supplied FRAME image.

The InternVL3.5-8B base model, tokenizer, DoRA adapter, canonical foreign-object
class names, and shared prompt are loaded once per batch. One ``focus.Response``
is written for every request.

IMPORTANT:
    This file expects ``model_utils.InternVLInferenceEngine.predict`` to accept
    an optional keyword argument ``image_label`` and to serialize the image as
    ``Image at <image_label>: <image>`` when that argument is provided.
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
from peft import PeftModel
from focus import Request, Response, load_requests, save_items

from answer_utils import normalize_answer
from image_utils import load_frame
from model_utils import InternVLInferenceEngine
from prompt_utils import (
    build_prompt,
    build_shared_prompt,
    load_fo_definitions,
    resolve_fo_class_names,
)

APP_PATH = Path(__file__).resolve().parent


def resolve_io_paths() -> tuple[Path, Path, str]:
    """Resolve input/output paths for Docker or direct local execution."""

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
FRAME_DIR = INPUT_PATH / "frames"

MODEL_PATH = APP_PATH / "resources" / "InternVL3_5-8B-Instruct"

_DEFAULT_DORA_ADAPTER_PATH = (
    APP_PATH
    / "resources"
    / "checkpoint-epoch-4"
    / "dora_adapter"
)

DORA_ADAPTER_PATH = Path(
    os.environ.get(
        "FRAME_DORA_ADAPTER_PATH",
        str(_DEFAULT_DORA_ADAPTER_PATH),
    )
).expanduser().resolve()

USE_FEW_SHOT_EXAMPLES = False
USE_SYSTEM_TURN = False

DEVICE = "cuda:0"
DTYPE = torch.float16

INPUT_SIZE = 448
MAX_TILES_PER_IMAGE = 4
USE_THUMBNAIL = True
MAX_NEW_TOKENS = 64
COLLECT_GPU_DIAGNOSTICS = False

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def frame_path_for(request: Request) -> Path:
    """Return the PNG frame belonging to one request."""

    return FRAME_DIR / f"{request.qID}.png"


def seconds_to_timestamp(seconds: float) -> str:
    """Convert procedure seconds to HH:MM:SS using the floored second."""

    seconds = float(seconds)
    if seconds < 0:
        raise ValueError(
            f"Frame timestamp must be non-negative, received {seconds}."
        )

    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def image_label_for(request: Request) -> str:
    """Return the original-procedure timestamp corresponding to the FRAME image."""

    return seconds_to_timestamp(float(request.start_time))


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


def validate_requests(
    requests: Sequence[Request],
) -> None:
    """Validate basic FRAME request invariants."""

    qids = [str(request.qID).strip() for request in requests]

    if any(not qid for qid in qids):
        raise ValueError("Every request must contain a non-empty qID.")

    if len(qids) != len(set(qids)):
        raise ValueError("request.json contains duplicate qIDs.")

    for request in requests:
        if not request.question.strip():
            raise ValueError(
                f"qID={request.qID} has an empty question."
            )

        if float(request.start_time) < 0:
            raise ValueError(
                f"qID={request.qID} has negative start_time="
                f"{request.start_time}."
            )

        if float(request.end_time) != float(request.start_time):
            log.warning(
                "FRAME qID=%s has start_time=%s and end_time=%s; "
                "using start_time as the supplied image timestamp.",
                request.qID,
                request.start_time,
                request.end_time,
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


def attach_dora_adapter(
    engine: InternVLInferenceEngine,
    adapter_path: Path,
) -> None:
    """Attach the fine-tuned FRAME DoRA adapter to InternVL's language model."""

    if not adapter_path.is_dir():
        raise FileNotFoundError(
            f"DoRA adapter directory does not exist: {adapter_path}"
        )

    adapter_config = adapter_path / "adapter_config.json"

    if not adapter_config.is_file():
        raise FileNotFoundError(
            f"Missing DoRA adapter config: {adapter_config}"
        )

    if engine.model is None:
        raise RuntimeError(
            "InternVL base model must be loaded before attaching DoRA."
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
    """Write structurally valid empty responses when CUDA is unavailable."""

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


def run() -> int:
    """Run FRAME inference for one complete request batch."""

    process_start = time.monotonic()
    output_path = OUTPUT_PATH / "answer.json"

    log.info("=== ORena SAVE FOCUS FRAME inference start ===")
    log.info("Execution mode: %s", EXECUTION_MODE)
    log.info("Input path: %s", INPUT_PATH)
    log.info("Output path: %s", OUTPUT_PATH)
    log.info("Use system turn: %s", USE_SYSTEM_TURN)
    log.info("Base model path: %s", MODEL_PATH)
    log.info("DoRA adapter path: %s", DORA_ADAPTER_PATH)
    log.info(
        "FRAME configuration: input_size=%d, "
        "max_tiles_per_image=%d, thumbnail=%s, max_new_tokens=%d",
        INPUT_SIZE,
        MAX_TILES_PER_IMAGE,
        USE_THUMBNAIL,
        MAX_NEW_TOKENS,
    )
    log.info("PyTorch: %s", torch.__version__)
    log.info("PyTorch CUDA runtime: %s", torch.version.cuda)
    log.info("CUDA available: %s", torch.cuda.is_available())

    if not REQUESTS_PATH.is_file():
        log.error("Missing request file: %s", REQUESTS_PATH)
        return 1

    if not FO_DEFINITIONS_PATH.is_file():
        log.error(
            "Missing foreign-object definitions: %s",
            FO_DEFINITIONS_PATH,
        )
        return 1

    if not FRAME_DIR.is_dir():
        log.error("Missing FRAME image directory: %s", FRAME_DIR)
        return 1

    try:
        requests = list(load_requests(REQUESTS_PATH))
        validate_requests(requests)
    except Exception:
        log.exception("Failed to load or validate request.json.")
        return 1

    if not requests:
        log.error("request.json contains no requests.")
        return 1

    log.info("Loaded %d request(s).", len(requests))
    log.info("Frame directory: %s", FRAME_DIR)

    try:
        fo_definitions = load_fo_definitions(FO_DEFINITIONS_PATH)
        shared_prompt, fo_class_names = build_shared_inference_prompt(
            fo_definitions
        )
    except Exception:
        log.exception(
            "Failed to load FO definitions or build the shared prompt."
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
        log.exception("Failed to write the initial answer.json.")
        return 1

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

    log.info(
        "Total GPU memory: %.2f GB",
        total_vram_gb,
    )

    engine = InternVLInferenceEngine(
        model_path=MODEL_PATH,
        device=DEVICE,
        dtype=DTYPE,
        input_size=INPUT_SIZE,
        max_tiles_per_image=MAX_TILES_PER_IMAGE,
        use_thumbnail=USE_THUMBNAIL,
        max_new_tokens=MAX_NEW_TOKENS,
        use_system_turn=USE_SYSTEM_TURN,
        collect_gpu_diagnostics=COLLECT_GPU_DIAGNOSTICS,
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
        log.exception(
            "InternVL3.5-8B / DoRA model loading failed."
        )
        try:
            engine.unload()
        except Exception:
            pass
        return 0

    try:
        for index, request in enumerate(
            requests,
            start=1,
        ):
            qid = str(request.qID)
            question_start = time.monotonic()
            frame_timestamp = image_label_for(request)

            log.info(
                "[%d/%d] qID=%s, videoID=%s, "
                "frame_time=%s (%.3f s)",
                index,
                len(requests),
                qid,
                request.videoID,
                frame_timestamp,
                float(request.start_time),
            )

            image = None
            prediction = None

            try:
                image_path = frame_path_for(request)
                image = load_frame(image_path)
                native_size = image.size

                prompt = build_prompt(
                    request=request,
                    shared_prompt=shared_prompt,
                )

                prediction = engine.predict(
                    image=image,
                    prompt=prompt,
                    image_label=frame_timestamp,
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
                    "  image_label=%s, native_size=%s, "
                    "images=%d, patches=%d, model_inference=%.2f s",
                    frame_timestamp,
                    native_size,
                    prediction.num_images,
                    prediction.num_patches,
                    prediction.inference_seconds,
                )

                if prediction.peak_gpu_memory_gb is not None:
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
        "=== FRAME inference complete in %.2f seconds ===",
        total_seconds,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(run())