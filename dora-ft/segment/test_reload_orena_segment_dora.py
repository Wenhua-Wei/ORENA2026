#!/usr/bin/env python3
"""Reload an ORena InternVL3.5-8B DoRA checkpoint and run one real generation."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

ORENA_ROOT = Path("/SAN/medic/Surgical_LLM_Agent/orena2026")
SEGMENT_ALGORITHM_DIR = ORENA_ROOT / "orena-docker" / "segment-algorithm"
if not SEGMENT_ALGORITHM_DIR.is_dir():
    raise FileNotFoundError(f"Missing SEGMENT algorithm directory: {SEGMENT_ALGORITHM_DIR}")
if str(SEGMENT_ALGORITHM_DIR) not in sys.path:
    sys.path.insert(0, str(SEGMENT_ALGORITHM_DIR))

from model_utils import prepare_frames  # noqa: E402

DEFAULT_MODEL_PATH = SEGMENT_ALGORITHM_DIR / "resources" / "InternVL3_5-8B-Instruct"
DEFAULT_CHECKPOINT_DIR = (
    ORENA_ROOT
    / "segment"
    / "dora-ft"
    / "outputs"
    / "internvl3_5_8b_dora_32frame_test"
    / "final"
)
DEFAULT_SFT_JSONL = (
    ORENA_ROOT
    / "data"
    / "segment"
    / "dora_8b_one32_test"
    / "train_sft.jsonl"
)
INPUT_SIZE = 448
MAX_TILES_PER_FRAME = 1
MAX_FRAMES = 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reload InternVL3.5-8B + saved DoRA + saved mlp1 and run one multimodal generation."
    )
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--sft-jsonl", type=Path, default=DEFAULT_SFT_JSONL)
    p.add_argument("--sample-id", type=str, default=None)
    p.add_argument("--precision", choices=("auto", "bf16", "fp16"), default="auto")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--compare-base",
        action="store_true",
        help="Generate with the untouched base before attaching the checkpoint.",
    )
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def choose_dtype(name: str) -> tuple[torch.dtype, str]:
    if name == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested but unsupported by this CUDA device")
        return torch.bfloat16, "bf16"
    if name == "fp16":
        return torch.float16, "fp16"
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bf16"
    return torch.float16, "fp16"


def load_record(path: Path, sample_id: str | None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"SFT JSONL not found: {path}")
    first = None
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                x = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
            if first is None:
                first = x
            if sample_id is not None and str(x.get("sample_id")) == sample_id:
                return x
    if sample_id is not None:
        raise KeyError(f"sample_id={sample_id!r} not found in {path}")
    if first is None:
        raise RuntimeError(f"Empty SFT JSONL: {path}")
    return first


def validate_record(x: dict[str, Any]) -> None:
    required = ("sample_id", "images", "num_frames", "user_text", "assistant_text")
    missing = [k for k in required if k not in x]
    if missing:
        raise KeyError(f"{x.get('sample_id', '<unknown>')}: missing {missing}")
    n = int(x["num_frames"])
    if not 1 <= n <= MAX_FRAMES:
        raise RuntimeError(f"Invalid num_frames={n}")
    if len(x["images"]) != n:
        raise RuntimeError(f"len(images)={len(x['images'])} != num_frames={n}")
    if str(x["user_text"]).count("<image>") != n:
        raise RuntimeError("<image> placeholder count does not match num_frames")
    if not str(x["assistant_text"]).strip():
        raise RuntimeError("assistant_text is empty")
    for raw in x["images"]:
        p = Path(str(raw))
        if not p.is_file() or p.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing/empty image: {p}")


def load_visual_input(x: dict[str, Any]) -> tuple[torch.Tensor, list[int]]:
    pil_images: list[Image.Image] = []
    try:
        for raw in x["images"]:
            with Image.open(str(raw)) as im:
                pil_images.append(im.convert("RGB"))
        pixel_values, num_patches_list = prepare_frames(
            pil_images,
            input_size=INPUT_SIZE,
            max_tiles_per_frame=MAX_TILES_PER_FRAME,
            use_thumbnail=True,
        )
    finally:
        for im in pil_images:
            try:
                im.close()
            except Exception:
                pass
    n = int(x["num_frames"])
    if num_patches_list != [1] * n:
        raise RuntimeError(f"Expected one tile/frame; got {num_patches_list}")
    if int(pixel_values.shape[0]) != n:
        raise RuntimeError(f"pixel_values tiles={pixel_values.shape[0]} != frames={n}")
    return pixel_values, num_patches_list


def validate_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    path = adapter_dir / "adapter_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing adapter config: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not bool(cfg.get("use_dora", False)):
        raise RuntimeError("adapter_config.json does not have use_dora=true")
    targets = set(cfg.get("target_modules", []))
    required = {"q_proj", "k_proj", "v_proj", "o_proj"}
    if not required.issubset(targets):
        raise RuntimeError(f"DoRA target mismatch: got {sorted(targets)}")
    logging.info("Adapter config PASSED: use_dora=True targets=%s", sorted(targets))
    if cfg.get("base_model_name_or_path"):
        logging.info("Adapter metadata base path: %s", cfg["base_model_name_or_path"])
        logging.info("This test ignores that hint and attaches the adapter to the explicitly loaded local InternVL LLM.")
    return cfg


def load_mlp1(model: Any, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing mlp1 checkpoint: {path}")
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    state = torch.load(path, **kwargs)
    if not isinstance(state, dict):
        raise TypeError("mlp1.pt is not a state_dict")
    incompatible = model.mlp1.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"mlp1 strict-load mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    logging.info("mlp1 reload PASSED: %d tensors loaded strictly", len(state))


def freeze_inference(model: Any) -> None:
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def count_trainable(model: Any) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.inference_mode()
def generate(
    *,
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
) -> dict[str, Any]:
    pixel_values, num_patches_list = load_visual_input(record)
    pixel_values = pixel_values.to(device=device, dtype=dtype, non_blocking=True)
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    answer = model.chat(
        tokenizer,
        pixel_values,
        str(record["user_text"]),
        generation_config,
        num_patches_list=num_patches_list,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    del pixel_values
    return {
        "prediction": str(answer).strip(),
        "generation_seconds": elapsed,
        "peak_gpu_memory_gb": peak_gb,
        "num_patches_list": num_patches_list,
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for InternVL3.5-8B reload testing")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    model_path = args.model_path.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    sft_jsonl = args.sft_jsonl.expanduser().resolve()
    adapter_dir = checkpoint_dir / "dora_adapter"
    mlp_path = checkpoint_dir / "mlp1.pt"
    tokenizer_dir = checkpoint_dir / "tokenizer"
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else checkpoint_dir / "reload_test_result.json"
    )

    if not model_path.is_dir():
        raise FileNotFoundError(f"Base model missing: {model_path}")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint_dir}")
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"DoRA adapter missing: {adapter_dir}")

    device = torch.device("cuda:0")
    dtype, precision = choose_dtype(args.precision)

    logging.info("=== InternVL3.5-8B DoRA reload test ===")
    logging.info("GPU: %s", torch.cuda.get_device_name(0))
    logging.info("Precision: %s", precision)
    logging.info("Base model: %s", model_path)
    logging.info("Checkpoint: %s", checkpoint_dir)

    adapter_cfg = validate_adapter_config(adapter_dir)
    record = load_record(sft_jsonl, args.sample_id)
    validate_record(record)
    logging.info(
        "Selected sample=%s frames=%d reference=%r",
        record["sample_id"],
        int(record["num_frames"]),
        str(record["assistant_text"]).strip(),
    )

    tokenizer_source = tokenizer_dir if tokenizer_dir.is_dir() else model_path
    logging.info("Loading tokenizer from %s", tokenizer_source)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source),
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )

    logging.info("Loading untouched InternVL base model...")
    model = AutoModel.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()

    base_result = None
    if args.compare_base:
        logging.info("Generating with untouched base model...")
        base_result = generate(
            model=model,
            tokenizer=tokenizer,
            record=record,
            device=device,
            dtype=dtype,
            max_new_tokens=args.max_new_tokens,
        )
        logging.info("Base prediction: %r", base_result["prediction"])

    logging.info("Attaching saved DoRA adapter to language_model...")
    model.language_model = PeftModel.from_pretrained(
        model.language_model,
        str(adapter_dir),
        is_trainable=False,
    )
    if not getattr(model.language_model, "peft_config", None):
        raise RuntimeError("Reloaded PEFT model has no peft_config")
    logging.info("DoRA adapter reload PASSED: adapters=%s", sorted(model.language_model.peft_config.keys()))

    logging.info("Loading saved mlp1 projector...")
    load_mlp1(model, mlp_path)

    freeze_inference(model)
    if count_trainable(model) != 0:
        raise RuntimeError("Reloaded inference model still has trainable parameters")
    model.vision_model.eval()
    model.mlp1.eval()
    model.language_model.eval()
    logging.info("Inference freeze PASSED: trainable parameters = 0")

    logging.info("Generating with reloaded DoRA + mlp1...")
    reloaded_result = generate(
        model=model,
        tokenizer=tokenizer,
        record=record,
        device=device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
    )

    logging.info("Reloaded prediction: %r", reloaded_result["prediction"])
    logging.info("Reference answer: %r", str(record["assistant_text"]).strip())
    logging.info(
        "Generation: %.2fs peak_mem=%.2fGB",
        reloaded_result["generation_seconds"],
        reloaded_result["peak_gpu_memory_gb"],
    )

    result = {
        "status": "passed",
        "sample_id": str(record["sample_id"]),
        "num_frames": int(record["num_frames"]),
        "reference_answer": str(record["assistant_text"]).strip(),
        "base_model": str(model_path),
        "checkpoint_dir": str(checkpoint_dir),
        "adapter_config": {
            "use_dora": bool(adapter_cfg.get("use_dora", False)),
            "target_modules": sorted(adapter_cfg.get("target_modules", [])),
            "base_model_name_or_path_metadata": adapter_cfg.get("base_model_name_or_path"),
        },
        "base_generation": base_result,
        "reloaded_generation": reloaded_result,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logging.info("Result written to %s", output_json)
    logging.info("=== RELOAD TEST PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())