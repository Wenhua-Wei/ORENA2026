"""
Train InternVL3.5-8B on ORena SAVE FOCUS SEGMENT using DoRA.

ORena-specific parts that cannot be
copied literally are retained: the ORena JSONL loader, up-to-32-frame visual
representation, Docker-aligned prompt, InternVL multimodal image-token expansion,
and the longer sequence length needed by InternVL3.5-8B.

Reference-aligned training design
---------------------------------
Model:
    vision_model              frozen
    mlp1 projector            frozen
    language-model base       frozen
    language-model attention  DoRA on q_proj/k_proj/v_proj/o_proj

DoRA / optimizer defaults:
    rank                      8
    alpha                     16
    dropout                   0.1
    AdamW learning rate       2e-5
    weight decay              0.0
    reference batch size      6
    ORena implementation      micro-batch 1 + gradient accumulation 6
    LR reduction              x0.8 after 3 validation epochs without improvement

ORena-specific input:
    simplified train_sft.jsonl / val_all_sft.jsonl produced by
    build_orena_segment_dora_jsonl.py

Visual representation:
    up to 32 ordered cached frames per VQA
    one visual tile per frame
    448 x 448
    ImageNet normalization
    same prepare_frames() implementation as the current Docker model_utils.py

Multimodal tokenization:
    follows InternVL's internvl2_5 SFT serialization:
      - each <image> becomes
        <img> + <IMG_CONTEXT> * model.num_image_token + </img>
      - system and user tokens are masked with label -100
      - only assistant-answer tokens contribute to the SFT loss
    Batch size is fixed to 1 because ORena VQAs have variable frame counts and
    therefore variable multimodal sequence lengths.

Checkpoint contents:
    <checkpoint>/dora_adapter/       PEFT DoRA adapter
    <checkpoint>/tokenizer/          tokenizer snapshot
    <checkpoint>/training_state.json metadata needed for reload/evaluation

The frozen base InternVL3.5-8B model, including its original mlp1 projector, is
not duplicated in each checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoTokenizer

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError as error:
    raise ImportError(
        "PEFT is required for DoRA. Install a PEFT version that supports "
        "`LoraConfig(use_dora=True)`."
    ) from error



ORENA_ROOT = Path("/SAN/medic/Surgical_LLM_Agent/orena2026")

SEGMENT_ALGORITHM_DIR = (
    ORENA_ROOT
    / "orena-docker"
    / "segment-algorithm"
)

if not SEGMENT_ALGORITHM_DIR.is_dir():
    raise FileNotFoundError(
        f"SEGMENT algorithm directory does not exist: {SEGMENT_ALGORITHM_DIR}"
    )

if str(SEGMENT_ALGORITHM_DIR) not in sys.path:
    sys.path.insert(0, str(SEGMENT_ALGORITHM_DIR))

# Reuse the exact 448x448/one-tile visual preprocessing used at inference.
from model_utils import prepare_frames  # noqa: E402


DEFAULT_DATA_ROOT = (
    ORENA_ROOT
    / "data"
    / "segment"
    / "dora_8b_balanced640"
)

DEFAULT_OUTPUT_DIR = (
    ORENA_ROOT
    / "segment"
    / "dora-ft"
    / "outputs"
    / "balanced640-r8-a16-reference2888888"
)

MODEL_NAME = "InternVL3_5-8B-Instruct"

MODEL_PATH_CANDIDATES = (
    SEGMENT_ALGORITHM_DIR / "resources" / MODEL_NAME,
    ORENA_ROOT / "resources" / MODEL_NAME,
    Path.cwd() / "resources" / MODEL_NAME,
)

TRAIN_JSONL = "train_sft.jsonl"
VAL_JSONL = "val_all_sft.jsonl"

INPUT_SIZE = 448
MAX_TILES_PER_FRAME = 1
MAX_FRAMES = 32

USE_SYSTEM_TURN = False

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IGNORE_INDEX = -100

DORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)

LOG_LEVEL = logging.INFO



@dataclass(frozen=True)
class RunConfig:
    model_path: str
    data_root: str
    output_dir: str
    precision: str
    max_seq_length: int
    epochs: int
    gradient_accumulation: int
    dora_rank: int
    dora_alpha: int
    dora_dropout: float
    dora_lr: float
    weight_decay: float
    lr_patience: int
    lr_shrink_factor: float
    seed: int
    use_flash_attn: bool
    use_system_turn: bool
    tiny_smoke: bool
    max_train_samples: int | None
    max_eval_samples: int | None
    max_optimizer_steps: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune InternVL3.5-8B for ORena SEGMENT using DoRA "
            "on language-model q/k/v/o projections, following DoRA_fine_tune.py "
            "as closely as practical."
        )
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Local InternVL3.5-8B model directory. If omitted, first use "
            "$INTERNVL_MODEL_PATH, then try a few standard ORena locations."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Directory containing *_sft.jsonl. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Checkpoint/output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )

    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16"),
        default="bf16",
        help="Training precision. Default bf16 to match the reference script.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=16384,
        help=(
            "Maximum multimodal token length. No silent truncation is used; "
            "a sample exceeding this length raises an error. Default: 16384."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs. Reference script default: 20.",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=6,
        help=(
            "Micro-samples per optimizer step. Physical batch size is 1; "
            "default 6 approximates the reference script's batch_size=6."
        ),
    )

    parser.add_argument("--dora-rank", type=int, default=8)
    parser.add_argument("--dora-alpha", type=int, default=16)
    parser.add_argument("--dora-dropout", type=float, default=0.1)
    parser.add_argument("--dora-lr", type=float, default=2e-5)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay. Reference script default: 0.0.",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=3,
        help=(
            "Reduce LR after this many consecutive validation epochs without "
            "improvement. Reference script default: 3."
        ),
    )
    parser.add_argument(
        "--lr-shrink-factor",
        type=float,
        default=0.8,
        help="Multiply LR by this factor after patience is reached. Default: 0.8.",
    )

    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument(
        "--use-flash-attn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Ask the InternVL remote code to use FlashAttention when available. "
            "If the model snapshot cannot use it, InternVL may fall back."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional deterministic limit on training records.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help=(
            "Optional cap on validation VQAs. Default None evaluates the whole "
            "validation JSONL, matching the reference script's full validation pass."
        ),
    )
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=None,
        help="Optional hard stop after this many optimizer steps.",
    )

    parser.add_argument(
        "--tiny-smoke",
        action="store_true",
        help=(
            "Pipeline test: use the 8 shortest training VQAs, 4 shortest "
            "validation VQAs, gradient accumulation 1, and at most 3 "
            "optimizer steps."
        ),
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip validation-loss computation.",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Allow a non-empty output directory.",
    )

    return parser.parse_args()



def configure_logging() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_model_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Model path does not exist: {path}")
        return path

    env_path = os.environ.get("INTERNVL_MODEL_PATH")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(
                f"$INTERNVL_MODEL_PATH does not exist: {path}"
            )
        return path

    existing = [
        candidate.expanduser().resolve()
        for candidate in MODEL_PATH_CANDIDATES
        if candidate.is_dir()
    ]

    if len(existing) == 1:
        return existing[0]

    if len(existing) > 1:
        logging.warning(
            "Multiple 8B model directories found; using %s",
            existing[0],
        )
        return existing[0]

    candidates_text = "\n  - ".join(
        str(path)
        for path in MODEL_PATH_CANDIDATES
    )
    raise FileNotFoundError(
        "Could not locate InternVL3.5-8B automatically. "
        "Run with --model-path /path/to/InternVL3_5-8B-Instruct "
        "or set INTERNVL_MODEL_PATH. Tried:\n  - "
        + candidates_text
    )


def choose_dtype(precision: str) -> tuple[torch.dtype, str]:
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested but this CUDA device does not support it.")
        return torch.bfloat16, "bf16"

    if precision == "fp16":
        return torch.float16, "fp16"

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bf16"

    return torch.float16, "fp16"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL does not exist: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} line {line_number}: {error}"
                ) from error

            if not isinstance(item, dict):
                raise TypeError(
                    f"{path} line {line_number} is not a JSON object."
                )

            records.append(item)

    return records


def validate_sft_record(record: dict[str, Any]) -> None:
    required = (
        "sample_id",
        "images",
        "num_frames",
        "user_text",
        "assistant_text",
    )

    missing = [
        key
        for key in required
        if key not in record
    ]

    if missing:
        raise KeyError(
            f"{record.get('sample_id', '<unknown>')}: missing keys {missing}"
        )

    sample_id = str(record["sample_id"])
    images = record["images"]

    if not isinstance(images, list) or not images:
        raise ValueError(
            f"{sample_id}: images must be a non-empty list."
        )

    num_frames = int(record["num_frames"])

    if len(images) != num_frames:
        raise RuntimeError(
            f"{sample_id}: len(images)={len(images)} != num_frames={num_frames}"
        )

    if not (1 <= num_frames <= MAX_FRAMES):
        raise RuntimeError(
            f"{sample_id}: invalid num_frames={num_frames}"
        )

    user_text = str(record["user_text"])
    assistant_text = str(record["assistant_text"]).strip()

    if user_text.count("<image>") != num_frames:
        raise RuntimeError(
            f"{sample_id}: <image> count={user_text.count('<image>')} "
            f"!= num_frames={num_frames}"
        )

    if not assistant_text:
        raise RuntimeError(
            f"{sample_id}: assistant_text is empty."
        )

    for image_path in images:
        path = Path(str(image_path))
        if not path.is_file():
            raise FileNotFoundError(
                f"{sample_id}: missing image: {path}"
            )
        if path.stat().st_size <= 0:
            raise RuntimeError(
                f"{sample_id}: empty image: {path}"
            )


def select_records(
    records: Sequence[dict[str, Any]],
    *,
    max_samples: int | None,
    prefer_short: bool,
    seed: int,
) -> list[dict[str, Any]]:
    selected = list(records)

    if prefer_short:
        selected.sort(
            key=lambda item: (
                int(item["num_frames"]),
                str(item["sample_id"]),
            )
        )
    else:
        rng = random.Random(seed)
        rng.shuffle(selected)

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive when provided.")
        selected = selected[:max_samples]

    return selected




def verify_image_special_tokens(tokenizer: Any) -> dict[str, int]:
    result: dict[str, int] = {}

    for token in (
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
    ):
        token_id = tokenizer.convert_tokens_to_ids(token)

        if token_id is None:
            raise RuntimeError(
                f"Tokenizer has no ID for required image token {token!r}."
            )

        if (
            tokenizer.unk_token_id is not None
            and token_id == tokenizer.unk_token_id
        ):
            raise RuntimeError(
                f"Required image token {token!r} maps to unk_token_id."
            )

        encoded = tokenizer(
            token,
            add_special_tokens=False,
        ).input_ids

        if len(encoded) != 1:
            raise RuntimeError(
                f"Required image token {token!r} is not a single tokenizer "
                f"token: encoded IDs={encoded}"
            )

        result[token] = int(token_id)

    return result


def expand_image_placeholders(
    user_text: str,
    *,
    num_images: int,
    num_image_token: int,
) -> str:
    if user_text.count("<image>") != num_images:
        raise RuntimeError(
            f"user_text contains {user_text.count('<image>')} <image> "
            f"placeholders, expected {num_images}."
        )

    expanded = user_text

    for _ in range(num_images):
        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * num_image_token
            + IMG_END_TOKEN
        )

        expanded = expanded.replace(
            "<image>",
            image_tokens,
            1,
        )

    if "<image>" in expanded:
        raise RuntimeError(
            "Unexpanded <image> placeholder remains after visual-token expansion."
        )

    return expanded


def tokenize_internvl2_5_sft(
    *,
    tokenizer: Any,
    system_message: str | None,
    user_text: str,
    assistant_text: str,
    num_images: int,
    num_image_token: int,
    max_seq_length: int,
) -> dict[str, torch.Tensor]:
    """
    Compact implementation of the official InternVL preprocess_internvl2_5 logic.

    System and human/user tokens are masked with -100.
    For the assistant turn, the role prefix and final newline are masked while
    the answer and <|im_end|> token remain supervised.
    """

    expanded_user_text = expand_image_placeholders(
        user_text,
        num_images=num_images,
        num_image_token=num_image_token,
    )

    batches: list[str] = []
    roles: list[str] = []

    if system_message is not None:
        batches.append(
            f"<|im_start|>system\n{system_message}<|im_end|>\n"
        )
        roles.append("system")

    batches.append(
        f"<|im_start|>user\n{expanded_user_text}<|im_end|>\n"
    )
    roles.append("human")

    batches.append(
        f"<|im_start|>assistant\n{assistant_text}<|im_end|>\n"
    )
    roles.append("gpt")

    add_bos_token = bool(
        getattr(
            tokenizer,
            "add_bos_token",
            False,
        )
    )

    if add_bos_token:
        if tokenizer.bos_token is None:
            raise RuntimeError(
                "tokenizer.add_bos_token=True but bos_token is None."
            )
        batches[0] = tokenizer.bos_token + batches[0]

    tokenized: list[torch.Tensor] = []

    for batch in batches:
        ids = tokenizer(
            batch,
            return_tensors="pt",
            padding=False,
            truncation=False,
        ).input_ids[0]

        if add_bos_token:
            ids = ids[1:]

        tokenized.append(ids)

    assistant_prefix_ids = tokenizer(
        "<|im_start|>assistant\n",
        return_tensors="pt",
        padding=False,
        truncation=False,
    ).input_ids[0]

    ignore_prefix_len = (
        int(assistant_prefix_ids.shape[0]) - 1
        if add_bos_token
        else int(assistant_prefix_ids.shape[0])
    )

    input_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []

    for role, ids in zip(
        roles,
        tokenized,
        strict=True,
    ):
        input_parts.append(ids)

        if role in {"system", "human"}:
            labels = torch.full_like(
                ids,
                IGNORE_INDEX,
            )
        elif role == "gpt":
            labels = ids.clone()

            labels[:ignore_prefix_len] = (
                IGNORE_INDEX
            )

            # Match official InternVL preprocessing: ignore final newline token.
            labels[-1:] = IGNORE_INDEX
        else:
            raise RuntimeError(
                f"Unexpected conversation role: {role}"
            )

        label_parts.append(labels)

    input_ids = torch.cat(
        input_parts,
        dim=0,
    )

    labels = torch.cat(
        label_parts,
        dim=0,
    )

    seq_length = int(
        input_ids.shape[0]
    )

    if seq_length > max_seq_length:
        raise RuntimeError(
            "Multimodal sequence is too long and will NOT be silently "
            f"truncated: sequence_length={seq_length}, "
            f"max_seq_length={max_seq_length}, num_images={num_images}, "
            f"num_image_token={num_image_token}. Increase --max-seq-length "
            "or change the visual representation."
        )

    if not bool(
        (labels != IGNORE_INDEX).any()
    ):
        raise RuntimeError(
            "All labels are masked; no assistant tokens would contribute loss."
        )

    attention_mask = torch.ones_like(
        input_ids,
        dtype=torch.long,
    )

    position_ids = torch.arange(
        seq_length,
        dtype=torch.long,
    )

    return {
        "input_ids": input_ids.unsqueeze(0),
        "labels": labels.unsqueeze(0),
        "attention_mask": attention_mask.unsqueeze(0),
        "position_ids": position_ids.unsqueeze(0),
    }



class OrenaSFTDataset:
    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        tokenizer: Any,
        system_message: str | None,
        num_image_token: int,
        max_seq_length: int,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.system_message = system_message
        self.num_image_token = int(
            num_image_token
        )
        self.max_seq_length = int(
            max_seq_length
        )

        for record in self.records:
            validate_sft_record(record)

    def __len__(self) -> int:
        return len(self.records)

    def prepare(
        self,
        index: int,
    ) -> dict[str, Any]:
        record = self.records[index]

        pil_images: list[Image.Image] = []

        try:
            for image_path in record["images"]:
                with Image.open(
                    str(image_path)
                ) as image:
                    pil_images.append(
                        image.convert("RGB")
                    )

            pixel_values, num_patches_list = (
                prepare_frames(
                    pil_images,
                    input_size=INPUT_SIZE,
                    max_tiles_per_frame=(
                        MAX_TILES_PER_FRAME
                    ),
                    use_thumbnail=True,
                )
            )
        finally:
            for image in pil_images:
                try:
                    image.close()
                except Exception:
                    pass

        num_frames = int(
            record["num_frames"]
        )

        expected_patches = [
            1
        ] * num_frames

        if num_patches_list != expected_patches:
            raise RuntimeError(
                f"{record['sample_id']}: expected one visual tile per frame, "
                f"got num_patches_list={num_patches_list}"
            )

        if int(pixel_values.shape[0]) != num_frames:
            raise RuntimeError(
                f"{record['sample_id']}: pixel_values has "
                f"{pixel_values.shape[0]} tiles, expected {num_frames}."
            )

        text_tensors = tokenize_internvl2_5_sft(
            tokenizer=self.tokenizer,
            system_message=self.system_message,
            user_text=str(
                record["user_text"]
            ),
            assistant_text=str(
                record["assistant_text"]
            ).strip(),
            num_images=num_frames,
            num_image_token=(
                self.num_image_token
            ),
            max_seq_length=(
                self.max_seq_length
            ),
        )

        # Use [num_patches, 1] so InternVL's `image_flags.squeeze(-1)` always
        # produces a 1-D vector, including the one-frame edge case.
        image_flags = torch.ones(
            (
                int(
                    pixel_values.shape[0]
                ),
                1,
            ),
            dtype=torch.long,
        )

        return {
            **text_tensors,
            "pixel_values": pixel_values,
            "image_flags": image_flags,
            "sample_id": str(
                record["sample_id"]
            ),
            "num_frames": num_frames,
            "seq_length": int(
                text_tensors[
                    "input_ids"
                ].shape[1]
            ),
        }



def load_model_and_tokenizer(
    *,
    model_path: Path,
    dtype: torch.dtype,
    use_flash_attn: bool,
    max_seq_length: int,
) -> tuple[Any, Any]:
    logging.info(
        "Loading tokenizer from %s",
        model_path,
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
            add_eos_token=False,
        )
    )

    tokenizer.model_max_length = (
        max_seq_length
    )

    logging.info(
        "Loading InternVL model in %s...",
        dtype,
    )

    model = AutoModel.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=use_flash_attn,
        trust_remote_code=True,
        local_files_only=True,
        device_map="cuda",
    )

    if not hasattr(
        model,
        "vision_model",
    ):
        raise RuntimeError(
            "Loaded model does not expose vision_model."
        )

    if not hasattr(
        model,
        "language_model",
    ):
        raise RuntimeError(
            "Loaded model does not expose language_model."
        )

    if not hasattr(
        model,
        "mlp1",
    ):
        raise RuntimeError(
            "Loaded model does not expose mlp1 projector."
        )

    if not hasattr(
        model,
        "num_image_token",
    ):
        raise RuntimeError(
            "Loaded model does not expose num_image_token."
        )

    template_name = str(
        getattr(
            model,
            "template",
            "",
        )
    )

    if template_name != "internvl2_5":
        raise RuntimeError(
            "This training script intentionally implements the official "
            "preprocess_internvl2_5 serialization, but the loaded model reports "
            f"template={template_name!r}. Refusing to silently use the wrong "
            "conversation template."
        )

    special_ids = (
        verify_image_special_tokens(
            tokenizer
        )
    )

    model.img_context_token_id = (
        special_ids[
            IMG_CONTEXT_TOKEN
        ]
    )

    logging.info(
        "Model template: %s",
        template_name,
    )
    logging.info(
        "num_image_token per one-tile frame: %d",
        int(
            model.num_image_token
        ),
    )
    logging.info(
        "Image special token IDs: %s",
        special_ids,
    )

    return model, tokenizer


def freeze_everything(
    model: nn.Module,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def detect_target_modules(
    language_model: nn.Module,
) -> list[str]:
    target_suffixes = tuple(
        f".{name}"
        for name in DORA_TARGET_MODULES
    )

    matched = [
        name
        for name, module
        in language_model.named_modules()
        if (
            name.endswith(
                DORA_TARGET_MODULES
            )
            or name.endswith(
                target_suffixes
            )
        )
        and isinstance(
            module,
            nn.Linear,
        )
    ]

    return matched


def configure_dora(
    *,
    model: Any,
    rank: int,
    alpha: int,
    dropout: float,
    gradient_checkpointing: bool,
) -> None:
    if "use_dora" not in inspect.signature(
        LoraConfig
    ).parameters:
        raise RuntimeError(
            "Installed PEFT does not expose LoraConfig(use_dora=...). "
            "Upgrade PEFT before training."
        )

    matched_targets = (
        detect_target_modules(
            model.language_model
        )
    )

    if not matched_targets:
        raise RuntimeError(
            "Could not find q_proj/k_proj/v_proj/o_proj Linear modules "
            "inside the language model."
        )

    logging.info(
        "Detected %d q/k/v/o projection modules for DoRA.",
        len(matched_targets),
    )

    # freeze the complete InternVL model first, then
    # attach DoRA only to the language model. vision_model and mlp1 stay frozen.
    freeze_everything(model)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(
            DORA_TARGET_MODULES
        ),
        bias="none",
        task_type=(
            TaskType.CAUSAL_LM
        ),
        use_dora=True,
    )

    model.language_model = (
        get_peft_model(
            model.language_model,
            lora_config,
        )
    )

    if hasattr(
        model.language_model,
        "enable_input_require_grads",
    ):
        model.language_model.enable_input_require_grads()

    for parameter in model.mlp1.parameters():
        parameter.requires_grad = False

    for parameter in model.vision_model.parameters():
        parameter.requires_grad = False

    if hasattr(
        model.language_model,
        "config",
    ):
        model.language_model.config.use_cache = False

    if hasattr(
        model.config,
        "llm_config",
    ):
        model.config.llm_config.use_cache = False

    if gradient_checkpointing:
        try:
            model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False,
                }
            )
        except TypeError:
            model.language_model.gradient_checkpointing_enable()
        except AttributeError:
            logging.warning(
                "language_model does not expose gradient_checkpointing_enable()."
            )

    model.language_model.print_trainable_parameters()


def count_parameters(
    parameters: Sequence[nn.Parameter],
) -> int:
    return sum(
        parameter.numel()
        for parameter in parameters
    )


def collect_optimizer_parameters(
    model: Any,
) -> list[nn.Parameter]:
    dora_params = [
        parameter
        for parameter
        in model.language_model.parameters()
        if parameter.requires_grad
    ]

    if not dora_params:
        raise RuntimeError(
            "No trainable DoRA language-model parameters found."
        )

    return dora_params


def verify_trainable_scope(
    model: Any,
) -> None:
    trainable_names = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    unexpected = [
        name
        for name in trainable_names
        if (
            "lora_" not in name
            and "magnitude" not in name.lower()
        )
    ]

    if unexpected:
        raise RuntimeError(
            "Unexpected trainable parameters outside DoRA:\n  - "
            + "\n  - ".join(
                unexpected[:100]
            )
        )

    vision_trainable = [
        name
        for name, parameter
        in model.vision_model.named_parameters()
        if parameter.requires_grad
    ]

    if vision_trainable:
        raise RuntimeError(
            "Vision model is not fully frozen:\n  - "
            + "\n  - ".join(
                vision_trainable[:50]
            )
        )

    mlp_trainable = [
        name
        for name, parameter
        in model.mlp1.named_parameters()
        if parameter.requires_grad
    ]

    if mlp_trainable:
        raise RuntimeError(
            "mlp1 is not fully frozen:\n  - "
            + "\n  - ".join(
                mlp_trainable[:50]
            )
        )

    logging.info(
        "Trainable scope verified: DoRA only; vision_model and mlp1 are frozen."
    )


# =============================================================================
# OPTIMIZER / REFERENCE-STYLE LR REDUCTION
# =============================================================================

def build_optimizer(
    *,
    model: Any,
    dora_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    dora_params = collect_optimizer_parameters(
        model
    )

    logging.info(
        "Trainable DoRA parameters: %s",
        f"{count_parameters(dora_params):,}",
    )

    optimizer = torch.optim.AdamW(
        dora_params,
        lr=dora_lr,
        weight_decay=weight_decay,
    )

    return optimizer


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    shrink_factor: float,
) -> None:
    if not (0.0 < shrink_factor < 1.0):
        raise ValueError(
            f"lr_shrink_factor must be in (0, 1), got {shrink_factor}."
        )

    old_lrs = [
        float(group["lr"])
        for group in optimizer.param_groups
    ]

    for group in optimizer.param_groups:
        group["lr"] = float(
            group["lr"]
        ) * shrink_factor

    new_lrs = [
        float(group["lr"])
        for group in optimizer.param_groups
    ]

    logging.info(
        "DECAYING learning rate by factor %.3f: %s -> %s",
        shrink_factor,
        old_lrs,
        new_lrs,
    )



def move_batch_to_device(
    batch: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    return {
        "input_ids": batch[
            "input_ids"
        ].to(
            device=device,
            non_blocking=True,
        ),
        "labels": batch[
            "labels"
        ].to(
            device=device,
            non_blocking=True,
        ),
        "attention_mask": batch[
            "attention_mask"
        ].to(
            device=device,
            non_blocking=True,
        ),
        "position_ids": batch[
            "position_ids"
        ].to(
            device=device,
            non_blocking=True,
        ),
        "pixel_values": batch[
            "pixel_values"
        ].to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        ),
        "image_flags": batch[
            "image_flags"
        ].to(
            device=device,
            non_blocking=True,
        ),
    }


def forward_loss(
    *,
    model: Any,
    batch: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    with torch.autocast(
        device_type="cuda",
        dtype=dtype,
    ):
        outputs = model(
            pixel_values=batch[
                "pixel_values"
            ],
            input_ids=batch[
                "input_ids"
            ],
            attention_mask=batch[
                "attention_mask"
            ],
            position_ids=batch[
                "position_ids"
            ],
            image_flags=batch[
                "image_flags"
            ],
            labels=batch[
                "labels"
            ],
            use_cache=False,
            return_dict=True,
        )

        loss = outputs.loss

    if loss is None:
        raise RuntimeError(
            "Model forward returned loss=None."
        )

    if not bool(
        torch.isfinite(loss)
    ):
        raise FloatingPointError(
            f"Non-finite loss: {loss.detach().float().item()}"
        )

    return loss


def verify_first_backward(
    model: Any,
) -> None:
    dora_gradients = [
        parameter.grad
        for parameter
        in model.language_model.parameters()
        if parameter.requires_grad
        and parameter.grad is not None
    ]

    if not dora_gradients:
        raise RuntimeError(
            "First backward produced no gradient for DoRA parameters."
        )

    if not all(
        bool(
            torch.isfinite(
                gradient
            ).all()
        )
        for gradient in dora_gradients
    ):
        raise FloatingPointError(
            "Non-finite gradient detected in DoRA parameters."
        )

    vision_grads = [
        name
        for name, parameter
        in model.vision_model.named_parameters()
        if parameter.grad is not None
    ]

    if vision_grads:
        raise RuntimeError(
            "Frozen vision model unexpectedly has gradients:\n  - "
            + "\n  - ".join(
                vision_grads[:50]
            )
        )

    mlp_grads = [
        name
        for name, parameter
        in model.mlp1.named_parameters()
        if parameter.grad is not None
    ]

    if mlp_grads:
        raise RuntimeError(
            "Frozen mlp1 unexpectedly has gradients:\n  - "
            + "\n  - ".join(
                mlp_grads[:50]
            )
        )

    logging.info(
        "First backward check PASSED: DoRA has finite gradients; "
        "vision_model and mlp1 remain frozen."
    )



@torch.no_grad()
def evaluate_loss(
    *,
    model: Any,
    dataset: OrenaSFTDataset,
    indices: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> float | None:
    if not indices:
        return None

    model.eval()
    model.vision_model.eval()

    losses: list[float] = []

    for number, index in enumerate(
        indices,
        start=1,
    ):
        raw_batch = dataset.prepare(
            index
        )

        device_batch = (
            move_batch_to_device(
                raw_batch,
                device=device,
                dtype=dtype,
            )
        )

        loss = forward_loss(
            model=model,
            batch=device_batch,
            dtype=dtype,
        )

        losses.append(
            float(
                loss.detach()
                .float()
                .cpu()
                .item()
            )
        )

        del loss
        del device_batch
        del raw_batch

        logging.info(
            "  eval %d/%d",
            number,
            len(indices),
        )

    mean_loss = (
        sum(losses)
        / len(losses)
    )

    return mean_loss




def save_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    checkpoint_name: str,
    run_config: RunConfig,
    global_step: int,
    epoch: int,
    train_loss: float | None,
    val_loss: float | None,
) -> Path:
    checkpoint_dir = (
        output_dir
        / checkpoint_name
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    adapter_dir = (
        checkpoint_dir
        / "dora_adapter"
    )

    model.language_model.save_pretrained(
        str(adapter_dir),
        safe_serialization=True,
    )


    tokenizer.save_pretrained(
        str(
            checkpoint_dir
            / "tokenizer"
        )
    )

    try:
        model.config.save_pretrained(
            str(
                checkpoint_dir
                / "internvl_config"
            )
        )
    except Exception as error:
        logging.warning(
            "Could not save InternVL config snapshot: %s",
            error,
        )

    state = {
        "global_step": global_step,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "run_config": asdict(
            run_config
        ),
        "model_template": str(
            getattr(
                model,
                "template",
                "",
            )
        ),
        "num_image_token": int(
            model.num_image_token
        ),
        "dora_target_modules": list(
            DORA_TARGET_MODULES
        ),
    }

    with (
        checkpoint_dir
        / "training_state.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            state,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    logging.info(
        "Saved checkpoint: %s",
        checkpoint_dir,
    )

    return checkpoint_dir




def train(
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: OrenaSFTDataset,
    val_dataset: OrenaSFTDataset | None,
    args: argparse.Namespace,
    run_config: RunConfig,
    device: torch.device,
    dtype: torch.dtype,
    gradient_accumulation: int,
    max_optimizer_steps: int | None,
    max_eval_samples: int | None,
) -> None:
    optimizer = build_optimizer(
        model=model,
        dora_lr=args.dora_lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.ceil(
        len(train_dataset)
        / gradient_accumulation
    )

    planned_steps = (
        steps_per_epoch
        * args.epochs
    )

    if max_optimizer_steps is not None:
        planned_steps = min(
            planned_steps,
            max_optimizer_steps,
        )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            dtype
            == torch.float16
        ),
    )

    global_step = 0
    first_backward_checked = False
    stopped_early = False

    # best-model selection by validation loss and reduce LR by a fixed factor after a patience window without improvement.
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_checkpoint: Path | None = None

    mean_train_loss: float | None = None
    val_loss: float | None = None
    epoch = 0


    # -------------------------------------------------------------------------
    # Pre-training validation: epoch 0
    # -------------------------------------------------------------------------
    baseline_val_loss: float | None = None

    if (
        not args.skip_eval
        and val_dataset is not None
        and len(val_dataset) > 0
    ):
        baseline_val_indices = list(
            range(len(val_dataset))
        )

        if max_eval_samples is not None:
            baseline_val_indices = baseline_val_indices[
                :max_eval_samples
            ]

        logging.info(
            "Evaluating pre-training CE loss on %d validation sample(s)...",
            len(baseline_val_indices),
        )

        baseline_val_loss = evaluate_loss(
            model=model,
            dataset=val_dataset,
            indices=baseline_val_indices,
            device=device,
            dtype=dtype,
        )

        logging.info(
            "epoch=0 train_loss=None validation_loss=%.6f lr=%.3e",
            baseline_val_loss,
            float(
                optimizer.param_groups[0]["lr"]
            ),
        )
    else:
        logging.info(
            "Pre-training validation skipped."
        )


    for epoch_index in range(
        args.epochs
    ):
        epoch = epoch_index + 1

        model.train()
        model.vision_model.eval()
        model.mlp1.eval()

        epoch_indices = list(
            range(
                len(
                    train_dataset
                )
            )
        )

        rng = random.Random(
            args.seed
            + epoch_index
        )

        if not args.tiny_smoke:
            rng.shuffle(
                epoch_indices
            )

        epoch_losses: list[float] = []

        for group_start in range(
            0,
            len(epoch_indices),
            gradient_accumulation,
        ):
            if (
                max_optimizer_steps
                is not None
                and global_step
                >= max_optimizer_steps
            ):
                stopped_early = True
                break

            group_indices = (
                epoch_indices[
                    group_start:
                    group_start
                    + gradient_accumulation
                ]
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(
                    device
                )

            group_losses: list[float] = []
            group_started = time.monotonic()

            for micro_number, sample_index in enumerate(
                group_indices,
                start=1,
            ):
                raw_batch = (
                    train_dataset.prepare(
                        sample_index
                    )
                )

                sample_id = (
                    raw_batch[
                        "sample_id"
                    ]
                )

                logging.info(
                    "epoch=%d step=%d micro=%d/%d sample=%s "
                    "frames=%d seq=%d",
                    epoch,
                    global_step + 1,
                    micro_number,
                    len(group_indices),
                    sample_id,
                    raw_batch[
                        "num_frames"
                    ],
                    raw_batch[
                        "seq_length"
                    ],
                )

                try:
                    device_batch = (
                        move_batch_to_device(
                            raw_batch,
                            device=device,
                            dtype=dtype,
                        )
                    )

                    loss = forward_loss(
                        model=model,
                        batch=device_batch,
                        dtype=dtype,
                    )

                    raw_loss_value = float(
                        loss.detach()
                        .float()
                        .cpu()
                        .item()
                    )

                    group_losses.append(
                        raw_loss_value
                    )
                    epoch_losses.append(
                        raw_loss_value
                    )

                    # Physical batch size is 1; averaging six micro-sample losses
                    # approximates the reference script's batch_size=6 update.
                    scaled_loss = (
                        loss
                        / len(
                            group_indices
                        )
                    )

                    scaler.scale(
                        scaled_loss
                    ).backward()

                except torch.OutOfMemoryError as error:
                    logging.error(
                        "CUDA OOM on sample=%s frames=%d seq=%d. "
                        "Try --tiny-smoke first, ensure FlashAttention is "
                        "available, or reduce visual/token length.",
                        sample_id,
                        raw_batch[
                            "num_frames"
                        ],
                        raw_batch[
                            "seq_length"
                        ],
                    )
                    raise error

                finally:
                    if "scaled_loss" in locals():
                        del scaled_loss
                    if "loss" in locals():
                        del loss
                    if "device_batch" in locals():
                        del device_batch
                    del raw_batch

            scaler.unscale_(
                optimizer
            )

            if not first_backward_checked:
                verify_first_backward(
                    model
                )
                first_backward_checked = True

            scaler.step(
                optimizer
            )
            scaler.update()

            global_step += 1

            peak_memory_gb = (
                torch.cuda.max_memory_allocated(
                    device
                )
                / (1024**3)
            )

            group_loss = (
                sum(group_losses)
                / len(group_losses)
            )

            elapsed = (
                time.monotonic()
                - group_started
            )

            current_lr = float(
                optimizer.param_groups[
                    0
                ]["lr"]
            )

            logging.info(
                "optimizer_step=%d/%d epoch=%d "
                "loss=%.6f lr=%.3e "
                "peak_mem=%.2fGB step_time=%.1fs",
                global_step,
                planned_steps,
                epoch,
                group_loss,
                current_lr,
                peak_memory_gb,
                elapsed,
            )

        mean_train_loss = (
            sum(epoch_losses)
            / len(epoch_losses)
            if epoch_losses
            else None
        )

        val_loss = None

        if (
            not args.skip_eval
            and val_dataset is not None
            and len(val_dataset) > 0
        ):
            val_indices = list(
                range(
                    len(
                        val_dataset
                    )
                )
            )

            if max_eval_samples is not None:
                val_indices = (
                    val_indices[
                        :max_eval_samples
                    ]
                )

            logging.info(
                "Evaluating CE loss on %d validation sample(s)...",
                len(val_indices),
            )

            val_loss = evaluate_loss(
                model=model,
                dataset=val_dataset,
                indices=val_indices,
                device=device,
                dtype=dtype,
            )

            logging.info(
                "epoch=%d train_loss=%s validation_loss=%.6f lr=%.3e",
                epoch,
                (
                    "None"
                    if mean_train_loss is None
                    else f"{mean_train_loss:.6f}"
                ),
                val_loss,
                float(
                    optimizer.param_groups[
                        0
                    ]["lr"]
                ),
            )

            improved = (
                val_loss
                < best_val_loss
            )

            if improved:
                best_val_loss = val_loss
                epochs_no_improve = 0

                best_checkpoint = save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=Path(
                        run_config.output_dir
                    ),
                    checkpoint_name="best",
                    run_config=run_config,
                    global_step=global_step,
                    epoch=epoch,
                    train_loss=(
                        mean_train_loss
                    ),
                    val_loss=val_loss,
                )

                logging.info(
                    "Best model updated at epoch %d: validation_loss=%.6f",
                    epoch,
                    best_val_loss,
                )
            else:
                epochs_no_improve += 1

                logging.info(
                    "Validation did not improve: %d/%d epoch(s) without improvement.",
                    epochs_no_improve,
                    args.lr_patience,
                )

                if (
                    epochs_no_improve
                    >= args.lr_patience
                ):
                    adjust_learning_rate(
                        optimizer,
                        args.lr_shrink_factor,
                    )
                    epochs_no_improve = 0

        else:
            logging.info(
                "Validation skipped for epoch %d.",
                epoch,
            )


        if not stopped_early:
            epoch_checkpoint = save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=Path(
                    run_config.output_dir
                ),
                checkpoint_name=(
                    f"checkpoint-epoch-{epoch}"
                ),
                run_config=run_config,
                global_step=global_step,
                epoch=epoch,
                train_loss=mean_train_loss,
                val_loss=val_loss,
            )

            logging.info(
                "Saved epoch %d checkpoint: %s",
                epoch,
                epoch_checkpoint,
            )
        else:
            logging.info(
                "Epoch %d stopped before completion; "
                "not saving it as a completed epoch checkpoint.",
                epoch,
            )


        if stopped_early:
            break


    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=Path(
            run_config.output_dir
        ),
        checkpoint_name="final",
        run_config=run_config,
        global_step=global_step,
        epoch=epoch,
        train_loss=(
            mean_train_loss
        ),
        val_loss=val_loss,
    )

    if best_checkpoint is not None:
        logging.info(
            "Best checkpoint: %s (validation_loss=%.6f)",
            best_checkpoint,
            best_val_loss,
        )
    elif not args.skip_eval:
        logging.warning(
            "No best checkpoint was created. Check validation configuration."
        )

    logging.info(
        "Training finished at optimizer step %d.",
        global_step,
    )




def main() -> int:
    args = parse_args()
    configure_logging()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for InternVL3.5-8B DoRA training."
        )

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be positive."
        )

    if args.gradient_accumulation <= 0:
        raise ValueError(
            "--gradient-accumulation must be positive."
        )

    if args.lr_patience <= 0:
        raise ValueError(
            "--lr-patience must be positive."
        )

    if not (0.0 < args.lr_shrink_factor < 1.0):
        raise ValueError(
            "--lr-shrink-factor must be in (0, 1)."
        )

    if args.max_seq_length <= 0:
        raise ValueError(
            "--max-seq-length must be positive."
        )

    set_seed(
        args.seed
    )

    torch.set_float32_matmul_precision(
        "high"
    )

    device = torch.device(
        "cuda:0"
    )

    model_path = resolve_model_path(
        args.model_path
    )

    data_root = (
        args.data_root
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Data root does not exist: {data_root}"
        )

    if (
        output_dir.exists()
        and any(
            output_dir.iterdir()
        )
        and not args.overwrite_output_dir
    ):
        raise FileExistsError(
            f"Output directory is non-empty: {output_dir}\n"
            "Use --overwrite-output-dir or choose a different --output-dir."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dtype, resolved_precision = (
        choose_dtype(
            args.precision
        )
    )

    gradient_accumulation = (
        args.gradient_accumulation
    )

    max_train_samples = (
        args.max_train_samples
    )

    max_eval_samples = (
        args.max_eval_samples
    )

    max_optimizer_steps = (
        args.max_optimizer_steps
    )

    if args.tiny_smoke:
        gradient_accumulation = 1

        if max_train_samples is None:
            max_train_samples = 8

        if (
            max_eval_samples is None
            or max_eval_samples > 4
        ):
            max_eval_samples = 4

        if (
            max_optimizer_steps is None
            or max_optimizer_steps > 3
        ):
            max_optimizer_steps = 3

    logging.info(
        "=== ORena InternVL3.5-8B DoRA training ==="
    )
    logging.info(
        "GPU: %s",
        torch.cuda.get_device_name(
            0
        ),
    )
    logging.info(
        "GPU memory: %.2f GB",
        torch.cuda.get_device_properties(
            0
        ).total_memory
        / (1024**3),
    )
    logging.info(
        "PyTorch: %s; CUDA runtime: %s",
        torch.__version__,
        torch.version.cuda,
    )

    try:
        peft_version = (
            package_version(
                "peft"
            )
        )
    except PackageNotFoundError:
        peft_version = "unknown"

    logging.info(
        "PEFT: %s",
        peft_version,
    )
    logging.info(
        "Model path: %s",
        model_path,
    )
    logging.info(
        "Data root: %s",
        data_root,
    )
    logging.info(
        "Output dir: %s",
        output_dir,
    )
    logging.info(
        "Precision: %s",
        resolved_precision,
    )
    logging.info(
        "tiny_smoke=%s; grad_accumulation=%d; "
        "max_train_samples=%s; max_optimizer_steps=%s",
        args.tiny_smoke,
        gradient_accumulation,
        max_train_samples,
        max_optimizer_steps,
    )

    train_records = load_jsonl(
        data_root
        / TRAIN_JSONL
    )

    val_records = load_jsonl(
        data_root
        / VAL_JSONL
    )

    train_records = select_records(
        train_records,
        max_samples=(
            max_train_samples
        ),
        prefer_short=(
            args.tiny_smoke
        ),
        seed=args.seed,
    )

    val_records = select_records(
        val_records,
        max_samples=(
            max_eval_samples
            if args.tiny_smoke
            else None
        ),
        prefer_short=(
            args.tiny_smoke
        ),
        seed=(
            args.seed
            + 1
        ),
    )

    logging.info(
        "Selected train records: %d",
        len(train_records),
    )
    logging.info(
        "Available validation records after selection: %d",
        len(val_records),
    )

    model, tokenizer = (
        load_model_and_tokenizer(
            model_path=model_path,
            dtype=dtype,
            use_flash_attn=(
                args.use_flash_attn
            ),
            max_seq_length=(
                args.max_seq_length
            ),
        )
    )

    # ORena no-system-turn training.
    # The full task instruction is already contained in user_text.
    system_message = (
        getattr(model, "system_message", None)
        if USE_SYSTEM_TURN
        else None
    )

    logging.info(
        "Use system turn for training: %s",
        USE_SYSTEM_TURN,
    )

    logging.info(
        "Training system message: %r",
        system_message,
    )

    configure_dora(
        model=model,
        rank=args.dora_rank,
        alpha=args.dora_alpha,
        dropout=args.dora_dropout,
        gradient_checkpointing=(
            args.gradient_checkpointing
        ),
    )

    verify_trainable_scope(
        model
    )

    logging.info(
        "About to move complete PEFT-wrapped InternVL model to %s...",
        device,
    )

    move_started = time.monotonic()

    model = model.to(
        device
    )

    torch.cuda.synchronize()

    logging.info(
        "Model move to GPU completed in %.2f s",
        time.monotonic() - move_started,
    )

    # Reference-aligned DoRA-only training: vision_model and mlp1 stay frozen.
    model.vision_model.eval()
    model.mlp1.eval()
    model.language_model.train()

    train_dataset = OrenaSFTDataset(
        train_records,
        tokenizer=tokenizer,
        system_message=(
            system_message
        ),
        num_image_token=int(
            model.num_image_token
        ),
        max_seq_length=(
            args.max_seq_length
        ),
    )

    val_dataset = (
        OrenaSFTDataset(
            val_records,
            tokenizer=tokenizer,
            system_message=(
                system_message
            ),
            num_image_token=int(
                model.num_image_token
            ),
            max_seq_length=(
                args.max_seq_length
            ),
        )
        if val_records
        else None
    )

    # Preflight one sample before constructing the optimizer/training loop.
    logging.info(
        "Running one-sample multimodal preflight..."
    )

    preflight = (
        train_dataset.prepare(
            0
        )
    )

    context_token_id = (
        tokenizer.convert_tokens_to_ids(
            IMG_CONTEXT_TOKEN
        )
    )

    context_count = int(
        (
            preflight[
                "input_ids"
            ]
            == context_token_id
        ).sum()
        .item()
    )

    expected_context_count = (
        preflight[
            "num_frames"
        ]
        * int(
            model.num_image_token
        )
    )

    if (
        context_count
        != expected_context_count
    ):
        raise RuntimeError(
            "Preflight visual-context-token count mismatch: "
            f"got {context_count}, expected {expected_context_count}."
        )

    logging.info(
        "Preflight PASSED: sample=%s frames=%d seq=%d "
        "IMG_CONTEXT=%d",
        preflight[
            "sample_id"
        ],
        preflight[
            "num_frames"
        ],
        preflight[
            "seq_length"
        ],
        context_count,
    )

    del preflight

    run_config = RunConfig(
        model_path=str(
            model_path
        ),
        data_root=str(
            data_root
        ),
        output_dir=str(
            output_dir
        ),
        precision=(
            resolved_precision
        ),
        max_seq_length=(
            args.max_seq_length
        ),
        epochs=args.epochs,
        gradient_accumulation=(
            gradient_accumulation
        ),
        dora_rank=(
            args.dora_rank
        ),
        dora_alpha=(
            args.dora_alpha
        ),
        dora_dropout=(
            args.dora_dropout
        ),
        dora_lr=(
            args.dora_lr
        ),
        weight_decay=(
            args.weight_decay
        ),
        lr_patience=(
            args.lr_patience
        ),
        lr_shrink_factor=(
            args.lr_shrink_factor
        ),
        seed=args.seed,
        use_flash_attn=(
            args.use_flash_attn
        ),
        use_system_turn=USE_SYSTEM_TURN,
        tiny_smoke=(
            args.tiny_smoke
        ),
        max_train_samples=(
            max_train_samples
        ),
        max_eval_samples=(
            max_eval_samples
        ),
        max_optimizer_steps=(
            max_optimizer_steps
        ),
    )

    try:
        train(
            model=model,
            tokenizer=tokenizer,
            train_dataset=(
                train_dataset
            ),
            val_dataset=(
                val_dataset
            ),
            args=args,
            run_config=(
                run_config
            ),
            device=device,
            dtype=dtype,
            gradient_accumulation=(
                gradient_accumulation
            ),
            max_optimizer_steps=(
                max_optimizer_steps
            ),
            max_eval_samples=(
                max_eval_samples
            ),
        )
    finally:
        del train_dataset
        if val_dataset is not None:
            del val_dataset
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    logging.info(
        "=== complete ==="
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
