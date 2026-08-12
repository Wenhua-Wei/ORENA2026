"""InternVL3.5 model utilities for local ORena FOCUS experiments.

This module:
1. loads OpenGVLab/InternVL3_5-4B-Instruct once;
2. converts a list of RGB PIL frames into InternVL visual tensors;
3. runs deterministic multi-frame inference; and
4. returns the raw generated answer.

For the Quadro RTX 6000, the default configuration uses FP16 and disables
FlashAttention.
"""

from __future__ import annotations
import os
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PredictionResult:
    """Raw model output and basic inference diagnostics."""

    answer: str
    inference_seconds: float
    num_frames: int
    num_patches: int
    peak_gpu_memory_gb: float | None


def build_transform(input_size: int = 448) -> T.Compose:
    """Build the image transform expected by InternVL."""

    return T.Compose(
        [
            T.Lambda(
                lambda image: (
                    image.convert("RGB")
                    if image.mode != "RGB"
                    else image
                )
            ),
            T.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Sequence[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """Choose the InternVL tile grid closest to an image's aspect ratio."""

    best_ratio = (1, 1)
    best_difference = float("inf")
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        difference = abs(aspect_ratio - target_aspect_ratio)

        if difference < best_difference:
            best_difference = difference
            best_ratio = ratio
        elif difference == best_difference:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    *,
    min_num: int = 1,
    max_num: int = 1,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    """Split one image into InternVL tiles.

    For the initial video experiment, keep ``max_num=1`` so each frame
    contributes exactly one visual tile. This is much lighter than dynamic
    multi-tile processing and follows the official InternVL video example.
    """

    if min_num <= 0:
        raise ValueError("min_num must be positive.")
    if max_num < min_num:
        raise ValueError("max_num must be greater than or equal to min_num.")

    image = image.convert("RGB")
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    target_ratios = {
        (columns, rows)
        for number_of_tiles in range(min_num, max_num + 1)
        for columns in range(1, number_of_tiles + 1)
        for rows in range(1, number_of_tiles + 1)
        if min_num <= columns * rows <= max_num
    }
    sorted_ratios = sorted(
        target_ratios,
        key=lambda ratio: ratio[0] * ratio[1],
    )

    target_ratio = find_closest_aspect_ratio(
        aspect_ratio=aspect_ratio,
        target_ratios=sorted_ratios,
        width=original_width,
        height=original_height,
        image_size=image_size,
    )

    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]
    number_of_blocks = target_ratio[0] * target_ratio[1]

    resized_image = image.resize(
        (target_width, target_height),
        resample=Image.Resampling.BICUBIC,
    )

    processed_images: list[Image.Image] = []

    for block_index in range(number_of_blocks):
        left = (block_index % target_ratio[0]) * image_size
        top = (block_index // target_ratio[0]) * image_size
        right = left + image_size
        bottom = top + image_size

        processed_images.append(
            resized_image.crop((left, top, right, bottom))
        )

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(
            image.resize(
                (image_size, image_size),
                resample=Image.Resampling.BICUBIC,
            )
        )

    return processed_images


def prepare_frames(
    images: Sequence[Image.Image],
    *,
    input_size: int = 448,
    max_tiles_per_frame: int = 1,
    use_thumbnail: bool = True,
) -> tuple[torch.Tensor, list[int]]:
    """Convert ordered PIL frames into InternVL pixel values."""

    if not images:
        raise ValueError("At least one frame is required.")

    if max_tiles_per_frame <= 0:
        raise ValueError("max_tiles_per_frame must be positive.")

    transform = build_transform(input_size=input_size)
    frame_tensors: list[torch.Tensor] = []
    num_patches_list: list[int] = []

    for image in images:
        tiles = dynamic_preprocess(
            image,
            image_size=input_size,
            max_num=max_tiles_per_frame,
            use_thumbnail=use_thumbnail,
        )
        pixel_values = torch.stack(
            [transform(tile) for tile in tiles]
        )
        frame_tensors.append(pixel_values)
        num_patches_list.append(pixel_values.shape[0])

    return torch.cat(frame_tensors, dim=0), num_patches_list


class InternVLInferenceEngine:
    """Load InternVL3.5 once and run multi-frame inference."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        input_size: int = 448,
        max_tiles_per_frame: int = 1,
        max_new_tokens: int = 32,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()

        if input_size <= 0:
            raise ValueError("input_size must be positive.")

        if max_tiles_per_frame <= 0:
            raise ValueError("max_tiles_per_frame must be positive.")

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")

        self.device = torch.device(
            device
            if device is not None
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype
        self.input_size = input_size
        self.max_tiles_per_frame = max_tiles_per_frame
        self.max_new_tokens = max_new_tokens

        self.model = None
        self.tokenizer = None

    @property
    def is_loaded(self) -> bool:
        """Whether model and tokenizer have been loaded."""

        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        """Load the tokenizer and model exclusively from local files."""

        if self.is_loaded:
            return

        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"InternVL model directory does not exist: "
                f"{self.model_path}"
            )

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {self.device} was requested, "
                "but torch.cuda.is_available() is False."
            )

        print(
            f"Loading tokenizer from: {self.model_path}",
            flush=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )

        print(
            f"Loading model on {self.device} with dtype={self.dtype} "
            "and FlashAttention disabled...",
            flush=True,
        )

        self.model = AutoModel.from_pretrained(
            str(self.model_path),
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
            local_files_only=True,
        )

        self.model = self.model.to(self.device).eval()

        print("InternVL model loaded.", flush=True)

    def predict(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        *,
        frame_labels: Sequence[str] | None = None,
        max_new_tokens: int | None = None,
    ) -> PredictionResult:
        """Run one deterministic multi-frame inference call."""

        if not self.is_loaded:
            raise RuntimeError("Call engine.load() before engine.predict().")

        if not prompt.strip():
            raise ValueError("prompt must not be empty.")

        if frame_labels is not None and len(frame_labels) != len(images):
            raise ValueError(
                "frame_labels must have the same length as images."
            )

        pixel_values, num_patches_list = prepare_frames(
            images,
            input_size=self.input_size,
            max_tiles_per_frame=self.max_tiles_per_frame,
            use_thumbnail=True,
        )
        pixel_values = pixel_values.to(
            device=self.device,
            dtype=self.dtype,
            non_blocking=True,
        )

        if frame_labels is None:
            video_prefix = "".join(
                f"Frame {index + 1}: <image>\n"
                for index in range(len(images))
            )
        else:
            video_prefix = "".join(
                f"{label}: <image>\n"
                for label in frame_labels
            )

#        else:
#            video_prefix = "".join(
#                f"Frame {index + 1} ({label}): <image>\n"
#                for index, label in enumerate(frame_labels)
#            )

        question = video_prefix + prompt.strip()

        generation_config = {
            "max_new_tokens": (
                max_new_tokens
                if max_new_tokens is not None
                else self.max_new_tokens
            ),
            "do_sample": False,
            "num_beams": 1,
        }

        peak_gpu_memory_gb: float | None = None
        response: str | None = None
        inference_seconds = 0.0

        try:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)

            start_time = time.perf_counter()

            with torch.inference_mode():
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    num_patches_list=num_patches_list,
                )

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                peak_gpu_memory_gb = (
                    torch.cuda.max_memory_allocated(self.device)
                    / (1024**3)
                )

            inference_seconds = time.perf_counter() - start_time

        finally:
            del pixel_values

        if response is None:
            raise RuntimeError("InternVL returned no response.")
       

        return PredictionResult(
            answer=str(response).strip(),
            inference_seconds=inference_seconds,
            num_frames=len(images),
            num_patches=sum(num_patches_list),
            peak_gpu_memory_gb=peak_gpu_memory_gb,
        )

    def unload(self) -> None:
        """Release model memory."""

        self.model = None
        self.tokenizer = None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    
def _check_model_snapshot(model_path: Path) -> None:
    """Verify that the essential offline model files are present."""

    required_files = (
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "configuration_intern_vit.py",
        "configuration_internvl_chat.py",
        "modeling_intern_vit.py",
        "modeling_internvl_chat.py",
        "conversation.py",
    )

    missing_files = [
        filename
        for filename in required_files
        if not (model_path / filename).is_file()
    ]

    if missing_files:
        formatted = "\n".join(
            f"  - {filename}"
            for filename in missing_files
        )
        raise FileNotFoundError(
            "The InternVL snapshot is incomplete. "
            f"Missing files:\n{formatted}"
        )


def _run_preprocessing_smoke_test() -> None:
    """Check image preprocessing without loading the large model."""

    images = [
        Image.new(
            "RGB",
            (640, 360),
            color=(0, 0, 0),
        ),
        Image.new(
            "RGB",
            (720, 576),
            color=(255, 255, 255),
        ),
    ]

    pixel_values, num_patches_list = prepare_frames(
        images,
        input_size=448,
        max_tiles_per_frame=1,
        use_thumbnail=True,
    )

    expected_shape = (2, 3, 448, 448)

    if tuple(pixel_values.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected preprocessing output shape: "
            f"{tuple(pixel_values.shape)}; "
            f"expected {expected_shape}."
        )

    if num_patches_list != [1, 1]:
        raise RuntimeError(
            "Unexpected num_patches_list: "
            f"{num_patches_list}; expected [1, 1]."
        )

    print("Preprocessing smoke test passed.")
    print(f"  pixel_values shape: {tuple(pixel_values.shape)}")
    print(f"  pixel_values dtype: {pixel_values.dtype}")
    print(f"  num_patches_list: {num_patches_list}")


def main() -> None:
    """Run lightweight checks and optionally one full model inference."""

    default_model_path = (
        Path(__file__).resolve().parent
        / "resources"
        / "InternVL3_5-4B-Instruct-a3fd3158"
    )

    model_path = Path(
        os.environ.get(
            "INTERNVL_MODEL_PATH",
            str(default_model_path),
        )
    ).expanduser().resolve()

    print("=" * 80)
    print("InternVL model-utils smoke test")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    print("\nChecking local model snapshot...")
    _check_model_snapshot(model_path)
    print("Model snapshot check passed.")

    print("\nChecking tokenizer offline...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )

    test_tokens = tokenizer(
        "Return exactly yes or no.",
        return_tensors="pt",
    )

    print("Tokenizer smoke test passed.")
    print(
        "  token count:",
        int(test_tokens["input_ids"].shape[-1]),
    )

    del tokenizer
    del test_tokens

    print("\nChecking visual preprocessing...")
    _run_preprocessing_smoke_test()

    run_full_test = (
        os.environ.get(
            "RUN_FULL_MODEL_TEST",
            "0",
        ).strip()
        == "1"
    )

    if not run_full_test:
        print(
            "\nLightweight smoke test passed.\n"
            "Set RUN_FULL_MODEL_TEST=1 to load the complete model "
            "and run one generation."
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "RUN_FULL_MODEL_TEST=1 was requested, but CUDA is "
            "unavailable. Do not run the full 4B model test in the "
            "Intel Mac CPU container."
        )

    print("\nLoading the complete model for a full GPU test...")

    engine = InternVLInferenceEngine(
        model_path=model_path,
        device="cuda:0",
        dtype=torch.float16,
        input_size=448,
        max_tiles_per_frame=1,
        max_new_tokens=16,
    )

    try:
        engine.load()

        images = [
            Image.new(
                "RGB",
                (640, 360),
                color=(0, 0, 0),
            ),
            Image.new(
                "RGB",
                (640, 360),
                color=(0, 0, 0),
            ),
        ]

        result = engine.predict(
            images=images,
            frame_labels=(
                "00:00:00",
                "00:00:01",
            ),
            prompt=(
                "Both supplied frames are synthetic black images. "
                "Are both frames black? Return exactly yes or no."
            ),
        )

        print("\nFull inference smoke test passed.")
        print(f"  answer: {result.answer!r}")
        print(
            f"  inference time: "
            f"{result.inference_seconds:.2f} seconds"
        )
        print(f"  frames: {result.num_frames}")
        print(f"  visual patches: {result.num_patches}")

        if result.peak_gpu_memory_gb is not None:
            print(
                "  peak allocated GPU memory: "
                f"{result.peak_gpu_memory_gb:.2f} GB"
            )

    finally:
        engine.unload()


if __name__ == "__main__":
    main()
