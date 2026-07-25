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
        model_id: str = "OpenGVLab/InternVL3_5-4B-Instruct",
        *,
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        input_size: int = 448,
        max_tiles_per_frame: int = 1,
        max_new_tokens: int = 32,
    ) -> None:
        self.model_id = model_id
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
        """Load tokenizer and model weights once."""

        if self.is_loaded:
            return

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {self.device} was requested, "
                "but torch.cuda.is_available() is False."
            )

        print(f"Loading tokenizer: {self.model_id}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
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
            self.model_id,
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
                f"Frame {index + 1} ({label}): <image>\n"
                for index, label in enumerate(frame_labels)
            )

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

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)


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

        # start_time = time.perf_counter()

        # with torch.inference_mode():
        #     response = self.model.chat(
        #         self.tokenizer,
        #         pixel_values,
        #         question,
        #         generation_config,
        #         num_patches_list=num_patches_list,
        #     )

        # if self.device.type == "cuda":
        #     torch.cuda.synchronize(self.device)
        #     peak_gpu_memory_gb = (
        #         torch.cuda.max_memory_allocated(self.device)
        #         / (1024**3)
        #     )

        # inference_seconds = time.perf_counter() - start_time

        # del pixel_values

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


if __name__ == "__main__":
    # Model-only smoke test using the JPEGs saved by video_utils.py.
    # This avoids reopening the long source AVI.
    project_root = Path(
        "/cs/student/projects1/aibh/2024/wenhuawe/ORENA"
    )
    debug_frames_dir = (
        project_root
        / "internvl_fewshot"
        / "video_utils_test_frames"
    )

    frame_paths = sorted(debug_frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(
            f"No debug frames found in {debug_frames_dir}. "
            "Run video_utils.py first."
        )

    frames = [
        Image.open(frame_path).convert("RGB")
        for frame_path in frame_paths
    ]

    engine = InternVLInferenceEngine(
        model_id="OpenGVLab/InternVL3_5-4B-Instruct",
        device="cuda:0",
        dtype=torch.float16,
        input_size=448,
        max_tiles_per_frame=1,
        max_new_tokens=32,
    )
    engine.load()

    result = engine.predict(
        images=frames,
        prompt=(
            "These are uniformly sampled frames from a laparoscopic "
            "surgical video segment. Briefly describe the visible surgical "
            "scene and mention any surgical foreign object you can identify. "
            "Answer in one concise sentence."
        ),
    )

    print("\nRaw answer:")
    print(result.answer)
    print(f"\nInference time: {result.inference_seconds:.2f} s")
    print(f"Frames: {result.num_frames}")
    print(f"Visual patches: {result.num_patches}")

    if result.peak_gpu_memory_gb is not None:
        print(
            "Peak allocated GPU memory during predict(): "
            f"{result.peak_gpu_memory_gb:.2f} GB"
        )
