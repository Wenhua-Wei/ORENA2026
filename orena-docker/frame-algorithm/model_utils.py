"""InternVL3.5 model utilities for ORena SAVE FOCUS FRAME inference.

This module:
1. loads ``InternVL3_5-8B-Instruct`` once from local files;
2. converts one native-resolution RGB laparoscopic image into aspect-ratio-aware
   InternVL visual tiles;
3. supports deterministic single-image inference either with InternVL's normal
   system turn or with the system turn disabled; and
4. returns the raw generated answer with lightweight diagnostics.

For the current FRAME submission, ``use_system_turn=False`` should be used so
the prompt is serialized directly as a user turn followed by the assistant turn,
matching the current SEGMENT submission setup.

The original FRAME PNG is kept at native resolution until InternVL preprocessing
selects a dynamic tile grid. Each final tile is resized to ``input_size`` square
pixels, normally 448 x 448.
"""

from __future__ import annotations

import gc
import json
import os
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

APP_PATH = Path(__file__).resolve().parent

SMOKE_TEST_MODEL_PATH = (
    APP_PATH
    / "resources"
    / "InternVL3_5-8B-Instruct"
)

SMOKE_TEST_IMAGE_PATH = (
    APP_PATH
    / "test"
    / "input"
    / "interface_1"
    / "frames"
    / "q0001.png"
)


@dataclass(frozen=True)
class PredictionResult:
    """Raw model output and basic single-image diagnostics."""

    answer: str
    inference_seconds: float
    num_images: int
    num_patches: int
    peak_gpu_memory_gb: float | None


def build_transform(input_size: int = 448) -> T.Compose:
    """Build the image transform expected by InternVL."""

    if input_size <= 0:
        raise ValueError("input_size must be positive.")

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
            T.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Sequence[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """Choose the InternVL tile grid closest to the source aspect ratio."""

    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be positive.")

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")

    if image_size <= 0:
        raise ValueError("image_size must be positive.")

    if not target_ratios:
        raise ValueError("target_ratios must not be empty.")

    best_ratio = (1, 1)
    best_difference = float("inf")
    source_area = width * height

    for columns, rows in target_ratios:
        target_aspect_ratio = columns / rows
        difference = abs(aspect_ratio - target_aspect_ratio)

        if difference < best_difference:
            best_difference = difference
            best_ratio = (columns, rows)

        elif difference == best_difference:
            target_area = (
                image_size
                * image_size
                * columns
                * rows
            )

            if source_area > 0.5 * target_area:
                best_ratio = (columns, rows)

    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    *,
    min_num: int = 1,
    max_num: int = 4,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    """Split one native-resolution image into InternVL visual tiles.

    The source image keeps its native aspect ratio until a tile grid is chosen.
    The image is then resized to the grid canvas and cropped into
    ``image_size x image_size`` local tiles.

    If more than one local tile is produced and ``use_thumbnail=True``, one
    additional global thumbnail is appended. Therefore, with ``max_num=4``,
    the model receives at most 5 visual patches for one FRAME image.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image.")

    if min_num <= 0:
        raise ValueError("min_num must be positive.")

    if max_num < min_num:
        raise ValueError(
            "max_num must be greater than or equal to min_num."
        )

    if image_size <= 0:
        raise ValueError("image_size must be positive.")

    image = image.convert("RGB")
    original_width, original_height = image.size

    if original_width <= 0 or original_height <= 0:
        raise ValueError(
            f"Invalid source image dimensions: {image.size}"
        )

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
        key=lambda ratio: (
            ratio[0] * ratio[1],
            ratio[0],
            ratio[1],
        ),
    )

    columns, rows = find_closest_aspect_ratio(
        aspect_ratio=aspect_ratio,
        target_ratios=sorted_ratios,
        width=original_width,
        height=original_height,
        image_size=image_size,
    )

    target_width = image_size * columns
    target_height = image_size * rows
    number_of_blocks = columns * rows

    resized_image = image.resize(
        (target_width, target_height),
        resample=Image.Resampling.BICUBIC,
    )

    processed_images: list[Image.Image] = []

    for block_index in range(number_of_blocks):
        left = (block_index % columns) * image_size
        top = (block_index // columns) * image_size
        right = left + image_size
        bottom = top + image_size

        tile = resized_image.crop(
            (left, top, right, bottom)
        )

        if tile.size != (image_size, image_size):
            raise RuntimeError(
                "Unexpected tile dimensions: "
                f"{tile.size}; expected {(image_size, image_size)}."
            )

        processed_images.append(tile)

    if use_thumbnail and number_of_blocks != 1:
        processed_images.append(
            image.resize(
                (image_size, image_size),
                resample=Image.Resampling.BICUBIC,
            )
        )

    if not processed_images:
        raise RuntimeError("No visual tiles were produced.")

    return processed_images


def prepare_image(
    image: Image.Image,
    *,
    input_size: int = 448,
    max_tiles_per_image: int = 4,
    use_thumbnail: bool = True,
) -> tuple[torch.Tensor, list[int]]:
    """Convert one PIL image into InternVL pixel values.

    Returns
    -------
    pixel_values:
        Tensor with shape ``[num_patches, 3, input_size, input_size]``.
    num_patches_list:
        A one-element list because FRAME has one logical image group.
        For example, ``[5]`` means one image represented by five patches.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image.")

    if input_size <= 0:
        raise ValueError("input_size must be positive.")

    if max_tiles_per_image <= 0:
        raise ValueError(
            "max_tiles_per_image must be positive."
        )

    transform = build_transform(
        input_size=input_size
    )

    tiles = dynamic_preprocess(
        image=image,
        image_size=input_size,
        max_num=max_tiles_per_image,
        use_thumbnail=use_thumbnail,
    )

    pixel_values = torch.stack(
        [transform(tile) for tile in tiles]
    )

    expected_shape = (
        len(tiles),
        3,
        input_size,
        input_size,
    )

    if tuple(pixel_values.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected preprocessing tensor shape: "
            f"{tuple(pixel_values.shape)}; "
            f"expected {expected_shape}."
        )

    return pixel_values, [len(tiles)]


class InternVLInferenceEngine:
    """Load InternVL3.5 once and run single-image FRAME inference."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        input_size: int = 448,
        max_tiles_per_image: int = 4,
        use_thumbnail: bool = True,
        max_new_tokens: int = 128,
        use_system_turn: bool = False,
        collect_gpu_diagnostics: bool = False,
    ) -> None:
        self.model_path = (
            Path(model_path)
            .expanduser()
            .resolve()
        )

        if input_size <= 0:
            raise ValueError("input_size must be positive.")

        if max_tiles_per_image <= 0:
            raise ValueError(
                "max_tiles_per_image must be positive."
            )

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be positive."
            )

        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda:0"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.dtype = dtype
        self.input_size = input_size
        self.max_tiles_per_image = max_tiles_per_image
        self.use_thumbnail = bool(use_thumbnail)
        self.max_new_tokens = max_new_tokens
        self.use_system_turn = bool(use_system_turn)
        self.collect_gpu_diagnostics = bool(
            collect_gpu_diagnostics
        )

        self.model = None
        self.tokenizer = None

    @property
    def is_loaded(self) -> bool:
        """Whether both model and tokenizer are loaded."""

        return (
            self.model is not None
            and self.tokenizer is not None
        )

    def load(self) -> None:
        """Load tokenizer and model exclusively from local files."""

        if self.is_loaded:
            return

        if not self.model_path.is_dir():
            raise FileNotFoundError(
                "InternVL model directory does not exist: "
                f"{self.model_path}"
            )

        if (
            self.device.type == "cuda"
            and not torch.cuda.is_available()
        ):
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
            f"Loading model on {self.device} with "
            f"dtype={self.dtype} and FlashAttention disabled...",
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

        self.model = (
            self.model
            .to(self.device)
            .eval()
        )

        print("InternVL model loaded.", flush=True)
        print(
            "Original InternVL system message: "
            f"{getattr(self.model, 'system_message', None)!r}",
            flush=True,
        )
        print(
            f"Use system turn: {self.use_system_turn}",
            flush=True,
        )

    def _generate_without_system(
        self,
        *,
        pixel_values: torch.Tensor,
        question: str,
        generation_config: dict,
        num_patches_list: Sequence[int],
    ) -> str:
        """Generate with InternVL user/assistant serialization and no system turn.

        This follows the same no-system-turn strategy as the current SEGMENT
        submission. ``question`` must contain exactly one ``<image>`` placeholder
        for the one FRAME image.
        """

        img_start_token = "<img>"
        img_end_token = "</img>"
        img_context_token = "<IMG_CONTEXT>"

        if question.count("<image>") != len(num_patches_list):
            raise RuntimeError(
                "Expected exactly one <image> placeholder per visual group: "
                f"placeholders={question.count('<image>')}, "
                f"groups={len(num_patches_list)}."
            )

        img_context_token_id = (
            self.tokenizer.convert_tokens_to_ids(
                img_context_token
            )
        )

        if img_context_token_id is None:
            raise RuntimeError(
                "Tokenizer does not contain <IMG_CONTEXT>."
            )

        self.model.img_context_token_id = int(
            img_context_token_id
        )

        # Intentionally no system turn.
        query = (
            "<|im_start|>user\n"
            + question
            + "<|im_end|>\n"
            + "<|im_start|>assistant\n"
        )

        for num_patches in num_patches_list:
            image_tokens = (
                img_start_token
                + img_context_token
                * int(self.model.num_image_token)
                * int(num_patches)
                + img_end_token
            )

            query = query.replace(
                "<image>",
                image_tokens,
                1,
            )

        if "<image>" in query:
            raise RuntimeError(
                "Unexpanded <image> placeholder remains."
            )

        model_inputs = self.tokenizer(
            query,
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"].to(
            self.device
        )
        attention_mask = model_inputs["attention_mask"].to(
            self.device
        )

        eos_token_id = (
            self.tokenizer.convert_tokens_to_ids(
                "<|im_end|>"
            )
        )

        if eos_token_id is None:
            raise RuntimeError(
                "Tokenizer does not contain <|im_end|>."
            )

        config = dict(generation_config)
        config["eos_token_id"] = int(eos_token_id)

        generation_output = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **config,
        )

        response = self.tokenizer.batch_decode(
            generation_output,
            skip_special_tokens=True,
        )[0]

        return response.split("<|im_end|>")[0].strip()

    def predict(
        self,
        image: Image.Image,
        prompt: str,
        *,
        image_label: str | None = None,
        max_new_tokens: int | None = None,
    ) -> PredictionResult:
        """Run one deterministic single-image FRAME inference call."""

        if not self.is_loaded:
            raise RuntimeError(
                "Call engine.load() before engine.predict()."
            )

        if not isinstance(image, Image.Image):
            raise TypeError(
                "image must be a PIL.Image.Image."
            )

        cleaned_prompt = prompt.strip()

        if not cleaned_prompt:
            raise ValueError("prompt must not be empty.")

        generation_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else self.max_new_tokens
        )

        if generation_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be positive."
            )

        pixel_values, num_patches_list = prepare_image(
            image=image,
            input_size=self.input_size,
            max_tiles_per_image=self.max_tiles_per_image,
            use_thumbnail=self.use_thumbnail,
        )

        pixel_values = pixel_values.to(
            device=self.device,
            dtype=self.dtype,
            non_blocking=True,
        )

        # FRAME has one logical visual input. The image may internally consist
        # of multiple InternVL tiles, but it still gets one <image> placeholder.
        if image_label is None:
            image_prefix = "Image: <image>\n"
        else:
            image_prefix = f"Image at {image_label}: <image>\n"

        question = image_prefix + cleaned_prompt

        generation_config = {
            "max_new_tokens": generation_tokens,
            "do_sample": False,
            "num_beams": 1,
        }

        response: str | None = None
        inference_seconds = 0.0
        peak_gpu_memory_gb: float | None = None

        try:
            if (
                self.device.type == "cuda"
                and self.collect_gpu_diagnostics
            ):
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(
                    self.device
                )

            start_time = time.perf_counter()

            with torch.inference_mode():
                if self.use_system_turn:
                    response = self.model.chat(
                        self.tokenizer,
                        pixel_values,
                        question,
                        generation_config,
                        num_patches_list=num_patches_list,
                    )
                else:
                    response = self._generate_without_system(
                        pixel_values=pixel_values,
                        question=question,
                        generation_config=generation_config,
                        num_patches_list=num_patches_list,
                    )

            if (
                self.device.type == "cuda"
                and self.collect_gpu_diagnostics
            ):
                torch.cuda.synchronize(self.device)
                peak_gpu_memory_gb = (
                    torch.cuda.max_memory_allocated(
                        self.device
                    )
                    / (1024**3)
                )

            inference_seconds = (
                time.perf_counter()
                - start_time
            )

        finally:
            del pixel_values

        if response is None:
            raise RuntimeError(
                "InternVL returned no response."
            )

        return PredictionResult(
            answer=str(response).strip(),
            inference_seconds=inference_seconds,
            num_images=1,
            num_patches=sum(num_patches_list),
            peak_gpu_memory_gb=peak_gpu_memory_gb,
        )

    def unload(self) -> None:
        """Release model and tokenizer memory."""

        self.model = None
        self.tokenizer = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _check_model_snapshot(
    model_path: Path,
) -> None:
    """Verify that the essential offline InternVL snapshot is present."""

    required_files = (
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
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

    index_path = (
        model_path
        / "model.safetensors.index.json"
    )

    if index_path.is_file():
        try:
            with index_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                index_data = json.load(file)

        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Failed to read model index: {index_path}"
            ) from error

        weight_map = index_data.get("weight_map")

        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(
                "model.safetensors.index.json contains no valid weight_map."
            )

        shard_names = sorted(
            {
                str(filename)
                for filename in weight_map.values()
            }
        )

        missing_files.extend(
            shard_name
            for shard_name in shard_names
            if not (model_path / shard_name).is_file()
        )

    if missing_files:
        unique_missing = list(
            dict.fromkeys(missing_files)
        )

        formatted = "\n".join(
            f"  - {filename}"
            for filename in unique_missing
        )

        raise FileNotFoundError(
            "The InternVL snapshot is incomplete. "
            f"Missing files:\n{formatted}"
        )


def _load_smoke_test_image() -> Image.Image:
    """Load q0001.png for preprocessing/inference smoke tests."""

    if not SMOKE_TEST_IMAGE_PATH.is_file():
        raise FileNotFoundError(
            "Smoke-test frame does not exist: "
            f"{SMOKE_TEST_IMAGE_PATH}"
        )

    try:
        with Image.open(
            SMOKE_TEST_IMAGE_PATH
        ) as opened_image:
            opened_image.load()
            image = opened_image.convert("RGB")

    except OSError as error:
        raise RuntimeError(
            "Failed to decode smoke-test frame: "
            f"{SMOKE_TEST_IMAGE_PATH}"
        ) from error

    return image


def _run_preprocessing_smoke_test(
    image: Image.Image,
) -> None:
    """Check FRAME visual preprocessing without loading the complete model."""

    pixel_values, num_patches_list = prepare_image(
        image=image,
        input_size=448,
        max_tiles_per_image=4,
        use_thumbnail=True,
    )

    num_patches = num_patches_list[0]

    if not 1 <= num_patches <= 5:
        raise RuntimeError(
            "Unexpected visual-patch count: "
            f"{num_patches}; expected between 1 and 5."
        )

    expected_shape = (
        num_patches,
        3,
        448,
        448,
    )

    if tuple(pixel_values.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected preprocessing output shape: "
            f"{tuple(pixel_values.shape)}; "
            f"expected {expected_shape}."
        )

    print("Preprocessing smoke test passed.")
    print(f"  native image size: {image.size}")
    print(f"  pixel_values shape: {tuple(pixel_values.shape)}")
    print(f"  pixel_values dtype: {pixel_values.dtype}")
    print(f"  num_patches_list: {num_patches_list}")


def main() -> None:
    """Run lightweight checks and optionally one complete GPU generation."""

    model_path = SMOKE_TEST_MODEL_PATH.resolve()

    print("=" * 80)
    print("ORena FOCUS FRAME InternVL model-utils smoke test")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Test image: {SMOKE_TEST_IMAGE_PATH}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(
            "CUDA device: "
            f"{torch.cuda.get_device_name(0)}"
        )

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

    print("\nLoading real FRAME image...")
    image = _load_smoke_test_image()
    print(
        f"Image loaded: mode={image.mode}, "
        f"size={image.size}"
    )

    print("\nChecking visual preprocessing...")
    _run_preprocessing_smoke_test(image)

    run_full_model_test = (
        os.environ.get(
            "RUN_FULL_MODEL_TEST",
            "0",
        ).strip()
        == "1"
    )

    if not run_full_model_test:
        print(
            "\nLightweight smoke test passed.\n"
            "Set RUN_FULL_MODEL_TEST=1 to load the complete model "
            "and run one generation."
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "RUN_FULL_MODEL_TEST=1 was requested, "
            "but CUDA is unavailable."
        )

    print("\nLoading the complete model for one GPU test...")

    engine = InternVLInferenceEngine(
        model_path=model_path,
        device="cuda:0",
        dtype=torch.float16,
        input_size=448,
        max_tiles_per_image=4,
        use_thumbnail=True,
        max_new_tokens=32,
        use_system_turn=False,
        collect_gpu_diagnostics=True,
    )

    try:
        engine.load()

        result = engine.predict(
            image=image,
            prompt=(
                "This is one laparoscopic image. "
                "Answer only yes or no: is there any visible content "
                "in the supplied image?"
            ),
            image_label="00:33:19",
        )

        print("\nFull inference smoke test passed.")
        print(f"  answer: {result.answer!r}")
        print(
            "  inference time: "
            f"{result.inference_seconds:.2f} seconds"
        )
        print(f"  images: {result.num_images}")
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