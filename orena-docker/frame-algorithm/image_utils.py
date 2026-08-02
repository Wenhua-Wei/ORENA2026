"""Image utilities for ORena FOCUS FRAME Docker inference.

Each input image is the single visual input for one FRAME-track question and is
stored at ``/input/frames/<qID>.png``.

This module deliberately preserves the image's native spatial resolution and
aspect ratio. Resizing, tiling, normalization, and conversion to model tensors
belong in ``model_utils.py`` rather than here.
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def _validate_loaded_frame(image: Image.Image, *, source_path: Path) -> None:
    """Validate a decoded FRAME image."""

    if image.mode != "RGB":
        raise RuntimeError(
            f"Expected an RGB image after conversion, received "
            f"mode={image.mode!r} from {source_path}."
        )

    if image.width <= 0 or image.height <= 0:
        raise RuntimeError(
            f"Invalid image dimensions {image.size} for {source_path}."
        )


def load_frame(
    image_path: str | Path,
    *,
    require_png: bool = True,
) -> Image.Image:
    """Load one FRAME-track image as an independent RGB PIL image.

    The image is fully decoded while the file handle is open, copied into
    memory, and returned at its native resolution. No resizing, cropping,
    padding, or model-specific preprocessing is performed.

    Args:
        image_path:
            Path to the input image, normally
            ``/input/frames/<qID>.png``.
        require_png:
            When ``True``, reject files whose decoded format is not PNG.

    Returns:
        A fully loaded ``PIL.Image.Image`` in RGB mode.

    Raises:
        FileNotFoundError:
            If the path does not exist or is not a regular file.
        ValueError:
            If the file is not PNG when ``require_png=True``.
        RuntimeError:
            If Pillow cannot identify or decode the image, or if the decoded
            dimensions are invalid.
    """

    path = Path(image_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Frame image does not exist: {path}")

    try:
        with Image.open(path) as opened_image:
            source_format = opened_image.format

            if require_png and source_format != "PNG":
                raise ValueError(
                    f"Expected a PNG frame, but Pillow detected "
                    f"{source_format!r}: {path}"
                )

            # Force decoding before the file handle is closed. Converting or
            # copying also detaches the returned image from the input file.
            opened_image.load()

            if opened_image.mode == "RGB":
                image = opened_image.copy()
            else:
                image = opened_image.convert("RGB")

    except ValueError:
        raise
    except UnidentifiedImageError as error:
        raise RuntimeError(
            f"Pillow could not identify the frame image: {path}"
        ) from error
    except (OSError, SyntaxError) as error:
        raise RuntimeError(
            f"Failed to decode frame image {path}: {error}"
        ) from error

    _validate_loaded_frame(image, source_path=path)
    return image


def save_debug_frame(
    image: Image.Image,
    output_path: str | Path,
) -> None:
    """Save an RGB copy of a loaded frame for local debugging."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    debug_image = (
        image.copy()
        if image.mode == "RGB"
        else image.convert("RGB")
    )

    _validate_loaded_frame(debug_image, source_path=path)
    debug_image.save(path, format="PNG")


def main() -> None:
    """Run a standalone smoke test on one FRAME-track PNG."""

    project_dir = Path(__file__).resolve().parent

    image_path = (
        project_dir
        / "test"
        / "input"
        / "interface_1"
        / "frames"
        / "q0001.png"
    ).resolve()

    debug_output_value = os.environ.get(
        "IMAGE_UTILS_TEST_OUTPUT",
        "",
    ).strip()

    print("=" * 80)
    print("ORena FOCUS image-utils smoke test")
    print("=" * 80)
    print(f"Image: {image_path}")

    image = load_frame(image_path)

    print("\nDecoded frame")
    print(f"  Mode: {image.mode}")
    print(f"  Width: {image.width}")
    print(f"  Height: {image.height}")
    print(f"  Size: {image.size}")
    print(
        "  Aspect ratio: "
        f"{image.width / image.height:.6f}"
    )

    if debug_output_value:
        debug_output_path = Path(
            debug_output_value
        ).expanduser().resolve()

        save_debug_frame(
            image=image,
            output_path=debug_output_path,
        )

        reloaded = load_frame(debug_output_path)

        if reloaded.size != image.size:
            raise RuntimeError(
                "Debug round-trip changed the image dimensions: "
                f"{image.size} -> {reloaded.size}."
            )

        print("\nDebug round-trip passed.")
        print(f"  Saved to: {debug_output_path}")

    print("\nAll image-utils smoke tests passed.")


if __name__ == "__main__":
    main()