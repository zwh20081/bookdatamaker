"""Image writing helpers with optional AVIF multi-threaded compression."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal

from PIL import Image

ImageFormat = Literal["avif", "jpeg", "png"]


def is_avif_supported() -> bool:
    """Return whether current Pillow build can encode AVIF images."""
    try:
        extensions = Image.registered_extensions()
    except Exception:
        extensions = {}

    if "AVIF" in Image.SAVE:
        return True
    return extensions.get(".avif") == "AVIF"


def _normalize_format(image_format: str) -> ImageFormat:
    fmt = str(image_format).strip().lower()
    if fmt in {"avif", "jpeg", "png"}:
        return fmt  # type: ignore[return-value]
    raise ValueError(f"Unsupported image format: {image_format}")


def extension_for_format(image_format: str) -> str:
    """Return filename extension without dot for configured image format."""
    fmt = _normalize_format(image_format)
    if fmt == "jpeg":
        return "jpg"
    return fmt


def build_save_path(base_path: Path, image_format: str) -> Path:
    """Build output path with the configured image format extension."""
    ext = extension_for_format(image_format)
    if base_path.suffix:
        return base_path.with_suffix(f".{ext}")
    return base_path.parent / f"{base_path.name}.{ext}"


def _save_kwargs(
    image_format: ImageFormat,
    *,
    quality: int,
    avif_speed: int,
    avif_max_threads: int,
) -> Dict[str, Any]:
    if image_format == "avif":
        return {
            "format": "AVIF",
            "quality": quality,
            "speed": avif_speed,
            "max_threads": avif_max_threads,
            "autotiling": True,
        }

    if image_format == "jpeg":
        return {
            "format": "JPEG",
            "quality": quality,
        }

    return {
        "format": "PNG",
        "compress_level": 6,
    }


def _normalize_image_for_format(image: Image.Image, image_format: ImageFormat) -> Image.Image:
    if image_format == "jpeg":
        if image.mode != "RGB":
            return image.convert("RGB")
        return image

    if image_format == "avif":
        if image.mode not in {"RGB", "RGBA"}:
            return image.convert("RGB")
        return image

    return image


def save_image(
    image: Image.Image,
    path_without_ext: Path,
    *,
    image_format: str,
    quality: int,
    avif_speed: int = 6,
    avif_max_threads: int = 4,
) -> Path:
    """Save image to configured format and return actual output path."""
    fmt = _normalize_format(image_format)
    out_path = build_save_path(path_without_ext, fmt)

    kwargs = _save_kwargs(
        fmt,
        quality=quality,
        avif_speed=avif_speed,
        avif_max_threads=avif_max_threads,
    )
    normalized = _normalize_image_for_format(image, fmt)
    normalized.save(out_path, **kwargs)
    return out_path


def save_page_image(
    image: Image.Image,
    path_without_ext: Path,
    *,
    image_format: str,
    quality: int,
    avif_speed: int = 6,
    avif_max_threads: int = 4,
) -> Path:
    """Save full-page image with configured format/compression settings."""
    return save_image(
        image,
        path_without_ext,
        image_format=image_format,
        quality=quality,
        avif_speed=avif_speed,
        avif_max_threads=avif_max_threads,
    )


def save_cropped_image(
    image: Image.Image,
    path_without_ext: Path,
    *,
    image_format: str,
    quality: int,
    avif_speed: int = 6,
    avif_max_threads: int = 4,
) -> Path:
    """Save cropped figure image with configured format/compression settings."""
    return save_image(
        image,
        path_without_ext,
        image_format=image_format,
        quality=quality,
        avif_speed=avif_speed,
        avif_max_threads=avif_max_threads,
    )


def save_overlay_image(
    image: Image.Image,
    path_without_ext: Path,
    *,
    image_format: str,
    quality: int,
    avif_speed: int = 6,
    avif_max_threads: int = 4,
) -> Path:
    """Save OCR overlay image with configured format/compression settings."""
    return save_image(
        image,
        path_without_ext,
        image_format=image_format,
        quality=quality,
        avif_speed=avif_speed,
        avif_max_threads=avif_max_threads,
    )
