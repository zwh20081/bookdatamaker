"""Shared image tool helpers for extracted page assets."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image


def _get_page_dir(extracted_dir: Path, page_number: int) -> Optional[Path]:
    """Get page directory for ``page_{N:03d}`` if it exists."""
    page_dir = extracted_dir / f"page_{page_number:03d}"
    if page_dir.is_dir():
        return page_dir
    return None


def list_page_images(extracted_dir: Path, page_number: int) -> Dict[str, Any]:
    """List full-page and cropped image assets for a page."""
    page_dir = _get_page_dir(extracted_dir, page_number)
    if page_dir is None:
        return {
            "ok": False,
            "error": f"Page {page_number} not found",
            "page_number": page_number,
            "images": [],
        }

    available: List[Dict[str, str]] = []

    # Full page image
    for ext in (".avif", ".png", ".jpg", ".jpeg"):
        page_img = page_dir / f"page_{page_number:03d}{ext}"
        if page_img.exists():
            available.append(
                {
                    "name": page_img.name,
                    "type": "full_page",
                    "path": str(page_img.resolve()),
                }
            )
            break

    # Cropped images
    images_subdir = page_dir / "images"
    if images_subdir.is_dir():
        for img_file in sorted(images_subdir.iterdir()):
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".avif"):
                available.append(
                    {
                        "name": f"images/{img_file.name}",
                        "type": "cropped",
                        "path": str(img_file.resolve()),
                    }
                )

    return {
        "ok": True,
        "page_number": page_number,
        "image_count": len(available),
        "images": available,
    }


def get_image_data_url(extracted_dir: Path, page_number: int, image_name: str) -> Dict[str, Any]:
    """Load a page image and return base64 data URL payload."""
    page_dir = _get_page_dir(extracted_dir, page_number)
    if page_dir is None:
        return {
            "ok": False,
            "error": f"Page {page_number} not found",
            "page_number": page_number,
            "image_name": image_name,
        }

    image_path = page_dir / image_name

    # Security: ensure path stays within page_dir
    try:
        image_path.resolve().relative_to(page_dir.resolve())
    except ValueError:
        return {
            "ok": False,
            "error": "Invalid image path",
            "page_number": page_number,
            "image_name": image_name,
        }

    if not image_path.exists():
        return {
            "ok": False,
            "error": f"Image '{image_name}' not found in page {page_number}",
            "page_number": page_number,
            "image_name": image_name,
        }

    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "ok": True,
        "page_number": page_number,
        "image_name": image_name,
        "data_url": f"data:image/jpeg;base64,{b64_str}",
    }
