"""Tests for shared image tool helpers."""

from pathlib import Path

from PIL import Image

from bookdatamaker.ocr.image_writer import is_avif_supported
from bookdatamaker.tools.image_tools import get_image_data_url, list_page_images


def _prepare_extracted_dir(tmp_path: Path) -> Path:
    extracted_dir = tmp_path / "extracted"
    page_dir = extracted_dir / "page_001"
    images_dir = page_dir / "images"
    images_dir.mkdir(parents=True)

    full_page = Image.new("RGB", (20, 20), color=(255, 255, 255))
    full_page.save(page_dir / "page_001.png")

    cropped = Image.new("RGB", (10, 10), color=(128, 128, 128))
    cropped.save(images_dir / "0.jpg")

    if is_avif_supported():
        alt_cropped = Image.new("RGB", (10, 10), color=(64, 64, 64))
        alt_cropped.save(images_dir / "1.avif", format="AVIF", quality=80)

    return extracted_dir


def test_list_page_images_success(tmp_path: Path) -> None:
    extracted_dir = _prepare_extracted_dir(tmp_path)

    result = list_page_images(extracted_dir, page_number=1)

    assert result["ok"] is True
    expected_count = 3 if is_avif_supported() else 2
    assert result["image_count"] == expected_count
    names = [item["name"] for item in result["images"]]
    assert "page_001.png" in names
    assert "images/0.jpg" in names
    if is_avif_supported():
        assert "images/1.avif" in names


def test_list_page_images_not_found(tmp_path: Path) -> None:
    extracted_dir = _prepare_extracted_dir(tmp_path)

    result = list_page_images(extracted_dir, page_number=999)

    assert result["ok"] is False
    assert "not found" in result["error"].lower()


def test_get_image_data_url_success(tmp_path: Path) -> None:
    extracted_dir = _prepare_extracted_dir(tmp_path)

    result = get_image_data_url(extracted_dir, page_number=1, image_name="images/0.jpg")

    assert result["ok"] is True
    assert result["data_url"].startswith("data:image/jpeg;base64,")


def test_get_image_data_url_path_escape_blocked(tmp_path: Path) -> None:
    extracted_dir = _prepare_extracted_dir(tmp_path)

    result = get_image_data_url(extracted_dir, page_number=1, image_name="../outside.jpg")

    assert result["ok"] is False
    assert "invalid image path" in result["error"].lower()
