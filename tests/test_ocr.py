"""Tests for OCR extractor."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
from bookdatamaker.ocr import OCRExtractor


class TestOCRExtractor:
    """Test OCR extraction functionality."""

    def test_split_paragraphs_double_newline(self):
        """Test splitting text by double newlines."""
        extractor = OCRExtractor("fake-key")
        text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        
        paragraphs = extractor.split_into_paragraphs(text)
        
        assert len(paragraphs) == 3
        assert paragraphs[0] == "Paragraph 1"
        assert paragraphs[1] == "Paragraph 2"
        assert paragraphs[2] == "Paragraph 3"

    def test_split_paragraphs_single_newline(self):
        """Test splitting text by single newlines when no double newlines."""
        extractor = OCRExtractor("fake-key")
        text = "Line 1\nLine 2\nLine 3"
        
        paragraphs = extractor.split_into_paragraphs(text)
        
        assert len(paragraphs) == 3
        assert paragraphs[0] == "Line 1"

    def test_split_paragraphs_empty(self):
        """Test splitting empty text."""
        extractor = OCRExtractor("fake-key")
        text = ""
        
        paragraphs = extractor.split_into_paragraphs(text)
        
        assert len(paragraphs) == 0


class TestOCRVersionConfig:
    """Test OCR version configuration."""

    def test_default_ocr_version_is_2(self):
        """Default OCR version should be 2."""
        extractor = OCRExtractor("fake-key", skip_model_load=True)
        assert extractor.ocr_version == "2"
        assert extractor.default_model_name == "deepseek-ai/DeepSeek-OCR-2"
        assert extractor.local_model_path == "deepseek-ai/DeepSeek-OCR-2"
        assert extractor.image_size == 768
        assert extractor.api_ngram_size == 20
        assert extractor.use_flash_attn is True

    def test_ocr_version_1(self):
        """OCR version 1 should use OCR-1 defaults."""
        extractor = OCRExtractor("fake-key", skip_model_load=True, ocr_version="1")
        assert extractor.ocr_version == "1"
        assert extractor.default_model_name == "deepseek-ai/DeepSeek-OCR"
        assert extractor.local_model_path == "deepseek-ai/DeepSeek-OCR"
        assert extractor.image_size == 640
        assert extractor.api_ngram_size == 30
        assert extractor.use_flash_attn is False

    def test_local_model_path_overrides_version_default(self):
        """Explicit local_model_path should override version default."""
        extractor = OCRExtractor(
            "fake-key",
            skip_model_load=True,
            ocr_version="2",
            local_model_path="/custom/model/path",
        )
        assert extractor.local_model_path == "/custom/model/path"
        # Other version-specific params should still reflect OCR-2
        assert extractor.default_model_name == "deepseek-ai/DeepSeek-OCR-2"
        assert extractor.image_size == 768

    def test_local_model_path_none_uses_version_default(self):
        """No local_model_path should use version default."""
        ext_v1 = OCRExtractor("fake-key", skip_model_load=True, ocr_version="1")
        ext_v2 = OCRExtractor("fake-key", skip_model_load=True, ocr_version="2")
        assert ext_v1.local_model_path == "deepseek-ai/DeepSeek-OCR"
        assert ext_v2.local_model_path == "deepseek-ai/DeepSeek-OCR-2"

    def test_api_mode_uses_version_model_name(self):
        """API mode should create client regardless of OCR version."""
        ext_v1 = OCRExtractor("fake-key", mode="api", ocr_version="1")
        ext_v2 = OCRExtractor("fake-key", mode="api", ocr_version="2")
        assert ext_v1.default_model_name == "deepseek-ai/DeepSeek-OCR"
        assert ext_v2.default_model_name == "deepseek-ai/DeepSeek-OCR-2"
        assert ext_v1.client is not None
        assert ext_v2.client is not None


class TestPostProcessOCROutput:
    """Test OCR post-processing: image cropping from ref/det annotations."""

    def _make_extractor(self):
        return OCRExtractor("fake-key", skip_model_load=True)

    def _make_image(self, width=999, height=999):
        """Create a test image."""
        return Image.new("RGB", (width, height), color=(200, 200, 200))

    def test_no_ref_det_returns_cleaned_text(self):
        """Text without ref/det annotations should be returned with empty lines removed."""
        ext = self._make_extractor()
        text = "Line one\n\nLine two\nLine three"
        with tempfile.TemporaryDirectory() as tmpdir:
            page_dir = Path(tmpdir) / "page_001"
            page_dir.mkdir()
            image = self._make_image()
            result = ext._post_process_ocr_output(text, image, page_dir)
        assert "Line one" in result
        assert "Line two" in result
        assert "Line three" in result
        # Empty lines should be removed
        assert "\n\n" not in result

    def test_image_ref_crops_and_replaces(self):
        """Image ref/det should crop the region and replace with markdown link."""
        ext = self._make_extractor()
        text = "Some text\n<|ref|>image<|/ref|><|det|>[[100,100],[500,500]]<|/det|>\nMore text"
        with tempfile.TemporaryDirectory() as tmpdir:
            page_dir = Path(tmpdir) / "page_001"
            page_dir.mkdir()
            image = self._make_image(999, 999)
            result = ext._post_process_ocr_output(text, image, page_dir)
        
            # Should have markdown image link
            assert "![](images/1.jpg)" in result
            # Should NOT have ref/det tags
            assert "<|ref|>" not in result
            assert "<|det|>" not in result
            # Image file should exist
            assert (page_dir / "images" / "1.jpg").exists()

    def test_non_image_ref_removed(self):
        """Non-image ref/det (e.g., title) should be removed entirely."""
        ext = self._make_extractor()
        text = "Hello <|ref|>title<|/ref|><|det|>[[0,0],[100,50]]<|/det|> world"
        with tempfile.TemporaryDirectory() as tmpdir:
            page_dir = Path(tmpdir) / "page_001"
            page_dir.mkdir()
            image = self._make_image()
            result = ext._post_process_ocr_output(text, image, page_dir)
        
            assert "<|ref|>" not in result
            # images dir should NOT exist (no image refs)
            assert not (page_dir / "images").exists()

    def test_multiple_image_refs(self):
        """Multiple image refs should produce sequential numbered files."""
        ext = self._make_extractor()
        text = (
            "<|ref|>image<|/ref|><|det|>[[0,0],[499,499]]<|/det|>\n"
            "middle\n"
            "<|ref|>image<|/ref|><|det|>[[500,500],[999,999]]<|/det|>"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            page_dir = Path(tmpdir) / "page_001"
            page_dir.mkdir()
            image = self._make_image(999, 999)
            result = ext._post_process_ocr_output(text, image, page_dir)
        
            assert "![](images/1.jpg)" in result
            assert "![](images/2.jpg)" in result
            assert (page_dir / "images" / "1.jpg").exists()
            assert (page_dir / "images" / "2.jpg").exists()

    def test_coordinate_scaling(self):
        """Coordinates should scale from 0-999 range to actual image dimensions."""
        ext = self._make_extractor()
        # Image is 2000x1000, coords are 0-999
        text = "<|ref|>image<|/ref|><|det|>[[0,0],[999,999]]<|/det|>"
        with tempfile.TemporaryDirectory() as tmpdir:
            page_dir = Path(tmpdir) / "page_001"
            page_dir.mkdir()
            image = self._make_image(2000, 1000)
            result = ext._post_process_ocr_output(text, image, page_dir)
        
            assert "![](images/1.jpg)" in result
            # Verify cropped image exists and has reasonable dimensions
            with Image.open(page_dir / "images" / "1.jpg") as cropped:
                assert cropped.width > 0
                assert cropped.height > 0

    def test_mixed_refs(self):
        """Mix of image and non-image refs should handle both correctly."""
        ext = self._make_extractor()
        text = (
            "Title: <|ref|>heading<|/ref|><|det|>[[0,0],[100,50]]<|/det|>\n"
            "Figure: <|ref|>image<|/ref|><|det|>[[200,200],[800,800]]<|/det|>\n"
            "Caption: <|ref|>text<|/ref|><|det|>[[0,900],[999,999]]<|/det|>"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            page_dir = Path(tmpdir) / "page_001"
            page_dir.mkdir()
            image = self._make_image(999, 999)
            result = ext._post_process_ocr_output(text, image, page_dir)
        
            # Only 1 image ref → 1 cropped file
            assert "![](images/1.jpg)" in result
            assert (page_dir / "images" / "1.jpg").exists()
            assert not (page_dir / "images" / "2.jpg").exists()
            # No ref/det tags remain
            assert "<|ref|>" not in result
            assert "<|det|>" not in result
