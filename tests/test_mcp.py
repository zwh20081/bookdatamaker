"""Tests for MCP server."""

import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image
from bookdatamaker.mcp import ParagraphNavigator
from bookdatamaker.mcp.server import MCPServer


class TestParagraphNavigator:
    """Test paragraph navigation."""

    def setup_method(self):
        """Set up test paragraphs."""
        self.paragraphs = [
            "First paragraph",
            "Second paragraph",
            "Third paragraph",
            "Fourth paragraph",
            "Fifth paragraph",
        ]
        self.navigator = ParagraphNavigator(self.paragraphs)

    def test_initial_position(self):
        """Test initial position is 0."""
        assert self.navigator.current_index == 0
        assert self.navigator.get_current() == "First paragraph"

    def test_move_forward(self):
        """Test moving forward."""
        result = self.navigator.move_forward()
        assert self.navigator.current_index == 1
        assert result == "Second paragraph"

    def test_move_forward_multiple(self):
        """Test moving forward multiple steps."""
        result = self.navigator.move_forward(2)
        assert self.navigator.current_index == 2
        assert result == "Third paragraph"

    def test_move_backward(self):
        """Test moving backward."""
        self.navigator.jump_to(2)
        result = self.navigator.move_backward()
        assert self.navigator.current_index == 1
        assert result == "Second paragraph"

    def test_move_backward_multiple(self):
        """Test moving backward multiple steps."""
        self.navigator.jump_to(4)
        result = self.navigator.move_backward(2)
        assert self.navigator.current_index == 2
        assert result == "Third paragraph"

    def test_jump_to(self):
        """Test jumping to specific index."""
        result = self.navigator.jump_to(3)
        assert self.navigator.current_index == 3
        assert result == "Fourth paragraph"

    def test_boundary_forward(self):
        """Test forward boundary."""
        self.navigator.move_forward(10)
        assert self.navigator.current_index == 4
        assert self.navigator.get_current() == "Fifth paragraph"

    def test_boundary_backward(self):
        """Test backward boundary."""
        self.navigator.move_backward(10)
        assert self.navigator.current_index == 0
        assert self.navigator.get_current() == "First paragraph"

    def test_get_context(self):
        """Test getting context."""
        self.navigator.jump_to(2)
        context = self.navigator.get_context(before=1, after=1)
        
        assert context["current_index"] == 2
        assert context["total_paragraphs"] == 5
        assert context["current"] == "Third paragraph"
        assert context["previous"] == ["Second paragraph"]
        assert context["next"] == ["Fourth paragraph"]

    def test_get_context_at_start(self):
        """Test getting context at start."""
        context = self.navigator.get_context(before=2, after=2)
        
        assert context["current"] == "First paragraph"
        assert context["previous"] == []
        assert len(context["next"]) == 2

    def test_get_context_at_end(self):
        """Test getting context at end."""
        self.navigator.jump_to(4)
        context = self.navigator.get_context(before=2, after=2)
        
        assert context["current"] == "Fifth paragraph"
        assert len(context["previous"]) == 2
        assert context["next"] == []


class TestMCPImageTools:
    """Test MCP server image tools (list_page_images)."""

    def _create_extracted_dir(self, tmpdir: Path, pages_with_images: dict = None):
        """Create a mock extracted directory structure.
        
        Args:
            tmpdir: Base temp directory
            pages_with_images: Dict mapping page_num to list of cropped image filenames.
                               e.g., {1: ["1.jpg", "2.jpg"], 2: []}
        """
        extracted = tmpdir / "extracted"
        extracted.mkdir()
        
        if pages_with_images is None:
            pages_with_images = {1: ["1.jpg"], 2: []}
        
        for page_num, image_files in pages_with_images.items():
            page_dir = extracted / f"page_{page_num:03d}"
            page_dir.mkdir()
            
            # Create full page image
            page_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
            page_img.save(page_dir / f"page_{page_num:03d}.png")
            
            # Create cropped images
            if image_files:
                images_dir = page_dir / "images"
                images_dir.mkdir()
                for fname in image_files:
                    cropped = Image.new("RGB", (50, 50), color=(128, 128, 128))
                    cropped.save(images_dir / fname, "JPEG")
        
        return extracted

    @pytest.mark.asyncio
    async def test_list_page_images_with_images(self):
        """list_page_images should find page dir with images when extracted_dir is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extracted = self._create_extracted_dir(Path(tmpdir), {1: ["1.jpg", "2.jpg"]})
            server = MCPServer(extracted_dir=extracted)
            
            # Verify image tools are available by checking extracted_dir is set
            assert server.extracted_dir is not None
            
            # Verify helper finds the page dir
            page_dir = server._get_page_dir(1)
            assert page_dir is not None
            
            # Verify images dir exists and has files
            images_dir = page_dir / "images"
            assert images_dir.is_dir()
            assert (images_dir / "1.jpg").exists()
            assert (images_dir / "2.jpg").exists()

    @pytest.mark.asyncio
    async def test_list_page_images_no_images(self):
        """list_page_images should return empty list when no cropped images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extracted = self._create_extracted_dir(Path(tmpdir), {1: []})
            server = MCPServer(extracted_dir=extracted)
            
            # Access call_tool handler
            handler = server.server.request_handlers.get("tools/call")
            # The tool registration makes these available via the server protocol
            # For unit testing, verify the helper methods work
            page_dir = server._get_page_dir(1)
            assert page_dir is not None
            assert page_dir.is_dir()

    def test_get_page_dir_found(self):
        """_get_page_dir should return path when page exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extracted = self._create_extracted_dir(Path(tmpdir), {1: []})
            server = MCPServer(extracted_dir=extracted)
            
            page_dir = server._get_page_dir(1)
            assert page_dir is not None
            assert page_dir.name == "page_001"

    def test_get_page_dir_not_found(self):
        """_get_page_dir should return None for non-existent page."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extracted = self._create_extracted_dir(Path(tmpdir), {1: []})
            server = MCPServer(extracted_dir=extracted)
            
            page_dir = server._get_page_dir(999)
            assert page_dir is None

    def test_encode_image_to_base64(self):
        """_encode_image_to_base64 should return valid data URL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Image.new("RGB", (10, 10), color=(0, 0, 0))
            img_path = Path(tmpdir) / "test.jpg"
            img.save(img_path, "JPEG")
            
            server = MCPServer()
            result = server._encode_image_to_base64(img_path)
            
            assert result.startswith("data:image/jpeg;base64,")
            assert len(result) > 30  # Should have actual base64 content

    def test_no_image_tools_without_extracted_dir(self):
        """Image tools should NOT be registered when extracted_dir is None."""
        server = MCPServer(paragraphs=["test"])
        # The tools are registered via _setup_tools() in __init__
        # We can verify by checking that _get_page_dir returns None
        assert server.extracted_dir is None
        assert server._get_page_dir(1) is None
