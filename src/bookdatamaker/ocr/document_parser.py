"""Document parser for PDF and EPUB files."""

import io
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image


class DocumentParser:
    """Parse PDF and EPUB documents into images or text."""

    @staticmethod
    def is_pdf(file_path: Path) -> bool:
        """Check if file is PDF."""
        return file_path.suffix.lower() == ".pdf"

    @staticmethod
    def is_epub(file_path: Path) -> bool:
        """Check if file is EPUB."""
        return file_path.suffix.lower() == ".epub"

    @staticmethod
    def parse_pdf_to_images(pdf_path: Path, dpi: int = 200) -> List[Tuple[int, Image.Image]]:
        """Convert PDF pages to images.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for rendering (default: 200)

        Returns:
            List of tuples (page_number, image)
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            images = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                # Render page to pixmap
                pix = page.get_pixmap(dpi=dpi)
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append((page_num + 1, img))

            doc.close()
            return images

        except ImportError:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF parsing. "
                "Install with: pip install pymupdf"
            )

    @staticmethod
    def parse_pdf_text(pdf_path: Path) -> List[Tuple[int, str]]:
        """Extract text directly from PDF (if available).

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of tuples (page_number, text)
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            pages_text = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                pages_text.append((page_num + 1, text))

            doc.close()
            return pages_text

        except ImportError:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF parsing. "
                "Install with: pip install pymupdf"
            )
    
    @staticmethod
    def parse_pdf_text_and_images(pdf_path: Path, dpi: int = 200) -> List[Tuple[int, tuple[str, Image.Image]]]:
        """Extract both text and images from PDF.

        Args:
            pdf_path: Path to PDF file
            dpi: DPI for rendering images

        Returns:
            List of tuples (page_number, (text, image))
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            pages = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                
                # Render page as image
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                pages.append((page_num + 1, (text, img)))

            doc.close()
            return pages

        except ImportError:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF parsing. "
                "Install with: pip install pymupdf"
            )

    @staticmethod
    def parse_epub_to_images(epub_path: Path, width: int = 800) -> List[Tuple[int, Image.Image]]:
        """Convert EPUB pages to images.

        Args:
            epub_path: Path to EPUB file
            width: Image width in pixels

        Returns:
            List of tuples (page_number, image)
        """
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
            from PIL import ImageDraw, ImageFont

            book = epub.read_epub(epub_path)
            images = []
            page_num = 1

            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    # Parse HTML content
                    soup = BeautifulSoup(item.get_content(), "html.parser")
                    text = soup.get_text()

                    # Create simple text image (for OCR simulation)
                    # In production, you might want to render HTML properly
                    img = Image.new("RGB", (width, 1000), color="white")
                    draw = ImageDraw.Draw(img)

                    try:
                        font = ImageFont.truetype("arial.ttf", 16)
                    except:
                        font = ImageFont.load_default()

                    # Simple text rendering
                    y_offset = 10
                    for line in text.split("\n")[:50]:  # Limit lines
                        if line.strip():
                            draw.text((10, y_offset), line[:100], fill="black", font=font)
                            y_offset += 20

                    images.append((page_num, img))
                    page_num += 1

            return images

        except ImportError:
            raise ImportError(
                "ebooklib and beautifulsoup4 are required for EPUB parsing. "
                "Install with: pip install ebooklib beautifulsoup4"
            )

    @staticmethod
    def parse_epub_text(epub_path: Path) -> List[Tuple[int, str]]:
        """Extract text directly from EPUB.

        Args:
            epub_path: Path to EPUB file

        Returns:
            List of tuples (chapter_number, text)
        """
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup

            book = epub.read_epub(epub_path)
            pages_text = []
            page_num = 1

            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), "html.parser")
                    text = soup.get_text()
                    pages_text.append((page_num, text))
                    page_num += 1

            return pages_text

        except ImportError:
            raise ImportError(
                "ebooklib and beautifulsoup4 are required for EPUB parsing. "
                "Install with: pip install ebooklib beautifulsoup4"
            )
    
    @staticmethod
    def parse_epub_text_and_images(epub_path: Path, width: int = 800) -> List[Tuple[int, tuple[str, Image.Image]]]:
        """Extract both text and rendered images from EPUB.

        Args:
            epub_path: Path to EPUB file
            width: Image width in pixels

        Returns:
            List of tuples (chapter_number, (text, image))
        """
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
            from PIL import ImageDraw, ImageFont

            book = epub.read_epub(epub_path)
            pages = []
            page_num = 1

            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    # Extract text
                    soup = BeautifulSoup(item.get_content(), "html.parser")
                    text = soup.get_text()

                    # Create simple text image for consistency
                    img = Image.new("RGB", (width, 1000), color="white")
                    draw = ImageDraw.Draw(img)

                    try:
                        font = ImageFont.truetype("arial.ttf", 16)
                    except:
                        font = ImageFont.load_default()

                    # Draw text on image (simplified rendering)
                    y_offset = 20
                    for line in text[:2000].split("\n"):  # Limit text for rendering
                        if y_offset > 950:
                            break
                        draw.text((20, y_offset), line[:80], fill="black", font=font)
                        y_offset += 20

                    pages.append((page_num, (text, img)))
                    page_num += 1

            return pages

        except ImportError:
            raise ImportError(
                "ebooklib and beautifulsoup4 are required for EPUB parsing. "
                "Install with: pip install ebooklib beautifulsoup4"
            )


def extract_document_pages(
    file_path: Path, prefer_text: bool = False
) -> List[Tuple[int, str | Image.Image]]:
    """Extract pages from document (PDF/EPUB).

    Args:
        file_path: Path to document file
        prefer_text: If True, extract text directly when possible

    Returns:
        List of tuples (page_number, content) where content is text or Image
    """
    parser = DocumentParser()

    if parser.is_pdf(file_path):
        if prefer_text:
            try:
                return parser.parse_pdf_text_and_images(file_path)
            except Exception:
                return parser.parse_pdf_to_images(file_path)
        else:
            return parser.parse_pdf_to_images(file_path)

    elif parser.is_epub(file_path):
        if prefer_text:
            return parser.parse_epub_text_and_images(file_path)
        else:
            return parser.parse_epub_to_images(file_path)

    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
