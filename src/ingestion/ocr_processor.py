"""OCR processor backed by AWS Textract.

Handles:
    - Single image files (PNG, JPG, TIFF, etc.) via detect_document_text
    - Individual pages of scanned PDF files (PDF page -> image -> Textract)
"""
from __future__ import annotations

import io
from typing import TYPE_CHECKING

import boto3
from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings


class OCRProcessor:
    """Wrapper around AWS Textract for text extraction."""

    def __init__(self, settings: "Settings") -> None:
        """Create a Textract client with project AWS credentials and region."""
        self._textract = boto3.client(
            "textract",
            region_name=settings.aws_default_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image file using Textract OCR.

        Args:
            image_path: Path to the image file.

        Returns:
            Extracted text string (empty string on failure).
        """
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            return self._extract_text_from_bytes(image_bytes)
        except Exception as exc:
            logger.error(f"OCR failed on image '{image_path}': {exc}")
            return ""

    def extract_text_from_pdf_page(self, pdf_path: str, page_number: int) -> str:
        """Extract text from a single PDF page using Textract.

        Textract synchronous API does not accept local PDF bytes directly,
        so we render the page to an image and send image bytes.

        Args:
            pdf_path: Path to the PDF file.
            page_number: 0-based page index.

        Returns:
            Extracted text string (empty string on failure).
        """
        try:
            from pdf2image import convert_from_path

            # convert_from_path uses 1-based page numbering
            images = convert_from_path(
                pdf_path,
                first_page=page_number + 1,
                last_page=page_number + 1,
                dpi=300,  # Higher DPI → better OCR accuracy
            )
            if not images:
                return ""

            # Convert PIL image to bytes for Textract
            buffer = io.BytesIO()
            images[0].save(buffer, format="PNG")
            return self._extract_text_from_bytes(buffer.getvalue())
        except Exception as exc:
            logger.error(
                f"OCR failed on PDF '{pdf_path}', page {page_number}: {exc}"
            )
            return ""

    def _extract_text_from_bytes(self, document_bytes: bytes) -> str:
        """Call Textract detect_document_text and assemble lines in order."""
        response = self._textract.detect_document_text(
            Document={"Bytes": document_bytes}
        )
        lines = [
            block["Text"]
            for block in response.get("Blocks", [])
            if block.get("BlockType") == "LINE" and block.get("Text")
        ]
        return "\n".join(lines).strip()

    def is_pdf_scanned(self, pdf_path: str, sample_pages: int = 3) -> bool:
        """Heuristic: returns True if the PDF is likely a scanned document
        with little extractable text.

        Args:
            pdf_path: Path to the PDF file.
            sample_pages: How many pages to test (from the start).
        """
        try:
            from pypdf import PdfReader

            reader = PdfReader(pdf_path)
            pages_to_check = min(sample_pages, len(reader.pages))
            if pages_to_check == 0:
                return False

            total_chars = sum(
                len((reader.pages[i].extract_text() or "").strip())
                for i in range(pages_to_check)
            )
            avg_chars = total_chars / pages_to_check
            return avg_chars < 100
        except Exception as exc:
            logger.warning(f"Could not determine if PDF is scanned: {exc}")
            return False
