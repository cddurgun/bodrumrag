"""
PDF text extractor with OCR fallback.

Strategy:
  1. Try ``pdfplumber`` to extract embedded text.
  2. If a page yields little/no text, fall back to OCR via
     ``pytesseract`` + ``pdf2image``.
  3. Concatenate all pages into a single string per document.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum characters per page to consider text extraction successful
_MIN_CHARS_PER_PAGE = 30


def _extract_with_pdfplumber(pdf_path: Path) -> Optional[str]:
    """
    Extract text using pdfplumber (works for text-based PDFs).

    Returns:
        The full text if extraction yields meaningful content, else None.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber not installed – cannot extract text")
        return None

    try:
        all_text: list[str] = []
        empty_pages = 0

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if len(text.strip()) < _MIN_CHARS_PER_PAGE:
                    empty_pages += 1
                all_text.append(text)

        combined = "\n\n".join(all_text).strip()

        # If more than half the pages are empty, text extraction likely failed
        total_pages = len(all_text)
        if total_pages > 0 and empty_pages / total_pages > 0.5:
            logger.info(
                "%s: %d/%d pages empty – likely scanned, will try OCR",
                pdf_path.name,
                empty_pages,
                total_pages,
            )
            if len(combined) < _MIN_CHARS_PER_PAGE * total_pages * 0.3:
                return None  # Signal to try OCR

        return combined if combined else None

    except Exception as exc:
        logger.warning("pdfplumber failed on %s: %s", pdf_path.name, exc)
        return None


def _extract_with_ocr(pdf_path: Path) -> Optional[str]:
    """
    Extract text via OCR (pytesseract + pdf2image).

    Requires system packages:
      - poppler (for pdf2image): ``brew install poppler``
      - tesseract + Turkish data: ``brew install tesseract tesseract-lang``
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError as exc:
        logger.warning(
            "OCR dependencies not available (%s) – skipping OCR for %s",
            exc,
            pdf_path.name,
        )
        return None

    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path, dpi=300)
        all_text: list[str] = []

        for i, img in enumerate(images):
            # Use Turkish + English language packs
            text = pytesseract.image_to_string(img, lang="tur+eng")
            all_text.append(text.strip())
            logger.debug(
                "OCR page %d/%d of %s: %d chars",
                i + 1,
                len(images),
                pdf_path.name,
                len(text),
            )

        combined = "\n\n".join(all_text).strip()
        return combined if combined else None

    except Exception as exc:
        logger.warning("OCR failed on %s: %s", pdf_path.name, exc)
        return None


def _extract_from_image(file_path: Path) -> Optional[str]:
    """
    If the 'PDF' is actually a raw image (e.g. JPEG/TIFF downloaded from Drive),
    extract text directly using pytesseract without pdf2image.
    """
    try:
        from PIL import Image, UnidentifiedImageError
        import pytesseract
    except ImportError:
        return None

    try:
        # Try to open it as a raw image (will fail if it's a real PDF)
        with Image.open(file_path) as img:
            text = pytesseract.image_to_string(img, lang="tur+eng")
            return text.strip() if text.strip() else None
    except UnidentifiedImageError:
        # Not a recognized image format (likely a real PDF)
        return None
    except Exception as exc:
        logger.debug("Raw image OCR failed on %s: %s", file_path.name, exc)
        return None


def extract_text(pdf_path: Path) -> str:
    """
    Extract text from a PDF file.

    Tries pdfplumber first, falls back to OCR if needed.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text (may be empty if all methods fail).
    """
    logger.info("Extracting text from: %s", pdf_path.name)

    # 0. Check if it's actually just an image renamed as a PDF
    image_text = _extract_from_image(pdf_path)
    if image_text:
        logger.info(
            "Direct image OCR extracted %d chars from %s", len(image_text), pdf_path.name
        )
        return image_text

    # 1. Try text-based extraction first
    text = _extract_with_pdfplumber(pdf_path)

    if text and len(text.strip()) > _MIN_CHARS_PER_PAGE:
        logger.info(
            "pdfplumber extracted %d chars from %s",
            len(text),
            pdf_path.name,
        )
        return text

    # Fall back to OCR
    logger.info("Trying OCR for %s", pdf_path.name)
    text = _extract_with_ocr(pdf_path)

    if text:
        logger.info(
            "OCR extracted %d chars from %s", len(text), pdf_path.name
        )
        return text

    logger.warning("Could not extract text from %s", pdf_path.name)
    return ""


def extract_all(
    pdf_dir: Path | str,
) -> List[Tuple[str, str]]:
    """
    Extract text from all PDF files in a directory.

    Args:
        pdf_dir: Directory containing PDF files.

    Returns:
        List of (filename, extracted_text) tuples.
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        logger.warning("PDF directory does not exist: %s", pdf_dir)
        return []

    logger.info("Extracting from directory: %s", pdf_dir)

    all_files = []
    for ext in ("*.pdf", "*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"):
        all_files.extend(pdf_dir.glob(ext))
        # Keep case insensitivity where possible extensions might be uppercase
        all_files.extend(pdf_dir.glob(ext.upper()))
    
    # deduplicate and sort
    pdf_files = sorted(list(set(all_files)))

    if not pdf_files:
        logger.warning("No PDF/Image files found in %s", pdf_dir)
        return []

    logger.info("Found %d document files in %s", len(pdf_files), pdf_dir)

    results: List[Tuple[str, str]] = []
    for pdf_file in pdf_files:
        text = extract_text(pdf_file)
        if text.strip():
            results.append((pdf_file.stem, text))
        else:
            logger.warning("No text extracted from %s – skipping", pdf_file.name)

    logger.info(
        "Extracted text from %d/%d PDFs", len(results), len(pdf_files)
    )
    return results


# ── CLI entry point ───────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    pdf_dir = sys.argv[1] if len(sys.argv) > 1 else "data/pdfs"
    results = extract_all(pdf_dir)
    for name, text in results:
        print(f"\n{'='*60}")
        print(f"File: {name}")
        print(f"Text length: {len(text)} characters")
        print(text[:500])
        print(f"{'='*60}")
