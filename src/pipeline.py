"""
Indexing pipeline — scrape → download PDFs → extract → clean → chunk → embed → store.

Run as:
    python -m src.pipeline

This will:
  1. Scrape the plan-notları page for document links.
  2. Download all linked PDFs from Google Drive.
  3. Extract text from each PDF (with OCR fallback).
  4. Clean and normalise the extracted text.
  5. Split into token-bounded chunks.
  6. Generate embeddings via NVIDIA NIM.
  7. Build and persist a FAISS index.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.chunker import split_into_chunks, TextChunk
from src.cleaner import clean
from src.config import settings
from src.embedder import embed_texts
from src.pdf_downloader import download_all
from src.pdf_extractor import extract_all
from src.scraper import scrape
from src.vector_db import VectorDB

logger = logging.getLogger(__name__)


def run_pipeline(url: str | None = None, skip_download: bool = False) -> VectorDB:
    """
    Execute the full indexing pipeline.

    Args:
        url:           Override the default scraping URL.
        skip_download: If True, skip PDF download (use existing files).

    Returns:
        A populated and persisted ``VectorDB`` instance.
    """
    # ─────────────────────────────────────────────────────────
    # 1 ▸ Scrape the page for links
    # ─────────────────────────────────────────────────────────
    logger.info("═══ Step 1/7: Scraping page ═══")
    page = scrape(url)
    logger.info(
        "Found %d document links on the page", len(page.document_links)
    )

    # ─────────────────────────────────────────────────────────
    # 2 ▸ Download PDFs from Google Drive
    # ─────────────────────────────────────────────────────────
    pdf_dir = Path(settings.data_dir) / "pdfs"

    if skip_download:
        logger.info("═══ Step 2/7: Skipping download (--skip-download) ═══")
    else:
        logger.info("═══ Step 2/7: Downloading PDFs ═══")
        downloaded = download_all(
            links=page.document_links,
            output_dir=str(pdf_dir),
            delay=1.0,
        )
        logger.info("Downloaded %d files", len(downloaded))

    # ─────────────────────────────────────────────────────────
    # 3 ▸ Extract text from PDFs
    # ─────────────────────────────────────────────────────────
    logger.info("═══ Step 3/7: Extracting text from PDFs ═══")
    pdf_texts = extract_all(pdf_dir)
    logger.info("Extracted text from %d PDFs", len(pdf_texts))

    # ─────────────────────────────────────────────────────────
    # 4 ▸ Clean all extracted text
    # ─────────────────────────────────────────────────────────
    logger.info("═══ Step 4/7: Cleaning text ═══")

    all_chunks: list[TextChunk] = []
    source_url = url or page.source_url

    # Add the web page text itself
    page_text = page.raw_text
    link_text_parts: list[str] = []
    for link in page.document_links:
        link_text_parts.append(f"{link.title}")
    if link_text_parts:
        page_text += "\n\n── Plan Notları Belge Listesi ──\n\n" + "\n".join(link_text_parts)

    cleaned_page = clean(page_text)
    if cleaned_page.strip():
        page_chunks = split_into_chunks(cleaned_page, source_url)
        all_chunks.extend(page_chunks)
        logger.info("Page text → %d chunks", len(page_chunks))

    # Process each PDF's text
    for pdf_name, pdf_text in pdf_texts:
        cleaned = clean(pdf_text)
        if not cleaned.strip():
            logger.warning("PDF '%s' produced no text after cleaning", pdf_name)
            continue

        pdf_source = f"{source_url}#pdf:{pdf_name}"
        chunks = split_into_chunks(cleaned, pdf_source)
        all_chunks.extend(chunks)
        logger.info("PDF '%s' → %d chunks (%d chars)", pdf_name, len(chunks), len(cleaned))

    # ─────────────────────────────────────────────────────────
    # 5 ▸ Re-index chunks sequentially
    # ─────────────────────────────────────────────────────────
    logger.info("═══ Step 5/7: Indexing chunks ═══")
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_index = i

    if not all_chunks:
        logger.error("No chunks produced – aborting pipeline")
        sys.exit(1)

    logger.info("Total: %d chunks ready for embedding", len(all_chunks))

    # ─────────────────────────────────────────────────────────
    # 6 ▸ Embed
    # ─────────────────────────────────────────────────────────
    logger.info("═══ Step 6/7: Embedding ═══")
    texts = [c.text for c in all_chunks]
    embeddings = embed_texts(texts, input_type="passage")
    logger.info("Embedding matrix shape: %s", embeddings.shape)

    # ─────────────────────────────────────────────────────────
    # 7 ▸ Store
    # ─────────────────────────────────────────────────────────
    logger.info("═══ Step 7/7: Building vector DB ═══")
    db = VectorDB()
    db.build(embeddings, all_chunks)
    db.save()
    logger.info("Pipeline complete ✓  Index saved with %d vectors", db.size)

    return db


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the indexing pipeline")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip PDF download, use existing files in data/pdfs/",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    run_pipeline(skip_download=args.skip_download)
