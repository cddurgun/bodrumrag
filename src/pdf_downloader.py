"""
Google Drive PDF downloader.

Downloads all plan-notu documents linked from the Bodrum Mimarlar
Odası plan-notları page.

Google Drive direct-download URL pattern:
    https://drive.google.com/uc?export=download&id=<FILE_ID>

For large files Google shows a virus-scan confirmation page;
this module handles that automatically.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from src.config import settings
from src.scraper import DocumentLink, scrape

logger = logging.getLogger(__name__)

# Where downloaded PDFs are stored
_PDF_DIR_NAME = "pdfs"

# Google Drive direct-download endpoint
_GD_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id={file_id}"

# Regex to extract file ID from various Google Drive URL formats
_GD_FILE_ID_RE = re.compile(
    r"(?:drive\.google\.com/file/d/|drive\.google\.com/open\?id=|"
    r"docs\.google\.com/.*?/d/)([a-zA-Z0-9_-]+)"
)


def _extract_file_id(url: str) -> Optional[str]:
    """Extract the Google Drive file ID from a sharing URL."""
    match = _GD_FILE_ID_RE.search(url)
    return match.group(1) if match else None


def _sanitise_filename(title: str) -> str:
    """
    Turn a document title into a safe filesystem name.

    Keeps Turkish characters intact, replaces problematic chars.
    """
    # Replace characters not allowed in filenames
    name = re.sub(r'[<>:"/\\|?*]', "_", title)
    # Collapse multiple underscores / spaces
    name = re.sub(r"[_\s]+", "_", name).strip("_")
    # Limit length
    if len(name) > 120:
        name = name[:120]
    return name


def _download_gdrive_file(
    file_id: str,
    dest_path: Path,
    timeout: int = 60,
) -> bool:
    """
    Download a single file from Google Drive.

    Handles the virus-scan confirmation redirect for large files.

    Returns:
        True if download succeeded, False otherwise.
    """
    session = requests.Session()
    url = _GD_DOWNLOAD_URL.format(file_id=file_id)

    try:
        response = session.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        # Check if Google is showing a confirmation page
        # (happens for files > ~100MB or flagged files)
        if b"download_warning" in response.content[:5000]:
            # Extract the confirmation token
            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    url = f"{url}&confirm={value}"
                    response = session.get(url, stream=True, timeout=timeout)
                    response.raise_for_status()
                    break

        # Check content type – skip if it's an HTML error page
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type and len(response.content) < 10000:
            # Likely a "can't download" page or permission error
            # Try to extract a confirm token from the HTML
            html_content = response.content.decode("utf-8", errors="ignore")

            # Look for confirm download link in the HTML
            confirm_match = re.search(
                r'href="(/uc\?export=download[^"]*confirm=[^"]*)"',
                html_content,
            )
            if confirm_match:
                confirm_url = "https://drive.google.com" + confirm_match.group(1).replace("&amp;", "&")
                response = session.get(confirm_url, stream=True, timeout=timeout)
                response.raise_for_status()
            else:
                # Check for the UUID/action form style
                confirm_match = re.search(
                    r'name="uuid" value="([^"]+)"',
                    html_content,
                )
                if confirm_match:
                    uuid_val = confirm_match.group(1)
                    post_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t&uuid={uuid_val}"
                    response = session.get(post_url, stream=True, timeout=timeout)
                    response.raise_for_status()
                else:
                    logger.warning(
                        "Could not resolve download for file_id=%s (HTML response)",
                        file_id,
                    )
                    # Save anyway – might still be useful
                    pass

        # Write the file
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        size_kb = dest_path.stat().st_size / 1024
        logger.info("Downloaded %s (%.1f KB)", dest_path.name, size_kb)

        # Validate: skip tiny files (likely error pages)
        if dest_path.stat().st_size < 1000:
            logger.warning(
                "File %s is suspiciously small (%.1f KB) – may be an error page",
                dest_path.name,
                size_kb,
            )
            return False

        return True

    except requests.RequestException as exc:
        logger.error("Failed to download file_id=%s: %s", file_id, exc)
        return False


def download_all(
    links: List[DocumentLink] | None = None,
    output_dir: str | None = None,
    delay: float = 1.0,
) -> List[Tuple[str, Path]]:
    """
    Download all plan-notu PDFs from Google Drive.

    Args:
        links:      List of DocumentLink objects. If None, scrapes the page.
        output_dir: Destination directory. Defaults to data/pdfs/.
        delay:      Seconds to wait between downloads (rate limiting).

    Returns:
        List of (title, local_path) tuples for successfully downloaded files.
    """
    if links is None:
        page = scrape()
        links = page.document_links

    pdf_dir = Path(output_dir or str(Path(settings.data_dir) / _PDF_DIR_NAME))
    pdf_dir.mkdir(parents=True, exist_ok=True)

    downloaded: List[Tuple[str, Path]] = []
    skipped = 0
    failed = 0

    for i, link in enumerate(links, 1):
        file_id = _extract_file_id(link.url)
        if not file_id:
            logger.warning(
                "[%d/%d] Could not extract file ID from: %s",
                i, len(links), link.url,
            )
            skipped += 1
            continue

        filename = _sanitise_filename(link.title)
        # We don't know the extension – try .pdf, could be image
        dest = pdf_dir / f"{filename}.pdf"

        # Skip if already downloaded
        if dest.exists() and dest.stat().st_size > 1000:
            logger.info(
                "[%d/%d] Already exists: %s", i, len(links), dest.name
            )
            downloaded.append((link.title, dest))
            continue

        logger.info(
            "[%d/%d] Downloading: %s", i, len(links), link.title
        )

        ok = _download_gdrive_file(file_id, dest)
        if ok:
            downloaded.append((link.title, dest))
        else:
            failed += 1

        # Rate limiting
        if i < len(links):
            time.sleep(delay)

    logger.info(
        "Download complete: %d succeeded, %d skipped, %d failed (out of %d)",
        len(downloaded),
        skipped,
        failed,
        len(links),
    )
    return downloaded


# ── CLI entry point ───────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    results = download_all()
    print(f"\n{'='*60}")
    print(f"Downloaded {len(results)} files:")
    for title, path in results:
        size = path.stat().st_size / 1024
        print(f"  • {title}  →  {path.name} ({size:.1f} KB)")
    print(f"{'='*60}")
