"""
Web scraper for Bodrum Mimarlar Odası plan-notları page.

The page contains:
  • Navigation / header / footer chrome  →  discarded
  • The main content area with a title "PLAN NOTLARI" listing
    Google Drive links to zoning-regulation PDFs.

This module:
  1. Fetches the HTML of the plan-notları page.
  2. Extracts all document links (title + URL).
  3. Extracts all visible text from the main content area.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

import requests
from bs4 import BeautifulSoup, Tag

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DocumentLink:
    """A single plan-notu document reference."""

    title: str
    url: str


@dataclass
class ScrapedPage:
    """Result of scraping the plan-notları page."""

    source_url: str
    page_title: str
    raw_text: str
    document_links: List[DocumentLink]


def fetch_html(url: str | None = None, timeout: int | None = None) -> str:
    """
    Download the HTML content of *url*.

    Returns:
        The decoded HTML string.

    Raises:
        requests.HTTPError: on 4xx / 5xx responses.
        requests.ConnectionError: on network failures.
    """
    url = url or settings.source_url
    timeout = timeout or settings.request_timeout
    logger.info("Fetching %s", url)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"

    logger.info("Fetched %d bytes from %s", len(response.text), url)
    return response.text


def _extract_document_links(soup: BeautifulSoup) -> List[DocumentLink]:
    """
    Pull every ``<a>`` that points to a Google Drive document.
    These are the actual plan-notu PDF/image links.
    """
    links: List[DocumentLink] = []
    seen_urls: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href: str = anchor["href"]
        if "drive.google.com" not in href:
            continue
        if href in seen_urls:
            continue
        seen_urls.add(href)

        title = anchor.get_text(strip=True) or "Unnamed Document"
        links.append(DocumentLink(title=title, url=href))

    logger.info("Extracted %d unique Google-Drive document links", len(links))
    return links


def _extract_main_text(soup: BeautifulSoup) -> str:
    """
    Extract the visible text from the main content section.

    Strategy:
      1. Try to locate the ``<article>`` or ``<main>`` element.
      2. Fall back to ``<div class="entry-content">`` (WordPress).
      3. Last resort: use the full ``<body>``.
    """
    # WordPress sites typically wrap page content in .entry-content
    content_area: Tag | None = (
        soup.find("div", class_="entry-content")
        or soup.find("article")
        or soup.find("main")
        or soup.find("body")
    )

    if content_area is None:
        logger.warning("Could not locate content area – using full body")
        content_area = soup  # type: ignore[assignment]

    # Remove script / style / nav / footer noise before extracting text
    for tag_name in ("script", "style", "nav", "footer", "header", "aside"):
        for tag in content_area.find_all(tag_name):  # type: ignore[union-attr]
            tag.decompose()

    raw = content_area.get_text(separator="\n")  # type: ignore[union-attr]
    # Collapse multiple blank lines
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def scrape(url: str | None = None) -> ScrapedPage:
    """
    High-level entry point: fetch, parse, and return structured data.

    Args:
        url: Override the default Plan Notları URL.

    Returns:
        A ``ScrapedPage`` with raw text and document links.
    """
    html = fetch_html(url)
    soup = BeautifulSoup(html, "lxml")

    page_title = soup.title.string.strip() if soup.title and soup.title.string else "Plan Notları"
    raw_text = _extract_main_text(soup)
    doc_links = _extract_document_links(soup)

    result = ScrapedPage(
        source_url=url or settings.source_url,
        page_title=page_title,
        raw_text=raw_text,
        document_links=doc_links,
    )

    logger.info(
        "Scraped page '%s': %d chars text, %d doc links",
        result.page_title,
        len(result.raw_text),
        len(result.document_links),
    )
    return result


# ── Quick smoke-test when run directly ────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    page = scrape()
    print(f"\n{'='*60}")
    print(f"Title : {page.page_title}")
    print(f"Text  : {len(page.raw_text)} characters")
    print(f"Links : {len(page.document_links)}")
    for link in page.document_links[:5]:
        print(f"  • {link.title}  →  {link.url}")
    print(f"{'='*60}")
