"""
Tests for the scraper module.

These tests use mocked HTTP responses to avoid hitting the real website.
"""

from unittest.mock import patch, MagicMock

import pytest

from src.scraper import (
    ScrapedPage,
    _extract_document_links,
    _extract_main_text,
    fetch_html,
    scrape,
)
from bs4 import BeautifulSoup


_SAMPLE_HTML = """\
<!DOCTYPE html>
<html lang="tr">
<head><title>PLAN NOTLARI – Mimarlar Odası</title></head>
<body>
  <nav><a href="/">ANA SAYFA</a></nav>
  <div class="entry-content">
    <h1>PLAN NOTLARI</h1>
    <p>Anasayfa</p>
    <p>>> <a href="https://drive.google.com/file/d/ABC/view">Gümüşlük LEJANT</a></p>
    <p>>> <a href="https://drive.google.com/file/d/DEF/view">Konacık Plan Hükümleri</a></p>
    <p>Yapılaşma koşulları: E=0.20, Hmaks=6.50m</p>
    <a href="https://example.com/not-drive">Non-Drive Link</a>
  </div>
  <footer>Footer content</footer>
</body>
</html>
"""


class TestExtractDocumentLinks:
    def test_finds_drive_links(self):
        soup = BeautifulSoup(_SAMPLE_HTML, "lxml")
        links = _extract_document_links(soup)
        assert len(links) == 2
        assert links[0].title == "Gümüşlük LEJANT"
        assert "drive.google.com" in links[0].url
        assert links[1].title == "Konacık Plan Hükümleri"

    def test_ignores_non_drive_links(self):
        soup = BeautifulSoup(_SAMPLE_HTML, "lxml")
        links = _extract_document_links(soup)
        urls = [l.url for l in links]
        assert not any("example.com" in u for u in urls)

    def test_deduplicates(self):
        html_dup = _SAMPLE_HTML + (
            '<p><a href="https://drive.google.com/file/d/ABC/view">'
            "Duplicate Link</a></p>"
        )
        soup = BeautifulSoup(html_dup, "lxml")
        links = _extract_document_links(soup)
        assert len(links) == 2  # ABC appears twice but should be deduped


class TestExtractMainText:
    def test_extracts_content(self):
        soup = BeautifulSoup(_SAMPLE_HTML, "lxml")
        text = _extract_main_text(soup)
        assert "PLAN NOTLARI" in text
        assert "Yapılaşma koşulları" in text

    def test_strips_nav_footer(self):
        soup = BeautifulSoup(_SAMPLE_HTML, "lxml")
        text = _extract_main_text(soup)
        assert "Footer content" not in text


class TestScrape:
    @patch("src.scraper.fetch_html", return_value=_SAMPLE_HTML)
    def test_returns_scraped_page(self, mock_fetch):
        result = scrape("https://test.example.com")
        assert isinstance(result, ScrapedPage)
        assert result.page_title == "PLAN NOTLARI – Mimarlar Odası"
        assert len(result.document_links) == 2
        assert "Yapılaşma" in result.raw_text
        mock_fetch.assert_called_once_with("https://test.example.com")
