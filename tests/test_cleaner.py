"""
Tests for the cleaner module.
"""

from src.cleaner import (
    clean,
    collapse_whitespace,
    normalise_turkish,
    remove_urls,
    strip_boilerplate,
    unescape_html,
)


class TestUnescapeHtml:
    def test_basic_entities(self):
        assert unescape_html("&amp;") == "&"
        assert unescape_html("&lt;div&gt;") == "<div>"
        assert unescape_html("&quot;test&quot;") == '"test"'

    def test_no_entities(self):
        assert unescape_html("plain text") == "plain text"


class TestNormaliseTurkish:
    def test_turkish_characters_preserved(self):
        text = "İstanbul Şehir Çevre Ğüzel Ödül Ünlü"
        result = normalise_turkish(text)
        assert "İ" in result
        assert "Ş" in result
        assert "Ç" in result
        assert "Ğ" in result
        assert "Ö" in result
        assert "Ü" in result

    def test_lowercase_turkish(self):
        text = "ışık şeker çiçek ğüzel özel ünvan ırmak"
        result = normalise_turkish(text)
        assert "ı" in result
        assert "ş" in result
        assert "ç" in result
        assert "ğ" in result
        assert "ö" in result
        assert "ü" in result


class TestStripBoilerplate:
    def test_removes_nav_lines(self):
        text = "ANA SAYFA\nPLAN NOTLARI\nKURUMSAL\nAnasayfa"
        result = strip_boilerplate(text)
        assert "ANA SAYFA" not in result
        assert "KURUMSAL" not in result
        assert "Anasayfa" not in result
        assert "PLAN NOTLARI" in result  # This is content, not boilerplate

    def test_keeps_content(self):
        text = "Yapılaşma koşulları E=0.20\nHmaks=6.50m"
        result = strip_boilerplate(text)
        assert "Yapılaşma" in result
        assert "Hmaks" in result


class TestRemoveUrls:
    def test_removes_http_urls(self):
        text = "Bkz: https://drive.google.com/file/xyz belge"
        result = remove_urls(text)
        assert "https://" not in result
        assert "belge" in result

    def test_removes_http(self):
        text = "Link: http://example.com burada"
        result = remove_urls(text)
        assert "http://" not in result


class TestCollapseWhitespace:
    def test_collapses_multiple_newlines(self):
        text = "A\n\n\n\n\nB"
        result = collapse_whitespace(text)
        assert result == "A\n\nB"

    def test_trims_trailing_spaces(self):
        text = "hello   \nworld   "
        result = collapse_whitespace(text)
        assert not any(line.endswith(" ") for line in result.split("\n"))


class TestCleanPipeline:
    def test_full_pipeline(self):
        raw = (
            "ANA SAYFA\n"
            "&amp; Şehir Planı\n"
            "https://example.com/test\n"
            "Yapılaşma koşulları E=0.20\n"
            "\n\n\n\n"
            "BİZE ULAŞIN"
        )
        result = clean(raw)
        assert "ANA SAYFA" not in result
        assert "BİZE ULAŞIN" not in result
        assert "& Şehir Planı" in result
        assert "Yapılaşma" in result
        assert "https://" not in result

    def test_empty_input(self):
        result = clean("")
        assert result == ""
