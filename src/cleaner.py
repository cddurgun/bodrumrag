"""
Text cleaning and Turkish-character normalisation.

Responsibilities:
  • Strip residual HTML entities.
  • Normalise Turkish characters (İ ı Ş ş Ç ç Ğ ğ Ö ö Ü ü).
  • Remove navigation / boilerplate lines.
  • Collapse whitespace while preserving paragraph breaks.
  • Keep zoning-regulation keywords intact.
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata
from typing import List

logger = logging.getLogger(__name__)

# ── Patterns that mark navigation / boilerplate ───────────────
_BOILERPLATE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"^\s*(ANA SAYFA|KURUMSAL|ÜYELİK|MESLEKİ HİZMETLER|GALERİ|BİZE ULAŞIN)\s*$", re.I),
    re.compile(r"^\s*(KURULUŞ VE AMAÇ|BODRUM TEMSİLCİLİĞİ|YÖNETİM KURULU)\s*$", re.I),
    re.compile(r"^\s*(ÇALIŞMA GRUPLARI|ÜYELERİMİZ|ETKİNLİK TAKVİMİ)\s*$", re.I),
    re.compile(r"^\s*(VİDEOLARIMIZ|FOTO GALERİMİZ|BİLGİ FORMLARI)\s*$", re.I),
    re.compile(r"^\s*(TMMOB ÜYE GİRİŞİ|KAYIT KOŞULLARI|İNSAN KAYNAKLARI)\s*$", re.I),
    re.compile(r"^\s*(ÖDENTİ ve BELGE HARÇLARI|ÖĞRENCİ ÜYELİK)\s*$", re.I),
    re.compile(r"^\s*(BÜRO TESCİL BELGESİ ALMA KOŞULLARI)\s*$", re.I),
    re.compile(r"^\s*(MİMARLAR ODASI MEVZUATI|HUKUK|HAKKIMIZDA)\s*$", re.I),
    re.compile(r"^\s*(YAPI YAKLAŞIK BİRİM MALİYETLERİ|EN AZ BEDEL HESABI)\s*$", re.I),
    re.compile(r"^\s*>>\s*$"),  # stray ">>" lines
    re.compile(r"^\s*Anasayfa\s*$", re.I),
]

# ── Turkish-specific mappings for robustness ──────────────────
_TR_UPPER_MAP = str.maketrans("İIŞÇĞÖÜ", "İIŞÇĞÖÜ")  # identity – kept for doc
_TR_NORMALISE = {
    "\u0130": "İ",   # İ  (capital I with dot above)
    "\u0131": "ı",   # ı  (lowercase dotless i)
    "\u015E": "Ş",   # Ş
    "\u015F": "ş",   # ş
    "\u00C7": "Ç",   # Ç
    "\u00E7": "ç",   # ç
    "\u011E": "Ğ",   # Ğ
    "\u011F": "ğ",   # ğ
    "\u00D6": "Ö",   # Ö
    "\u00F6": "ö",   # ö
    "\u00DC": "Ü",   # Ü
    "\u00FC": "ü",   # ü
}


def normalise_turkish(text: str) -> str:
    """
    Ensure Turkish special characters use their canonical Unicode forms.

    This prevents issues where the *same* visual glyph may be encoded
    with different codepoints (composed vs. decomposed).
    """
    # NFC normalisation merges combining marks into precomposed chars
    text = unicodedata.normalize("NFC", text)
    for src, dst in _TR_NORMALISE.items():
        text = text.replace(src, dst)
    return text


def unescape_html(text: str) -> str:
    """Decode HTML entities (``&amp;`` → ``&``, etc.)."""
    return html.unescape(text)


def strip_boilerplate(text: str) -> str:
    """
    Remove lines that match known navigation / boilerplate patterns.
    """
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        if any(p.match(line) for p in _BOILERPLATE_PATTERNS):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def collapse_whitespace(text: str) -> str:
    """
    • Trim trailing whitespace on each line.
    • Replace runs of 3+ newlines with a double newline.
    • Strip leading/trailing whitespace from the whole text.
    """
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_urls(text: str) -> str:
    """Strip inline URLs that leaked through the HTML→text conversion."""
    return re.sub(r"https?://\S+", "", text)


def clean(raw_text: str) -> str:
    """
    Full cleaning pipeline:

    1. Unescape HTML entities.
    2. Normalise Turkish characters.
    3. Remove leaked URLs.
    4. Strip navigation boilerplate lines.
    5. Collapse whitespace.

    Args:
        raw_text: The text extracted by the scraper.

    Returns:
        Cleaned, human-readable text ready for chunking.
    """
    text = unescape_html(raw_text)
    text = normalise_turkish(text)
    text = remove_urls(text)
    text = strip_boilerplate(text)
    text = collapse_whitespace(text)
    logger.info("Cleaned text: %d → %d characters", len(raw_text), len(text))
    return text


# ── Quick smoke-test ──────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "ANA SAYFA\n"
        "PLAN NOTLARI\nGümüşlük_ LEJANT\n"
        ">>  Gümüşlük_PLAN NOTLARI 001\n"
        "https://drive.google.com/file/d/XXXX\n"
        "Yapılaşma koşulları: E=0.20, Hmaks=6.50m\n"
        "&amp; Şehir Planı"
    )
    print(clean(sample))
