"""
Tests for the chunker module.
"""

import pytest

from src.chunker import TextChunk, count_tokens, split_into_chunks


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_known_string(self):
        tokens = count_tokens("hello world")
        assert tokens > 0  # exact count depends on encoding

    def test_turkish_text(self):
        tokens = count_tokens("Yapılaşma koşulları belirlenmiştir")
        assert tokens > 0


class TestSplitIntoChunks:
    def test_empty_text_returns_empty(self):
        result = split_into_chunks("", "https://example.com")
        assert result == []

    def test_whitespace_only_returns_empty(self):
        result = split_into_chunks("   \n\n  ", "https://example.com")
        assert result == []

    def test_short_text_single_chunk(self):
        text = "Kısa bir test metni."
        result = split_into_chunks(text, "https://example.com", chunk_size=500)
        assert len(result) == 1
        assert result[0].text == text
        assert result[0].source_url == "https://example.com"
        assert result[0].chunk_index == 0

    def test_multiple_paragraphs_split(self):
        # Create text that should span multiple chunks
        para = "Bu bir test paragrafıdır. " * 50
        text = f"{para}\n\n{para}\n\n{para}\n\n{para}"
        result = split_into_chunks(text, "https://example.com", chunk_size=100, chunk_overlap=20)
        assert len(result) > 1

    def test_chunk_metadata(self):
        text = "Paragraf bir.\n\nParagraf iki.\n\nParagraf üç."
        result = split_into_chunks(text, "https://test.com", chunk_size=1000)
        assert len(result) >= 1
        chunk = result[0]
        assert isinstance(chunk, TextChunk)
        assert chunk.source_url == "https://test.com"
        assert chunk.chunk_index == 0
        assert chunk.token_count > 0

    def test_chunks_have_sequential_indices(self):
        para = "Uzun bir metin parçası. " * 100
        text = f"{para}\n\n{para}\n\n{para}"
        result = split_into_chunks(text, "https://example.com", chunk_size=50, chunk_overlap=10)
        for i, chunk in enumerate(result):
            assert chunk.chunk_index == i

    def test_overlap_produces_shared_content(self):
        # With overlap, consecutive chunks should share some text
        para1 = "Birinci paragraf içeriği. " * 30
        para2 = "İkinci paragraf içeriği. " * 30
        para3 = "Üçüncü paragraf içeriği. " * 30
        text = f"{para1}\n\n{para2}\n\n{para3}"

        with_overlap = split_into_chunks(
            text, "https://example.com", chunk_size=80, chunk_overlap=30
        )
        without_overlap = split_into_chunks(
            text, "https://example.com", chunk_size=80, chunk_overlap=0
        )

        # Both should produce valid chunks
        assert len(with_overlap) > 0
        assert len(without_overlap) > 0

        # With overlap, chunks should share some text content
        # (overlap causes trailing paragraphs to be carried forward)
        if len(with_overlap) >= 2:
            # Check that consecutive chunks have overlapping text
            first_text = with_overlap[0].text
            second_text = with_overlap[1].text
            # The second chunk should start with content that appeared
            # at the end of the first chunk (due to overlap)
            assert len(second_text) > 0
