"""
Tests for the retriever module.

Uses mocked embeddings and vector DB to test the RAG prompt
construction and answer generation without calling real APIs.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.retriever import build_rag_prompt, retrieve


class TestBuildRagPrompt:
    def test_includes_context_and_question(self):
        chunks = [
            ({"text": "TAKS=0.20, KAKS=0.40", "source_url": "https://example.com"}, 0.95),
            ({"text": "Hmaks=6.50m", "source_url": "https://example.com"}, 0.88),
        ]
        prompt = build_rag_prompt("Yapılaşma koşulları nelerdir?", chunks)
        assert "TAKS=0.20" in prompt
        assert "Hmaks=6.50m" in prompt
        assert "Yapılaşma koşulları nelerdir?" in prompt
        assert "Kaynak 1" in prompt
        assert "Kaynak 2" in prompt

    def test_empty_chunks(self):
        prompt = build_rag_prompt("Test sorusu?", [])
        assert "Test sorusu?" in prompt


class TestRetrieve:
    @patch("src.retriever.embed_query")
    def test_retrieve_calls_embed_and_search(self, mock_embed):
        mock_embed.return_value = np.random.randn(64).astype(np.float32)

        mock_db = MagicMock()
        mock_db.search.return_value = [
            ({"text": "chunk1", "source_url": "u1"}, 0.9),
        ]

        results = retrieve(mock_db, "test query", top_k=3)
        mock_embed.assert_called_once_with("test query")
        mock_db.search.assert_called_once()
        assert len(results) == 1
