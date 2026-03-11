"""
Tests for the vector_db module.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.chunker import TextChunk
from src.vector_db import VectorDB


def _make_chunks(n: int) -> list[TextChunk]:
    """Create *n* dummy TextChunk objects."""
    return [
        TextChunk(
            text=f"Chunk {i}: Yapılaşma koşulları E=0.{i:02d}",
            source_url="https://example.com",
            chunk_index=i,
            start_char=i * 100,
            end_char=(i + 1) * 100,
            token_count=50,
        )
        for i in range(n)
    ]


def _make_vectors(n: int, dim: int = 64) -> np.ndarray:
    """Create random float32 vectors."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, dim)).astype(np.float32)


class TestVectorDBBuild:
    def test_build_creates_index(self):
        db = VectorDB("/tmp/test_vdb_build")
        vecs = _make_vectors(5)
        chunks = _make_chunks(5)
        db.build(vecs, chunks)
        assert db.size == 5
        assert db.dimension == 64

    def test_build_mismatch_raises(self):
        db = VectorDB("/tmp/test_vdb_mismatch")
        vecs = _make_vectors(5)
        chunks = _make_chunks(3)
        with pytest.raises(ValueError, match="Mismatch"):
            db.build(vecs, chunks)


class TestVectorDBSearch:
    def test_search_returns_results(self):
        db = VectorDB("/tmp/test_vdb_search")
        n, dim = 10, 64
        vecs = _make_vectors(n, dim)
        chunks = _make_chunks(n)
        db.build(vecs, chunks)

        q = np.random.randn(dim).astype(np.float32)
        results = db.search(q, top_k=3)
        assert len(results) == 3
        for meta, score in results:
            assert "text" in meta
            assert "source_url" in meta
            assert isinstance(score, float)

    def test_search_empty_index(self):
        db = VectorDB("/tmp/test_vdb_empty")
        results = db.search(np.zeros(64, dtype=np.float32), top_k=3)
        assert results == []

    def test_search_top_k_larger_than_index(self):
        db = VectorDB("/tmp/test_vdb_topk")
        vecs = _make_vectors(3, 32)
        chunks = _make_chunks(3)
        db.build(vecs, chunks)
        results = db.search(np.random.randn(32).astype(np.float32), top_k=10)
        assert len(results) <= 3


class TestVectorDBPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_index")
            db = VectorDB(path)
            n, dim = 8, 48
            vecs = _make_vectors(n, dim)
            chunks = _make_chunks(n)
            db.build(vecs, chunks)
            db.save()

            # Load into a fresh instance
            db2 = VectorDB(path)
            db2.load()
            assert db2.size == n
            assert db2.dimension == dim

            # Metadata should match
            q = np.random.randn(dim).astype(np.float32)
            results = db2.search(q, top_k=2)
            assert len(results) == 2

    def test_load_missing_raises(self):
        db = VectorDB("/tmp/nonexistent_index_xyz")
        with pytest.raises(FileNotFoundError):
            db.load()

    def test_exists_false_initially(self):
        db = VectorDB("/tmp/totally_new_index_abc")
        assert not db.exists()

    def test_exists_true_after_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "exist_test")
            db = VectorDB(path)
            db.build(_make_vectors(3, 16), _make_chunks(3))
            db.save()
            assert db.exists()


class TestVectorDBAdd:
    def test_incremental_add(self):
        db = VectorDB("/tmp/test_vdb_add")
        dim = 32
        # Build initial
        db.build(_make_vectors(3, dim), _make_chunks(3))
        assert db.size == 3
        # Add more
        db.add(_make_vectors(2, dim), _make_chunks(2))
        assert db.size == 5
