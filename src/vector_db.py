"""
FAISS-based vector database for chunk storage and retrieval.

Stores:
  • A FAISS ``IndexFlatIP`` (inner-product / cosine similarity on
    L2-normalised vectors).
  • A side-car JSON file with chunk metadata (text, source URL,
    offsets, token count).

Persistence:
  Vectors → ``<faiss_index_path>.index``
  Metadata → ``<faiss_index_path>_meta.json``
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from src.chunker import TextChunk
from src.config import settings

logger = logging.getLogger(__name__)


class VectorDB:
    """Thin wrapper around a FAISS flat index with JSON metadata."""

    def __init__(self, index_path: str | None = None) -> None:
        self._index_path = Path(index_path or settings.faiss_index_path)
        self._index: Optional[faiss.IndexFlatIP] = None
        self._metadata: List[Dict[str, Any]] = []

    # ── Properties ────────────────────────────────────────────

    @property
    def index_file(self) -> Path:
        return self._index_path.with_suffix(".index")

    @property
    def meta_file(self) -> Path:
        return self._index_path.parent / f"{self._index_path.name}_meta.json"

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal if self._index else 0

    @property
    def dimension(self) -> int:
        """Vector dimension."""
        return self._index.d if self._index else 0

    # ── Build / Add ───────────────────────────────────────────

    def build(
        self,
        embeddings: np.ndarray,
        chunks: List[TextChunk],
    ) -> None:
        """
        Create a new index from scratch.

        Args:
            embeddings: ``(N, D)`` float32 array of L2-normalised vectors.
            chunks:     Corresponding ``TextChunk`` metadata list.
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} vectors vs {len(chunks)} chunks"
            )

        # L2-normalise so inner product = cosine similarity
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._metadata = [asdict(c) for c in chunks]

        logger.info(
            "Built FAISS index: %d vectors × %d dimensions",
            self._index.ntotal,
            dim,
        )

    def add(
        self,
        embeddings: np.ndarray,
        chunks: List[TextChunk],
    ) -> None:
        """Incrementally add vectors + metadata to an existing index."""
        if self._index is None:
            self.build(embeddings, chunks)
            return

        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        self._metadata.extend(asdict(c) for c in chunks)
        logger.info("Added %d vectors (total now %d)", embeddings.shape[0], self.size)

    # ── Search ────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int | None = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Return the *top_k* nearest chunks to ``query_vector``.

        Args:
            query_vector: 1-D float32, shape ``(D,)``.
            top_k:        Number of results (default from config).

        Returns:
            List of ``(metadata_dict, similarity_score)`` tuples,
            sorted by descending similarity.
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("Search on empty index – returning []")
            return []

        top_k = top_k or settings.top_k

        qv = query_vector.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(qv)

        distances, indices = self._index.search(qv, top_k)

        results: List[Tuple[Dict[str, Any], float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue  # fewer results than top_k
            results.append((self._metadata[idx], float(dist)))

        logger.info("Search returned %d results", len(results))
        return results

    # ── Persistence ───────────────────────────────────────────

    def save(self) -> None:
        """Write index + metadata to disk."""
        if self._index is None:
            raise RuntimeError("Cannot save – no index has been built")

        self._index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(self.index_file))
        with open(self.meta_file, "w", encoding="utf-8") as fh:
            json.dump(self._metadata, fh, ensure_ascii=False, indent=2)

        logger.info(
            "Saved FAISS index (%d vectors) to %s",
            self._index.ntotal,
            self.index_file,
        )

    def load(self) -> None:
        """Load a previously saved index + metadata from disk."""
        if not self.index_file.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_file}")
        if not self.meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_file}")

        self._index = faiss.read_index(str(self.index_file))
        with open(self.meta_file, "r", encoding="utf-8") as fh:
            self._metadata = json.load(fh)

        logger.info(
            "Loaded FAISS index: %d vectors × %d dimensions from %s",
            self._index.ntotal,
            self._index.d,
            self.index_file,
        )

    def exists(self) -> bool:
        """Check whether a saved index is available on disk."""
        return self.index_file.exists() and self.meta_file.exists()


# ── Quick smoke-test ──────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    dim = 128
    n = 10
    vecs = np.random.randn(n, dim).astype(np.float32)
    dummy_chunks = [
        TextChunk(
            text=f"chunk {i}",
            source_url="https://example.com",
            chunk_index=i,
            start_char=i * 100,
            end_char=(i + 1) * 100,
            token_count=50,
        )
        for i in range(n)
    ]
    db = VectorDB("/tmp/test_faiss_index")
    db.build(vecs, dummy_chunks)
    db.save()
    db.load()
    q = np.random.randn(dim).astype(np.float32)
    results = db.search(q, top_k=3)
    for meta, score in results:
        print(f"  {meta['text']}  (score={score:.4f})")
