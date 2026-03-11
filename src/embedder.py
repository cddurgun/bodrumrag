"""
Dense-vector embedder via NVIDIA NIM API.

Uses the ``nvidia/llama-nemotron-embed-1b-v2`` model through the
OpenAI-compatible ``/v1/embeddings`` endpoint hosted on
``integrate.api.nvidia.com``.

Key features:
  • Batched embedding calls to respect rate limits.
  • Automatic retry with exponential back-off.
  • Input type prefix support (``passage:`` / ``query:``).
"""

from __future__ import annotations

import logging
import time
from typing import List

import numpy as np
from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)

# Maximum texts per single API call (NVIDIA NIM limit)
_BATCH_SIZE = 50

# Retry configuration
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds


def _get_client() -> OpenAI:
    """Build an OpenAI client pointed at the NVIDIA embedding endpoint."""
    if not settings.nvidia_api_key:
        raise ValueError(
            "NVIDIA_API_KEY is not set. "
            "Please set it in your .env file or environment."
        )

    return OpenAI(
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_embed_base_url,
    )


def _embed_batch(
    client: OpenAI,
    texts: List[str],
    input_type: str = "passage",
) -> List[List[float]]:
    """
    Embed a single batch of texts.

    Args:
        client:     OpenAI client configured for NVIDIA NIM.
        texts:      Up to ``_BATCH_SIZE`` strings.
        input_type: ``"passage"`` for documents, ``"query"`` for queries.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(
                model=settings.nvidia_embed_model,
                input=texts,
                encoding_format="float",
                extra_body={"input_type": input_type, "truncate": "END"},
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                "Embedding attempt %d/%d failed: %s.  Retrying in %.1fs …",
                attempt,
                _MAX_RETRIES,
                exc,
                wait,
            )
            if attempt == _MAX_RETRIES:
                raise
            time.sleep(wait)

    # unreachable, but keeps mypy happy
    raise RuntimeError("Embedding failed after retries")  # pragma: no cover


def embed_texts(
    texts: List[str],
    input_type: str = "passage",
) -> np.ndarray:
    """
    Embed an arbitrary number of texts, batching automatically.

    Args:
        texts:      List of text strings (documents or queries).
        input_type: ``"passage"`` for indexing, ``"query"`` for retrieval.

    Returns:
        A NumPy array of shape ``(len(texts), embedding_dim)``
        with ``float32`` dtype.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    client = _get_client()
    all_embeddings: List[List[float]] = []

    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        logger.info(
            "Embedding batch %d–%d of %d texts",
            start,
            start + len(batch) - 1,
            len(texts),
        )
        embeddings = _embed_batch(client, batch, input_type=input_type)
        all_embeddings.extend(embeddings)

    matrix = np.array(all_embeddings, dtype=np.float32)
    logger.info("Produced embedding matrix of shape %s", matrix.shape)
    return matrix


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns:
        A 1-D NumPy array of shape ``(embedding_dim,)``.
    """
    result = embed_texts([query], input_type="query")
    return result[0]


# ── Quick smoke-test ──────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_texts = [
        "Bodrum plan notları imar koşulları",
        "Yapılaşma koşulları E=0.20 Hmaks=6.50m",
    ]
    embeddings = embed_texts(test_texts)
    print(f"Shape: {embeddings.shape}")
    print(f"First vector (first 5 dims): {embeddings[0][:5]}")
