"""
Centralised configuration loaded from environment variables / .env file.

All tunables live here so every other module imports from one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Config:
    """Immutable application configuration."""

    # ── Scraping ───────────────────────────────────────────────
    source_url: str = "https://bodrummimarlarodasi.org.tr/plan-notlari/"
    request_timeout: int = 30  # seconds

    # ── NVIDIA Embedding API ───────────────────────────────────
    nvidia_api_key: str = field(
        default_factory=lambda: os.getenv("NVIDIA_API_KEY", "")
    )
    nvidia_embed_base_url: str = field(
        default_factory=lambda: os.getenv(
            "NVIDIA_EMBED_BASE_URL",
            "https://integrate.api.nvidia.com/v1",
        )
    )
    nvidia_embed_model: str = field(
        default_factory=lambda: os.getenv(
            "NVIDIA_EMBED_MODEL",
            "nvidia/llama-nemotron-embed-1b-v2",
        )
    )

    # ── LLM (generative) ──────────────────────────────────────
    llm_base_url: str = field(
        default_factory=lambda: os.getenv(
            "LLM_BASE_URL",
            "https://integrate.api.nvidia.com/v1",
        )
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv(
            "LLM_MODEL",
            "nvidia/llama-3.3-nemotron-super-49b-v1",
        )
    )

    # ── Chunking ───────────────────────────────────────────────
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "600"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100"))
    )

    # ── Retrieval ──────────────────────────────────────────────
    top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K", "5"))
    )

    # ── Vector DB ──────────────────────────────────────────────
    faiss_index_path: str = field(
        default_factory=lambda: os.getenv(
            "FAISS_INDEX_PATH",
            str(_PROJECT_ROOT / "data" / "faiss_index"),
        )
    )

    # ── Data directory ─────────────────────────────────────────
    data_dir: str = field(
        default_factory=lambda: str(_PROJECT_ROOT / "data")
    )


# Singleton-ish instance used across the project
settings = Config()
