"""
Bodrum RAG — Retrieval-Augmented Generation for Bodrum Zoning Regulations.

Modules:
    scraper   – Fetches and parses plan-notları pages
    cleaner   – Text cleaning and Turkish character normalization
    chunker   – Token-aware text splitting
    embedder  – Dense vector generation via NVIDIA NIM API
    vector_db – FAISS index management
    retriever – Query embedding + similarity search + RAG prompt
    config    – Centralised configuration
"""

__version__ = "1.0.0"
