"""
Retriever — ties together embedding, vector search, and RAG prompt construction.

Workflow:
  1. Embed the user query with ``input_type="query"``.
  2. Search the FAISS index for top-k nearest chunks.
  3. Build a RAG prompt that feeds context + question to the LLM.
  4. Call the LLM and return the generated answer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from src.config import settings
from src.embedder import embed_query
from src.vector_db import VectorDB

logger = logging.getLogger(__name__)

# ── RAG Prompt Template ───────────────────────────────────────
_SYSTEM_PROMPT = """\
Sen, Bodrum ilçesindeki imar planları ve yapılaşma koşulları konusunda \
uzman bir yapay zeka asistanısın. Sana verilen bağlam bilgilerini \
kullanarak soruları Türkçe olarak doğru ve detaylı şekilde yanıtla.

Kurallar:
• Yalnızca verilen bağlam bilgilerine dayanarak cevap ver.
• Bağlamda bulunmayan bilgiyi uydurma.
• Eğer bağlamda yeterli bilgi yoksa, bunu açıkça belirt.
• Teknik terimleri (TAKS, KAKS, Emsal, Hmaks, vb.) doğru kullan.
• Cevabın sonunda hangi kaynaklardan yararlandığını belirt.
"""

_USER_PROMPT_TEMPLATE = """\
BAĞLAM:
{context}

SORU:
{question}

Lütfen yukarıdaki bağlam bilgilerine dayanarak soruyu yanıtla.
"""


def retrieve(
    db: VectorDB,
    query: str,
    top_k: int | None = None,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Embed the query and retrieve the top-k nearest chunks.

    Args:
        db:    An initialised (and loaded) ``VectorDB``.
        query: User's Turkish-language question.
        top_k: Override for settings.top_k.

    Returns:
        List of ``(chunk_metadata, cosine_score)`` tuples.
    """
    logger.info("Embedding query: '%s'", query[:80])
    q_vec = embed_query(query)
    results = db.search(q_vec, top_k=top_k)
    return results


def build_rag_prompt(
    query: str,
    retrieved_chunks: List[Tuple[Dict[str, Any], float]],
) -> str:
    """
    Construct the user-part of the RAG prompt by concatenating
    retrieved context chunks.

    Returns:
        The formatted user prompt string.
    """
    context_parts: list[str] = []
    for i, (meta, score) in enumerate(retrieved_chunks, 1):
        text = meta.get("text", "")
        source = meta.get("source_url", "?")
        context_parts.append(
            f"[Kaynak {i} | Benzerlik: {score:.3f} | URL: {source}]\n{text}"
        )

    context_block = "\n\n---\n\n".join(context_parts)
    return _USER_PROMPT_TEMPLATE.format(context=context_block, question=query)


def generate_answer(
    query: str,
    retrieved_chunks: List[Tuple[Dict[str, Any], float]],
) -> str:
    """
    Send the RAG prompt to the LLM and return the generated text.

    Uses the NVIDIA NIM API (OpenAI-compatible) with the model
    configured in ``settings.llm_model``.
    """
    api_key = settings.llm_api_key or settings.nvidia_api_key
    if not api_key:
        raise ValueError(
            "LLM_API_KEY (or NVIDIA_API_KEY) is not set. "
            "Please configure it in your .env file."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=settings.llm_base_url,
        timeout=120.0,  # seconds – prevent indefinite hangs
    )

    user_prompt = build_rag_prompt(query, retrieved_chunks)

    logger.info(
        "Sending RAG prompt to %s (%d context chunks)",
        settings.llm_model,
        len(retrieved_chunks),
    )

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content or ""
    logger.info("Generated answer: %d characters", len(answer))
    return answer.strip()


def ask(
    db: VectorDB,
    query: str,
    top_k: int | None = None,
) -> Dict[str, Any]:
    """
    End-to-end RAG pipeline: embed → retrieve → generate.

    Returns:
        Dict with keys ``answer``, ``sources``, ``query``.
    """
    chunks = retrieve(db, query, top_k=top_k)

    if not chunks:
        return {
            "query": query,
            "answer": (
                "Üzgünüm, veritabanında bu soruyla ilgili herhangi bir "
                "bilgi bulunamadı. Lütfen farklı bir soru deneyin."
            ),
            "sources": [],
        }

    answer = generate_answer(query, chunks)

    sources = [
        {
            "chunk_index": meta.get("chunk_index"),
            "score": round(score, 4),
            "text_preview": meta.get("text", "")[:200],
            "source_url": meta.get("source_url", ""),
        }
        for meta, score in chunks
    ]

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
    }
