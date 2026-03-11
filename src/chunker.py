"""
Token-aware text chunker.

Splits cleaned text into fixed-size chunks (measured in *tokens*,
not characters) with a configurable overlap so that context is not
lost at chunk boundaries.

Token counting uses ``tiktoken`` with the ``cl100k_base`` encoding
(GPT-4 / text-embedding family).  This is a reasonable proxy for
sub-word token counts on multilingual text even when the actual
model uses a different tokeniser.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import tiktoken

from src.config import settings

logger = logging.getLogger(__name__)

# Use the cl100k_base encoding as a token-count approximation
_ENCODER = tiktoken.get_encoding("cl100k_base")


@dataclass
class TextChunk:
    """One chunk of text with positional metadata."""

    text: str
    source_url: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int


def count_tokens(text: str) -> int:
    """Return the number of tokens in *text*."""
    return len(_ENCODER.encode(text))


def split_into_chunks(
    text: str,
    source_url: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[TextChunk]:
    """
    Split *text* into token-bounded chunks with overlap.

    The algorithm works on **paragraph boundaries** first (split on
    ``\\n\\n``), then accumulates paragraphs until the next one would
    exceed ``chunk_size`` tokens.  When that happens the current
    accumulator is emitted as a chunk and we rewind by
    ``chunk_overlap`` tokens worth of paragraphs.

    Args:
        text:          Cleaned text to split.
        source_url:    Original URL for metadata.
        chunk_size:    Max tokens per chunk (default from config).
        chunk_overlap: Overlap in tokens between consecutive chunks.

    Returns:
        Ordered list of ``TextChunk`` objects.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if not text.strip():
        logger.warning("Empty text – no chunks produced")
        return []

    paragraphs = text.split("\n\n")
    # Remove empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: List[TextChunk] = []
    current_paras: list[str] = []
    current_tokens = 0
    char_offset = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # If a single paragraph is bigger than chunk_size, split by sentences
        if para_tokens > chunk_size:
            # Flush any accumulated paragraphs first
            if current_paras:
                chunk_text = "\n\n".join(current_paras)
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        source_url=source_url,
                        chunk_index=len(chunks),
                        start_char=char_offset,
                        end_char=char_offset + len(chunk_text),
                        token_count=current_tokens,
                    )
                )
                char_offset += len(chunk_text) + 2  # +2 for \n\n separator
                current_paras = []
                current_tokens = 0

            # Break the oversized paragraph into sentence-level pieces
            _split_oversized_paragraph(
                para, source_url, chunk_size, chunk_overlap, chunks, char_offset
            )
            char_offset += len(para) + 2
            continue

        # Would adding this paragraph exceed the limit?
        if current_tokens + para_tokens > chunk_size and current_paras:
            chunk_text = "\n\n".join(current_paras)
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    source_url=source_url,
                    chunk_index=len(chunks),
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                    token_count=current_tokens,
                )
            )
            char_offset += len(chunk_text) + 2

            # Rewind for overlap: keep trailing paragraphs whose total
            # token count is <= chunk_overlap
            overlap_paras: list[str] = []
            overlap_tokens = 0
            for prev_para in reversed(current_paras):
                pt = count_tokens(prev_para)
                if overlap_tokens + pt > chunk_overlap:
                    break
                overlap_paras.insert(0, prev_para)
                overlap_tokens += pt

            current_paras = overlap_paras
            current_tokens = overlap_tokens

        current_paras.append(para)
        current_tokens += para_tokens

    # Flush remainder
    if current_paras:
        chunk_text = "\n\n".join(current_paras)
        chunks.append(
            TextChunk(
                text=chunk_text,
                source_url=source_url,
                chunk_index=len(chunks),
                start_char=char_offset,
                end_char=char_offset + len(chunk_text),
                token_count=current_tokens,
            )
        )

    logger.info(
        "Split %d characters into %d chunks (target %d tokens, overlap %d)",
        len(text),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def _split_oversized_paragraph(
    para: str,
    source_url: str,
    chunk_size: int,
    chunk_overlap: int,
    chunks: List[TextChunk],
    char_offset: int,
) -> None:
    """
    Handle a single paragraph that exceeds ``chunk_size`` tokens
    by splitting it on sentence boundaries (period, newline).
    """
    import re

    sentences = re.split(r"(?<=[.!?])\s+|\n", para)
    buf: list[str] = []
    buf_tokens = 0

    for sentence in sentences:
        s_tokens = count_tokens(sentence)
        if buf_tokens + s_tokens > chunk_size and buf:
            chunk_text = " ".join(buf)
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    source_url=source_url,
                    chunk_index=len(chunks),
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                    token_count=buf_tokens,
                )
            )
            char_offset += len(chunk_text) + 1

            # Overlap: keep last sentence(s)
            overlap_buf: list[str] = []
            overlap_t = 0
            for s in reversed(buf):
                t = count_tokens(s)
                if overlap_t + t > chunk_overlap:
                    break
                overlap_buf.insert(0, s)
                overlap_t += t
            buf = overlap_buf
            buf_tokens = overlap_t

        buf.append(sentence)
        buf_tokens += s_tokens

    if buf:
        chunk_text = " ".join(buf)
        chunks.append(
            TextChunk(
                text=chunk_text,
                source_url=source_url,
                chunk_index=len(chunks),
                start_char=char_offset,
                end_char=char_offset + len(chunk_text),
                token_count=buf_tokens,
            )
        )


# ── Quick smoke-test ──────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    sample = "Bu bir test cümlesidir. " * 200
    result = split_into_chunks(sample, "https://example.com")
    for c in result:
        print(f"Chunk {c.chunk_index}: {c.token_count} tokens, chars {c.start_char}-{c.end_char}")
