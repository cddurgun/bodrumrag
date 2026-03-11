#!/usr/bin/env python3
"""
Bodrum RAG — Interactive command-line interface.

Usage:
    python -m src.cli index      # Scrape + embed + build index
    python -m src.cli ask        # Interactive Q&A loop
    python -m src.cli query "…"  # Single-shot question
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap

from src.config import settings
from src.retriever import ask
from src.vector_db import VectorDB

# ── ANSI colours for a nicer terminal experience ─────────────
_BOLD = "\033[1m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_DIM = "\033[2m"
_RESET = "\033[0m"

_BANNER = f"""
{_CYAN}{_BOLD}╔══════════════════════════════════════════════════════════╗
║          🏛  BODRUM PLAN NOTLARI RAG SİSTEMİ  🏛          ║
║     Bodrum imar planları için akıllı soru-cevap sistemi    ║
╚══════════════════════════════════════════════════════════╝{_RESET}
"""


def _print_answer(result: dict) -> None:
    """Pretty-print a RAG answer result."""
    print(f"\n{_GREEN}{_BOLD}CEVAP:{_RESET}")
    print(textwrap.fill(result["answer"], width=80))

    sources = result.get("sources", [])
    if sources:
        print(f"\n{_YELLOW}{_BOLD}KAYNAKLAR:{_RESET}")
        for i, src in enumerate(sources, 1):
            score = src.get("score", 0)
            preview = src.get("text_preview", "")[:120].replace("\n", " ")
            print(f"  {_DIM}{i}. [Benzerlik: {score:.3f}] {preview}…{_RESET}")
    print()


def cmd_index(args: argparse.Namespace) -> None:
    """Run the indexing pipeline."""
    from src.pipeline import run_pipeline

    print(f"{_CYAN}▶ İndeksleme başlatılıyor…{_RESET}")
    db = run_pipeline(skip_download=getattr(args, 'skip_download', False))
    print(f"{_GREEN}✓ İndeksleme tamamlandı: {db.size} vektör kaydedildi.{_RESET}")


def cmd_query(args: argparse.Namespace) -> None:
    """Answer a single question from the command line."""
    db = _load_db()
    question = " ".join(args.question)
    if not question.strip():
        print(f"{_YELLOW}⚠ Lütfen bir soru yazın.{_RESET}")
        sys.exit(1)

    result = ask(db, question, top_k=args.top_k)
    _print_answer(result)


def cmd_ask(args: argparse.Namespace) -> None:
    """Interactive Q&A REPL."""
    db = _load_db()
    print(_BANNER)
    print(f"{_DIM}Çıkmak için 'q' veya 'exit' yazın.{_RESET}\n")

    while True:
        try:
            query = input(f"{_CYAN}{_BOLD}Soru ▶ {_RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_DIM}Hoşça kalın! 👋{_RESET}")
            break

        if not query:
            continue
        if query.lower() in {"q", "exit", "quit", "çıkış", "çık"}:
            print(f"{_DIM}Hoşça kalın! 👋{_RESET}")
            break

        try:
            result = ask(db, query, top_k=args.top_k)
            _print_answer(result)
        except Exception as exc:
            print(f"{_YELLOW}⚠ Hata: {exc}{_RESET}")
            logging.getLogger(__name__).exception("Error during ask()")


def _load_db() -> VectorDB:
    """Load the persisted FAISS index or exit with a helpful message."""
    db = VectorDB()
    if not db.exists():
        print(
            f"{_YELLOW}⚠ Henüz bir indeks bulunamadı.\n"
            f"  Lütfen önce indekslemeyi çalıştırın:\n"
            f"    python -m src.cli index{_RESET}"
        )
        sys.exit(1)
    db.load()
    print(f"{_DIM}İndeks yüklendi: {db.size} vektör, {db.dimension} boyut{_RESET}")
    return db


def main() -> None:
    """Parse arguments and dispatch to the right sub-command."""
    parser = argparse.ArgumentParser(
        prog="bodrum-rag",
        description="Bodrum Plan Notları RAG Sistemi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Örnekler:
              python -m src.cli index
              python -m src.cli ask
              python -m src.cli query "Gümüşlük'te yapılaşma koşulları nelerdir?"
        """),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── index ─────────────────────────────────────────────────
    index_p = sub.add_parser("index", help="Sayfayı tara, PDF indir, parçala, gömvektörle ve indeksle")
    index_p.add_argument(
        "--skip-download",
        action="store_true",
        help="PDF indirmeyi atla, mevcut dosyaları kullan",
    )

    # ── ask ───────────────────────────────────────────────────
    ask_p = sub.add_parser("ask", help="İnteraktif soru-cevap döngüsü")
    ask_p.add_argument(
        "-k", "--top-k",
        type=int,
        default=None,
        help=f"Döndürülecek en yakın chunk sayısı (varsayılan: {settings.top_k})",
    )

    # ── query ─────────────────────────────────────────────────
    query_p = sub.add_parser("query", help="Tek seferlik soru")
    query_p.add_argument("question", nargs="+", help="Sorunuz")
    query_p.add_argument(
        "-k", "--top-k",
        type=int,
        default=None,
        help=f"Döndürülecek en yakın chunk sayısı (varsayılan: {settings.top_k})",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    dispatch = {
        "index": cmd_index,
        "ask": cmd_ask,
        "query": cmd_query,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
