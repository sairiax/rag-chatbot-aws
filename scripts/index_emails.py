"""
CLI script to batch-index emails from a local directory into ChromaDB.
Orchestrates the LegalMail pipeline: Parser -> Curator -> Chunker -> ChromaDB.

Usage:
    python scripts/index_emails.py --dir data/clean --curate --clear
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so config / src are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from config.settings import Settings
from src.embeddings.aws_embeddings import get_embeddings
from src.ingestion.email_parser import EmailParser
from src.ingestion.email_curator import EmailCurator
from src.ingestion.text_splitter import EmailAwareTextSplitter
from src.vectorstore.chroma_store import ChromaVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index legal emails into ChromaDB")
    parser.add_argument("--dir", metavar="PATH", help="Directory containing .txt emails")
    parser.add_argument("--curate", action="store_true", help="Enable LLM spam filtering")
    parser.add_argument("--clear", action="store_true", help="Clear existing ChromaDB collection first")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    
    source_dir = Path(args.dir) if args.dir else Path(settings.email_data_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        logger.error(f"Data directory not found: {source_dir}")
        sys.exit(1)

    logger.info("Loading settings and initializing Vector Store...")
    embeddings = get_embeddings(settings)
    vector_store = ChromaVectorStore(embeddings, settings)

    if args.clear:
        logger.warning("--clear flag detected: dropping existing collection...")
        vector_store.delete_collection()

    # 1. Parse emails
    logger.info(f"Scanning directory: {source_dir.resolve()}")
    parser = EmailParser()
    documents = parser.parse_directory(source_dir)
    
    if not documents:
        logger.warning("No emails found. Nothing to index.")
        return

    # 2. Curate/Filter Spam using LLM
    do_curation = args.curate or settings.enable_curation
    if do_curation:
        curator = EmailCurator(settings)
        documents = curator.filter_documents(documents)
    
    if not documents:
        logger.warning("All emails filtered out. Nothing to index.")
        return

    # 3. Chunk
    logger.info("Chunking and enriching documents...")
    splitter = EmailAwareTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    # 4. Index
    logger.info(f"Indexing {len(chunks)} chunk(s) into ChromaDB...")
    vector_store.add_documents(chunks)

    stats = vector_store.get_collection_stats()
    logger.success(f"Done! Collection '{stats['name']}' now contains {stats['count']} chunk(s).")


if __name__ == "__main__":
    main()
