"""
CLI script to batch-index documents from a local directory into ChromaDB.

Usage examples:
    # Index all documents in data/raw/ with OCR enabled
    python scripts/index_documents.py --dir data/raw --ocr

    # Reset the collection and re-index
    python scripts/index_documents.py --dir data/raw --ocr --clear

    # Non-recursive (top-level files only)
    python scripts/index_documents.py --dir data/raw --no-recursive
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
from src.ingestion.document_loader import MultiFormatDocumentLoader
from src.ingestion.ocr_processor import OCRProcessor
from src.ingestion.text_splitter import MetadataAwareTextSplitter
from src.vectorstore.chroma_store import ChromaVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index documents from a directory into ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dir",
        required=True,
        metavar="PATH",
        help="Directory containing the documents to index",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR for scanned PDFs and image files (requires Tesseract)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="⚠️  Clear the existing ChromaDB collection before indexing",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only index files in the top-level directory (skip sub-folders)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()

    logger.info("Loading settings and initialising components…")
    embeddings = get_embeddings(settings)
    vector_store = ChromaVectorStore(embeddings, settings)

    if args.clear:
        logger.warning("--clear flag detected: dropping existing collection…")
        vector_store.delete_collection()

    ocr_processor = OCRProcessor(settings) if args.ocr else None
    if args.ocr:
        logger.info("OCR enabled")

    loader = MultiFormatDocumentLoader(ocr_processor=ocr_processor)
    splitter = MetadataAwareTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    source_dir = Path(args.dir)
    if not source_dir.exists():
        logger.error(f"Directory not found: {source_dir}")
        sys.exit(1)

    logger.info(f"Scanning directory: {source_dir.resolve()}")
    documents = loader.load_directory(source_dir, recursive=not args.no_recursive)

    if not documents:
        logger.warning("No supported documents found. Nothing to index.")
        return

    logger.info(f"Splitting {len(documents)} document(s)…")
    chunks = splitter.split_documents(documents)

    logger.info(f"Indexing {len(chunks)} chunk(s) into ChromaDB…")
    vector_store.add_documents(chunks)

    stats = vector_store.get_collection_stats()
    logger.success(
        f"Done! Collection '{stats['name']}' now contains {stats['count']} chunk(s)."
    )


if __name__ == "__main__":
    main()
