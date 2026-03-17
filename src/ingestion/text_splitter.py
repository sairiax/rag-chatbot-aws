"""
Metadata-aware text splitter.

Splits LangChain Documents into smaller chunks using RecursiveCharacterTextSplitter
and enriches each chunk with positional metadata (chunk_index, total_chunks, etc.)
so downstream retrieval can surface provenance information.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


class MetadataAwareTextSplitter:
    """Splits documents and propagates + enriches metadata on every chunk."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Preference order for split points
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of Documents and enrich chunk metadata.

        Returns a flat list of chunks, each carrying:
          - All original metadata from the parent document.
          - ``chunk_index``   — 0-based position within the parent document.
          - ``total_chunks``  — Total number of chunks for the parent document.
          - ``chunk_size``    — Character length of this chunk.
        """
        all_chunks: List[Document] = []

        for doc in documents:
            chunks = self._splitter.split_documents([doc])
            total = len(chunks)

            for idx, chunk in enumerate(chunks):
                chunk.metadata.update(
                    {
                        "chunk_index": idx,
                        "total_chunks": total,
                        "chunk_size": len(chunk.page_content),
                    }
                )
                all_chunks.append(chunk)

        logger.info(
            f"Split {len(documents)} document(s) → {len(all_chunks)} chunk(s)"
        )
        return all_chunks
