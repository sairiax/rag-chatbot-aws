"""
Metadata-aware text splitter.

Splits LangChain Documents into smaller chunks using RecursiveCharacterTextSplitter
and enriches each chunk with positional metadata (chunk_index, total_chunks, etc.)
so downstream retrieval can surface provenance information.

Additionally provides EmailAwareTextSplitter to inject context in split emails.
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
        """Split a list of Documents and enrich chunk metadata."""
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


class EmailAwareTextSplitter(MetadataAwareTextSplitter):
    """Splits emails while prepending contextual header to each chunk."""

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split email documents and inject context headers if they are partitioned."""
        all_chunks: List[Document] = []

        for doc in documents:
            chunks = self._splitter.split_documents([doc])
            total = len(chunks)

            is_email = doc.metadata.get("file_type") == "email"

            for idx, chunk in enumerate(chunks):
                # Only prepend context to chunks IF the document is a multi-chunk email
                if total > 1 and is_email:
                    sender = doc.metadata.get("from_name") or doc.metadata.get("from_email", "Desconocido")
                    recipient = doc.metadata.get("to_name") or doc.metadata.get("to_email", "Desconocido")
                    date = doc.metadata.get("date", "Sin fecha")
                    subject = doc.metadata.get("subject", "Sin asunto")

                    header = f"[Contexto fragmento {idx + 1}/{total}: Email de {sender} a {recipient} | {date} | Asunto: {subject}]\n---\n"
                    chunk.page_content = f"{header}{chunk.page_content.lstrip()}"
                
                chunk.metadata.update(
                    {
                        "chunk_index": idx,
                        "total_chunks": total,
                        "chunk_size": len(chunk.page_content),
                    }
                )
                all_chunks.append(chunk)

        logger.info(
            f"EmailAware split: {len(documents)} document(s) → {len(all_chunks)} chunk(s)"
        )
        return all_chunks
