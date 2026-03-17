"""
Smart retriever with optional metadata filtering.

Wraps ChromaVectorStore and exposes a configurable LangChain BaseRetriever
that can filter by any metadata field (e.g. file_type, source, slide_number).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings
    from src.vectorstore.chroma_store import ChromaVectorStore


class SmartRetriever:
    """Configurable retriever with metadata-filter support."""

    def __init__(self, vector_store: "ChromaVectorStore", settings: "Settings") -> None:
        self.vector_store = vector_store
        self.settings = settings

    # ──────────────────────────────────────────────────────────────────────
    # LangChain retriever interface
    # ──────────────────────────────────────────────────────────────────────

    def get_retriever(
        self,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        search_type: str = "similarity",
    ):
        """Return a LangChain-compatible BaseRetriever.

        Args:
            k: Number of documents to retrieve (defaults to settings.top_k).
            filter_dict: ChromaDB metadata filter, e.g. ``{"file_type": "pdf"}``.
                         Multiple conditions: ``{"$and": [{"file_type": "pdf"}, ...]}``.
            search_type: "similarity" | "mmr" | "similarity_score_threshold".
        """
        top_k = k or self.settings.top_k

        search_kwargs: Dict[str, Any] = {"k": top_k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        logger.debug(
            f"Building retriever — search_type={search_type}, k={top_k}, filter={filter_dict}"
        )
        return self.vector_store.store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Direct search helpers (useful for scripts / debugging)
    # ──────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Retrieve documents for *query* with optional metadata filter."""
        top_k = k or self.settings.top_k
        return self.vector_store.similarity_search(query, k=top_k, filter=filter_dict)

    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents with cosine-relevance scores."""
        top_k = k or self.settings.top_k
        return self.vector_store.similarity_search_with_score(
            query, k=top_k, filter=filter_dict
        )
