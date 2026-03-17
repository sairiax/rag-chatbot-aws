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

        def _get_docs_with_scores(query: str) -> List[Document]:
            # 1. Similarity Search
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, k=top_k, filter=filter_dict
            )
            
            docs = []
            for doc, score in docs_with_scores:
                doc.metadata["similarity_score"] = score
                docs.append(doc)
            
            # 2. LLM Reranking & Filtering
            if docs:
                logger.info(f"Reranking {len(docs)} documents for query: {query}")
                docs = self._rerank_documents(query, docs)
                
            return docs

        logger.debug(
            f"Building retriever (with Reranking) — search_type={search_type}, k={top_k}, filter={filter_dict}"
        )
        
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(_get_docs_with_scores)

    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """Use LLM to score and filter documents by relevance."""
        if not documents:
            return []

        # We use a simple prompt for the LLM to act as a reranker
        from langchain_aws import ChatBedrock
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = ChatBedrock(
            model_id=self.settings.llm_model_id,
            model_kwargs={"temperature": 0},
            region_name=self.settings.aws_default_region,
            aws_access_key_id=self.settings.aws_access_key_id,
            aws_secret_access_key=self.settings.aws_secret_access_key,
        )

        prompt = ChatPromptTemplate.from_template("""
        Eres un experto legal encargado de evaluar la relevancia de fragmentos de correos electrónicos para una consulta específica.
        
        Consulta: {query}
        
        Evalúa cada fragmento a continuación. Para cada uno, determina si aporta información ÚTIL para responder a la consulta.
        Responde exclusivamente con una lista de índices (0, 1, 2...) de los fragmentos que SI son relevantes, separados por comas.
        Si NINGUNO es relevante, responde con "NINGUNO".
        
        Fragmentos:
        {fragments}
        
        Índices relevantes:""")

        fragments_text = ""
        for i, doc in enumerate(documents):
            fragments_text += f"[{i}] RELEVANZA_INICIAL: {doc.metadata.get('similarity_score', 0):.2f}\nCONTENIDO: {doc.page_content}\n---\n"

        try:
            response = llm.invoke(prompt.format(query=query, fragments=fragments_text))
            content = response.content.strip()
            
            if "NINGUNO" in content.upper():
                logger.warning("Reranker decided NO documents are relevant")
                return []
            
            # Parse indices
            import re
            indices = [int(i.strip()) for i in re.findall(r'\d+', content)]
            
            reranked_docs = []
            for idx in indices:
                if 0 <= idx < len(documents):
                    reranked_docs.append(documents[idx])
            
            logger.info(f"Reranker filtered {len(documents)} -> {len(reranked_docs)} documents")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error during Reranking: {e}")
            # Fallback to original documents if reranking fails
            return documents

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
