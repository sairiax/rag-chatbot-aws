"""
Conversational RAG chain with per-session memory.

Architecture (LCEL):
  1. history_aware_retriever  — rewrites the user question as a standalone
                                question given the chat history, then retrieves
                                relevant chunks from ChromaDB.
  2. question_answer_chain    — stuffs retrieved chunks into a QA prompt and
                                calls the Bedrock LLM.
  3. RunnableWithMessageHistory — persists the chat history per session_id,
                                making the chatbot stateful and conversational.

The internal message store (_store) is kept separate from the LangChain chain
object so that rebuilding the chain (e.g. when the retrieval filter changes)
does NOT reset the conversation history.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_aws import ChatBedrock
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings
    from src.vectorstore.chroma_store import ChromaVectorStore

# ── Prompt templates ──────────────────────────────────────────────────────────

_CONTEXTUALIZE_SYSTEM = """
You are a helpful assistant. Given the conversation history and the user's latest \
question (which may refer to previous messages), reformulate the question as a \
clear, self-contained question that can be understood without any prior context.

Do NOT answer the question — only rewrite it if necessary. \
If it is already self-contained, return it as-is.
""".strip()

_QA_SYSTEM = """
Eres un asistente legal corporativo senior (LegalMail RAG). Tienes acceso a un archivo estructurado de correos electrónicos y expedientes.

INSTRUCCIONES CRÍTICAS:
1. Usa ÚNICAMENTE el contexto recuperado a continuación para responder.
2. Si la respuesta no está en el contexto, dilo claramente y no inventes información.
3. Sé analítico, exhaustivo y profesional. No te limites a dar respuestas cortas de una línea; sintetiza la información, explica el contexto de la situación y detalla las acciones u opiniones mencionadas en los correos.
4. Usa formato Markdown (negritas para nombres/fechas, listas con viñetas) para hacer tu respuesta fácil de leer.
5. Cita SIEMPRE tus fuentes mencionando explícitamente el remitente, el destinatario o el asunto del correo (ej. "Según el correo enviado por Jaime Cortés a Isabel...").

--- CONTEXTO RECUPERADO ---
{context}
""".strip()


class ConversationalRAGChain:
    """Stateful RAG chatbot backed by AWS Bedrock and ChromaDB."""

    def __init__(
        self,
        vector_store: "ChromaVectorStore",
        settings: "Settings",
        message_store: Optional[Dict[str, ChatMessageHistory]] = None,
    ) -> None:
        self.vector_store = vector_store
        self.settings = settings

        # Shared message store — survives chain rebuilds (e.g. on filter change)
        self._store: Dict[str, ChatMessageHistory] = message_store or {}

        self._current_filter: Optional[Dict[str, Any]] = None
        self._chain = self._build_chain()

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def invoke(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        """Process a user question and return the answer + source documents."""
        logger.info(f"[session={session_id}] Invoking RAG chain")

        # 1. Parse query for metadata filters
        from src.retrieval.query_parser import QueryParser
        from src.utils.helpers import build_metadata_filter
        
        parser = QueryParser(self.settings)
        parsed = parser.parse_query(question)
        
        semantic_query = parsed.get("semantic_query", question)
        filters = parsed.get("filters", {})
        
        # 2. Update filter dict for the chain (combining user UI filters and LLM filters)
        # We can merge them. Currently self._current_filter holds the UI filter.
        # Let's keep it simple: merge UI filter and LLM filters
        merged_filters = {}
        if self._current_filter:
            merged_filters.update(self._current_filter)
        if filters:
            # Rebuild using our helper to get the proper Chromadb operators
            llm_filter = build_metadata_filter(
                from_name=filters.get("from_name"),
                thread_id=filters.get("thread_id"),
                date=filters.get("date")
            )
            if llm_filter:
                if merged_filters:
                    # ChromaDB uses {"$and": [{}, {}]} for combining
                    merged_filters = {"$and": [merged_filters, llm_filter]}
                else:
                    merged_filters = llm_filter
                
        # Only rebuild chain if combined filter changed
        if merged_filters != self._current_filter:
            # We temporarily swap it just for this invocation, or permanently?
            # It's better to build a temporary retriever for execution, but since
            # LCEL chains bake the retriever in, we must rebuild the chain if we change the filter.
            self._chain = self._build_chain(merged_filters)

        result = self._chain.invoke(
            {"input": semantic_query},
            config={"configurable": {"session_id": session_id}},
        )
        
        # Restore chain using just the UI filter if we merged
        if merged_filters != self._current_filter:
            self._chain = self._build_chain(self._current_filter)

        return {
            "answer": result["answer"],
            "source_documents": result.get("context", []),
            "session_id": session_id,
            "parsed_query": parsed
        }

    def set_filter(self, filter_dict: Optional[Dict[str, Any]]) -> None:
        """Update the metadata filter and rebuild the internal chain.

        The message history is preserved across rebuilds.

        Args:
            filter_dict: ChromaDB where-clause, e.g. ``{"file_type": "pdf"}``.
                         Pass ``None`` to remove the filter.
        """
        if filter_dict != self._current_filter:
            self._current_filter = filter_dict
            self._chain = self._build_chain()
            logger.info(f"Retrieval filter updated: {filter_dict}")

    def clear_session(self, session_id: str) -> None:
        """Clear the chat history for a specific session."""
        if session_id in self._store:
            self._store[session_id].clear()
            logger.info(f"Cleared history for session '{session_id}'")

    def get_session_messages(self, session_id: str) -> List[BaseMessage]:
        """Return all messages in a session's history."""
        return self._get_session_history(session_id).messages

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]

    def _build_llm(self) -> ChatBedrock:
        return ChatBedrock(
            model=self.settings.llm_model_id,
            region=self.settings.aws_default_region,
            aws_access_key_id=self.settings.aws_access_key_id,
            aws_secret_access_key=self.settings.aws_secret_access_key,
            model_kwargs={
                "temperature": self.settings.llm_temperature,
                "max_tokens": self.settings.llm_max_tokens,
            },
        )

    def _build_retriever(self, filter_dict: Optional[Dict[str, Any]] = None):
        from src.retrieval.retriever import SmartRetriever

        smart = SmartRetriever(self.vector_store, self.settings)
        # Fall back to self._current_filter if none provided explicitly
        active_filter = filter_dict if filter_dict is not None else self._current_filter
        return smart.get_retriever(filter_dict=active_filter)

    def _build_chain(self, filter_dict: Optional[Dict[str, Any]] = None) -> RunnableWithMessageHistory:
        llm = self._build_llm()
        retriever = self._build_retriever(filter_dict=filter_dict)

        # Step 1 — History-aware retriever
        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _CONTEXTUALIZE_SYSTEM),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_prompt
        )

        # Step 2 — QA chain
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _QA_SYSTEM),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Step 3 — Full retrieval chain
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        # Step 4 — Wrap with session-aware message history
        return RunnableWithMessageHistory(
            rag_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
