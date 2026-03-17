"""
LegalMail RAG — Streamlit conversational interface.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st

# ── Make sure the project root is on sys.path ────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from config.settings import Settings
from src.chains.rag_chain import ConversationalRAGChain
from src.embeddings.aws_embeddings import get_embeddings
from src.utils.helpers import build_metadata_filter
from src.vectorstore.chroma_store import ChromaVectorStore

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LegalMail RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS tweaks ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .source-badge {
        display: inline-block;
        background: #2E4053;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: .5px;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    .source-badge-spam {
        background: #C0392B;
    }
    .source-preview {
        color: #ddd;
        font-size: 13px;
        border-left: 3px solid #3498DB;
        padding-left: 8px;
        margin-top: 4px;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Conectando a AWS Bedrock y ChromaDB…")
def _load_base_components():
    settings = Settings()
    embeddings = get_embeddings(settings)
    vector_store = ChromaVectorStore(embeddings, settings)
    return settings, vector_store

def _get_or_create_rag_chain(
    settings: Settings, vector_store: ChromaVectorStore
) -> ConversationalRAGChain:
    if "rag_chain" not in st.session_state:
        st.session_state["message_store"] = {}
        st.session_state["rag_chain"] = ConversationalRAGChain(
            vector_store=vector_store,
            settings=settings,
            message_store=st.session_state["message_store"],
        )
    return st.session_state["rag_chain"]

def _get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]

# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(vector_store: ChromaVectorStore) -> Dict[str, Any]:
    st.sidebar.title("⚖️ LegalMail RAG")
    st.sidebar.caption("Inteligencia Operativa para Correos Legales")
    st.sidebar.divider()

    # ── Collection stats ──────────────────────────────────────────────────
    try:
        stats = vector_store.get_collection_stats()
        st.sidebar.metric("📚 Fragmentos indexados", stats["count"])
    except Exception:
        st.sidebar.info("Base de conocimiento vacía")

    st.sidebar.divider()

    # ── Advanced Filters ──────────────────────────────────────────────────
    st.sidebar.subheader("🔍 Filtros manuales")
    st.sidebar.caption("El LLM también aplica filtros automáticamente según tu pregunta.")
    
    # 1. Thread Filter
    unique_threads = ["Todos"] + vector_store.get_unique_metadata_values("thread_id")
    selected_thread = st.sidebar.selectbox("Filtro por Hilo (Proyecto/Asunto)", unique_threads)
    
    # 2. Sender Filter
    unique_senders = ["Todos"] + vector_store.get_unique_metadata_values("from_name")
    selected_sender = st.sidebar.selectbox("Filtro por Remitente", unique_senders)

    # 3. Date Filter (Mes)
    unique_dates = vector_store.get_unique_metadata_values("date")
    unique_months = sorted(list(set(d[:7] for d in unique_dates if len(d) >= 7)))
    unique_months = ["Todos"] + unique_months
    selected_month = st.sidebar.selectbox("Filtro por Mes (YYYY-MM)", unique_months)

    filters = {}
    if selected_thread != "Todos":
        filters["thread_id"] = selected_thread
    if selected_sender != "Todos":
        filters["from_name"] = selected_sender
    if selected_month != "Todos":
        filters["date"] = selected_month

    st.sidebar.divider()

    # ── Session management ────────────────────────────────────────────────
    st.sidebar.subheader("🗑️ Conversación")
    if st.sidebar.button("Limpiar historial", use_container_width=True):
        st.session_state["rag_chain"].clear_session(_get_session_id())
        st.session_state["messages"] = []
        st.rerun()
        
    return filters

# ── Chat ──────────────────────────────────────────────────────────────────────

def render_chat(rag_chain: ConversationalRAGChain, manual_filters: Dict[str, Any]) -> None:
    # Setup base manual filters
    filter_dict = build_metadata_filter(**manual_filters) if manual_filters else None
    rag_chain.set_filter(filter_dict)

    st.title("💬 Búsqueda de Correos y Expedientes")
    if manual_filters:
        st.caption(f"Aplicando filtros manuales: {manual_filters}")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        
        # Suggested initial questions
        with st.chat_message("assistant"):
            st.markdown("¡Hola! Soy tu asistente legal corporativo. Puedes preguntarme sobre proyectos, contratos o due diligence. Por ejemplo:")
            st.markdown("- *¿Cuáles son los riesgos del Proyecto Ámbar?*")
            st.markdown("- *¿Qué dijo Jaime Cortés sobre los trabajadores falsos autónomos?*")
            st.markdown("- *Resume el hilo de correos sobre la due diligence de Tecnalia.*")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("sources"):
                    _render_sources(msg["sources"])

    if prompt := st.chat_input("Busca información en los correos..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analizando pregunta y buscando expedientes…"):
                try:
                    result = rag_chain.invoke(prompt, session_id=_get_session_id())
                    answer = result["answer"]
                    sources = result["source_documents"]
                    st.markdown(answer)
                        
                    if sources:
                        _render_sources(sources)

                    st.session_state["messages"].append(
                        {
                            "role": "assistant", 
                            "content": answer, 
                            "sources": sources
                        }
                    )

                except Exception as exc:
                    err = f"⚠️ Error al generar respuesta: {exc}"
                    st.error(err)
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": err, "sources": []}
                    )

def _render_sources(sources: List) -> None:
    if not sources:
        return
    with st.expander(f"📎 Correos consultados ({len(sources)})", expanded=False):
        for i, doc in enumerate(sources, start=1):
            meta = doc.metadata
            sender = meta.get("from_name") or meta.get("from_email", "Desconocido")
            date = meta.get("date", "Sin fecha")
            subject = meta.get("subject", "Sin asunto")
            thread = meta.get("thread_id", "ninguno")
            is_spam = meta.get("is_spam", False)

            col_info, col_badge = st.columns([5, 1])
            with col_info:
                st.markdown(f"**{i}. De: {sender}** (Fecha: {date})")
                st.caption(f"Asunto: *{subject}*")
                
                parts = []
                if "chunk_index" in meta:
                    parts.append(f"Fragmento {meta['chunk_index'] + 1}/{meta.get('total_chunks', '?')}")
                if meta.get("has_attachments"):
                    parts.append(f"📎 {meta.get('attachments')}")
                
                # Format cosine similarity score as percentage
                if "cosine_score" in meta:
                    score = meta["cosine_score"]
                    # Usually Chroma cosine distance is returned as distance (0 = identical, 2 = opposite direction)
                    # Convert distance to similarity percentage: (2 - distance) / 2 * 100
                    # or if it returns 1 - cosine_similarity (0 = identical, 1 = orthogonal, 2 = opposite)
                    # We assume it's distance, so similarity is roughly (1 - score/2) * 100
                    # or simpler: (1 - score) * 100 if score is between 0 and 1
                    try:
                        sim_pct = max(0, min(100, (1.0 - float(score)) * 100.0))
                        parts.append(f"🎯 Relevancia: {sim_pct:.1f}%")
                    except (ValueError, TypeError):
                        pass
                
                if parts:
                    st.caption(" · ".join(parts))
                    
                st.markdown(
                    f'<div class="source-preview">{doc.page_content[:400].strip()}…</div>',
                    unsafe_allow_html=True,
                )
            with col_badge:
                st.markdown(
                    f'<span class="source-badge">Hilo: {thread}</span>',
                    unsafe_allow_html=True,
                )
                if is_spam:
                    st.markdown(
                        f'<span class="source-badge source-badge-spam">SPAM</span>',
                        unsafe_allow_html=True,
                    )

            if i < len(sources):
                st.divider()

def main() -> None:
    try:
        settings, vector_store = _load_base_components()
        rag_chain = _get_or_create_rag_chain(settings, vector_store)
        manual_filters = render_sidebar(vector_store)
        render_chat(rag_chain, manual_filters)
    except Exception as exc:
        st.error(f"❌ Error al inicializar LegalMail RAG: {exc}")
        st.info("Asegúrate de que la DB de Chroma esté accesible y el índice esté generado.")

if __name__ == "__main__":
    main()
