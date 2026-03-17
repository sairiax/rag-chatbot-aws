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
    page_title="LegalRAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS tweaks ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Project header ── */
    .project-header {
        background: linear-gradient(135deg, #1a2a3a 0%, #2E4053 100%);
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid #3d5166;
    }
    .project-header-left h1 {
        margin: 0;
        font-size: 24px;
        color: #ffffff;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    .project-header-left p {
        margin: 4px 0 0 0;
        font-size: 13px;
        color: #a8bfd0;
        max-width: 620px;
        line-height: 1.5;
    }
    .project-header-right {
        text-align: right;
        flex-shrink: 0;
        margin-left: 20px;
    }
    .version-badge {
        display: inline-block;
        background: #17a589;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .stack-pills {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        justify-content: flex-end;
        margin-top: 6px;
    }
    .stack-pill {
        background: rgba(255,255,255,0.08);
        color: #c8dce8;
        padding: 3px 9px;
        border-radius: 10px;
        font-size: 10px;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.12);
    }

    /* ── Source badges ── */
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

    /* ── Suggested questions ── */
    .suggested-label {
        font-size: 12px;
        font-weight: 600;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 8px;
    }
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 20px !important;
        font-size: 13px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Suggested questions ───────────────────────────────────────────────────────
SUGGESTED_QUESTIONS = [
    "¿Cuáles son los riesgos del Proyecto Ámbar?",
    "¿Qué dijo Jaime Cortés sobre los falsos autónomos?",
    "Resume el hilo sobre la due diligence de Tecnalia.",
    "¿Qué cláusulas aparecen en el contrato de NDA?",
    "¿Quién aprobó los honorarios del Q3?",
    "Lista los correos con archivos adjuntos de enero.",
]


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


# ── Project header ────────────────────────────────────────────────────────────


def render_project_header() -> None:
    st.markdown(
        """
        <div class="project-header">
            <div class="project-header-left">
                <h1>⚖️ LegalRAG</h1>
                <p>
                    Sistema de Recuperación Aumentada de Generación (RAG) para
                    departamentos legales. Analiza miles de correos, contratos y
                    expedientes en segundos mediante lenguaje natural, con memoria
                    conversacional y citación de fuentes.
                </p>
            </div>
            <div class="project-header-right">
                <div class="version-badge">v0.1 · Beta</div>
                <div class="stack-pills">
                    <span class="stack-pill">AWS Bedrock</span>
                    <span class="stack-pill">Claude 3.5</span>
                    <span class="stack-pill">ChromaDB</span>
                    <span class="stack-pill">LangChain</span>
                    <span class="stack-pill">Streamlit</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────


def render_sidebar(vector_store: ChromaVectorStore) -> Dict[str, Any]:
    st.sidebar.title("⚖️ LegalRAG")
    st.sidebar.caption("Inteligencia Operativa para Correos Legales")
    st.sidebar.divider()

    # ── Collection stats ──────────────────────────────────────────────────
    try:
        stats = vector_store.get_collection_stats()
        st.sidebar.metric("📚 Fragmentos indexados", stats["count"])
    except Exception:
        st.sidebar.info("Base de conocimiento vacía")

    st.sidebar.divider()

    # ── Advanced Filters (below stats) ───────────────────────────────────
    st.sidebar.subheader("🔍 Filtros manuales")
    st.sidebar.caption(
        "El LLM también aplica filtros automáticamente según tu pregunta."
    )

    # 1. Thread Filter
    unique_threads = ["Todos"] + vector_store.get_unique_metadata_values("thread_id")
    selected_thread = st.sidebar.selectbox(
        "Filtro por Hilo (Proyecto/Asunto)", unique_threads
    )

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


def render_chat(
    rag_chain: ConversationalRAGChain, manual_filters: Dict[str, Any]
) -> None:
    # Setup base manual filters
    filter_dict = build_metadata_filter(**manual_filters) if manual_filters else None
    rag_chain.set_filter(filter_dict)

    st.title("💬 Búsqueda de Correos y Expedientes")
    if manual_filters:
        st.caption(f"Aplicando filtros manuales: {manual_filters}")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # ── Suggested questions (only shown when chat is empty) ───────────────
    if not st.session_state["messages"]:
        with st.chat_message("assistant"):
            st.markdown(
                "¡Hola! Soy tu asistente legal corporativo. "
                "Puedes preguntarme sobre proyectos, contratos o due diligence. "
                "Escribe tu pregunta o elige una sugerencia:"
            )

        st.markdown(
            '<p class="suggested-label">💡 Preguntas sugeridas</p>',
            unsafe_allow_html=True,
        )

        # Render in rows of 3
        rows = [
            SUGGESTED_QUESTIONS[i : i + 3]
            for i in range(0, len(SUGGESTED_QUESTIONS), 3)
        ]
        for row in rows:
            cols = st.columns(len(row))
            for col, question in zip(cols, row):
                with col:
                    if st.button(question, use_container_width=True):
                        st.session_state["pending_question"] = question
                        st.rerun()

        st.markdown("---")

    # ── Render existing messages ──────────────────────────────────────────
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    # ── Handle pending question from suggested button ─────────────────────
    if "pending_question" in st.session_state:
        prompt = st.session_state.pop("pending_question")
        _process_prompt(prompt, rag_chain)

    # ── Chat input ────────────────────────────────────────────────────────
    if prompt := st.chat_input("Busca información en los correos..."):
        _process_prompt(prompt, rag_chain)


def _process_prompt(prompt: str, rag_chain: ConversationalRAGChain) -> None:
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
                    {"role": "assistant", "content": answer, "sources": sources}
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
                    parts.append(
                        f"Fragmento {meta['chunk_index'] + 1}/{meta.get('total_chunks', '?')}"
                    )
                if meta.get("has_attachments"):
                    parts.append(f"📎 {meta.get('attachments')}")

                if "similarity_score" in meta:
                    try:
                        score = float(meta["similarity_score"])
                        sim_pct = max(0, min(100, score * 100.0))
                        if sim_pct >= 85:
                            color_emoji = "🟢"
                        elif sim_pct >= 70:
                            color_emoji = "🟡"
                        elif sim_pct >= 50:
                            color_emoji = "🟠"
                        else:
                            color_emoji = "🔴"
                        parts.append(f"{color_emoji} Relevancia: {sim_pct:.1f}%")
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
                        '<span class="source-badge source-badge-spam">SPAM</span>',
                        unsafe_allow_html=True,
                    )

            if i < len(sources):
                st.divider()


def main() -> None:
    try:
        settings, vector_store = _load_base_components()
        rag_chain = _get_or_create_rag_chain(settings, vector_store)
        render_project_header()
        manual_filters = render_sidebar(vector_store)
        render_chat(rag_chain, manual_filters)
    except Exception as exc:
        st.error(f"❌ Error al inicializar LegalRAG: {exc}")
        st.info(
            "Asegúrate de que la DB de Chroma esté accesible y el índice esté generado."
        )


if __name__ == "__main__":
    main()
