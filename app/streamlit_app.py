"""
RAG University — Streamlit conversational interface.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

import streamlit as st

# ── Make sure the project root is on sys.path ────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from config.settings import Settings
from src.chains.rag_chain import ConversationalRAGChain
from src.embeddings.aws_embeddings import get_embeddings
from src.ingestion.document_loader import MultiFormatDocumentLoader
from src.ingestion.ocr_processor import OCRProcessor
from src.ingestion.text_splitter import MetadataAwareTextSplitter
from src.utils.helpers import build_metadata_filter
from src.vectorstore.chroma_store import ChromaVectorStore

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG University",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS tweaks ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .source-badge {
        display: inline-block;
        background: #0066cc;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: .5px;
    }
    .source-preview {
        color: #555;
        font-size: 13px;
        border-left: 3px solid #0066cc;
        padding-left: 8px;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached resource: heavy components ────────────────────────────────────────

@st.cache_resource(show_spinner="Conectando a AWS Bedrock y ChromaDB…")
def _load_base_components():
    """Load Settings, embeddings and vector store once per process."""
    settings = Settings()
    embeddings = get_embeddings(settings)
    vector_store = ChromaVectorStore(embeddings, settings)
    return settings, vector_store


def _get_or_create_rag_chain(
    settings: Settings, vector_store: ChromaVectorStore
) -> ConversationalRAGChain:
    """Return the RAG chain stored in session state (creates it on first call)."""
    if "rag_chain" not in st.session_state:
        # Share a single message store so memory survives filter changes
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

def render_sidebar(
    settings: Settings,
    vector_store: ChromaVectorStore,
    rag_chain: ConversationalRAGChain,
) -> Optional[str]:
    """Render the sidebar. Returns the selected file-type filter (or None)."""

    st.sidebar.image(
        "https://img.icons8.com/color/96/graduation-cap.png", width=60
    )
    st.sidebar.title("RAG University")
    st.sidebar.caption("Asistente inteligente sobre documentos")
    st.sidebar.divider()

    # ── Collection stats ──────────────────────────────────────────────────
    try:
        stats = vector_store.get_collection_stats()
        st.sidebar.metric("📚 Fragmentos indexados", stats["count"])
    except Exception:
        st.sidebar.info("Base de conocimiento vacía")

    st.sidebar.divider()

    # ── Document upload ───────────────────────────────────────────────────
    st.sidebar.subheader("📂 Cargar documentos")
    uploaded_files = st.sidebar.file_uploader(
        "Selecciona uno o varios archivos",
        accept_multiple_files=True,
        type=["pdf", "docx", "doc", "pptx", "ppt", "txt", "md", "png", "jpg", "jpeg"],
        help="PDF, Word, PowerPoint, texto, imágenes",
    )

    use_ocr = st.sidebar.toggle(
        "Activar OCR (para PDFs escaneados e imágenes)",
        value=True,
        help="Usa AWS Textract para extraer texto de documentos escaneados",
    )

    if st.sidebar.button("📥 Indexar documentos", type="primary", use_container_width=True):
        if not uploaded_files:
            st.sidebar.warning("Selecciona al menos un archivo.")
        else:
            _index_uploaded_files(uploaded_files, use_ocr, settings, vector_store)

    st.sidebar.divider()

    # ── Retrieval filter ──────────────────────────────────────────────────
    st.sidebar.subheader("🔍 Filtro de recuperación")
    filter_options = ["Todos los tipos", "pdf", "docx", "pptx", "txt", "png/jpg"]
    selected = st.sidebar.selectbox("Filtrar por tipo de archivo", filter_options)
    active_filter = None if selected == "Todos los tipos" else selected.split("/")[0]

    st.sidebar.divider()

    # ── Session management ────────────────────────────────────────────────
    st.sidebar.subheader("🗑️ Conversación")
    if st.sidebar.button("Limpiar historial", use_container_width=True):
        rag_chain.clear_session(_get_session_id())
        st.session_state["messages"] = []
        st.rerun()

    return active_filter


# ── Document indexing ─────────────────────────────────────────────────────────

def _index_uploaded_files(
    uploaded_files,
    use_ocr: bool,
    settings: Settings,
    vector_store: ChromaVectorStore,
) -> None:
    ocr_processor = OCRProcessor(settings) if use_ocr else None
    loader = MultiFormatDocumentLoader(ocr_processor=ocr_processor)
    splitter = MetadataAwareTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    progress = st.sidebar.progress(0, text="Iniciando…")
    status = st.sidebar.empty()
    all_chunks = []

    for i, uploaded_file in enumerate(uploaded_files):
        status.text(f"⏳ Procesando: {uploaded_file.name}")

        # Save upload to a temp file so loaders can access it by path
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            docs = loader.load_file(tmp_path)
            # Restore the original filename in metadata
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
                doc.metadata["filename"] = uploaded_file.name
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
        except Exception as exc:
            st.sidebar.error(f"❌ Error en '{uploaded_file.name}': {exc}")
        finally:
            os.unlink(tmp_path)

        progress.progress((i + 1) / len(uploaded_files), text=f"{i+1}/{len(uploaded_files)} archivos")

    if all_chunks:
        vector_store.add_documents(all_chunks)
        status.success(f"✅ {len(all_chunks)} fragmentos indexados correctamente")
        progress.empty()
    else:
        status.warning("No se extrajeron fragmentos. Revisa los archivos.")
        progress.empty()


# ── Chat ──────────────────────────────────────────────────────────────────────

def render_chat(rag_chain: ConversationalRAGChain, active_filter: Optional[str]) -> None:
    # Apply (or remove) retrieval filter
    filter_dict = build_metadata_filter(file_type=active_filter)
    rag_chain.set_filter(filter_dict)

    st.title("💬 RAG University — Asistente inteligente")
    if active_filter:
        st.caption(f"🔍 Filtrando por documentos de tipo: **{active_filter.upper()}**")

    # Initialise message history in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Render existing messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    # New user input
    if prompt := st.chat_input("Escribe tu pregunta aquí…"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Buscando en la base de conocimiento…"):
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
    with st.expander(f"📎 Fuentes consultadas ({len(sources)})", expanded=False):
        for i, doc in enumerate(sources, start=1):
            meta = doc.metadata
            filename = meta.get("filename") or meta.get("source", "Desconocido")
            file_type = meta.get("file_type", "?").upper()

            col_info, col_badge = st.columns([5, 1])
            with col_info:
                st.markdown(f"**{i}. {filename}**")
                detail_parts = []
                if "page" in meta:
                    detail_parts.append(f"Página {meta['page'] + 1}")
                if "slide_number" in meta:
                    detail_parts.append(f"Diapositiva {meta['slide_number']}")
                if "chunk_index" in meta:
                    detail_parts.append(
                        f"Fragmento {meta['chunk_index'] + 1}/{meta.get('total_chunks', '?')}"
                    )
                if "ocr_applied" in meta:
                    detail_parts.append("OCR aplicado")
                if detail_parts:
                    st.caption(" · ".join(detail_parts))
                st.markdown(
                    f'<p class="source-preview">{doc.page_content[:350].strip()}…</p>',
                    unsafe_allow_html=True,
                )
            with col_badge:
                st.markdown(
                    f'<span class="source-badge">{file_type}</span>',
                    unsafe_allow_html=True,
                )

            if i < len(sources):
                st.divider()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    try:
        settings, vector_store = _load_base_components()
        rag_chain = _get_or_create_rag_chain(settings, vector_store)
        active_filter = render_sidebar(settings, vector_store, rag_chain)
        render_chat(rag_chain, active_filter)
    except Exception as exc:
        st.error(f"❌ Error al inicializar el sistema: {exc}")
        st.info(
            "Asegúrate de que el archivo `.env` existe y tiene las credenciales de AWS configuradas. "
            "Consulta `.env.example` para ver las variables necesarias."
        )
        with st.expander("Detalle del error"):
            st.exception(exc)


if __name__ == "__main__":
    main()
