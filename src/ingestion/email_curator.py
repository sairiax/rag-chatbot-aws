"""
LLM-based Email Curator for LegalMail RAG.

Uses AWS Bedrock and LangChain to classify emails as LEGÍTIMO or SPAM,
discarding phishing, newsletters, and irrelevant content before indexing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from langchain_aws import ChatBedrock
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings

_CURATION_SYSTEM_PROMPT = """
Eres un analista de datos legales. Tu tarea es analizar el siguiente correo electrónico y clasificarlo.
Debes determinar si es un correo relevante para el contexto empresarial/legal o si es correo basura.

Clasificaciones permitidas:
- LEGITIMO: Correo profesional, corporativo, legal, de proyectos, contratos, due diligence, litigios, etc.
- SPAM: Publicidad, phishing, estafas, correos personales sin valor laboral, notificaciones irrelevantes.

Responde ÚNICAMENTE con un objeto JSON válido con este esquema exacto:
{{"classification": "LEGITIMO", "reason": "Breve justificación"}}
o
{{"classification": "SPAM", "reason": "Breve justificación"}}

Email a evaluar:
---
{email_content}
---
"""


class EmailCurator:
    """Classifies emails to filter out noise before RAG indexing."""

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings
        self.llm = ChatBedrock(
            model_id=settings.llm_model_id,
            region_name=settings.aws_default_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            model_kwargs={
                "temperature": 0.0,
                "max_tokens": 150,
            },
        )
        self.prompt = ChatPromptTemplate.from_template(_CURATION_SYSTEM_PROMPT)
        self.chain = self.prompt | self.llm

    def classify_document(self, doc: Document) -> bool:
        """Classify a single document. Returns True if LEGITIMO, False if SPAM."""
        content = doc.page_content
        meta = doc.metadata
        
        subject = meta.get("subject", "Sin asunto")
        sender = meta.get("from_name") or meta.get("from_email", "Desconocido")

        email_preview = f"De: {sender}\nAsunto: {subject}\n\nCuerpo:\n{content[:800]}"

        try:
            response = self.chain.invoke({"email_content": email_preview})
            text = response.content.strip()
            
            # Clean markdown JSON formatting if the LLM output it
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            elif text.startswith("```"):
                text = text.replace("```", "").strip()

            result = json.loads(text)
            is_legit = result.get("classification", "").strip().upper() == "LEGITIMO"
            
            doc.metadata["is_spam"] = not is_legit
            doc.metadata["curation_reason"] = result.get("reason", "No reason provided")
            
            return is_legit
        except Exception as exc:
            logger.warning(f"Curation failed for '{meta.get('filename')}': {exc}")
            # Fallback defensively: return True so we don't accidentally drop valid data if LLM fails
            doc.metadata["is_spam"] = False
            doc.metadata["curation_reason"] = "Error in LLM curation fallback"
            return True

    def filter_documents(
        self, documents: List[Document], report_path: str | Path = "data/curation_report.json"
    ) -> List[Document]:
        """Filter out SPAM documents and write a curation report to disk."""
        valid_docs: List[Document] = []
        report: List[Dict[str, Any]] = []

        logger.info(f"Curating {len(documents)} document(s) via LLM...")
        
        for idx, doc in enumerate(documents, start=1):
            filename = doc.metadata.get("filename", f"doc_{idx}")
            
            is_legit = self.classify_document(doc)
            
            entry = {
                "filename": filename,
                "subject": doc.metadata.get("subject", ""),
                "classification": "LEGITIMO" if is_legit else "SPAM",
                "reason": doc.metadata.get("curation_reason", "")
            }
            report.append(entry)
            
            if is_legit:
                valid_docs.append(doc)
            else:
                logger.info(f"🗑️ Spam detected: {filename} — {entry['reason']}")

        try:
            out_path = Path(report_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.error(f"Failed to save curation report: {exc}")
        
        logger.success(f"Curation complete: kept {len(valid_docs)}/{len(documents)} documents.")
        return valid_docs
