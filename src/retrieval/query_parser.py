"""
LLM-based Query Parser for LegalMail RAG.

Transforms natural language questions into structured queries with metadata filters
(sender, date range, thread, etc.) for Hybrid Search.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict

from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings

_QUERY_PARSER_PROMPT = """
Eres un asistente que procesa consultas de búsqueda sobre correos electrónicos legales corporativos.
Tu tarea es analizar la pregunta del usuario y extraer los filtros de búsqueda y la consulta semántica.

Reglas de extracción:
1. semantic_query: La pura intención de búsqueda (el tema principal, excluyendo nombres o fechas si se usan solo como filtro).
2. from_name: Si el usuario menciona quién envió el correo (ej. "Enviado por Jaime", "Qué dijo Isabel"). Usa solo el nombre de pila o apellido.
3. thread_id: Si el usuario menciona un proyecto, litigio, expediente o empresa (ej. "Proyecto Ámbar", "2026/SOC/1187"). 
   - IMPORTANTE: Si es un número de expediente (ej. "2026/SOC/1187" o "2026/ARR/0341"), devuélvelo EXACTAMENTE COMO APARECE, no lo conviertas a snake_case.
   - Si es un nombre de proyecto (ej. "Proyecto Ámbar"), conviértelo a snake_case (ej. "proyecto_ambar").
4. date: Si el usuario menciona un año y mes. Usa formato YYYY-MM.

Responde ÚNICAMENTE con un objeto JSON válido con este esquema exacto (omite las claves de filtros que no apliquen, NO las pongas nulas, simplemente no las incluyas en "filters"):
{{
    "semantic_query": "texto a buscar sin filtros",
    "filters": {{
        "from_name": "nombre",
        "thread_id": "2026/SOC/1187",
        "date": "2026-03"
    }}
}}

Pregunta del usuario: {query}
"""

class QueryParser:
    """Parses natural language into structured RAG queries with metadata filters."""

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings
        self.llm = ChatBedrock(
            model_id=settings.llm_model_id,
            region_name=settings.aws_default_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            model_kwargs={"temperature": 0.0, "max_tokens": 150},
        )
        self.prompt = ChatPromptTemplate.from_template(_QUERY_PARSER_PROMPT)
        self.chain = self.prompt | self.llm

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language into semantic query + filters."""
        default_result = {"semantic_query": query, "filters": {}}
        try:
            response = self.chain.invoke({"query": query})
            text = response.content.strip()
            
            # Clean markdown JSON format if present
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            elif text.startswith("```"):
                text = text.replace("```", "").strip()
            
            result = json.loads(text)
            
            # Ensure proper structure
            if "semantic_query" not in result:
                result["semantic_query"] = query
            if "filters" not in result or not isinstance(result["filters"], dict):
                result["filters"] = {}
                
            logger.info(f"Query parsed: {result}")
            return result
        except Exception as exc:
            logger.warning(f"Failed to parse query, falling back to pure semantic: {exc}")
            return default_result
