"""Shared utility helpers."""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document


def format_sources_text(documents: List[Document]) -> str:
    """Format source documents as a human-readable string."""
    if not documents:
        return "No sources found."

    lines: List[str] = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata
        name = meta.get("filename") or meta.get("source", "Unknown")
        lines.append(f"{i}. {name}")
        if "page" in meta:
            lines.append(f"   Page: {meta['page'] + 1}")
        if "slide_number" in meta:
            lines.append(f"   Slide: {meta['slide_number']}")
        lines.append(f"   Preview: {doc.page_content[:200].strip()}…")
        lines.append("")

    return "\n".join(lines)


def build_metadata_filter(
    file_type: str | None = None,
    source: str | None = None,
    from_name: str | None = None,
    thread_id: str | None = None,
    date: str | None = None,
) -> dict | None:
    """Build a ChromaDB metadata filter dict from optional field values.

    Returns ``None`` when no filter is active (no restriction).
    """
    conditions: List[dict] = []

    if file_type:
        conditions.append({"file_type": file_type})
    if source:
        conditions.append({"source": source})
        
    # Exact match operators
    if from_name:
        conditions.append({"from_name": from_name})
    if thread_id:
        conditions.append({"thread_id": thread_id})
        
    # Chroma standard operators: exact match for year and month
    if date:
        try:
            # Expected format: "YYYY-MM"
            parts = date.split("-")
            if len(parts) == 2:
                conditions.append({"year": int(parts[0])})
                conditions.append({"month": int(parts[1])})
            else:
                # Fallback to exact string match if format is different
                conditions.append({"date": date})
        except (ValueError, TypeError):
            # Fallback if parsing fails
            conditions.append({"date": date})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
