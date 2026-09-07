from typing import Any, Dict, List


def build_context(
    retrieved_results: List[Dict[str, Any]],
    *,
    application: bool = False,
) -> str:
    parts = []
    for index, item in enumerate(retrieved_results, start=1):
        document = str(item.get("document", ""))
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        source = (
            metadata.get("source")
            or metadata.get("document")
            or metadata.get("source_document")
            or "Unknown"
        )
        page = metadata.get("page") or metadata.get("source_page") or "Unknown"
        if application:
            content_type = metadata.get("content_type") or "text"
            extra = f"\nContent type: {content_type}\nChunk: {metadata.get('chunk_id', 'Unknown')}"
            if content_type == "table":
                extra += f"\nTable index: {metadata.get('table_index')}\nTable title: {metadata.get('table_title') or ''}"
            block = f"--- SOURCE {index} ---\nDocument: {source}\nPage: {page}{extra}\n\n{document}\n\n--- END SOURCE {index} ---"
        else:
            block = f"--- SOURCE {index} ---\nDocument: {source}\nPage: {page}\n\n{document}\n\n--- END SOURCE {index} ---"
        parts.append(block.strip())
    return "\n\n".join(parts)


def build_prompt(
    question: str,
    context: str,
    *,
    application: bool = False,
) -> str:
    if not application:
        return f"""
You are an enterprise document
question-answering assistant.

Answer the question using ONLY
the supplied evidence.

Do not invent facts.

If the evidence is insufficient,
clearly state that the available
evidence is insufficient.

QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}

ANSWER:
""".strip()
    if not str(context or "").strip():
        return f"""
You are answering from uploaded application documents only.
There is no retrieved evidence.
Reply with exactly:
The information needed to answer this question was not found in the accessible documents.

QUESTION:
{question}
""".strip()
    return f"""
You are an enterprise knowledge assistant for uploaded application documents only.

GROUNDING INSTRUCTIONS:
- Answer only from RETRIEVED CONTEXT below.
- Do not use unsupported outside knowledge.
- Do not fabricate numbers, dates, percentages, or citations.
- Preserve units and percentages exactly as written.
- When using tables, reason from table headers and cell values.
- When multiple documents are required, combine only supported evidence.
- Mention conflicts if sources disagree.
- Cite document name, page, and table index when table evidence is used.

ABSTENTION INSTRUCTIONS:
- If evidence is insufficient, say: The information needed to answer this question was not found in the accessible documents.
- If sources conflict, say so and do not pick a value without evidence.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
""".strip()
