from typing import Any, Dict, List


def build_context(
    retrieved_results: List[Dict[str, Any]],
) -> str:

    parts = []

    for index, item in enumerate(
        retrieved_results,
        start=1,
    ):

        document = str(
            item.get(
                "document",
                "",
            )
        )

        metadata = item.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):

            metadata = {}


        source = (
            metadata.get("source")
            or metadata.get("document")
            or metadata.get("source_document")
            or "Unknown"
        )

        page = (
            metadata.get("page")
            or metadata.get("source_page")
            or "Unknown"
        )


        parts.append(
            f"""
--- SOURCE {index} ---
Document: {source}
Page: {page}

{document}

--- END SOURCE {index} ---
""".strip()
        )


    return "\n\n".join(
        parts
    )


def build_prompt(
    question: str,
    context: str,
) -> str:

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