"""Common ranked-chunk retriever used by both fixed and adaptive flows."""

from __future__ import annotations

from typing import Any

ROLE_ACCESS_LEVELS = {
    "student": {"student"},
    "teacher": {"student", "teacher"},
    "office": {"student", "teacher", "office"},
    "admin": {"student", "teacher", "office"},
}

ABSTAIN_NO_ACCESS = "No accessible documents are available in the application knowledge base."
ABSTAIN_NO_EVIDENCE = "The information needed to answer this question was not found in the accessible documents."


def allowed_access_levels(user_role: str) -> list[str]:
    role = str(user_role or "").strip().lower()
    if role not in ROLE_ACCESS_LEVELS:
        raise ValueError(f"Unsupported user role: {role}")
    return sorted(ROLE_ACCESS_LEVELS[role])


def application_where(user_role: str) -> dict[str, Any]:
    return {
        "$and": [
            {"scope": "application"},
            {"access_level": {"$in": allowed_access_levels(user_role)}},
        ]
    }


class ChromaRetriever:
    def __init__(self, collection: Any, embedding_model: Any):
        self.collection = collection
        self.embedding_model = embedding_model

    def has_accessible_documents(self, user_role: str = "student") -> bool:
        where = application_where(user_role)
        try:
            if hasattr(self.collection, "get"):
                result = self.collection.get(where=where, limit=1)
                ids = result.get("ids", []) if isinstance(result, dict) else []
                if ids:
                    return True
        except Exception:
            pass
        try:
            return bool(self.retrieve("document", 1, user_role=user_role))
        except Exception:
            return False

    def retrieve(self, query: str, k: int, user_role: str = "student") -> list[dict[str, Any]]:
        embedding = self.embedding_model.encode([query])[0]
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=max(int(k), 1),
            where=application_where(user_role),
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ranked = []
        for index, (text, metadata, distance) in enumerate(zip(documents, metadatas, distances), start=1):
            meta = metadata or {}
            ranked.append(
                {
                    "document": text,
                    "metadata": meta,
                    "distance": distance,
                    "source": {
                        "document": meta.get("document"),
                        "page": meta.get("page"),
                        "chunk_id": meta.get("chunk_id"),
                        "content_type": meta.get("content_type", "text"),
                        "table_index": meta.get("table_index"),
                        "table_title": meta.get("table_title"),
                        "rank": index,
                        "distance": distance,
                        "text": (text or "")[:240],
                    },
                }
            )
        return ranked

    def delete_document(self, document_id: str) -> None:
        if hasattr(self.collection, "delete"):
            self.collection.delete(where={"document_id": str(document_id)})
