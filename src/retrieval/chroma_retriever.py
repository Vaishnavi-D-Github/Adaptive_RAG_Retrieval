"""Common ranked-chunk retriever used by both fixed and adaptive flows."""

from __future__ import annotations

from typing import Any


class ChromaRetriever:
    def __init__(self, collection: Any, embedding_model: Any):
        self.collection = collection
        self.embedding_model = embedding_model

    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        embedding = self.embedding_model.encode([query])[0].tolist()
        result = self.collection.query(query_embeddings=[embedding], n_results=int(k))
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            {"document": text, "metadata": metadata or {}, "distance": distance}
            for text, metadata, distance in zip(documents, metadatas, distances)
        ]
