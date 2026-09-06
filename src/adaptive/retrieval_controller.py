from typing import Any, Dict, List


class RetrievalController:
    def __init__(self, retriever):
        self.retriever = retriever

    def retrieve(self, query: str, k: int, user_role: str = "student") -> List[Dict[str, Any]]:
        if hasattr(self.retriever, "retrieve"):
            try:
                results = self.retriever.retrieve(query, k, user_role=user_role)
            except TypeError:
                results = self.retriever.retrieve(query, k)
        elif hasattr(self.retriever, "search"):
            try:
                results = self.retriever.search(query, k, user_role=user_role)
            except TypeError:
                results = self.retriever.search(query, k)
        else:
            raise AttributeError("Retriever must expose retrieve(query, k) or search(query, k).")
        if results is None:
            return []
        return list(results)
