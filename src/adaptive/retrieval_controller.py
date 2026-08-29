import time
from typing import Any, Dict, List


class RetrievalController:

    def __init__(
        self,
        retriever,
    ):

        self.retriever = retriever


    def retrieve(
        self,
        query: str,
        k: int,
    ) -> List[Dict[str, Any]]:

        """
        Adapter around the existing retrieval system.

        The existing project retriever must expose one of:

            retriever.retrieve(query, k)
            retriever.search(query, k)

        and return a list of dictionaries.

        Each dictionary should ideally contain:

            document
            metadata
            distance
        """

        if hasattr(
            self.retriever,
            "retrieve",
        ):

            results = self.retriever.retrieve(
                query,
                k,
            )

        elif hasattr(
            self.retriever,
            "search",
        ):

            results = self.retriever.search(
                query,
                k,
            )

        else:

            raise AttributeError(
                "Retriever must expose "
                "retrieve(query, k) or "
                "search(query, k)."
            )


        if results is None:

            return []


        return list(results)