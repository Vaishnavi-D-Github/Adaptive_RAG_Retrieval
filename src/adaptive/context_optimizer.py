from typing import Any, Dict, List


def optimize_context(
    retrieved_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    seen = set()

    optimized = []

    for item in retrieved_results:

        document = str(
            item.get(
                "document",
                "",
            )
        ).strip()

        if not document:

            continue


        normalized = " ".join(
            document.split()
        ).lower()


        if normalized in seen:

            continue


        seen.add(
            normalized
        )

        optimized.append(
            item
        )


    return optimized