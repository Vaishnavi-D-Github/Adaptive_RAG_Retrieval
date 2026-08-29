from typing import Any, Dict, List

from .schemas import VerificationResult


def _get_source_id(
    item: Dict[str, Any],
):

    metadata = item.get(
        "metadata",
        {},
    )

    if not isinstance(
        metadata,
        dict,
    ):

        return None

    return (
        metadata.get("source")
        or metadata.get("document")
        or metadata.get("source_document")
        or metadata.get("file")
    )


def verify_retrieval(
    query: str,
    retrieved_results: List[Dict[str, Any]],
    *,
    minimum_results: int = 1,
    relevance_threshold: float = 0.35,
) -> VerificationResult:

    if not retrieved_results:

        return VerificationResult(

            sufficient=False,

            relevance_score=0.0,

            coverage_score=0.0,

            evidence_count=0,

            source_diversity=0,

            reason=(
                "No evidence was retrieved."
            ),
        )


    distances = []

    for item in retrieved_results:

        distance = item.get(
            "distance"
        )

        if distance is None:
            continue

        try:

            distances.append(
                float(distance)
            )

        except (
            TypeError,
            ValueError,
        ):

            continue


    if distances:

        best_distance = min(
            distances
        )

        relevance_score = 1.0 / (
            1.0 + max(
                best_distance,
                0.0,
            )
        )

    else:

        # No distance available.
        # Do not pretend we know relevance.
        relevance_score = 0.0


    evidence_count = len(
        retrieved_results
    )


    source_ids = {
        _get_source_id(item)
        for item in retrieved_results
    }

    source_ids.discard(None)

    source_diversity = len(
        source_ids
    )


    coverage_score = min(
        evidence_count / 5.0,
        1.0,
    )


    sufficient = (
        evidence_count >= minimum_results
        and relevance_score >= relevance_threshold
    )


    if sufficient:

        reason = (
            "Retrieved evidence passes "
            "baseline relevance and "
            "evidence-count checks."
        )

    else:

        reason = (
            "Retrieved evidence does not "
            "meet the baseline verification "
            "threshold."
        )


    return VerificationResult(

        sufficient=sufficient,

        relevance_score=(
            float(relevance_score)
        ),

        coverage_score=(
            float(coverage_score)
        ),

        evidence_count=evidence_count,

        source_diversity=source_diversity,

        reason=reason,
    )