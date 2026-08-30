import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from .schemas import (
    QueryFeatures,
    RetrievalPlan,
    VerificationResult,
)


def count_tokens(
    tokenizer,
    text: str,
) -> int:

    if tokenizer is None:

        return 0

    if not text:

        return 0

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )

    return len(
        encoded["input_ids"]
    )


def build_telemetry(
    *,
    query: str,
    query_features: QueryFeatures,
    retrieval_plan: RetrievalPlan,
    selected_k: int,
    retrieval_iterations: int,
    retrieved_documents,
    retrieved_chunk_count: int | None = None,
    context: str,
    prompt: str,
    tokenizer=None,
    retrieval_time_ms: float = 0.0,
    verification_time_ms: float = 0.0,
    optimization_time_ms: float = 0.0,
    generation_time_ms: float = 0.0,
    generated_text: str = "",
    verification: Optional[
        VerificationResult
    ] = None,
    fallback_used: bool = False,
) -> Dict[str, Any]:

    prompt_tokens = count_tokens(
        tokenizer,
        prompt,
    )

    generated_tokens = count_tokens(
        tokenizer,
        generated_text,
    )

    total_tokens = (
        prompt_tokens
        + generated_tokens
    )


    unique_documents = set()

    for item in retrieved_documents:

        metadata = item.get(
            "metadata",
            {},
        )

        if isinstance(
            metadata,
            dict,
        ):

            source = (
                metadata.get("source")
                or metadata.get("document")
                or metadata.get("source_document")
            )

            if source:
                unique_documents.add(
                    str(source)
                )


    verification_score = 0.0

    verification_passed = False

    if verification is not None:

        verification_score = (
            verification.relevance_score
        )

        verification_passed = (
            verification.sufficient
        )


    total_latency_ms = (
        retrieval_time_ms
        + verification_time_ms
        + optimization_time_ms
        + generation_time_ms
    )

    retrieved_sources = []
    for rank, item in enumerate(retrieved_documents, start=1):
        metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        retrieved_sources.append({
            "document": metadata.get("document") or metadata.get("source") or metadata.get("source_document"),
            "page": metadata.get("page") or metadata.get("source_page"),
            "rank": rank,
            "distance": item.get("distance") if isinstance(item, dict) else None,
        })


    return {

        "query": query,

        "query_intent": (
            query_features.intent
        ),

        "query_complexity": (
            query_features.complexity
        ),

        "query_word_count": (
            query_features.word_count
        ),

        "query_entity_count": (
            query_features.entity_count
        ),

        "is_multi_hop": (
            query_features.is_multi_hop
        ),

        "requires_multiple_sources": (
            query_features.requires_multiple_sources
        ),

        "initial_k": (
            retrieval_plan.initial_k
        ),

        "predicted_k": (
            retrieval_plan.predicted_k
        ),

        "predicted_k_confidence": (
            retrieval_plan.k_confidence
        ),

        "predicted_k_probabilities": retrieval_plan.k_probabilities,

        "selected_k": selected_k,

        "maximum_k": (
            retrieval_plan.maximum_k
        ),

        "retrieval_iterations": (
            retrieval_iterations
        ),

        "retrieval_strategy": (
            retrieval_plan.strategy
        ),

        "num_retrieved_chunks": retrieved_chunk_count if retrieved_chunk_count is not None else len(retrieved_documents),
        "num_chunks_used": len(retrieved_documents),

        "unique_documents": (
            len(unique_documents)
        ),

        "retrieval_time_ms": (
            retrieval_time_ms
        ),

        "generation_time_ms": (
            generation_time_ms
        ),
        "optimization_time_ms": optimization_time_ms,
        "verification_time_ms": verification_time_ms,

        "total_latency_ms": (
            total_latency_ms
        ),

        "prompt_tokens": (
            prompt_tokens
        ),

        "generated_tokens": (
            generated_tokens
        ),

        "total_tokens": (
            total_tokens
        ),

        "verification_passed": (
            verification_passed
        ),

        "verification_score": (
            verification_score
        ),

        "verification_performed": verification is not None,
        "verification_result": verification_passed if verification is not None else None,
        "verification_reason": verification.reason if verification is not None else None,
        "k_escalated": selected_k > retrieval_plan.initial_k,
        "query_features": asdict(query_features),
        "retrieved_sources": retrieved_sources,

        "fallback_used": fallback_used,

        "context_characters": (
            len(context)
        ),

        "prompt_characters": (
            len(prompt)
        ),
    }
