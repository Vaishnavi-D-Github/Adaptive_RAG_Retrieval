import re

from .schemas import QueryFeatures


COMPARISON_TERMS = {
    "compare",
    "comparison",
    "difference",
    "differences",
    "versus",
    "vs",
    "similar",
    "similarities",
}


PROCEDURAL_TERMS = {
    "how",
    "steps",
    "procedure",
    "process",
    "implement",
    "implementation",
    "configure",
    "setup",
    "install",
}


POLICY_TERMS = {
    "policy",
    "policies",
    "requirement",
    "requirements",
    "required",
    "must",
    "allowed",
    "prohibited",
    "regulation",
    "rules",
}


SUMMARY_TERMS = {
    "summarize",
    "summary",
    "overview",
    "main points",
    "key points",
}


DEFINITION_TERMS = {
    "what is",
    "what are",
    "define",
    "definition",
    "meaning",
}


MULTI_HOP_TERMS = {
    "why",
    "how does",
    "how do",
    "relationship",
    "impact",
    "effect",
    "based on",
    "according to",
    "compare",
    "difference",
}


def _contains_any(
    text: str,
    terms: set[str],
) -> bool:

    return any(
        term in text
        for term in terms
    )


def _estimate_intent(
    text: str,
) -> str:

    if _contains_any(
        text,
        COMPARISON_TERMS,
    ):
        return "comparison"

    if _contains_any(
        text,
        PROCEDURAL_TERMS,
    ):
        return "procedural"

    if _contains_any(
        text,
        POLICY_TERMS,
    ):
        return "policy"

    if _contains_any(
        text,
        SUMMARY_TERMS,
    ):
        return "summary"

    if _contains_any(
        text,
        DEFINITION_TERMS,
    ):
        return "definition"

    return "general"


def _estimate_complexity(
    *,
    word_count: int,
    entity_count: int,
    question_count: int,
    is_comparison: bool,
    is_multi_hop: bool,
    requires_multiple_sources: bool,
) -> str:

    score = 0

    if word_count >= 15:
        score += 1

    if word_count >= 25:
        score += 1

    if entity_count >= 2:
        score += 1

    if entity_count >= 4:
        score += 1

    if question_count >= 2:
        score += 2

    if is_comparison:
        score += 2

    if is_multi_hop:
        score += 1

    if requires_multiple_sources:
        score += 1

    if score >= 6:
        return "high"

    if score >= 3:
        return "medium"

    return "low"


def extract_features(
    query: str,
) -> QueryFeatures:

    query = str(query).strip()

    lower = query.lower()

    words = re.findall(
        r"\b\w+\b",
        query,
    )

    word_count = len(words)

    query_length = len(query)

    question_count = query.count("?")

    capitalized_entities = re.findall(
        r"\b[A-Z][A-Za-z0-9-]{2,}\b",
        query,
    )

    entity_count = len(
        set(capitalized_entities)
    )

    is_comparison = _contains_any(
        lower,
        COMPARISON_TERMS,
    )

    is_procedural = _contains_any(
        lower,
        PROCEDURAL_TERMS,
    )

    is_policy = _contains_any(
        lower,
        POLICY_TERMS,
    )

    is_summary = _contains_any(
        lower,
        SUMMARY_TERMS,
    )

    is_definition = _contains_any(
        lower,
        DEFINITION_TERMS,
    )

    is_multi_hop = (
        is_comparison
        or _contains_any(
            lower,
            MULTI_HOP_TERMS,
        )
        or question_count >= 2
    )

    requires_multiple_sources = (
        is_comparison
        or is_multi_hop
        or entity_count >= 3
    )

    complexity = _estimate_complexity(
        word_count=word_count,
        entity_count=entity_count,
        question_count=question_count,
        is_comparison=is_comparison,
        is_multi_hop=is_multi_hop,
        requires_multiple_sources=(
            requires_multiple_sources
        ),
    )

    if (
        requires_multiple_sources
        or complexity == "high"
    ):
        evidence_breadth = "high"

    elif complexity == "medium":

        evidence_breadth = "medium"

    else:

        evidence_breadth = "low"

    return QueryFeatures(
        query=query,
        query_length=query_length,
        word_count=word_count,
        entity_count=entity_count,
        question_count=question_count,
        intent=_estimate_intent(lower),
        complexity=complexity,
        is_comparison=is_comparison,
        is_procedural=is_procedural,
        is_policy=is_policy,
        is_summary=is_summary,
        is_definition=is_definition,
        is_multi_hop=is_multi_hop,
        requires_multiple_sources=(
            requires_multiple_sources
        ),
        expected_evidence_breadth=(
            evidence_breadth
        ),
    )