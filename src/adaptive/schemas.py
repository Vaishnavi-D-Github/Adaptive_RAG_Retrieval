from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QueryFeatures:

    query: str

    query_length: int
    word_count: int
    unique_word_count: int
    average_word_length: float

    entity_count: int

    question_count: int
    question_mark_indicator: bool
    number_indicator: bool
    uppercase_token_count: int

    starts_with_what: bool
    starts_with_why: bool
    starts_with_how: bool
    starts_with_when: bool
    starts_with_where: bool
    starts_with_who: bool
    starts_with_which: bool

    intent: str

    complexity: Optional[str]

    is_comparison: bool
    is_procedural: bool
    is_policy: bool
    is_summary: bool
    is_definition: bool

    is_multi_hop: bool

    requires_multiple_sources: bool

    expected_evidence_breadth: str
    is_table_question: bool = False


@dataclass
class RetrievalPlan:

    initial_k: int
    maximum_k: int

    escalation_step: int

    strategy: str

    predicted_complexity: Optional[str] = None

    complexity_confidence: Optional[float] = None

    predicted_k: Optional[int] = None

    k_confidence: Optional[float] = None

    k_probabilities: Optional[Dict[str, float]] = None

    decision_reason: str = ""


@dataclass
class VerificationResult:

    sufficient: bool

    relevance_score: float

    coverage_score: float

    evidence_count: int

    source_diversity: int

    reason: str


@dataclass
class RetrievalAttempt:

    k: int

    success: bool

    verification: VerificationResult

    retrieval_time_ms: float

    num_results: int


@dataclass
class AdaptiveResult:

    query: str

    optimized_query: str

    selected_k: int

    initial_k: int

    maximum_k: int

    retrieval_iterations: int

    retrieval_strategy: str

    query_features: QueryFeatures

    retrieval_plan: RetrievalPlan

    verification: VerificationResult

    retrieved_documents: List[Dict[str, Any]]

    attempts: List[RetrievalAttempt] = field(
        default_factory=list
    )

    context: str = ""

    prompt: str = ""

    telemetry: Dict[str, Any] = field(
        default_factory=dict
    )

    generation_result: Optional[str] = None
