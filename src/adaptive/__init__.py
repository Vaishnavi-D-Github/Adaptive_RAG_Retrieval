"""
Adaptive Retrieval Framework.

Pipeline:

Query
    ↓
Feature Extraction
    ↓
Random Forest Complexity Model
    ↓
Random Forest K Model / Bootstrap Policy
    ↓
Retrieval
    ↓
Verification
    ↓
K Escalation if required
    ↓
Context Optimization
    ↓
Prompt Orchestration
    ↓
Generation
    ↓
Telemetry
"""

from .schemas import (
    QueryFeatures,
    RetrievalPlan,
    VerificationResult,
    RetrievalAttempt,
    AdaptiveResult,
)

from .feature_extractor import extract_features
from .complexity_model import ComplexityModel
from .k_model import KModel
from .k_policy import BootstrapKPolicy
from .query_optimizer import optimize_query
from .verifier import verify_retrieval
from .context_optimizer import optimize_context
from .prompt_orchestrator import build_prompt
from .adaptive_pipeline import AdaptivePipeline

__all__ = [
    "QueryFeatures",
    "RetrievalPlan",
    "VerificationResult",
    "RetrievalAttempt",
    "AdaptiveResult",
    "extract_features",
    "ComplexityModel",
    "KModel",
    "BootstrapKPolicy",
    "optimize_query",
    "verify_retrieval",
    "optimize_context",
    "build_prompt",
    "AdaptivePipeline",
]