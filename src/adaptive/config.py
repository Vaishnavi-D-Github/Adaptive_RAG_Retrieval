"""Central configuration for the Adaptive-K research runtime."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MIN_K = int(os.getenv("AE_RAG_MIN_K", "1"))
MAX_K = int(os.getenv("AE_RAG_MAX_K", "10"))
DEFAULT_K = int(os.getenv("AE_RAG_DEFAULT_K", "5"))
FIXED_K_VALUES = (3, 5, 10)
ESCALATION_STEP = int(os.getenv("AE_RAG_ESCALATION_STEP", "2"))
VERIFICATION_THRESHOLD = float(os.getenv("AE_RAG_VERIFICATION_THRESHOLD", "0.35"))
CHROMA_PATH = Path(os.getenv("AE_RAG_CHROMA_PATH", ROOT / "chroma_db"))
MODEL_PATH = Path(os.getenv("AE_RAG_MODEL_PATH", ROOT / "models"))
GENERATION_MODEL = os.getenv("AE_RAG_GENERATION_MODEL", "qwen3:8b")
OLLAMA_BASE_URL = os.getenv("AE_RAG_OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("AE_RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
