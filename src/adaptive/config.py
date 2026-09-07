"""Central configuration for the Adaptive-K research runtime."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_project_env(root: Path | None = None) -> None:
    """Load KEY=VALUE and PowerShell $env:KEY=VALUE lines without overriding the process."""

    env_path = (root or ROOT) / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("$env:"):
            line = line[5:]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


load_project_env()

MIN_K = int(os.getenv("AE_RAG_MIN_K", "1"))
MAX_K = int(os.getenv("AE_RAG_MAX_K", "10"))
DEFAULT_K = int(os.getenv("AE_RAG_DEFAULT_K", "5"))
FIXED_K_VALUES = (3, 5, 10)
ESCALATION_STEP = int(os.getenv("AE_RAG_ESCALATION_STEP", "2"))
VERIFICATION_THRESHOLD = float(os.getenv("AE_RAG_VERIFICATION_THRESHOLD", "0.35"))
CHROMA_PATH = Path(os.getenv("AE_RAG_CHROMA_PATH", ROOT / "chroma_db"))
APPLICATION_CHROMA_PATH = Path(os.getenv("AE_RAG_APP_CHROMA_PATH", ROOT / "data" / "application_chroma_db"))
APPLICATION_COLLECTION_NAME = os.getenv("AE_RAG_APP_COLLECTION", "application_uploaded_documents")
MODEL_PATH = Path(os.getenv("AE_RAG_MODEL_PATH", ROOT / "models"))
GENERATION_MODEL = os.getenv("AE_RAG_GENERATION_MODEL", "qwen3:8b")
OLLAMA_BASE_URL = os.getenv("AE_RAG_OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("AE_RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
UPLOAD_ROOT = Path(os.getenv("AE_RAG_UPLOAD_ROOT", ROOT / "data" / "uploads"))
HISTORY_ROOT = Path(os.getenv("AE_RAG_HISTORY_ROOT", ROOT / "data" / "query_history"))
