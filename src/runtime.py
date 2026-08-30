"""Concrete runtime wiring shared by the API and controlled experiment."""

from __future__ import annotations

from pathlib import Path

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

from adaptive.adaptive_pipeline import AdaptivePipeline
from adaptive.config import (
    CHROMA_PATH,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    MAX_K,
    MODEL_PATH,
    OLLAMA_BASE_URL,
)
from adaptive.fixed_pipeline import FixedKPipeline
from adaptive.k_model import KModel
from adaptive.mysql_store import MySQLTelemetryStore
from retrieval.chroma_retriever import ChromaRetriever


class GenerationResult(str):
    """Generated text with optional provider-native metadata attached."""

    def __new__(
        cls,
        text: str,
        *,
        ollama_metadata: dict | None = None,
    ):
        instance = super().__new__(
            cls,
            text,
        )

        instance.ollama_metadata = (
            ollama_metadata or {}
        )

        return instance


class OllamaGenerator:
    def __init__(self):
        self.client = ollama.Client(
            host=OLLAMA_BASE_URL
        )

    def __call__(
        self,
        prompt: str,
    ) -> GenerationResult:

        response = self.client.chat(
            model=GENERATION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            think=False,
            options={
                "temperature": 0,
                "num_predict": 512,
            },
        )

        metadata_fields = (
            "prompt_eval_count",
            "eval_count",
            "prompt_eval_duration",
            "eval_duration",
            "total_duration",
        )

        metadata = {
            field: response.get(field)
            for field in metadata_fields
            if field in response
        }

        return GenerationResult(
            response["message"]["content"],
            ollama_metadata=metadata,
        )


def load_runtime(
    *,
    require_generator: bool = True,
    enable_telemetry_db: bool = False,
):
    """Create pipelines from the existing Chroma collection and model.

    ``enable_telemetry_db`` is deliberately False by default.

    Therefore existing research scripts that call:

        load_runtime()

    continue to behave exactly as before and do not write to MySQL.

    The production API can explicitly enable persistence with:

        load_runtime(enable_telemetry_db=True)
    """

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_collection(
        "enterprise_documents"
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    retriever = ChromaRetriever(
        collection,
        embedding_model,
    )

    model_files = sorted(
        Path(MODEL_PATH).glob(
            "adaptive_k_random_forest_*.joblib"
        )
    )

    k_model = (
        KModel().load(model_files[-1])
        if model_files
        else None
    )

    generator = (
        OllamaGenerator()
        if require_generator
        else None
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # Research experiments remain database-free by default.
    # Only explicitly enabled production runtime gets MySQL.
    # ---------------------------------------------------------

    telemetry_store = (
        MySQLTelemetryStore()
        if enable_telemetry_db
        else None
    )

    adaptive_pipeline = AdaptivePipeline(
        retriever,
        k_model=k_model,
        generator=generator,
        max_k=MAX_K,
        telemetry_store=telemetry_store,
    )

    fixed_pipeline = FixedKPipeline(
        retriever,
        generator=generator,
        max_k=MAX_K,
        telemetry_store=telemetry_store,
    )

    return (
        adaptive_pipeline,
        fixed_pipeline,
        collection,
        embedding_model,
    )