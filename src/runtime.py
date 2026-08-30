"""Concrete runtime wiring shared by the API and controlled experiment."""

from __future__ import annotations

from pathlib import Path

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

from adaptive.adaptive_pipeline import AdaptivePipeline
from adaptive.config import CHROMA_PATH, EMBEDDING_MODEL, GENERATION_MODEL, MAX_K, MODEL_PATH, OLLAMA_BASE_URL
from adaptive.fixed_pipeline import FixedKPipeline
from adaptive.k_model import KModel
from retrieval.chroma_retriever import ChromaRetriever


class OllamaGenerator:
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_BASE_URL)

    def __call__(self, prompt: str) -> str:
        response = self.client.chat(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            think=False,
            options={"temperature": 0, "num_predict": 512},
        )
        return response["message"]["content"]


def load_runtime(*, require_generator: bool = True):
    """Create pipelines from the project's existing Chroma collection and model.

    Model loading remains lazy: importing API modules never downloads models or contacts Ollama.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection("enterprise_documents")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    retriever = ChromaRetriever(collection, embedding_model)
    model_files = sorted(Path(MODEL_PATH).glob("adaptive_k_random_forest_*.joblib"))
    k_model = KModel().load(model_files[-1]) if model_files else None
    generator = OllamaGenerator() if require_generator else None
    return (
        AdaptivePipeline(retriever, k_model=k_model, generator=generator, max_k=MAX_K),
        FixedKPipeline(retriever, generator=generator, max_k=MAX_K),
        collection,
        embedding_model,
    )
