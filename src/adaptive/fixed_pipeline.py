"""Fixed-K baseline execution.

The implementation intentionally does not invoke Adaptive-K logic.
MySQL telemetry persistence is optional and disabled by default.
"""

from __future__ import annotations

import time

from .context_optimizer import optimize_context
from .feature_extractor import extract_features
from .prompt_orchestrator import build_context, build_prompt
from .retrieval_controller import RetrievalController
from .telemetry import build_telemetry
from .schemas import RetrievalPlan


class FixedKPipeline:
    def __init__(
        self,
        retriever,
        generator=None,
        tokenizer=None,
        max_k: int = 10,
        telemetry_store=None,
    ):
        self.retriever = RetrievalController(
            retriever
        )

        self.generator = generator
        self.tokenizer = tokenizer
        self.max_k = max_k

        # Optional production telemetry persistence.
        # None means existing experiment behavior is unchanged.
        self.telemetry_store = telemetry_store

    def run(
        self,
        query: str,
        k: int,
        *,
        generate: bool = True,
    ) -> dict:

        if k not in {3, 5, 10}:
            raise ValueError(
                "Fixed-K experiments support the "
                "configured baselines: 3, 5, and 10."
            )

        start = time.perf_counter()

        retrieved = self.retriever.retrieve(
            query,
            k,
        )

        retrieval_time_ms = (
            time.perf_counter() - start
        ) * 1000

        optimize_start = time.perf_counter()

        used = optimize_context(
            retrieved
        )

        optimization_time_ms = (
            time.perf_counter() - optimize_start
        ) * 1000

        context = build_context(
            used
        )

        prompt = build_prompt(
            query,
            context,
        )

        generated_text = ""
        generation_time_ms = 0.0
        generation_metadata = {}

        if generate and self.generator:

            generation_start = time.perf_counter()

            generation_response = self.generator(
                prompt
            )

            generated_text = str(
                generation_response
            )

            generation_metadata = getattr(
                generation_response,
                "ollama_metadata",
                {},
            )

            generation_time_ms = (
                time.perf_counter()
                - generation_start
            ) * 1000

        features = extract_features(
            query
        )

        plan = RetrievalPlan(
            k,
            k,
            0,
            "fixed_baseline",
            predicted_k=k,
            decision_reason=(
                f"Fixed K={k} baseline."
            ),
        )

        telemetry = build_telemetry(
            query=query,
            query_features=features,
            retrieval_plan=plan,
            selected_k=k,
            retrieval_iterations=1,
            retrieved_documents=used,
            retrieved_chunk_count=len(
                retrieved
            ),
            context=context,
            prompt=prompt,
            tokenizer=self.tokenizer,
            retrieval_time_ms=retrieval_time_ms,
            optimization_time_ms=(
                optimization_time_ms
            ),
            generation_time_ms=(
                generation_time_ms
            ),
            generated_text=generated_text,
            generation_metadata=(
                generation_metadata
            ),
        )

        telemetry["mode"] = "fixed"
        telemetry["verification_performed"] = False
        telemetry["verification_result"] = None

        # -----------------------------------------------------
        # OPTIONAL MYSQL TELEMETRY PERSISTENCE
        #
        # Disabled unless a telemetry_store was explicitly
        # supplied by the runtime.
        # -----------------------------------------------------

        if self.telemetry_store is not None:

            try:
                self.telemetry_store.save_run(
                    telemetry,
                    mode=f"fixed_{k}",
                    run_type="production",
                )

            except Exception as exc:

                # Database failure must never prevent the
                # RAG answer from being returned.
                print(
                    "[WARNING] MySQL telemetry "
                    f"persistence failed: {exc}"
                )

        return {
            "answer": generated_text,
            "sources": [
                item.get("metadata", {})
                for item in used
            ],
            "telemetry": telemetry,
        }