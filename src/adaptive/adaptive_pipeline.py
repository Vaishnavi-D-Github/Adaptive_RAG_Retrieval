import time

from .feature_extractor import extract_features
from .k_policy import (
    BootstrapKPolicy,
    MLPredictiveKPolicy,
)
from .query_optimizer import optimize_query
from .retrieval_controller import (
    RetrievalController,
)
from .verifier import verify_retrieval
from .context_optimizer import (
    optimize_context,
)
from .prompt_orchestrator import (
    build_context,
    build_prompt,
)
from .telemetry import build_telemetry

from .schemas import AdaptiveResult


class AdaptivePipeline:

    def __init__(
        self,
        retriever,
        complexity_model=None,
        k_model=None,
        generator=None,
        tokenizer=None,
        max_k=10,
        telemetry_store=None,
        application: bool = False,
    ):

        self.retrieval_controller = (
            RetrievalController(
                retriever
            )
        )

        self.complexity_model = (
            complexity_model
        )

        self.k_model = k_model

        self.generator = generator

        self.tokenizer = tokenizer

        self.max_k = max_k

        self.telemetry_store = telemetry_store
        self.application = application


        # -----------------------------------------------------
        # Select K policy
        # -----------------------------------------------------

        if k_model is not None and k_model.trained:

            self.k_policy = (
                MLPredictiveKPolicy(
                    complexity_model,
                    k_model,
                    max_k=max_k,
                )
            )

            self.policy_mode = "ml"

        else:

            self.k_policy = (
                BootstrapKPolicy(
                    max_k=max_k,
                )
            )

            self.policy_mode = (
                "bootstrap"
            )


    def _generate(
        self,
        prompt,
    ):

        if self.generator is None:

            return "", 0.0, {}

        start = time.perf_counter()

        if hasattr(
            self.generator,
            "invoke",
        ):

            response = (
                self.generator.invoke(
                    prompt
                )
            )

            if hasattr(
                response,
                "content",
            ):

                text = response.content

            else:

                text = str(response)

        elif callable(
            self.generator
        ):

            response = self.generator(
                prompt
            )
            text = str(response)

        else:

            raise TypeError(
                "Generator must be callable "
                "or expose invoke()."
            )


        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000

        generation_metadata = getattr(
            response,
            "ollama_metadata",
            {},
        )

        return str(text), elapsed_ms, generation_metadata


    def run(
    self,
    query: str,
    *,
    generate=True,
    user_role: str = "student",
    ) -> AdaptiveResult:

        # =====================================================
        # 1. QUERY ANALYSIS
        # =====================================================

        features = extract_features(
            query
        )


        # =====================================================
        # 2. K DECISION
        # =====================================================

        plan = self.k_policy.predict(
            features
        )


        # =====================================================
        # 3. QUERY OPTIMIZATION
        # =====================================================

        optimized_query = optimize_query(
            query,
            features,
        )


        # =====================================================
        # 4. ITERATIVE RETRIEVAL
        # =====================================================

        current_k = plan.initial_k

        attempts = []

        final_results = []

        total_retrieval_time_ms = 0.0
        total_verification_time_ms = 0.0

        while True:

            start = time.perf_counter()

            retrieved = (
                self.retrieval_controller
                .retrieve(
                    optimized_query,
                    current_k,
                    user_role=user_role,
                )
            )

            retrieval_time_ms = (
                time.perf_counter()
                - start
            ) * 1000

            total_retrieval_time_ms += (
                retrieval_time_ms
            )


            # -------------------------------------------------
            # 5. VERIFICATION
            # -------------------------------------------------

            verification_start = time.perf_counter()
            verification = (
                verify_retrieval(
                    query,
                    retrieved,
                    is_table_question=bool(getattr(features, "is_table_question", False)),
                    requires_multiple_sources=bool(features.requires_multiple_sources),
                    strict_coverage=self.application,
                )
            )
            verification_time_ms = (time.perf_counter() - verification_start) * 1000
            total_verification_time_ms += verification_time_ms


            attempts.append({

                "k": current_k,

                "success": True,

                "verification": verification,

                "retrieval_time_ms": (
                    retrieval_time_ms
                ),

                "num_results": len(
                    retrieved
                ),
                "verification_time_ms": verification_time_ms,
            })


            final_results = retrieved


            # -------------------------------------------------
            # 6. STOP IF VERIFIED
            # -------------------------------------------------

            if verification.sufficient:

                break


            # -------------------------------------------------
            # 7. STOP AT MAXIMUM K
            # -------------------------------------------------

            if current_k >= plan.maximum_k:

                break


            # -------------------------------------------------
            # 8. ESCALATE K
            # -------------------------------------------------

            next_k = min(

                current_k
                + plan.escalation_step,

                plan.maximum_k,

            )


            if next_k == current_k:

                break


            current_k = next_k


        # =====================================================
        # 9. CONTEXT OPTIMIZATION
        # =====================================================

        optimization_start = time.perf_counter()
        optimized_results = optimize_context(final_results)
        optimization_time_ms = (time.perf_counter() - optimization_start) * 1000


        # =====================================================
        # 10. BUILD CONTEXT
        # =====================================================

        context = build_context(
            optimized_results,
            application=self.application,
        )


        # =====================================================
        # 11. BUILD PROMPT
        # =====================================================

        prompt = build_prompt(
            query,
            context,
            application=self.application,
        )


        # =====================================================
        # 12. GENERATION
        # =====================================================

        generated_text = ""

        generation_time_ms = 0.0
        generation_metadata = {}

        if self.application and not optimized_results:
            generated_text = "The information needed to answer this question was not found in the accessible documents."
        elif generate:

            generated_text, generation_time_ms, generation_metadata = (
                self._generate(
                    prompt
                )
            )


        # =====================================================
        # 13. TELEMETRY
        # =====================================================

        telemetry = build_telemetry(

            query=query,

            query_features=features,

            retrieval_plan=plan,

            selected_k=current_k,

            retrieval_iterations=len(
                attempts
            ),

            retrieved_documents=(
                optimized_results
            ),
            retrieved_chunk_count=len(final_results),

            context=context,

            prompt=prompt,

            tokenizer=self.tokenizer,

            retrieval_time_ms=(
                total_retrieval_time_ms
            ),
            verification_time_ms=total_verification_time_ms,
            optimization_time_ms=optimization_time_ms,

            generation_time_ms=(
                generation_time_ms
            ),

            generated_text=(
                generated_text
            ),
            generation_metadata=(
                generation_metadata
            ),

            verification=verification,

            fallback_used=(
                len(attempts) > 1
            ),
        )


        telemetry["policy_mode"] = (
            self.policy_mode
        )

        # -----------------------------------------------------
        # OPTIONAL MYSQL TELEMETRY PERSISTENCE
        # -----------------------------------------------------

        if self.telemetry_store is not None:
            try:
                self.telemetry_store.save_run(
                    telemetry,
                    mode="adaptive",
                    run_type="production",
                )
            except Exception as exc:
                print(
                    "[WARNING] MySQL telemetry "
                    f"persistence failed: {exc}"
                )


        return AdaptiveResult(

            query=query,

            optimized_query=(
                optimized_query
            ),

            selected_k=current_k,

            initial_k=plan.initial_k,

            maximum_k=plan.maximum_k,

            retrieval_iterations=len(
                attempts
            ),

            retrieval_strategy=(
                plan.strategy
            ),

            query_features=features,

            retrieval_plan=plan,

            verification=verification,

            retrieved_documents=(
                optimized_results
            ),

            attempts=attempts,

            context=context,

            prompt=prompt,

            telemetry=telemetry,

            generation_result=(
                generated_text
            ),
        )
