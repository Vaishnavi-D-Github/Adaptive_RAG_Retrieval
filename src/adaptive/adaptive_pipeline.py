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


        # -----------------------------------------------------
        # Select K policy
        # -----------------------------------------------------

        if (
            complexity_model is not None
            and k_model is not None
            and complexity_model.trained
            and k_model.trained
        ):

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

            return ""

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

            text = self.generator(
                prompt
            )

        else:

            raise TypeError(
                "Generator must be callable "
                "or expose invoke()."
            )


        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000


        return str(text), elapsed_ms


    def run(
        self,
        query: str,
        *,
        generate=True,
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

        while True:

            start = time.perf_counter()

            retrieved = (
                self.retrieval_controller
                .retrieve(
                    optimized_query,
                    current_k,
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

            verification = (
                verify_retrieval(
                    query,
                    retrieved,
                )
            )


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

        optimized_results = (
            optimize_context(
                final_results
            )
        )


        # =====================================================
        # 10. BUILD CONTEXT
        # =====================================================

        context = build_context(
            optimized_results
        )


        # =====================================================
        # 11. BUILD PROMPT
        # =====================================================

        prompt = build_prompt(
            query,
            context,
        )


        # =====================================================
        # 12. GENERATION
        # =====================================================

        generated_text = ""

        generation_time_ms = 0.0

        if generate:

            generated_text, generation_time_ms = (
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

            context=context,

            prompt=prompt,

            tokenizer=self.tokenizer,

            retrieval_time_ms=(
                total_retrieval_time_ms
            ),

            generation_time_ms=(
                generation_time_ms
            ),

            generated_text=(
                generated_text
            ),

            verification=verification,

            fallback_used=(
                len(attempts) > 1
            ),
        )


        telemetry["policy_mode"] = (
            self.policy_mode
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