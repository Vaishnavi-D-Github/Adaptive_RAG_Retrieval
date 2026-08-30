from .schemas import QueryFeatures, RetrievalPlan


class BootstrapKPolicy:
    """
    Temporary policy used until the 959-record evaluation
    produces Optimal_K labels.

    IMPORTANT:
    These values are engineering defaults.
    They are NOT the final experimentally learned policy.
    """

    def __init__(
        self,
        min_k=1,
        max_k=10,
        escalation_step=2,
    ):

        self.min_k = min_k
        self.max_k = max_k
        self.escalation_step = (
            escalation_step
        )


    def predict(
        self,
        features: QueryFeatures,
    ) -> RetrievalPlan:

        if features.complexity == "low":

            initial_k = 2

        elif features.complexity == "medium":

            initial_k = 4

        else:

            initial_k = 6


        if features.is_comparison:

            initial_k = max(
                initial_k,
                5,
            )


        if features.is_multi_hop:

            initial_k = max(
                initial_k,
                5,
            )


        if features.requires_multiple_sources:

            initial_k = max(
                initial_k,
                5,
            )


        initial_k = min(
            max(
                initial_k,
                self.min_k,
            ),
            self.max_k,
        )


        return RetrievalPlan(

            initial_k=initial_k,

            maximum_k=self.max_k,

            escalation_step=(
                self.escalation_step
            ),

            strategy=(
                "hybrid"
                if (
                    features.is_comparison
                    or features.is_multi_hop
                )
                else "semantic"
            ),

            predicted_complexity=(
                features.complexity
            ),

            complexity_confidence=None,

            predicted_k=initial_k,

            k_confidence=None,

            decision_reason=(
                "Bootstrap policy; "
                "will be replaced by "
                "trained Random Forest."
            ),
        )


class MLPredictiveKPolicy:

    def __init__(
        self,
        complexity_model,
        k_model,
        max_k=10,
    ):

        self.complexity_model = (
            complexity_model
        )

        self.k_model = k_model

        self.max_k = max_k


    def predict(
        self,
        features: QueryFeatures,
    ) -> RetrievalPlan:

        if self.complexity_model is not None and self.complexity_model.trained:
            complexity, complexity_confidence = self.complexity_model.predict(features.query)
        else:
            # This is a transparent query-derived heuristic, not a dataset
            # annotation. It keeps the initial K decision available at runtime.
            complexity, complexity_confidence = features.complexity, None

        probability_result = None
        if hasattr(self.k_model, "predict_with_probabilities"):
            probability_result = self.k_model.predict_with_probabilities(features.query, complexity)
            predicted_k, k_confidence = probability_result["prediction"], probability_result["confidence"]
        else:
            predicted_k, k_confidence = self.k_model.predict(features.query, complexity)

        predicted_k = max(
            1,
            min(
                int(predicted_k),
                self.max_k,
            ),
        )


        if predicted_k <= 3:

            strategy = "semantic"

        elif (
            features.is_comparison
            or features.is_multi_hop
        ):

            strategy = "hybrid"

        else:

            strategy = "semantic"


        return RetrievalPlan(

            initial_k=predicted_k,

            maximum_k=self.max_k,

            escalation_step=2,

            strategy=strategy,

            predicted_complexity=complexity,

            complexity_confidence=(
                complexity_confidence
            ),

            predicted_k=predicted_k,

            k_confidence=k_confidence,

            k_probabilities=(probability_result["probabilities"] if probability_result else None),

            decision_reason=(
                "Random Forest complexity "
                "and K prediction."
            ),
        )
