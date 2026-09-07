import unittest

from adaptive.adaptive_pipeline import AdaptivePipeline
from adaptive.fixed_pipeline import FixedKPipeline
from adaptive.feature_extractor import extract_features
from adaptive.schemas import RetrievalPlan
from adaptive.telemetry import build_telemetry
from runtime import GenerationResult, OllamaGenerator


OLLAMA_METADATA = {
    "prompt_eval_count": 29,
    "eval_count": 9,
    "prompt_eval_duration": 1039855000,
    "eval_duration": 1068776000,
    "total_duration": 2294719100,
}


class _MockOllamaClient:
    def chat(self, **kwargs):
        return {
            "message": {"content": "answer text"},
            **OLLAMA_METADATA,
        }


class _Retriever:
    def retrieve(self, query, k):
        return [
            {
                "document": f"evidence for {query}",
                "metadata": {"document": "demo.pdf", "page": 1},
                "distance": 0.1,
            }
        ] * k


class _KModel:
    trained = True

    def predict(self, query, complexity=None):
        return 3, 0.9


class _MetadataGenerator:
    def __call__(self, prompt):
        return GenerationResult("generated answer", ollama_metadata=OLLAMA_METADATA)


class OllamaTelemetryTests(unittest.TestCase):
    def test_ollama_generator_preserves_text_and_metadata(self):
        generator = OllamaGenerator.__new__(OllamaGenerator)
        generator.client = _MockOllamaClient()

        result = generator("prompt")

        self.assertIsInstance(result, str)
        self.assertEqual(str(result), "answer text")
        self.assertEqual(result.ollama_metadata["prompt_eval_count"], 29)
        self.assertEqual(result.ollama_metadata["eval_count"], 9)
        self.assertEqual(result.ollama_metadata["prompt_eval_duration"], 1039855000)
        self.assertEqual(result.ollama_metadata["eval_duration"], 1068776000)
        self.assertEqual(result.ollama_metadata["total_duration"], 2294719100)

    def test_telemetry_uses_native_ollama_counts_and_milliseconds(self):
        features = extract_features("What is the policy?")
        plan = RetrievalPlan(3, 3, 0, "fixed_baseline", predicted_k=3)

        telemetry = build_telemetry(
            query="What is the policy?",
            query_features=features,
            retrieval_plan=plan,
            selected_k=3,
            retrieval_iterations=1,
            retrieved_documents=[],
            context="context",
            prompt="prompt",
            generated_text="answer",
            generation_metadata=OLLAMA_METADATA,
        )

        self.assertEqual(telemetry["prompt_tokens"], 29)
        self.assertEqual(telemetry["generated_tokens"], 9)
        self.assertEqual(telemetry["total_tokens"], 38)
        self.assertEqual(
            telemetry["ollama_prompt_eval_duration_ms"],
            1039.855,
        )
        self.assertEqual(
            telemetry["ollama_eval_duration_ms"],
            1068.776,
        )
        self.assertEqual(
            telemetry["ollama_total_duration_ms"],
            2294.7191,
        )

    def test_fixed_pipeline_passes_generation_metadata_to_telemetry(self):
        result = FixedKPipeline(_Retriever(), generator=_MetadataGenerator()).run(
            "What is the policy?",
            3,
        )

        self.assertEqual(result["answer"], "generated answer")
        self.assertEqual(result["telemetry"]["prompt_tokens"], 29)
        self.assertEqual(result["telemetry"]["generated_tokens"], 9)
        self.assertEqual(result["telemetry"]["total_tokens"], 38)

    def test_adaptive_pipeline_passes_generation_metadata_to_telemetry(self):
        result = AdaptivePipeline(
            _Retriever(),
            k_model=_KModel(),
            generator=_MetadataGenerator(),
        ).run("What is the policy?")

        self.assertEqual(result.generation_result, "generated answer")
        self.assertEqual(result.telemetry["prompt_tokens"], 29)
        self.assertEqual(result.telemetry["generated_tokens"], 9)
        self.assertEqual(result.telemetry["total_tokens"], 38)

    def test_telemetry_falls_back_when_ollama_metadata_unavailable(self):
        features = extract_features("What is the policy?")
        plan = RetrievalPlan(3, 3, 0, "fixed_baseline", predicted_k=3)

        telemetry = build_telemetry(
            query="What is the policy?",
            query_features=features,
            retrieval_plan=plan,
            selected_k=3,
            retrieval_iterations=1,
            retrieved_documents=[],
            context="context",
            prompt="prompt",
            tokenizer=None,
            generated_text="answer",
        )

        self.assertEqual(telemetry["prompt_tokens"], 0)
        self.assertEqual(telemetry["generated_tokens"], 0)
        self.assertEqual(telemetry["total_tokens"], 0)
        self.assertIsNone(telemetry["ollama_prompt_eval_duration_ms"])
        self.assertIsNone(telemetry["ollama_eval_duration_ms"])
        self.assertIsNone(telemetry["ollama_total_duration_ms"])


if __name__ == "__main__":
    unittest.main()
