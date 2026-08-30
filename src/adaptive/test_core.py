import unittest

from adaptive.feature_extractor import extract_features
from adaptive.fixed_pipeline import FixedKPipeline
from adaptive.adaptive_pipeline import AdaptivePipeline


class _Retriever:
    def retrieve(self, query, k):
        return [{"document": f"evidence for {query}", "metadata": {"document": "demo.pdf", "page": 1}, "distance": 0.1}] * k


class _KModel:
    trained = True

    def predict(self, query, complexity=None):
        return 3, 0.9


class FeatureAndBaselineTests(unittest.TestCase):
    def test_pre_retrieval_feature_contract(self):
        features = extract_features("HOW many policies apply in 2026?")
        self.assertTrue(features.question_mark_indicator)
        self.assertTrue(features.number_indicator)
        self.assertEqual(features.uppercase_token_count, 1)
        self.assertTrue(features.starts_with_how)
        self.assertGreater(features.unique_word_count, 0)

    def test_fixed_baseline_does_not_adapt_k(self):
        result = FixedKPipeline(_Retriever()).run("What is the policy?", 3, generate=False)
        self.assertEqual(result["telemetry"]["mode"], "fixed")
        self.assertEqual(result["telemetry"]["initial_k"], 3)
        self.assertEqual(result["telemetry"]["selected_k"], 3)
        with self.assertRaises(ValueError):
            FixedKPipeline(_Retriever()).run("What is the policy?", 4, generate=False)

    def test_adaptive_pipeline_uses_query_based_initial_k(self):
        result = AdaptivePipeline(_Retriever(), k_model=_KModel()).run("What is the policy?", generate=False)
        self.assertEqual(result.initial_k, 3)
        self.assertEqual(result.telemetry["policy_mode"], "ml")
        self.assertEqual(result.telemetry["num_chunks_used"], 1)


if __name__ == "__main__":
    unittest.main()
