import unittest

from evaluation_metrics import evaluate_creator, negative_only_threshold
from prepare_research_data import derive_label, text_hash
from summarize_transfer import cluster_draw
import numpy as np


class EvaluationTests(unittest.TestCase):
    def test_cluster_draw_keeps_siblings_together(self):
        weights = cluster_draw(["a", "a", "b", "c"], np.random.default_rng(7))
        self.assertEqual(weights[0], weights[1])
        self.assertEqual(weights[[0, 2, 3]].sum(), 3)
        np.testing.assert_array_equal(
            weights, cluster_draw(["a", "a", "b", "c"], np.random.default_rng(7)))

    def test_threshold_ties_are_conservative(self):
        result = negative_only_threshold([0.8] * 100)
        self.assertGreater(result["threshold"], 0.8)
        self.assertEqual(result["actual_false_positives"], 0)

    def test_threshold_budget_and_small_samples(self):
        result = negative_only_threshold([i / 100 for i in range(100)])
        self.assertEqual(result["actual_false_positives"], 1)
        self.assertEqual(negative_only_threshold([0.2, 1.0])["actual_false_positives"], 0)

    def test_invalid_threshold_inputs(self):
        for values in ([], [float("nan")], [-1], [1.1]):
            with self.assertRaises(ValueError):
                negative_only_threshold(values)

    def test_known_binary_metrics(self):
        result = evaluate_creator([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.95)
        self.assertEqual(result["auroc"], 1)
        self.assertEqual(result["at_threshold"]["fixed_0_5"]["macro_f1"], 1)
        self.assertEqual(result["at_threshold"]["hart_validation_locked"]["tp"], 0)

    def test_provenance_mapping(self):
        cases = [
            ("news-a", "human", "human", "human_written"),
            ("rep/news-a", "human", "rephrase:x", "human_ai_polished"),
            ("gen/news-a", "machine:x", "machine:x", "ai_generated"),
            ("hum/gen/news-a", "machine:x", "humanize:human", "ai_humanized"),
            ("hum/gen/news-a", "machine:x", "humanize:tool", "ai_humanized"),
            ("hum/gen/news-a", "machine:x", "humanize:other-model", "ai_generated"),
            ("unknown/news-a", "human", "unknown", None),
        ]
        for identifier, content, language, expected in cases:
            self.assertEqual(derive_label({"id": identifier, "content_source": content,
                                          "language_source": language}), expected)
        self.assertEqual(text_hash("Ａ BC\n"), text_hash("a  bc"))


if __name__ == "__main__":
    unittest.main()
