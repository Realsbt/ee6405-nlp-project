import json
from pathlib import Path
import tempfile
import unittest

from analyze_saved_probabilities import analyze_file, binary_metrics, probability_vector


class MetricsTests(unittest.TestCase):
    def test_perfect_and_wrong(self):
        perfect = binary_metrics([(0, 0.0), (1, 1.0)])
        self.assertEqual(perfect["binary_brier"], 0)
        self.assertEqual(perfect["positive_class_ece_10_bins"], 0)
        self.assertEqual(perfect["reliability_bins"][9]["count"], 1)
        wrong = binary_metrics([(0, 1.0), (1, 0.0)])
        self.assertEqual(wrong["binary_brier"], 1)
        self.assertEqual(wrong["positive_class_ece_10_bins"], 1)

    def test_half_and_single_class(self):
        result = binary_metrics([(0, 0.5), (1, 0.5)])
        self.assertEqual(result["binary_brier"], 0.25)
        self.assertEqual(result["positive_class_ece_10_bins"], 0)
        self.assertEqual(result["fpr_at_fixed_0_5"], 1)
        self.assertIsNone(binary_metrics([(0, 0.1)])["tpr_at_fixed_0_5"])

    def test_float32_rounding(self):
        values = probability_vector({"human_written": 0, "human_ai_polished": 0,
                                     "ai_generated": 0.9999, "ai_humanized": 0.00010001})
        self.assertAlmostEqual(sum(values), 1)
        self.assertLessEqual(values[2] + values[3], 1)
        with self.assertRaises(ValueError):
            probability_vector({"human_written": 0, "human_ai_polished": 0,
                                "ai_generated": 0.9, "ai_humanized": 0.2})

    def test_reject_invalid(self):
        for pairs in ([], [(2, 0.2)], [(1, float("nan"))], [(1, 1.01)]):
            with self.assertRaises(ValueError):
                binary_metrics(pairs)
        with self.assertRaises(ValueError):
            probability_vector({"human_written": 1})

    def test_multiclass_mapping_and_duplicate_rejection(self):
        row = {"item_id": "one", "true_label": "human_ai_polished",
               "probabilities": {"human_written": 0.1, "human_ai_polished": 0.6,
                                 "ai_generated": 0.2, "ai_humanized": 0.1}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.jsonl"
            path.write_text(json.dumps(row) + "\n")
            result = analyze_file(path)
            self.assertAlmostEqual(result["binary_brier"], 0.09)
            self.assertAlmostEqual(result["multiclass_brier_sum_over_classes"], 0.22)
            path.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaises(ValueError):
                analyze_file(path)

    def test_binary_score_mismatch(self):
        row = {"item_id": "one", "true_binary_label": 1, "ai_creator_score": 0.9,
               "four_class_scores": {"human_written": 1, "human_ai_polished": 0,
                                     "ai_generated": 0, "ai_humanized": 0}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.jsonl"
            path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                analyze_file(path, binary=True)


if __name__ == "__main__":
    unittest.main()
