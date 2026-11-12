import unittest

from diagnose_model_gap import compare


class GapTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"item_id": "a", "label": "human_written", "domain": "news", "split_block_id": "g"},
            {"item_id": "b", "label": "human_written", "domain": "news", "split_block_id": "g"},
        ]
        self.text = {
            "a": {"true_label": "human_written", "domain": "news", "prediction": "ai_generated"},
            "b": {"true_label": "human_written", "domain": "news", "prediction": "human_written"},
        }
        self.graph = {
            "b": {"true_label": "human_written", "domain": "news", "prediction": "human_written"},
            "a": {"true_label": "human_written", "domain": "news", "prediction": "human_written"},
        }

    def test_alignment_and_recall(self):
        result = compare(self.rows, self.text, self.graph)
        self.assertEqual(result["accuracy_delta_pp"], 50)
        self.assertEqual(result["race_only_correct"], 1)
        self.assertEqual(result["source_blocks"], 1)
        self.assertNotIn("macro_f1_delta_pp", result)

    def test_mismatched_truth_rejected(self):
        self.graph["a"]["true_label"] = "ai_generated"
        with self.assertRaises(ValueError):
            compare(self.rows, self.text, self.graph)

    def test_missing_item_rejected(self):
        del self.graph["a"]
        with self.assertRaises(KeyError):
            compare(self.rows, self.text, self.graph)


if __name__ == "__main__":
    unittest.main()
