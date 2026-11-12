import unittest

import numpy as np

from scripts.analyze_validation_errors import diagnose
from experiments.common import LABELS


class ValidationDiagnosticsTests(unittest.TestCase):
    def test_overlap_and_no_raw_text_export(self):
        records = [{"item_id": str(i), "group_id": str(i), "domain": "essay",
                    "label": label, "article": "must not appear"}
                   for i, label in enumerate(LABELS)]
        a = np.eye(4)
        b = np.eye(4)
        a[3], b[3] = [0, 0, 1, 0], [0, 1, 0, 0]
        result = diagnose(records, {"a": a, "b": b})
        self.assertEqual(result["pairwise_errors"]["a__b"]["both_wrong"], 1)
        self.assertEqual(result["pairwise_errors"]["a__b"]["wrong_set_jaccard"], 1)
        self.assertEqual(len(result["all_models_wrong"]), 1)
        self.assertNotIn("article", result["all_models_wrong"][0])
        self.assertFalse(result["test_opened"])


if __name__ == "__main__":
    unittest.main()
