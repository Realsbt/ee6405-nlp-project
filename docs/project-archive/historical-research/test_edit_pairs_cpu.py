import unittest

import numpy as np

from analyze_edit_pairs_cpu import pair_groups, paired_summary


class EditPairTests(unittest.TestCase):
    def test_creator_change_rejected(self):
        rows = [
            {"group_id": "g", "item_id": "o", "operation": "original", "label": "human_written"},
            {"group_id": "g", "item_id": "e", "operation": "edit", "label": "ai_humanized"},
        ]
        with self.assertRaises(ValueError):
            pair_groups(rows)

    def test_equal_source_not_variant_weighting(self):
        groups = [
            ({"item_id": "a"}, [{"item_id": "b"}, {"item_id": "c"}]),
            ({"item_id": "d"}, [{"item_id": "e"}]),
        ]
        p = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0],
                      [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float)
        s = paired_summary(groups, p, dict(zip("abcde", range(5))), .5)
        self.assertEqual(s["paired_mean_ai_score_delta"], .5)
        self.assertEqual(s["edited_variants"], 3)
        self.assertEqual(s["source_balanced_fraction_ai_edited_at_threshold"], .5)


if __name__ == "__main__":
    unittest.main()
