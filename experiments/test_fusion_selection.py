import copy
import unittest

from cloud.kaggle.run_edu_fusion import select_candidate


class FusionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.screen = {
            name: {
                "condition": name, "seed": 3407, "status": "validation_completed",
                "test_evaluated": False, "test": None,
                "validation": {"macro_f1": value}, "parameters": {"trainable": parameters},
            } for name, value, parameters in (
                ("fusion_cls", 0.92, 100), ("fusion_mean", 0.90, 100),
                ("fusion_attention", 0.91, 200), ("fusion_gated", 0.91, 210),
            )
        }

    def test_best_fusion_is_frozen_even_when_baseline_is_better(self):
        selected = select_candidate(self.screen)
        self.assertEqual(selected["selected"], "fusion_attention")
        self.assertAlmostEqual(selected["validation_gap"], -0.01)
        self.assertFalse(selected["test_used_for_selection"])

    def test_higher_validation_wins_over_parameter_tie_break(self):
        self.screen["fusion_gated"]["validation"]["macro_f1"] = 0.93
        self.assertEqual(select_candidate(self.screen)["selected"], "fusion_gated")

    def test_missing_candidate_test_access_or_mismatched_seed_is_rejected(self):
        missing = copy.deepcopy(self.screen)
        del missing["fusion_mean"]
        with self.assertRaises(ValueError):
            select_candidate(missing)
        for key, value in (("test_evaluated", True), ("test", {"macro_f1": 1}), ("seed", 42)):
            changed = copy.deepcopy(self.screen)
            changed["fusion_mean"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                select_candidate(changed)


if __name__ == "__main__":
    unittest.main()
