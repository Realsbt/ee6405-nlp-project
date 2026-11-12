import unittest

from experiments.fusion_dfs_policy import SEEDS, continuation_gate, extension_tree
from experiments.fusion_extensions import extension_config


class FusionDFSPolicyTests(unittest.TestCase):
    def test_pilot_prunes_small_or_negative_gains(self):
        for score in (0.89, 0.9, 0.901):
            self.assertEqual(continuation_gate({3407: score}, {3407: 0.9}, True)["decision"],
                             "prune")
        self.assertEqual(continuation_gate({3407: 0.902}, {3407: 0.9}, True)["decision"],
                         "continue")

    def test_confirmation_requires_mean_majority_and_no_large_regression(self):
        baseline = dict.fromkeys(SEEDS, 0.9)
        for values in ((0.901, 0.901, 0.901), (0.92, 0.899, 0.899), (0.91, 0.91, 0.895)):
            self.assertEqual(continuation_gate(dict(zip(SEEDS, values)), baseline)["decision"],
                             "prune")
        self.assertEqual(continuation_gate(dict(zip(SEEDS, (0.906, 0.904, 0.898))),
                                           baseline)["decision"], "continue")

    def test_missing_seeds_and_invalid_scores_fail(self):
        with self.assertRaises(ValueError):
            continuation_gate({3407: 0.91}, {3407: 0.9})
        for value in (float("nan"), float("inf"), 2, -1):
            with self.assertRaises(ValueError):
                continuation_gate({3407: value}, {3407: 0.9}, True)

    def test_cls_prunes_meaningless_segmentation(self):
        calls = []
        winner = extension_tree("fusion_cls", lambda c, r: calls.append((c, r)) or False)
        self.assertEqual(winner, "ext_edu_ce2")
        self.assertEqual(calls, [("ext_edu_consistent", "ext_edu_ce2")])

    def test_successful_branch_is_expanded_depth_first(self):
        calls = []
        winner = extension_tree("fusion_gated", lambda c, r: calls.append((c, r)) or True)
        self.assertEqual(winner, "ext_chunks_consistent")
        self.assertEqual(calls, [
            ("ext_edu_consistent", "ext_edu_ce2"),
            ("ext_chunks_consistent", "ext_edu_consistent"),
        ])

    def test_failed_combination_backtracks_to_independent_architecture(self):
        calls = []
        outcomes = iter((True, False, True))
        winner = extension_tree("fusion_attention", lambda c, r: calls.append((c, r)) or next(outcomes),
                                prefer=lambda candidate, incumbent: True)
        self.assertEqual(winner, "ext_chunks_ce2")
        self.assertEqual(calls[-1], ("ext_chunks_ce2", "ext_edu_ce2"))

    def test_weaker_backtrack_cannot_overwrite_confirmed_incumbent(self):
        outcomes = iter((True, False, True))
        winner = extension_tree("fusion_gated", lambda c, r: next(outcomes),
                                prefer=lambda candidate, incumbent: False)
        self.assertEqual(winner, "ext_edu_consistent")

    def test_two_local_passes_do_not_guarantee_terminal_stability(self):
        reference = dict.fromkeys(SEEDS, 0.8)
        first = dict(zip(SEEDS, (0.8079, 0.8001, 0.798)))
        combined = dict(zip(SEEDS, (0.8158, 0.7981, 0.7981)))
        self.assertEqual(continuation_gate(first, reference)["decision"], "continue")
        self.assertEqual(continuation_gate(combined, first)["decision"], "continue")
        self.assertEqual(continuation_gate(combined, reference)["decision"], "prune")

    def test_training_parent_is_explicit_not_always_gated(self):
        for parent in ("fusion_cls", "fusion_mean", "fusion_attention", "fusion_gated"):
            config = extension_config("ext_edu_consistent", 42, parent)
            self.assertEqual(config["extension"]["parent"], parent)
            self.assertEqual(config["seed"], 42)


if __name__ == "__main__":
    unittest.main()
