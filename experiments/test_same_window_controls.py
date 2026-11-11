import copy
from collections import Counter
from types import SimpleNamespace
import unittest

import torch

from experiments.nfkc_alignment import remap_verified
from experiments.same_window_controls import config_for, mean_edu, shuffled_relations


def node(text, left=None, right=None):
    return SimpleNamespace(text=text, start=0, end=len(text), left=left, right=right)


class SameWindowControlTests(unittest.TestCase):
    def setUp(self):
        self.item = {
            "id": "sample",
            "label": 3,
            "parent_child_edges": [(20, 0)],
            "root_node_ids": [20],
            "nodes": [{"id": index, "type": "relation_node", "relation": relation}
                      for index, relation in enumerate(["Cause", "Joint", "Contrast", "Joint"] * 8)]
                     + [{"id": 100, "type": "edu_node"}],
        }

    def test_shuffle_is_deterministic_label_independent_and_histogram_preserving(self):
        original = copy.deepcopy(self.item)
        first = shuffled_relations(self.item)
        second = shuffled_relations(self.item)
        self.assertEqual(first, second)
        self.assertEqual(self.item, original)
        self.assertEqual(Counter(node["relation"] for node in first["nodes"][:-1]),
                         Counter(node["relation"] for node in self.item["nodes"][:-1]))
        self.assertNotEqual(first["nodes"], self.item["nodes"])
        self.item["label"] = 0
        self.assertEqual(first["nodes"], shuffled_relations(self.item)["nodes"])

    def test_uniform_relation_graph_is_unchanged(self):
        for relation_node in self.item["nodes"][:-1]:
            relation_node["relation"] = "Joint"
        self.assertEqual(shuffled_relations(self.item), self.item)

    def test_edu_mean_has_equal_edu_weight_and_gradients(self):
        hidden = torch.tensor([[1.0], [3.0], [9.0]], requires_grad=True)
        value = mean_edu(hidden, [(0, 2), (2, 3)])
        self.assertEqual(value.item(), 5.5)
        value.backward()
        torch.testing.assert_close(hidden.grad, torch.tensor([[0.25], [0.25], [0.5]]))
        with self.assertRaises(ValueError):
            mean_edu(hidden, [(0, 4)])

    def test_configs_use_ce_and_fixed_control_seed(self):
        configs = [config_for(condition) for condition in ("text", "graph", "shuffled", "edu")]
        self.assertTrue(all(not config["use_supcon"] for config in configs))
        self.assertEqual(configs[1]["architecture"], configs[2]["architecture"])
        self.assertEqual(configs[1]["control_seed"], config_for("shuffled", 42)["control_seed"])

    def test_nfkc_expansion_preserves_original_offsets(self):
        tree = node("A... B", node("A..."), node("B"))
        result = remap_verified(None, tree, None, None, "A\u2026 B")
        self.assertEqual(result["mode"], "per_character_nfkc")
        self.assertEqual((tree.left.start, tree.left.end, tree.left.text), (0, 2, "A\u2026"))
        self.assertEqual(tree.text, "A\u2026 B")

    def test_nfkc_rejects_split_expansion_and_content_change(self):
        tree = node("A...B", node("A."), node("..B"))
        with self.assertRaisesRegex(ValueError, "boundary"):
            remap_verified(None, tree, None, None, "A\u2026B")
        with self.assertRaisesRegex(ValueError, "differs"):
            remap_verified(None, node("AXB"), None, None, "A\u2026B")

    def test_audited_deleted_character_preserves_original(self):
        tree = node("AB", node("A"), node("B"))
        result = remap_verified(None, tree, None, None, "A\u200bB")
        self.assertEqual(result["mode"], "audited_projection")
        self.assertEqual(tree.left.text, "A\u200b")
        self.assertEqual(tree.right.text, "B")


if __name__ == "__main__":
    unittest.main()
