from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from window_controls import config_for, mean_edu, shuffled_relations
from diagnose_window_alignment import differences
from repair_window_unknowns import token_offset_spans


class WindowControlTests(unittest.TestCase):
    def setUp(self):
        self.item = {
            "id": "sample", "label": 3, "parent_child_edges": [(20, 0)], "root_node_ids": [20],
            "edu_token_spans": [[1, 3]], "full_text_input_ids": [0, 4, 5, 2],
            "nodes": [{"id": i, "type": "relation_node", "relation": r}
                      for i, r in enumerate(["Cause", "Joint", "Contrast", "Joint"] * 8)]
                     + [{"id": 100, "type": "edu_node"}],
        }

    def test_shuffle_deterministic_histogram_and_topology_preserved(self):
        original = copy.deepcopy(self.item)
        a, b = shuffled_relations(self.item), shuffled_relations(self.item)
        self.assertEqual(a, b)
        self.assertEqual(self.item, original)
        self.assertEqual(Counter(n["relation"] for n in a["nodes"][:-1]),
                         Counter(n["relation"] for n in self.item["nodes"][:-1]))
        self.assertNotEqual(a["nodes"], self.item["nodes"])
        for key in self.item:
            if key != "nodes":
                self.assertEqual(a[key], self.item[key])

    def test_shuffle_does_not_read_label(self):
        a = shuffled_relations(self.item)
        self.item["label"] = 0
        self.assertEqual(a["nodes"], shuffled_relations(self.item)["nodes"])

    def test_uniform_relation_graph_unchanged(self):
        for n in self.item["nodes"][:-1]:
            n["relation"] = "Joint"
        self.assertEqual(shuffled_relations(self.item), self.item)

    def test_edu_mean_and_gradients(self):
        hidden = torch.tensor([[1.], [3.], [9.]], requires_grad=True)
        value = mean_edu(hidden, [(0, 2), (2, 3)])
        self.assertEqual(value.item(), 5.5)
        value.sum().backward()
        torch.testing.assert_close(hidden.grad, torch.tensor([[.25], [.25], [.5]]))
        with self.assertRaises(ValueError):
            mean_edu(hidden, [(0, 4)])

    def test_ce_only_and_fixed_control_seed(self):
        configs = [config_for(c) for c in ("text", "graph", "shuffled", "edu")]
        self.assertTrue(all(not c["use_supcon"] for c in configs))
        self.assertEqual(configs[1]["architecture"], configs[2]["architecture"])
        self.assertEqual(configs[1]["control_seed"], config_for("shuffled", 42)["control_seed"])

    def test_alignment_diagnostic_preserves_unknown_character_failure(self):
        report = differences("cost \u20bf 10", "cost <unk> 10")
        self.assertFalse(report["nfkc_nonspace_equal"])
        self.assertEqual(report["differences"][0]["original_codepoints"], ["U+20BF"])
        self.assertTrue(differences("x \ufb01", "x fi")["nfkc_nonspace_equal"])

    def test_smoke_relation_audit_handles_tensor_graphs(self):
        from audit_window_controls import audit
        from window_controls import ROOT
        report = audit(ROOT / "resources/experiments/same_window_smoke_nfkc_v2", smoke=True)
        self.assertEqual(report["totals"]["graphs"], 32)
        self.assertGreater(report["changed_relation_node_fraction"], 0)

    def test_unknown_offset_mapping_and_rejections(self):
        args = (["a<unk>", "b"], ["a", "<unk>", "b"], [(0, 1), (1, 2), (2, 3)], [0, 1, 2], "a\u225cb")
        self.assertEqual(token_offset_spans(*args), [(0, 2), (2, 3)])
        with self.assertRaisesRegex(ValueError, "Leaf strings differ"):
            token_offset_spans(["a", "b"], *args[1:])
        with self.assertRaisesRegex(ValueError, "splits a tokenizer piece"):
            token_offset_spans(["a<u", "nk>b"], *args[1:])
        with self.assertRaisesRegex(ValueError, "omitted"):
            token_offset_spans(*args[:-1], "a\u225cb!")
        with self.assertRaisesRegex(ValueError, "overlapping"):
            token_offset_spans(["a<unk>", "b"], args[1], [(0, 1), (1, 2), (1, 3)], args[3], args[4])

    def test_indexed_cache_roundtrip_and_partial_tamper_guards(self):
        from same_window_cache import atomic_json, compile_graphs
        from screen_structure_cpu import digest, rows
        from window_controls import ROOT, WindowDataset, tokenizer_local
        record = rows(ROOT / "resources/experiments/same_window_smoke_nfkc_v2/parsed.jsonl")[0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            for name in ("parsed.jsonl", "inputs.jsonl"):
                (path / name).write_text(json.dumps(record) + "\n")
            atomic_json(path / "protocol.json", {"inputs_sha256": digest(path / "inputs.jsonl")})
            compile_graphs(path, {record["item_id"]: record}, tokenizer_local())
            atomic_json(path / "manifest.json", {
                "full_cache_ready": False, "graphs_sha256": digest(path / "graphs.jsonl"),
                "index_sha256": digest(path / "index.json"), "protocol_sha256": digest(path / "protocol.json"),
            })
            with self.assertRaisesRegex(ValueError, "complete, failure-free"):
                WindowDataset(path, record["source_split"], "graph")
            dataset = WindowDataset(path, record["source_split"], "graph", smoke=True)
            try:
                item = dataset[0]
                self.assertEqual(item["full_text_input_ids"].tolist(), record["visible_token_ids"])
                self.assertIsInstance(item["domain_label"], torch.Tensor)
                self.assertIsInstance(item["label"], torch.Tensor)
                self.assertTrue(all(isinstance(k, int) for k in item["descendant_map"]))
            finally:
                if dataset._handle:
                    dataset._handle.close()
            with (path / "graphs.jsonl").open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "cache changed"):
                WindowDataset(path, record["source_split"], "graph", smoke=True)

    def test_prediction_output_alignment_and_preservation(self):
        from train_window_controls import save_predictions
        from experiments.common import LABELS
        data = SimpleNamespace(records=[{"item_id": "x", "group_id": "g", "domain": "essay",
                                         "label": LABELS[0]}])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.jsonl"
            with self.assertRaisesRegex(ValueError, "order/size"):
                save_predictions(path, data, np.array([[1., 0., 0., 0.]]), ["y"])
            self.assertFalse(path.exists())
            save_predictions(path, data, np.array([[1., 0., 0., 0.]]), ["x"])
            self.assertEqual(json.loads(path.read_text())["prediction"], LABELS[0])
            with self.assertRaises(FileExistsError):
                save_predictions(path, data, np.array([[1., 0., 0., 0.]]), ["x"])


if __name__ == "__main__":
    unittest.main()
