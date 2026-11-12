import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments.edu_fusion import (
    EDUFusion, EDUFusionModel, FUSION_CONDITIONS, fusion_config, fusion_dataset,
)
from experiments.common import file_hash
from experiments.evaluate_edu_fusion import explanation_rows, verify_run
from experiments.train_same_window import evaluate_partitions


class EDUFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3407)
        self.hidden = torch.randn(2, 7, 8)
        self.mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]])
        self.spans = [[(1, 3), (3, 6)], [(1, 2), (2, 5)]]

    def test_cls_baseline_is_exact_cls(self):
        output = EDUFusion(8, "fusion_cls")(self.hidden, self.mask, self.spans)
        torch.testing.assert_close(output["pooled"], self.hidden[:, 0])
        self.assertEqual(output["edu_weights"], [])
        self.assertEqual(output["edu_gate"].sum(), 0)

    def test_mean_is_cls_plus_equal_weight_edu_mean(self):
        output = EDUFusion(8, "fusion_mean")(self.hidden, self.mask, self.spans)
        expected = torch.stack([
            row[0] + torch.stack([row[a:b].mean(0) for a, b in spans]).mean(0)
            for row, spans in zip(self.hidden, self.spans)
        ])
        torch.testing.assert_close(output["pooled"], expected)
        torch.testing.assert_close(output["edu_gate"], torch.ones(2, 1))

    def test_zero_attention_equals_mean_and_weights_sum_to_one(self):
        fusion = EDUFusion(8, "fusion_attention")
        nn.init.zeros_(fusion.attention[-1].weight)
        output = fusion(self.hidden, self.mask, self.spans)
        mean = EDUFusion(8, "fusion_mean")(self.hidden, self.mask, self.spans)
        torch.testing.assert_close(output["pooled"], mean["pooled"])
        for weights in output["edu_weights"]:
            torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
            self.assertTrue(torch.all(weights >= 0))

    def test_padding_values_and_batch_composition_do_not_affect_pooling(self):
        fusion = EDUFusion(8, "fusion_gated")
        original = fusion(self.hidden, self.mask, self.spans)["pooled"]
        changed = self.hidden.clone()
        changed[1, 5:] = 100000
        torch.testing.assert_close(original, fusion(changed, self.mask, self.spans)["pooled"])
        single = fusion(self.hidden[1:2], self.mask[1:2], self.spans[1:])["pooled"]
        torch.testing.assert_close(original[1:], single)

    def test_gate_initial_value_and_gradients(self):
        hidden = self.hidden.clone().requires_grad_()
        fusion = EDUFusion(8, "fusion_gated")
        output = fusion(hidden, self.mask, self.spans)
        torch.testing.assert_close(output["edu_gate"], torch.full((2, 1), torch.sigmoid(torch.tensor(-2.)).item()))
        output["pooled"].square().sum().backward()
        for parameter in fusion.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        torch.testing.assert_close(hidden.grad[1, 5:], torch.zeros(2, 8))
        self.assertGreater(hidden.grad[:, 0].abs().sum().item(), 0)
        self.assertGreater(hidden.grad[:, 1:5].abs().sum().item(), 0)

    def test_shared_boundary_tokens_are_valid(self):
        output = EDUFusion(8, "fusion_mean")(
            self.hidden[:1], self.mask[:1], [[(1, 4), (3, 6)]],
        )
        self.assertTrue(torch.isfinite(output["pooled"]).all())

    def test_invalid_spans_and_masks_fail(self):
        fusion = EDUFusion(8, "fusion_attention")
        for spans in ([], [(1, 8)], [(3, 5), (1, 3)], [(1, 1)], [(-1, 3)]):
            with self.subTest(spans=spans), self.assertRaises(ValueError):
                fusion(self.hidden[:1], self.mask[:1], [spans])
        with self.assertRaisesRegex(ValueError, "padding"):
            fusion(self.hidden[1:], self.mask[1:], [[(1, 6)]])
        with self.assertRaisesRegex(ValueError, "per item"):
            fusion(self.hidden, self.mask, self.spans[:1])
        with self.assertRaisesRegex(ValueError, "matching"):
            fusion(self.hidden, self.mask[:, :2], self.spans)
        with self.assertRaisesRegex(ValueError, "CLS"):
            fusion(self.hidden, torch.zeros_like(self.mask), self.spans)

    def test_configs_share_recipe_and_original_controls_are_unchanged(self):
        configs = [fusion_config(condition, 42) for condition in FUSION_CONDITIONS]
        for config in configs:
            self.assertFalse(config["use_supcon"])
            self.assertEqual(config["seed"], 42)
            self.assertEqual(config["architecture"], configs[0]["architecture"])
        from experiments.same_window_controls import CONDITIONS, config_for
        self.assertEqual(CONDITIONS, ("text", "graph", "shuffled", "edu"))
        self.assertNotIn("fusion", config_for("text"))
        with self.assertRaises(ValueError):
            fusion_config("not_a_condition")

    def test_dataset_uses_original_text_without_relation_changes(self):
        with patch("experiments.edu_fusion.IndexedSameWindowDataset") as dataset:
            fusion_dataset("cache", "val", "fusion_gated")
        dataset.assert_called_once_with("cache", "val", "text")

    def test_common_encoder_and_head_initialization_is_matched(self):
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = nn.Embedding(16, 8)

            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(last_hidden_state=self.embeddings(input_ids))

        def baseline(config):
            return SimpleNamespace(bert_model=Backbone(), _init_classifier=lambda module: None)

        models = []
        with patch("experiments.edu_fusion.build_model", side_effect=baseline):
            for condition in FUSION_CONDITIONS:
                torch.manual_seed(3407)
                config = fusion_config(condition)
                config["architecture"].update(bert_dim=8, gnn_hidden_dim=4)
                models.append(EDUFusionModel(config))
        for model in models[1:]:
            for name, tensor in models[0].state_dict().items():
                torch.testing.assert_close(tensor, model.state_dict()[name])
        batch = {
            "full_text_input_ids": torch.zeros(2, 7, dtype=torch.long),
            "full_text_attention_mask": self.mask,
            "edu_token_spans": self.spans,
        }
        for model in models:
            model.eval()
            output = model(batch)
            self.assertEqual(output["logits"].shape, (2, 4))
            output["logits"].sum().backward()


class FusionEvaluationTests(unittest.TestCase):
    def test_screening_never_constructs_test_dataset(self):
        data = SimpleNamespace(records=[{
            "item_id": "v1", "group_id": "g1", "domain": "essay", "label": "human_written",
        }])
        evaluation = ({"macro_f1": 0.5}, np.array([[0.7, 0.1, 0.1, 0.1]]), ["v1"])
        with tempfile.TemporaryDirectory() as directory:
            for validation_only in (True, False):
                with patch("experiments.train_same_window.evaluate", return_value=evaluation), \
                     patch("experiments.train_same_window.write_jsonl"), \
                     patch("experiments.train_same_window.prediction_rows", return_value=[]) as rows, \
                     patch("experiments.edu_fusion.fusion_dataset", return_value=data) as factory:
                    validation, test = evaluate_partitions(
                        None, data, cache=Path("cache"), condition="fusion_gated",
                        batch_size=4, output=Path(directory), dataset_factory=factory,
                        validation_only=validation_only,
                    )
                self.assertEqual(validation, evaluation[0])
                if validation_only:
                    factory.assert_not_called()
                    self.assertIsNone(test)
                    self.assertEqual(rows.call_count, 1)
                else:
                    factory.assert_called_once_with(Path("cache"), "test", "fusion_gated")
                    self.assertEqual(test, evaluation[0])
                    self.assertEqual(rows.call_count, 2)

    def test_reload_verifies_config_cache_checkpoint_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            cache = Path(directory) / "cache"
            run.mkdir()
            cache.mkdir()
            (cache / "manifest.json").write_text("{}")
            config = fusion_config("fusion_gated")
            torch.save({"config": config, "model_state_dict": {}}, run / "best.pt")
            recorded = {
                "training_only_smoke": False, "config": config,
                "cache_manifest_sha256": file_hash(cache / "manifest.json"),
            }
            result = {"status": "validation_completed",
                      "best_checkpoint_sha256": file_hash(run / "best.pt")}
            source = Path(__file__).with_name("edu_fusion.py")
            (run / "config.json").write_text(json.dumps(recorded))
            (run / "result.json").write_text(json.dumps(result))
            (run / "source_hashes.json").write_text(json.dumps({source.name: file_hash(source)}))
            checkpoint, loaded_result = verify_run(run, cache)
            self.assertEqual(checkpoint["config"], config)
            self.assertEqual(loaded_result, result)
            (run / "source_hashes.json").write_text(json.dumps({source.name: "changed"}))
            with self.assertRaisesRegex(ValueError, "source changed"):
                verify_run(run, cache)
            (run / "source_hashes.json").write_text(json.dumps({source.name: file_hash(source)}))
            (cache / "manifest.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                verify_run(run, cache)
            (cache / "manifest.json").write_text("{}")
            torch.save({"config": config, "model_state_dict": {"changed": 1}}, run / "best.pt")
            with self.assertRaisesRegex(ValueError, "checkpoint hash"):
                verify_run(run, cache)

    def test_smoke_checkpoint_cannot_be_used_as_performance_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "config.json").write_text(json.dumps({"training_only_smoke": True}))
            with self.assertRaisesRegex(ValueError, "Smoke checkpoints"):
                verify_run(run, run)

    def test_explanation_sampling_is_first_per_class_and_cls_has_no_weights(self):
        class Dataset:
            labels = np.array([3, 0, 1, 2, 0])
            records = [{"item_id": str(i), "label": str(label)}
                       for i, label in enumerate(labels)]

            def __getitem__(self, index):
                return {
                    "id": str(index),
                    "label": torch.tensor(int(self.labels[index])),
                    "full_text_input_ids": torch.tensor([0, 4, 5, 2]),
                    "full_text_attention_mask": torch.ones(4, dtype=torch.long),
                    "edu_token_spans": [(1, 3)],
                }

        class Model(nn.Module):
            def forward(self, batch):
                return {
                    "logits": torch.zeros(1, 4), "edu_weights": [],
                    "edu_gate": torch.zeros(1, 1),
                }

        tokenizer = SimpleNamespace(decode=lambda ids: "decoded")
        rows = explanation_rows(Model(), Dataset(), tokenizer, 1)
        self.assertEqual([row["item_id"] for row in rows], ["1", "2", "3", "0"])
        self.assertIsNone(rows[0]["edus"][0]["attention_weight"])
        self.assertEqual(rows[0]["edus"][0]["decoded_excerpt"], "decoded")


if __name__ == "__main__":
    unittest.main()
