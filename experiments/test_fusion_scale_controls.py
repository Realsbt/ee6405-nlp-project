import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments.edu_fusion import EDUFusion
from experiments.evaluate_fusion_scale import reload_scale_run
from experiments.fusion_scale_controls import (
    INITIAL_LOGIT, SCALE_CONDITIONS, InputIndependentGate, ScaleControlModel,
    scale_config, scale_dataset,
)


class ScaleControlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.hidden = torch.randn(2, 6, 8)
        self.mask = torch.ones(2, 6, dtype=torch.long)
        self.spans = [[(1, 3), (3, 5)], [(1, 2), (2, 5)]]

    def test_fixed_and_global_initially_match_adaptive_gate(self):
        adaptive = EDUFusion(8, "fusion_gated")
        expected = adaptive(self.hidden, self.mask, self.spans)
        for learnable in (False, True):
            with self.subTest(learnable=learnable):
                fusion = EDUFusion(8, "fusion_gated")
                fusion.load_state_dict(adaptive.state_dict())
                fusion.gate = InputIndependentGate(learnable)
                actual = fusion(self.hidden, self.mask, self.spans)
                torch.testing.assert_close(actual["pooled"], expected["pooled"])
                torch.testing.assert_close(actual["edu_gate"], expected["edu_gate"])
                for left, right in zip(actual["edu_weights"], expected["edu_weights"]):
                    torch.testing.assert_close(left, right)

    def test_fixed_buffer_has_no_parameter_and_global_has_one(self):
        fixed, shared = InputIndependentGate(False), InputIndependentGate(True)
        self.assertEqual(sum(p.numel() for p in fixed.parameters()), 0)
        self.assertEqual(sum(p.numel() for p in shared.parameters()), 1)
        self.assertIn("logit", fixed.state_dict())
        self.assertEqual(fixed.logit.item(), INITIAL_LOGIT)
        features = torch.randn(3, 16)
        torch.testing.assert_close(shared(features), shared(features * 100))
        shared(features).sigmoid().sum().backward()
        self.assertTrue(torch.isfinite(shared.logit.grad))
        self.assertGreater(shared.logit.grad.item(), 0)

    def test_fixed_scale_still_trains_attention_and_content(self):
        fusion = EDUFusion(8, "fusion_gated")
        fusion.gate = InputIndependentGate(False)
        hidden = self.hidden.clone().requires_grad_()
        fusion(hidden, self.mask, self.spans)["pooled"].square().sum().backward()
        self.assertGreater(fusion.attention[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(hidden.grad.abs().sum().item(), 0)
        self.assertIsNone(fusion.gate.logit.grad)

    def test_recipe_is_separate_and_dataset_preserves_input(self):
        from experiments.edu_fusion import FUSION_CONDITIONS, fusion_config
        for condition in SCALE_CONDITIONS:
            config = scale_config(condition, 2026)
            self.assertEqual(config["seed"], 2026)
            self.assertFalse(config["use_supcon"])
            self.assertNotEqual(config["recipe_id"], fusion_config("fusion_gated")["recipe_id"])
            with patch("experiments.fusion_scale_controls.IndexedSameWindowDataset") as dataset:
                scale_dataset("cache", "val", condition)
            dataset.assert_called_once_with("cache", "val", "text")
        self.assertEqual(len(FUSION_CONDITIONS), 4)
        for function in (scale_config, scale_dataset):
            with self.assertRaises(ValueError):
                if function is scale_config:
                    function("invalid")
                else:
                    function("cache", "val", "invalid")

    def test_common_initialization_and_strict_reload(self):
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
            for condition in SCALE_CONDITIONS:
                torch.manual_seed(3407)
                config = scale_config(condition)
                config["architecture"].update(bert_dim=8, gnn_hidden_dim=4)
                models.append(ScaleControlModel(config))
        batch = {
            "full_text_input_ids": torch.arange(6).repeat(2, 1),
            "full_text_attention_mask": self.mask, "edu_token_spans": self.spans,
        }
        for model in models:
            model.eval()
        expected = models[0](batch)["logits"]
        for model in models:
            torch.testing.assert_close(model(batch)["logits"], expected)
            for name, tensor in models[0].state_dict().items():
                if not name.startswith("fusion.gate."):
                    torch.testing.assert_close(tensor, model.state_dict()[name])
            model.load_state_dict(model.state_dict(), strict=True)

    def test_reload_defaults_to_validation_only(self):
        self.assertFalse(inspect.signature(reload_scale_run).parameters["evaluate_test"].default)
        record = {
            "item_id": "v1", "group_id": "g1", "domain": "essay",
            "label": "human_written",
        }
        data = SimpleNamespace(records=[record])
        evaluation = ({"macro_f1": 0.5}, np.array([[0.7, 0.1, 0.1, 0.1]]), ["v1"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("{}")
            checkpoint = {"config": scale_config("scale_fixed"), "model_state_dict": {}}
            with patch("experiments.evaluate_fusion_scale.verify_run",
                       return_value=(checkpoint, {"best_checkpoint_sha256": "hash"})), \
                 patch("experiments.evaluate_fusion_scale.ScaleControlModel"), \
                 patch("experiments.evaluate_fusion_scale.scale_dataset", return_value=data) as factory, \
                 patch("experiments.train_same_window.evaluate", return_value=evaluation):
                result = reload_scale_run(root, root, root / "output")
            factory.assert_called_once_with(root, "val", "scale_fixed")
            self.assertFalse(result["test_evaluated"])
            self.assertEqual(set(result["splits"]), {"validation"})
            self.assertFalse((root / "output/test_predictions.jsonl").exists())
            self.assertEqual(json.loads((root / "output/result.json").read_text()), result)


if __name__ == "__main__":
    unittest.main()
