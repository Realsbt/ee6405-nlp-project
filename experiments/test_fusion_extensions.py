import unittest

import torch
from torch.nn import functional as F
from unittest.mock import patch

from experiments.fusion_extensions import (
    EXTENSION_CONDITIONS, chunk_spans, extension_config, extension_dataset,
    paired_dropout_loss, validation_partition,
)


class FusionExtensionTests(unittest.TestCase):
    def test_chunk_boundaries_exclude_special_and_padding_tokens(self):
        ids = torch.tensor([[0, 4, 5, 6, 7, 8, 2], [0, 4, 2, 1, 1, 1, 1]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0]])
        self.assertEqual(chunk_spans(ids, mask, 2), [[(1, 3), (3, 5), (5, 6)], [(1, 2)]])
        for row, spans in zip(mask, chunk_spans(ids, mask, 2)):
            self.assertEqual([i for a, b in spans for i in range(a, b)], list(range(1, int(row.sum()) - 1)))

    def test_chunk_rejects_invalid_masks_and_empty_body(self):
        for ids, mask in (
            ([[0, 2]], [[1, 1]]), ([[0, 5, 2]], [[1, 0, 1]]),
            ([[0, 5, 3]], [[1, 1, 1]]), ([[0, 5, 2]], [[1, 2, 0]]),
        ):
            with self.assertRaises(ValueError):
                chunk_spans(torch.tensor(ids), torch.tensor(mask))
        with self.assertRaises(ValueError):
            chunk_spans(torch.tensor([[0, 4, 2]]), torch.ones(1, 3), 0)

    def test_ce_control_is_the_same_two_forward_average(self):
        first = torch.tensor([[1., 2., 3., 4.]], requires_grad=True)
        second = torch.tensor([[4., 3., 2., 1.]], requires_grad=True)
        labels = torch.tensor([1])
        loss, ce, kl = paired_dropout_loss(first, second, labels, 0)
        torch.testing.assert_close(loss, (F.cross_entropy(first, labels) + F.cross_entropy(second, labels)) / 2)
        self.assertGreater(kl.item(), 0)
        loss.backward()
        self.assertIsNotNone(first.grad)
        self.assertIsNotNone(second.grad)

    def test_consistency_is_symmetric_finite_and_zero_for_identical_logits(self):
        torch.manual_seed(42)
        first, second = torch.randn(4, 4, requires_grad=True), torch.randn(4, 4, requires_grad=True)
        labels = torch.arange(4)
        loss, ce, kl = paired_dropout_loss(first, second, labels, 1)
        reverse = paired_dropout_loss(second, first, labels, 1)
        torch.testing.assert_close(loss, reverse[0])
        torch.testing.assert_close(loss, ce + kl)
        self.assertGreaterEqual(kl.item(), 0)
        self.assertAlmostEqual(paired_dropout_loss(first, first, labels, 1)[2].item(), 0, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(first.grad).all())
        self.assertTrue(torch.isfinite(second.grad).all())
        with self.assertRaises(ValueError):
            paired_dropout_loss(first, second, labels, -1)

    def test_matrix_changes_only_declared_factors(self):
        configs = [extension_config(condition, 42) for condition in EXTENSION_CONDITIONS]
        for config in configs:
            self.assertEqual(config["extension"]["training_forwards_per_batch"], 2)
            self.assertEqual(config["architecture"], configs[0]["architecture"])
            self.assertEqual(config["fusion"], configs[0]["fusion"])
            self.assertEqual(config["seed"], 42)
        self.assertEqual([c["extension"]["consistency_weight"] for c in configs], [0, 0, 1, 1])
        with patch("experiments.fusion_extensions.IndexedSameWindowDataset") as factory:
            extension_dataset("cache", "val", "ext_chunks_ce2")
        factory.assert_called_once_with("cache", "val", "text")

    def test_validation_partition_is_group_disjoint_and_order_independent(self):
        records = [
            {"item_id": f"{group}:{label}", "group_id": str(group), "label_id": label}
            for group in range(12) for label in range(4)
        ]
        selection, calibration = validation_partition(records)
        selected_groups = {records[i]["group_id"] for i in selection}
        calibration_groups = {records[i]["group_id"] for i in calibration}
        self.assertFalse(selected_groups & calibration_groups)
        self.assertEqual(set(selection + calibration), set(range(len(records))))
        reversed_records = list(reversed(records))
        reversed_selection, _ = validation_partition(reversed_records)
        self.assertEqual(selected_groups, {reversed_records[i]["group_id"] for i in reversed_selection})
        with self.assertRaises(ValueError):
            validation_partition(records[:4])


if __name__ == "__main__":
    unittest.main()
