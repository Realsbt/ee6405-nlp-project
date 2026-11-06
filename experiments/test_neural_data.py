"""Checks for neural sampler balance and correct RoBERTa padding."""

import unittest
import numpy as np
import torch

from experiments.neural import BalancedBatchSampler, collate_batch


class NeuralDataChecks(unittest.TestCase):
    def test_balanced_cycles_and_reproducible_epoch(self):
        labels = np.repeat(np.arange(4), [8, 12, 16, 20])
        sampler = BalancedBatchSampler(labels, 16, 42)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        self.assertEqual(len(first), len(labels) // 16)
        for batch in first:
            self.assertEqual(np.bincount(labels[batch], minlength=4).tolist(), [4, 4, 4, 4])
        sampler.epoch = 1
        self.assertNotEqual(first, list(sampler))

    def test_roberta_pad_id_and_mask(self):
        items = [{"id": "a", "label": torch.tensor(0), "full_text_input_ids": torch.tensor([0, 8, 2]),
                  "full_text_attention_mask": torch.ones(3, dtype=torch.long)},
                 {"id": "b", "label": torch.tensor(1), "full_text_input_ids": torch.tensor([0, 2]),
                  "full_text_attention_mask": torch.ones(2, dtype=torch.long)}]
        batch = collate_batch(items)
        self.assertEqual(batch["full_text_input_ids"][1].tolist(), [0, 2, 1])
        self.assertEqual(batch["full_text_attention_mask"][1].tolist(), [1, 1, 0])
        self.assertEqual(batch["labels"].tolist(), [0, 1])
        self.assertEqual(batch["id"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
