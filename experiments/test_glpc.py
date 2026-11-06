import unittest

import torch

from experiments.glpc_data import collate_glpc, hard_source_pairs, sentence_spans, visible_token_spans
from experiments.glpc_model import CREATOR_LABELS, consistency_js, pair_ranking_loss


class GLPCChecks(unittest.TestCase):
    def test_creator_mapping(self):
        self.assertEqual(CREATOR_LABELS.tolist(), [0, 0, 1, 1])

    def test_sentence_edge_cases(self):
        self.assertEqual(len(sentence_spans("One sentence")), 1)
        self.assertEqual(len(sentence_spans("One. Two!")), 2)
        with self.assertRaises(ValueError):
            sentence_spans(" \n ")

    def test_alignment_excludes_special_and_truncates(self):
        class FakeTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [0, 5, 6, 2], "attention_mask": [1, 1, 1, 1],
                        "offset_mapping": [(0, 0), (0, 3), (4, 7), (0, 0)],
                        "special_tokens_mask": [1, 0, 0, 1]}
        encoded, spans = visible_token_spans("One two", FakeTokenizer())
        self.assertEqual([x["token_indices"] for x in spans], [[1, 2]])
        self.assertEqual(encoded["input_ids"][0], 0)

    def test_consistency_is_differentiable(self):
        four = torch.randn(3, 4, requires_grad=True)
        creator = torch.randn(3, 2, requires_grad=True)
        loss = consistency_js(four, creator)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(four.grad)
        self.assertIsNotNone(creator.grad)

    def test_same_source_pairs_only_use_requested_hard_boundaries(self):
        records = [
            {"group_id": "g", "label_id": 1, "article": "a", "item_id": "g1"},
            {"group_id": "g", "label_id": 2, "article": "b", "item_id": "g2"},
            {"group_id": "g", "label_id": 3, "article": "c", "item_id": "g3"},
            {"group_id": "other", "label_id": 0, "article": "d", "item_id": "o0"},
        ]
        pairs = hard_source_pairs(records)
        self.assertEqual({pair["pair_type"] for pair in pairs},
                         {"polished_vs_generated", "humanized_vs_generated"})
        self.assertEqual({pair["group_id"] for pair in pairs}, {"g"})

    def test_pair_ranking_loss_has_gradient(self):
        negative = torch.zeros(2, 4, requires_grad=True)
        positive = torch.zeros(2, 4, requires_grad=True)
        loss = pair_ranking_loss(
            negative, positive,
            ["polished_vs_generated", "humanized_vs_generated"], margin=0.1,
        )
        loss.backward()
        self.assertGreater(float(loss), 0)
        self.assertIsNotNone(negative.grad)
        self.assertIsNotNone(positive.grad)


if __name__ == "__main__":
    unittest.main()
