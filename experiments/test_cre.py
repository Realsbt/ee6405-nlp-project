"""Focused CRE unit and integration checks."""

import json
from pathlib import Path
import unittest

import torch

from experiments.common import LABELS
from experiments.cre_data import CREDataset, collate_cre, encode_record, pair_inventory
from experiments.cre_losses import combine_condition_scores, condition_scores_from_logits, pair_ranking_loss


class CREChecks(unittest.TestCase):
    def test_label_order(self):
        self.assertEqual(LABELS, ["human_written", "human_ai_polished", "ai_generated", "ai_humanized"])

    def test_zero_delta_matches_softmax_random_and_extreme(self):
        for logits in (torch.randn(7, 4, dtype=torch.float64), torch.tensor([[1000., -1000., 500., -500.]], dtype=torch.float64)):
            s_c, s_h, s_a = condition_scores_from_logits(logits)
            logp = combine_condition_scores(s_c, s_h, s_a)
            self.assertTrue(torch.allclose(logp.exp(), logits.softmax(1), atol=1e-6, rtol=1e-6))
            self.assertTrue(torch.allclose(logp.logsumexp(1), torch.zeros(logp.size(0), dtype=logp.dtype), atol=1e-10))

    def test_condition_probability_marginal(self):
        logits = torch.randn(10, 4)
        s_c, s_h, s_a = condition_scores_from_logits(logits)
        probabilities = combine_condition_scores(s_c, s_h, s_a).exp()
        self.assertTrue(torch.allclose(probabilities[:, 2:].sum(1), s_c.sigmoid(), atol=1e-6))

    def test_pair_loss_direction_and_empty(self):
        good_logp = torch.log_softmax(torch.tensor([[3., 0., 0., 0.], [0., 3., 0., 0.]]), dim=1)
        bad_logp = torch.log_softmax(torch.tensor([[0., 3., 0., 0.], [3., 0., 0., 0.]]), dim=1)
        good = pair_ranking_loss(good_logp, [(0, 1)], ["H-P"])
        bad = pair_ranking_loss(bad_logp, [(0, 1)], ["H-P"])
        self.assertLess(float(good), float(bad))
        self.assertEqual(float(pair_ranking_loss(good_logp, [], [])), 0.0)

    def test_pairs_use_group_and_three_boundaries(self):
        records = [{"group_id": "g", "item_id": f"{i}", "label_id": i, "article": "x", "domain": "d"} for i in range(4)]
        pairs = pair_inventory(records)
        self.assertEqual([p["pair_type"] for p in pairs], ["H-P", "P-G", "G-U"])
        self.assertTrue(all(p["group_id"] == "g" for p in pairs))

    def test_no_special_tokens_or_padding_in_alignment(self):
        class FakeTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [0, 5, 6, 2, 1], "attention_mask": [1, 1, 1, 1, 0],
                        "offset_mapping": [(0, 0), (0, 3), (4, 7), (0, 0), (0, 0)],
                        "special_tokens_mask": [1, 0, 0, 1, 1]}
        record = {"article": "One two", "item_id": "x", "group_id": "g", "domain": "d", "label_id": 0}
        item = encode_record(record, FakeTokenizer())
        self.assertEqual(item["visible_token_indices"], [1, 2])
        self.assertEqual([x["token_indices"] for x in item["sentence_tokens"]], [[1, 2]])

    def test_pair_loss_does_not_use_g_u_as_positive_same_class(self):
        records = [{"group_id": "g", "item_id": str(i), "label_id": i, "article": "x", "domain": "d"} for i in range(4)]
        self.assertEqual(pair_inventory(records)[2]["pair_type"], "G-U")


if __name__ == "__main__":
    unittest.main()
