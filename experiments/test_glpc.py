import unittest

import torch

from experiments.glpc_data import collate_glpc, sentence_spans, visible_token_spans
from experiments.glpc_model import CREATOR_LABELS, consistency_js


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


if __name__ == "__main__":
    unittest.main()
