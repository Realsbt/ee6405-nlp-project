"""Targeted checks for source/text leakage, ambiguous labels and missing classes."""

import unittest

from experiments.common import classification_metrics
from experiments.prepare_data import clean_and_group, text_hash


def row(item_id, group, text, label="human_written", has_rst=True):
    return {"item_id": item_id, "group_id": group, "label": label, "has_rst": has_rst,
            "normalized_text_sha256": text_hash(text)}


class ProtocolChecks(unittest.TestCase):
    def test_duplicate_link_preserves_other_variants_as_one_block(self):
        rows = [row("a", "news-a", "Same text"), row("b", "news-b", "same   text"),
                row("c", "news-b", "A different variant", "ai_generated")]
        retained, excluded = clean_and_group(rows)
        self.assertEqual([r["item_id"] for r in retained], ["a", "c"])
        self.assertEqual(len({r["split_block_id"] for r in retained}), 1)
        self.assertEqual(excluded[0]["reason"], "same_label_normalized_duplicate")

    def test_conflicting_labels_are_all_excluded(self):
        retained, excluded = clean_and_group([row("a", "news-a", "same"),
                                             row("b", "news-a", "same", "human_ai_polished")])
        self.assertEqual(retained, [])
        self.assertEqual(len(excluded), 2)
        self.assertTrue(all(r["reason"] == "conflicting_normalized_text_labels" for r in excluded))

    def test_normalization_handles_unicode_and_whitespace(self):
        self.assertEqual(text_hash("ＡＢＣ\n  Text"), text_hash("abc text"))

    def test_missing_class_is_undefined_not_zero_auc(self):
        metrics = classification_metrics([0, 1, 2], [[.8, .1, .05, .05], [.1, .8, .05, .05], [.05, .05, .8, .1]])
        self.assertIsNone(metrics["macro_auroc_ovr"])
        self.assertIsNone(metrics["per_class"]["ai_humanized"]["auroc_ovr"])
        self.assertEqual(metrics["per_class"]["ai_humanized"]["support"], 0)


if __name__ == "__main__":
    unittest.main()
