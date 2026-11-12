import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.common import LABELS
from experiments.train_stacking import aligned_probabilities, model_for, stack_features


class StackingTests(unittest.TestCase):
    def test_features_match_historical_centered_log_recipe(self):
        p = np.array([[.1, .2, .3, .4], [0, .5, .5, 0]])
        x = stack_features([p, p])
        z = np.log(np.clip(p, 1e-12, 1))
        np.testing.assert_allclose(x[:, :4], z - z.mean(1, keepdims=True))
        np.testing.assert_allclose(x[:, 4:8], p)
        self.assertEqual(x.shape, (2, 16))
        self.assertTrue(np.isfinite(x).all())

    def test_alignment_reorders_and_rejects_changed_truth(self):
        records = [{"item_id": name, "label": LABELS[0], "group_id": name}
                   for name in ("a", "b")]
        predictions = [{
            "item_id": row["item_id"], "true_label": row["label"],
            "group_id": row["group_id"], "probabilities": dict(zip(LABELS, [.1, .2, .3, .4])),
        } for row in reversed(records)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in predictions))
            self.assertEqual(aligned_probabilities(records, path).shape, (2, 4))
            predictions[0]["true_label"] = LABELS[1]
            path.write_text("\n".join(json.dumps(row) for row in predictions))
            with self.assertRaises(ValueError):
                aligned_probabilities(records, path)

    def test_selected_meta_model_is_cpu_logistic(self):
        model = model_for({"kind": "logistic", "C": 1})
        self.assertEqual(model[-1].C, 1)
        self.assertEqual(model[-1].random_state, 3407)

    def test_historical_unlabeled_predictions_use_frozen_metadata(self):
        records = [{"item_id": "a", "label": LABELS[0], "group_id": "a"}]
        row = {"item_id": "a", "probabilities": dict(zip(LABELS, [.1, .2, .3, .4]))}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(json.dumps(row))
            self.assertEqual(aligned_probabilities(records, path).shape, (1, 4))
            path.write_text(json.dumps(row) + "\n" + json.dumps(row))
            with self.assertRaises(ValueError):
                aligned_probabilities(records, path)


if __name__ == "__main__":
    unittest.main()
