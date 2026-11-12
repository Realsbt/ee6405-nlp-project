import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.report_fusion_matrix import (
    CONDITIONS, SEEDS, arrays, fit_calibration, macro_f1_counts, main,
    temperature_probabilities,
)
from experiments.common import LABELS, classification_metrics, file_hash, write_json, write_jsonl


class FusionMatrixReportingTests(unittest.TestCase):
    def test_confusion_f1_matches_known_values(self):
        perfect = np.eye(4)
        self.assertEqual(macro_f1_counts(perfect), 1)
        self.assertEqual(macro_f1_counts(np.zeros((4, 4))), 0)
        np.testing.assert_allclose(macro_f1_counts(np.stack([perfect, perfect])), [1, 1])

    def test_temperature_preserves_argmax_and_normalization(self):
        values = np.array([[0.1, 0.2, 0.3, 0.4], [0.8, 0.1, 0.05, 0.05]])
        for temperature in (0.2, 1, 3):
            actual = temperature_probabilities(values, temperature)
            np.testing.assert_allclose(actual.sum(1), 1)
            np.testing.assert_array_equal(actual.argmax(1), values.argmax(1))
        with self.assertRaises(ValueError):
            temperature_probabilities(values, 0)

    def test_invalid_predictions_fail(self):
        row = {"item_id": "a", "true_label": LABELS[0],
               "probabilities": dict(zip(LABELS, [1, 0, 0, 0]))}
        with self.assertRaises(ValueError):
            arrays([row, row])
        with self.assertRaises(ValueError):
            arrays([{**row, "probabilities": dict(zip(LABELS, [1, 1, 0, 0]))}])

    def test_temperature_and_thresholds_do_not_use_test_labels(self):
        calibration = []
        for index in range(80):
            label = index % 4
            values = np.full(4, 0.1)
            values[label] = 0.7
            calibration.append({
                "item_id": str(index), "true_label": LABELS[label],
                "probabilities": dict(zip(LABELS, values)),
            })
        test = [{**row, "item_id": "t" + row["item_id"]} for row in calibration]
        changed_test = [{**row, "true_label": LABELS[(i + 1) % 4]} for i, row in enumerate(test)]
        first, second = fit_calibration(calibration, test), fit_calibration(calibration, changed_test)
        for field in ("temperature", "creator_threshold", "confidence_threshold"):
            self.assertEqual(first[field], second[field])
        self.assertLessEqual(first["calibration"]["actual_fpr"], 0.01)

    def make_matrix(self, root, kind):
        rows = []
        for index in range(3194):
            label = index % 4
            probabilities = np.full(4, 0.1)
            probabilities[label] = 0.7
            rows.append({
                "item_id": f"test-{index}", "true_label": LABELS[label],
                "group_id": f"test-group-{index // 4}", "domain": "essay",
                "probabilities": dict(zip(LABELS, probabilities)),
            })
        labels, probabilities = arrays(rows)
        metrics = classification_metrics(labels, probabilities)
        validation = [{**row, "item_id": f"val-{i}", "group_id": f"val-group-{i // 4}"}
                      for i, row in enumerate(rows[:16])]
        partition = {
            "selection_ids": [row["item_id"] for row in validation[:8]],
            "calibration_ids": [row["item_id"] for row in validation[8:]],
            "selection_group_ids": ["val-group-0", "val-group-1"],
            "calibration_group_ids": ["val-group-2", "val-group-3"],
            "calibration_used_for_checkpoint_selection": False,
        }
        runs = []
        for condition in CONDITIONS[kind]:
            for seed in SEEDS:
                name = f"{condition}_s{seed}"
                training = root / "runs" / name
                evaluation = root / "evaluation" / name
                training.mkdir(parents=True)
                checkpoint = training / "best.pt"
                checkpoint.write_bytes(b"synthetic-report-test-only")
                digest = file_hash(checkpoint)
                result = {
                    "best_checkpoint_sha256": digest, "best_epoch": 2,
                    "elapsed_seconds": 10, "parameters": {"trainable": 123},
                    "peak_cuda_allocated_mib": 100,
                }
                if kind == "extensions":
                    write_json(training / "validation_partitions.json", partition)
                    result["validation_partitions_sha256"] = file_hash(
                        training / "validation_partitions.json")
                    write_jsonl(training / "validation_predictions.jsonl", validation)
                write_json(training / "result.json", result)
                write_json(evaluation / "result.json", {
                    "checkpoint_sha256": digest,
                    **({"splits": {"test": {"metrics": metrics}}}
                       if kind == "scale" else {"test": metrics}),
                })
                write_jsonl(evaluation / "test_predictions.jsonl", rows)
                runs.append({"condition": condition, "seed": seed, "test_macro_f1": 1.0})
        write_json(root / "summary.json", {
            "status": "completed", "test_used_for_selection": False,
            "formal_training_runs": len(runs), "runs": runs,
        })

    def test_complete_matrices_report_end_to_end(self):
        for kind in CONDITIONS:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "inputs"
                output = Path(directory) / "report"
                self.make_matrix(root, kind)
                with patch("sys.argv", ["report", "--kind", kind, "--input", str(root),
                                        "--output", str(output)]), contextlib.redirect_stdout(io.StringIO()):
                    main()
                verification = json.loads((output / "verification.json").read_text())
                self.assertEqual(verification["status"], "passed")
                self.assertEqual(verification["checkpoints_verified"], len(CONDITIONS[kind]) * 3)
                contrasts = json.loads((output / "contrasts.json").read_text())
                for contrast in contrasts.values():
                    self.assertEqual(contrast["mean_pp"], 0)
                self.assertEqual((output / "calibration_and_rejection.json").exists(),
                                 kind == "extensions")

    def test_corrupted_checkpoint_rejects_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "inputs"
            output = Path(directory) / "report"
            self.make_matrix(root, "scale")
            (root / "runs" / "scale_fixed_s3407" / "best.pt").write_bytes(b"corrupted")
            with patch("sys.argv", ["report", "--kind", "scale", "--input", str(root),
                                    "--output", str(output)]):
                with self.assertRaisesRegex(ValueError, "checkpoint hash"):
                    main()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
