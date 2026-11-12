"""Synthetic fixtures only: never treat these values as experimental results."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import LABELS, classification_metrics
from scripts.collect_fusion_run import output_path
from scripts.report_fusion_results import main


class CollectorPathTests(unittest.TestCase):
    def test_only_own_evidence_is_downloaded(self):
        root = Path("/tmp/fixture")
        self.assertIsNone(output_path("other_job/result.json", root, True))
        self.assertIsNone(output_path("edu_fusion_experiment_v1/smoke/best.pt", root, True))
        self.assertIsNone(output_path("edu_fusion_experiment_v1/screen/best.pt", root))
        self.assertEqual(
            output_path("edu_fusion_experiment_v1/screen/best.pt", root, True),
            root / "edu_fusion_experiment_v1/screen/best.pt",
        )

    def test_unsafe_paths_are_rejected(self):
        for name in ("/tmp/result.json", "edu_fusion_experiment_v1/../result.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                output_path(name, Path("/tmp/fixture"))


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-fusion-report-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "input"
        self.source.mkdir()
        self.destination = self.root / "report"
        seeds = [3407, 42, 2026]
        labels = np.arange(3194) % 4
        probabilities = np.full((3194, 4), 0.1)
        probabilities[np.arange(3194), labels] = 0.7
        metrics = classification_metrics(labels, probabilities)
        self.write(self.source / "selection.json", {
            "baseline": "fusion_cls", "selected": "fusion_gated", "seeds": seeds,
            "test_used_for_selection": False,
            "all_validation_macro_f1": {
                "fusion_cls": 1, "fusion_mean": 1, "fusion_attention": 1, "fusion_gated": 1,
            },
        })
        summary = {"status": "completed", "test": {},
                   "selection": json.loads((self.source / "selection.json").read_text())}
        for condition in ("fusion_cls", "fusion_gated"):
            summary["test"][condition] = {}
            for seed in seeds:
                directory = self.source / f"evaluation/{condition}_s{seed}"
                self.write(directory / "result.json", {"splits": {"test": {"metrics": metrics}}})
                rows = [{
                    "item_id": str(index), "true_label": LABELS[int(label)],
                    "group_id": str(index // 4), "domain": "fixture",
                    "probabilities": dict(zip(LABELS, probability.tolist())),
                } for index, (label, probability) in enumerate(zip(labels, probabilities))]
                (directory / "test_predictions.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
                source = self.source / ("screen" if seed == 3407 else "confirmation")
                self.write(source / f"{condition}_s{seed}/result.json", {
                    "elapsed_seconds": 60, "peak_cuda_allocated_mib": 1024,
                    "best_epoch": 1, "parameters": {"trainable": 100},
                })
                summary["test"][condition][str(seed)] = metrics
        self.write(self.source / "summary.json", summary)
        self.write(
            self.source / "evaluation/fusion_gated_s3407/validation_explanations.local.json",
            {"items": [{
                "true_label": "ai_humanized", "residual_gate": 0.2,
                "edus": [{"decoded_excerpt": "Synthetic example, not real data.",
                          "attention_weight": 1.0}],
            }]},
        )

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def run_report(self):
        with patch.object(sys, "argv", [
            "report", "--input", str(self.source), "--output", str(self.destination),
        ]):
            main()

    def test_complete_report_and_zero_gain_disclosure(self):
        self.run_report()
        verification = json.loads((self.destination / "verification.json").read_text())
        self.assertEqual(verification["status"], "passed")
        self.assertEqual(len(verification["test_predictions_sha256"]), 6)
        self.assertIn("did not improve", (self.destination / "REPORT.md").read_text())
        for name in ("01_validation_screen.png", "02_paired_test.png", "03_class_f1.png",
                     "04_validation_attention.local.png", "resource_costs.csv", "domain_metrics.json"):
            self.assertGreater((self.destination / name).stat().st_size, 0)

    def test_modified_prediction_fails_before_report_is_created(self):
        path = self.source / "evaluation/fusion_gated_s2026/test_predictions.jsonl"
        rows = path.read_text().splitlines()
        row = json.loads(rows[0])
        row["probabilities"] = dict(zip(LABELS, [0.1, 0.7, 0.1, 0.1]))
        rows[0] = json.dumps(row)
        path.write_text("\n".join(rows) + "\n")
        with self.assertRaisesRegex(ValueError, "metrics differ"):
            self.run_report()
        self.assertFalse(self.destination.exists())

    def test_selection_using_test_is_rejected(self):
        path = self.source / "selection.json"
        selection = json.loads(path.read_text())
        selection["test_used_for_selection"] = True
        self.write(path, selection)
        with self.assertRaisesRegex(ValueError, "validation-selected"):
            self.run_report()


if __name__ == "__main__":
    unittest.main()
