import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.common import LABELS, classification_metrics, file_hash, write_json, write_jsonl
from experiments.fusion_dfs_policy import SEEDS, continuation_gate
from scripts.report_fusion_dfs import main as report
from scripts.report_fusion_matrix import arrays
from scripts.supervise_fusion_dfs import root_decision


def predictions(count, prefix, imperfect=False):
    rows = []
    for i in range(count):
        label = i % 4
        predicted = (label + 1) % 4 if imperfect and i % 8 == 0 else label
        probabilities = np.full(4, 0.1)
        probabilities[predicted] = 0.7
        rows.append({
            "item_id": f"{prefix}-{i}", "group_id": f"{prefix}-group-{i // 4}",
            "true_label": LABELS[label], "domain": "essay",
            "probabilities": dict(zip(LABELS, probabilities)),
        })
    return rows


def metrics(rows):
    return classification_metrics(*arrays(rows))


class FusionDFSTests(unittest.TestCase):
    def make_search(self, root, positive):
        reference, candidate = "ext_edu_ce2", "ext_edu_consistent"
        runs, scores, evaluations = [], {reference: {}, candidate: {}}, []
        seeds = SEEDS if positive else SEEDS[:1]
        for condition in (reference, candidate):
            for seed in seeds:
                name = f"{condition}_s{seed}"
                folder = root / "runs" / name
                folder.mkdir(parents=True)
                checkpoint = folder / "best.pt"
                checkpoint.write_bytes(f"synthetic-test-only-{name}".encode())
                digest = file_hash(checkpoint)
                rows = predictions(16, "val", imperfect=positive and condition == reference)
                selection = rows[:8]
                score = metrics(selection)["macro_f1"]
                scores[condition][seed] = score
                partition = {
                    "selection_ids": [r["item_id"] for r in selection],
                    "calibration_ids": [r["item_id"] for r in rows[8:]],
                    "selection_group_ids": ["val-group-0", "val-group-1"],
                    "calibration_group_ids": ["val-group-2", "val-group-3"],
                    "calibration_used_for_checkpoint_selection": False,
                }
                write_json(folder / "validation_partitions.json", partition)
                write_jsonl(folder / "validation_predictions.jsonl", rows)
                write_json(folder / "result.json", {
                    "best_checkpoint_sha256": digest, "test_evaluated": False,
                    "parameters": {"trainable": 100},
                    "validation": metrics(rows), "selection_validation": metrics(selection),
                    "validation_partitions_sha256": file_hash(folder / "validation_partitions.json"),
                })
                runs.append({
                    "kind": "extensions", "condition": condition, "seed": seed,
                    "checkpoint_sha256": digest, "test_evaluated_during_search": False,
                    "selection_validation_macro_f1": score, "elapsed_seconds": 10,
                    "trainable_parameters": 100, "peak_cuda_allocated_mib": 200,
                })
                if positive:
                    target = root / "evaluation" / name
                    test = predictions(3194, "test", imperfect=condition == reference)
                    actual = metrics(test)
                    write_jsonl(target / "test_predictions.jsonl", test)
                    write_json(target / "result.json", {
                        "checkpoint_sha256": digest, "test": actual,
                    })
                    evaluations.append({
                        "condition": condition, "seed": seed,
                        "test_macro_f1": actual["macro_f1"], "checkpoint_sha256": digest,
                    })
        decisions = []
        for pilot in ((True, False) if positive else (True,)):
            used = SEEDS[:1] if pilot else SEEDS
            decision = continuation_gate(
                {s: scores[candidate][s] for s in used},
                {s: scores[reference][s] for s in used}, pilot=pilot,
            )
            decisions.append({**decision, "candidate": candidate, "reference": reference})
        if positive:
            decisions.append({
                "branch": "backtrack_finalist_choice", "decision": candidate,
                "reason": "Synthetic rank-audit fixture", "compared_conditions": [reference, candidate],
            })
        write_json(root / "summary.json", {
            "status": "completed", "parent": "fusion_cls", "search_frozen": True,
            "test_used_for_decisions": False, "runs": runs, "decisions": decisions,
            "formal_training_runs": len(runs), "evaluations": evaluations,
            "selected_extension": candidate if positive else reference, "selected_scale": None,
        })

    def test_negative_pilot_and_positive_confirmation_both_report(self):
        for positive in (False, True):
            with self.subTest(positive=positive), tempfile.TemporaryDirectory() as directory:
                root, output = Path(directory) / "inputs", Path(directory) / "report"
                self.make_search(root, positive)
                with patch("sys.argv", ["report", "--input", str(root), "--output", str(output)]):
                    with contextlib.redirect_stdout(io.StringIO()):
                        report()
                actual = json.loads((output / "verification.json").read_text())
                self.assertEqual(actual["attempted_training_runs"], 6 if positive else 2)
                self.assertEqual(actual["final_evaluations"], 6 if positive else 0)
                self.assertTrue(actual["negative_and_pruned_results_retained"])

    def test_report_rejects_omitted_negative_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory) / "inputs", Path(directory) / "report"
            self.make_search(root, False)
            summary = json.loads((root / "summary.json").read_text())
            summary["runs"].pop()
            summary["formal_training_runs"] -= 1
            write_json(root / "summary.json", summary)
            with patch("sys.argv", ["report", "--input", str(root), "--output", str(output)]):
                with self.assertRaisesRegex(ValueError, "omitted"):
                    report()
            self.assertFalse(output.exists())

    def test_report_rejects_weaker_backtrack_finalist(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory) / "inputs", Path(directory) / "report"
            self.make_search(root, True)
            summary = json.loads((root / "summary.json").read_text())
            summary["decisions"][-1]["decision"] = "ext_edu_ce2"
            write_json(root / "summary.json", summary)
            with patch("sys.argv", ["report", "--input", str(root), "--output", str(output)]):
                with self.assertRaisesRegex(ValueError, "Backtrack finalist"):
                    report()
            self.assertFalse(output.exists())

    def test_root_parent_uses_only_matched_validation_scores(self):
        for positive in (False, True):
            with self.subTest(positive=positive), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_json(root / "selection.json", {"selected": "fusion_gated"})
                for condition in ("fusion_cls", "fusion_mean", "fusion_attention", "fusion_gated"):
                    for seed in (SEEDS if condition in ("fusion_cls", "fusion_gated") else SEEDS[:1]):
                        folder = root / ("screen" if seed == 3407 else "confirmation") / f"{condition}_s{seed}"
                        rows = predictions(16, "val", imperfect=positive and condition == "fusion_cls")
                        write_jsonl(folder / "validation_predictions.jsonl", rows)
                        write_json(folder / "result.json", {
                            "condition": condition, "seed": seed, "test": None,
                            "test_evaluated": False, "validation": metrics(rows),
                            "elapsed_seconds": 10, "best_checkpoint_sha256": "test-only",
                            "parameters": {"trainable": 100},
                        })
                actual = root_decision(root)
                self.assertEqual(actual["parent"], "fusion_gated" if positive else "fusion_cls")
                self.assertFalse(actual["test_used_for_decision"])


if __name__ == "__main__":
    unittest.main()
