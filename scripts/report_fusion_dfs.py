"""Audit every attempted DFS run, including pruned routes; never hide negatives."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import classification_metrics, file_hash, write_json
from experiments.fusion_dfs_policy import SEEDS, continuation_gate
from scripts.report_fusion_matrix import arrays, bootstrap_contrasts, fit_calibration, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior reports")
    summary = json.loads((args.input / "summary.json").read_text())
    if summary["status"] != "completed" or summary["test_used_for_decisions"] or not summary["search_frozen"]:
        raise ValueError("DFS search is incomplete or used test")
    scores, all_metrics, evaluations, calibration, checkpoints = {}, {}, {}, {}, {}
    test_records, domain_metrics = {}, {}
    parameters = {}
    validation_partition = None
    if len(summary["runs"]) != summary["formal_training_runs"]:
        raise ValueError("Incorrect attempted-run count")
    declared_runs = {f"{r['condition']}_s{r['seed']}" for r in summary["runs"]}
    actual_runs = {path.parent.name for path in (args.input / "runs").glob("*/result.json")}
    if declared_runs != actual_runs:
        raise ValueError("An attempted completed run was omitted or fabricated")
    for row in summary["runs"]:
        condition, seed = row["condition"], row["seed"]
        name = f"{condition}_s{seed}"
        if name in all_metrics:
            raise ValueError("Duplicate training run")
        folder = args.input / "runs" / name
        result = json.loads((folder / "result.json").read_text())
        checkpoint = file_hash(folder / "best.pt")
        if checkpoint != row["checkpoint_sha256"] or checkpoint != result["best_checkpoint_sha256"]:
            raise ValueError("DFS checkpoint hash differs")
        checkpoints[name] = checkpoint
        if result["parameters"]["trainable"] != row["trainable_parameters"]:
            raise ValueError("Reported parameter count differs from training")
        parameters[condition] = row["trainable_parameters"]
        if result["test_evaluated"] or row["test_evaluated_during_search"]:
            raise ValueError("Training search inspected test")
        rows = read_jsonl(folder / "validation_predictions.jsonl")
        if row["kind"] == "extensions":
            path = folder / "validation_partitions.json"
            if file_hash(path) != result["validation_partitions_sha256"]:
                raise ValueError("Validation partition changed")
            partition = json.loads(path.read_text())
            if validation_partition is None:
                validation_partition = partition
            if partition != validation_partition:
                raise ValueError("Extension runs use different selection/calibration partitions")
            if (set(partition["selection_group_ids"]) & set(partition["calibration_group_ids"])
                    or set(partition["selection_ids"]) & set(partition["calibration_ids"])
                    or partition["calibration_used_for_checkpoint_selection"]):
                raise ValueError("Calibration leaked into checkpoint selection")
            ids = set(partition["selection_ids"])
            rows = [item for item in rows if item["item_id"] in ids]
            if {item["item_id"] for item in rows} != ids:
                raise ValueError("Missing selection predictions")
        labels, probabilities = arrays(rows)
        metrics = classification_metrics(labels, probabilities)
        expected = result["validation" if row["kind"] == "scale" else "selection_validation"]["macro_f1"]
        if (abs(metrics["macro_f1"] - row["selection_validation_macro_f1"]) > 1e-12
                or abs(metrics["macro_f1"] - expected) > 1e-12):
            raise ValueError("DFS selection metrics differ from actual predictions")
        all_metrics[name] = metrics
        scores.setdefault(condition, {})[seed] = metrics["macro_f1"]
    for decision in summary["decisions"]:
        if decision.get("branch") == "backtrack_finalist_choice":
            expected = min(decision["compared_conditions"], key=lambda c: (
                -np.mean([scores[c][s] for s in SEEDS]), parameters[c], c))
            if decision["decision"] != expected:
                raise ValueError("Backtrack finalist does not have the best confirmed validation rank")
        if "candidate" not in decision:
            continue
        pilot = decision["phase"] == "single_seed_screen"
        seeds = SEEDS[:1] if pilot else SEEDS
        actual = continuation_gate(
            {s: scores[decision["candidate"]][s] for s in seeds},
            {s: scores[decision["reference"]][s] for s in seeds}, pilot=pilot,
        )
        if actual != {name: decision[name] for name in actual}:
            raise ValueError("Recorded DFS continuation decision differs from frozen policy")
    selected = summary["selected_extension"]
    if selected != "ext_edu_ce2":
        if continuation_gate(scores[selected], scores["ext_edu_ce2"])["decision"] != "continue":
            raise ValueError("Final extension fails the root-reference confirmation gate")
    expected_evaluations = set()
    if summary.get("selected_scale"):
        expected_evaluations |= {(c, seed) for c in ("scale_adaptive", summary["selected_scale"])
                                 for seed in SEEDS}
    if summary["selected_extension"] != "ext_edu_ce2":
        expected_evaluations |= {(c, seed) for c in ("ext_edu_ce2", summary["selected_extension"])
                                 for seed in SEEDS}
    if ({(r["condition"], r["seed"]) for r in summary["evaluations"]} != expected_evaluations
            or len(summary["evaluations"]) != len(expected_evaluations)):
        raise ValueError("Missing, duplicate or unexpected frozen evaluation")
    reference_ids = None
    for row in summary["evaluations"]:
        name = f"{row['condition']}_s{row['seed']}"
        folder = args.input / "evaluation" / name
        rows = read_jsonl(folder / "test_predictions.jsonl")
        ids = [(r["item_id"], r["true_label"], r["group_id"]) for r in rows]
        if len(rows) != 3194 or (reference_ids is not None and ids != reference_ids):
            raise ValueError("Final test predictions must be aligned and complete")
        reference_ids = ids
        labels, probabilities = arrays(rows)
        actual = classification_metrics(labels, probabilities)
        recorded = json.loads((folder / "result.json").read_text())
        saved_metrics = recorded["splits"]["test"]["metrics"] if "splits" in recorded else recorded["test"]
        if (abs(actual["macro_f1"] - row["test_macro_f1"]) > 1e-12
                or actual["confusion_matrix"] != saved_metrics["confusion_matrix"]
                or recorded["checkpoint_sha256"] != row["checkpoint_sha256"]
                or row["checkpoint_sha256"] != checkpoints[name]):
            raise ValueError("Final test score/checkpoint mismatch")
        evaluations[name] = actual
        test_records.setdefault(row["condition"], {})[row["seed"]] = rows
        domain_metrics[name] = {}
        for domain in sorted({r["domain"] for r in rows}):
            mask = np.array([r["domain"] == domain for r in rows])
            domain_metrics[name][domain] = classification_metrics(labels[mask], probabilities[mask])
        if row["condition"].startswith("ext_"):
            training = args.input / "runs" / name
            partition = json.loads((training / "validation_partitions.json").read_text())
            ids = set(partition["calibration_ids"])
            calibration_rows = [r for r in read_jsonl(training / "validation_predictions.jsonl")
                                if r["item_id"] in ids]
            if {r["item_id"] for r in calibration_rows} != ids:
                raise ValueError("Missing calibration predictions")
            calibration[name] = fit_calibration(calibration_rows, rows)
            calibration[name]["history_warning"] = (
                "Parent architecture and historical checkpoint selection used full validation. "
                "Calibration IDs are excluded only from current-run checkpoint and DFS gate selection; "
                "this is not untouched end-to-end independent calibration."
            )
    contrasts = {}
    if summary.get("selected_scale"):
        contrasts["selected-scale-minus-adaptive"] = {
            summary["selected_scale"]: 1, "scale_adaptive": -1}
    if summary["selected_extension"] != "ext_edu_ce2":
        contrasts["selected-extension-minus-ce2"] = {
            summary["selected_extension"]: 1, "ext_edu_ce2": -1}
    intervals = bootstrap_contrasts(test_records, sorted(test_records), contrasts) if contrasts else {}
    for name, coefficients in contrasts.items():
        differences = [100 * sum(
            coefficient * evaluations[f"{condition}_s{seed}"]["macro_f1"]
            for condition, coefficient in coefficients.items()) for seed in SEEDS]
        intervals[name].update(per_seed_pp=differences, mean_pp=float(np.mean(differences)))
    args.output.mkdir(parents=True)
    write_json(args.output / "all_attempts.json", summary["runs"])
    write_json(args.output / "all_selection_metrics.json", all_metrics)
    write_json(args.output / "all_branch_decisions.json", summary["decisions"])
    write_json(args.output / "final_test_metrics.json", evaluations)
    write_json(args.output / "final_domain_metrics.json", domain_metrics)
    write_json(args.output / "final_paired_contrasts.json", intervals)
    if calibration:
        write_json(args.output / "calibration_and_rejection.json", calibration)
    write_json(args.output / "verification.json", {
        "status": "passed", "attempted_training_runs": len(all_metrics),
        "recomputed_branch_gates": sum("candidate" in d for d in summary["decisions"]),
        "final_evaluations": len(evaluations), "summary_sha256": file_hash(args.input / "summary.json"),
        "negative_and_pruned_results_retained": True, "test_used_for_search": False,
    })
    lines = [
        "# Validation-gated DFS experiment", "",
        f"Parent: {summary['parent']}. Final extension: {summary['selected_extension']}.",
        "Every attempted run is listed, including negative and pruned directions.", "",
        "| Condition | Seed | Selection Macro-F1 | Minutes | Trainable parameters |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["runs"]:
        lines.append(f"| {row['condition']} | {row['seed']} | "
                     f"{100 * row['selection_validation_macro_f1']:.4f}% | "
                     f"{row['elapsed_seconds'] / 60:.2f} | {row['trainable_parameters']} |")
    lines += ["", "## Decisions", ""]
    for row in summary["decisions"]:
        if "candidate" in row:
            lines.append(f"- {row['candidate']} vs {row['reference']}, {row['phase']}: "
                         f"{row['mean_validation_gain_pp']:+.4f} pp; {row['decision']}.")
        else:
            lines.append(f"- {row['branch']}: {row['decision']}; {row['reason']}.")
    lines += [
        "", "A 0.2pp gate is a compute-allocation heuristic, not statistical significance.",
        "Single-seed pruning can discard useful directions. Adaptive validation search is",
        "optimistically biased; this repeatedly inspected HART holdout is not independent confirmation.",
        "Scale runs use full validation; extension runs use a frozen source-group selection half.",
        "Do not compare those selection scores directly or claim calibrated 1% test FPR.",
        "Parent architecture selection used historical full validation, including current calibration IDs.",
        "Calibration is isolated only within current-run checkpoint/gate fitting, not the entire workflow.",
        "Final tests are descriptive only and cannot reopen the search.",
        "Group-bootstrap intervals condition on these fitted models, omit training/selection",
        "uncertainty and are not adjusted for multiple comparisons.",
    ]
    if evaluations:
        lines += ["", "## Frozen Final Tests", ""]
        for condition in sorted({r["condition"] for r in summary["evaluations"]}):
            values = [v["macro_f1"] * 100 for name, v in evaluations.items()
                      if name.rsplit("_s", 1)[0] == condition]
            lines.append(f"- {condition}: mean {np.mean(values):.4f}%, "
                         f"sample SD {np.std(values, ddof=1):.4f} pp ({len(values)} seeds).")
        for name, result in intervals.items():
            low, high = result["conditional_group_bootstrap_95ci_pp"]
            lines.append(f"- {name}: {result['mean_pp']:+.4f} pp; conditional group interval "
                         f"[{low:+.4f}, {high:+.4f}] pp.")
    else:
        lines += ["", "No extension passed confirmation; no extra test evaluations were requested."]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "verified", "runs": len(all_metrics),
                      "final_evaluations": len(evaluations)}), flush=True)


if __name__ == "__main__":
    main()
