"""Recompute all matrix predictions, group uncertainty and held-out calibration."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import LABELS, classification_metrics, file_hash, write_json


SEEDS = (3407, 42, 2026)
CONDITIONS = {
    "scale": ("scale_fixed", "scale_global", "scale_adaptive"),
    "extensions": ("ext_edu_ce2", "ext_chunks_ce2", "ext_edu_consistent", "ext_chunks_consistent"),
}
CONTRASTS = {
    "scale": {
        "global-minus-fixed": {"scale_global": 1, "scale_fixed": -1},
        "adaptive-minus-global": {"scale_adaptive": 1, "scale_global": -1},
        "adaptive-minus-fixed": {"scale_adaptive": 1, "scale_fixed": -1},
    },
    "extensions": {
        "chunks-minus-edu-ce2": {"ext_chunks_ce2": 1, "ext_edu_ce2": -1},
        "consistency-minus-ce2-edu": {"ext_edu_consistent": 1, "ext_edu_ce2": -1},
        "consistency-minus-ce2-chunks": {"ext_chunks_consistent": 1, "ext_chunks_ce2": -1},
        "interaction": {
            "ext_chunks_consistent": 1, "ext_chunks_ce2": -1,
            "ext_edu_consistent": -1, "ext_edu_ce2": 1,
        },
    },
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def arrays(rows):
    ids = [row["item_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Predictions must have nonempty unique IDs")
    probabilities = np.array([[row["probabilities"][label] for label in LABELS] for row in rows])
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Invalid probability distribution")
    labels = np.array([LABELS.index(row["true_label"]) for row in rows])
    return labels, probabilities


def macro_f1_counts(counts):
    tp = np.diagonal(counts, axis1=-2, axis2=-1)
    denominator = counts.sum(-1) + counts.sum(-2)
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float),
                     where=denominator > 0).mean(-1)


def bootstrap_contrasts(records, conditions, contrasts, repetitions=1000):
    reference = records[conditions[0]][SEEDS[0]]
    groups = sorted({row["group_id"] for row in reference})
    group_index = {name: i for i, name in enumerate(groups)}
    group_ids = np.array([group_index[row["group_id"]] for row in reference])
    counts = np.zeros((len(conditions), len(SEEDS), len(groups), 4, 4))
    for condition_index, condition in enumerate(conditions):
        for seed_index, seed in enumerate(SEEDS):
            labels, probabilities = arrays(records[condition][seed])
            np.add.at(counts[condition_index, seed_index],
                      (group_ids, labels, probabilities.argmax(1)), 1)
    values = {name: [] for name in contrasts}
    coefficients = {
        name: np.array([contrast.get(condition, 0) for condition in conditions])
        for name, contrast in contrasts.items()
    }
    rng = np.random.default_rng(20261111)
    for _ in range(repetitions):
        weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))
        sampled = np.einsum("g,csgij->csij", weights, counts, optimize=True)
        f1 = macro_f1_counts(sampled)
        for name, coefficient in coefficients.items():
            values[name].append(float((coefficient[:, None] * f1).sum(0).mean()))
    return {
        name: {
            "conditional_group_bootstrap_95ci_pp": (100 * np.quantile(samples, [0.025, 0.975])).tolist(),
            "repetitions": repetitions, "source_groups": len(groups),
            "scope": "Shared source-group resampling across fixed fitted seeds; excludes training/selection uncertainty",
        } for name, samples in values.items()
    }


def temperature_probabilities(probabilities, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    logits = np.log(np.clip(probabilities, 1e-12, 1)) / temperature
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def fit_calibration(calibration_rows, test_rows):
    labels, probabilities = arrays(calibration_rows)
    test_labels, test_probabilities = arrays(test_rows)
    if len(np.unique(labels)) != 4:
        raise ValueError("Calibration must contain all four labels")
    def objective(log_temperature):
        values = temperature_probabilities(probabilities, np.exp(log_temperature))
        return float(-np.log(np.clip(values[np.arange(len(labels)), labels], 1e-12, 1)).mean())
    fitted = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
    if not fitted.success:
        raise ValueError("Temperature fitting failed")
    temperature = float(np.exp(fitted.x))
    cal = temperature_probabilities(probabilities, temperature)
    test = temperature_probabilities(test_probabilities, temperature)
    negative_scores = np.sort(cal[labels < 2, 2:].sum(1))
    index = max(0, int(np.ceil(0.99 * len(negative_scores))) - 1)
    creator_threshold = float(np.nextafter(negative_scores[index], np.inf))
    confidence_threshold = float(np.quantile(cal.max(1), 0.2, method="higher"))

    def metrics(y, p):
        score = p[:, 2:].sum(1)
        creator = y >= 2
        detected = score >= creator_threshold
        accepted = p.max(1) >= confidence_threshold
        return {
            "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
            "creator_brier": float(np.square(score - creator).mean()),
            "human_creator_support": int((~creator).sum()),
            "ai_creator_support": int(creator.sum()),
            "creator_false_positives": int((detected & ~creator).sum()),
            "creator_true_positives": int((detected & creator).sum()),
            "actual_fpr": float(detected[~creator].mean()),
            "actual_tpr": float(detected[creator].mean()),
            "coverage": float(accepted.mean()),
            "accepted_error_rate": float((p.argmax(1)[accepted] != y[accepted]).mean()) if accepted.any() else None,
        }
    return {
        "temperature": temperature, "log_temperature_bounds": [-3, 3],
        "creator_threshold": creator_threshold,
        "creator_threshold_rule": "Next float above empirical calibration-negative 99th percentile order statistic",
        "confidence_threshold": confidence_threshold,
        "confidence_threshold_rule": "Calibration confidence 20th percentile; target about 80% coverage, ties retained",
        "calibration": metrics(labels, cal), "test_calibrated": metrics(test_labels, test),
        "test_uncalibrated_nll": float(-np.log(np.clip(
            test_probabilities[np.arange(len(test_labels)), test_labels], 1e-12, 1)).mean()),
        "test_uncalibrated_creator_brier": float(np.square(
            test_probabilities[:, 2:].sum(1) - (test_labels >= 2)).mean()),
        "scope": "Only pre-held-out calibration IDs fitted temperature and thresholds; test never fitted them",
        "warning": "Empirical 1% calibration FPR is not a test/deployment guarantee; report counts and actual FPR",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=CONDITIONS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous reports")
    summary = json.loads((args.input / "summary.json").read_text())
    conditions = CONDITIONS[args.kind]
    if summary["status"] != "completed" or summary["test_used_for_selection"]:
        raise ValueError("Need a completed predeclared matrix")
    if summary["formal_training_runs"] != len(conditions) * len(SEEDS):
        raise ValueError("Matrix run count differs from the protocol")
    expected_runs = {(condition, seed) for condition in conditions for seed in SEEDS}
    if {(row["condition"], row["seed"]) for row in summary["runs"]} != expected_runs:
        raise ValueError("Missing or unexpected matrix condition/seed")
    metrics, records, hashes, domain_metrics, costs, calibration = {}, {}, {}, {}, [], {}
    reference = None
    validation_partition = None
    for condition in conditions:
        metrics[condition], records[condition], domain_metrics[condition] = {}, {}, {}
        for seed in SEEDS:
            name = f"{condition}_s{seed}"
            evaluation = args.input / "evaluation" / name
            path = evaluation / "test_predictions.jsonl"
            rows = read_jsonl(path)
            if len(rows) != 3194:
                raise ValueError("Expected all 3,194 HART test rows")
            identity = [(r["item_id"], r["true_label"], r["group_id"], r["domain"]) for r in rows]
            if reference is None:
                reference = identity
            if identity != reference:
                raise ValueError("Conditions/seeds differ in ordered test records")
            labels, probabilities = arrays(rows)
            recalculated = classification_metrics(labels, probabilities)
            saved = json.loads((evaluation / "result.json").read_text())
            expected = saved["splits"]["test"]["metrics"] if args.kind == "scale" else saved["test"]
            if (abs(recalculated["macro_f1"] - expected["macro_f1"]) > 1e-12
                    or recalculated["confusion_matrix"] != expected["confusion_matrix"]):
                raise ValueError("Saved test metrics differ from actual predictions")
            training_dir = args.input / "runs" / name
            training = json.loads((training_dir / "result.json").read_text())
            checkpoint = training_dir / "best.pt"
            if file_hash(checkpoint) != training["best_checkpoint_sha256"]:
                raise ValueError("Downloaded checkpoint hash differs from training")
            if saved["checkpoint_sha256"] != training["best_checkpoint_sha256"]:
                raise ValueError("Evaluation used a different checkpoint")
            summary_row = next(row for row in summary["runs"] if (row["condition"], row["seed"]) == (condition, seed))
            if abs(recalculated["macro_f1"] - summary_row["test_macro_f1"]) > 1e-12:
                raise ValueError("Summary score differs from predictions")
            metrics[condition][seed], records[condition][seed] = recalculated, rows
            hashes[str(path.relative_to(args.input))] = file_hash(path)
            domain_metrics[condition][seed] = {}
            for domain in sorted({row["domain"] for row in rows}):
                mask = np.array([row["domain"] == domain for row in rows])
                domain_metrics[condition][seed][domain] = classification_metrics(labels[mask], probabilities[mask])
            costs.append({
                "condition": condition, "seed": seed, "best_epoch": training["best_epoch"],
                "elapsed_seconds": training["elapsed_seconds"],
                "trainable_parameters": training["parameters"]["trainable"],
                "peak_cuda_allocated_mib": training["peak_cuda_allocated_mib"],
            })
            if args.kind == "extensions":
                partition_path = training_dir / "validation_partitions.json"
                if file_hash(partition_path) != training["validation_partitions_sha256"]:
                    raise ValueError("Calibration partition hash differs")
                partition = json.loads(partition_path.read_text())
                if partition["calibration_used_for_checkpoint_selection"]:
                    raise ValueError("Calibration was used for checkpoint selection")
                if set(partition["selection_group_ids"]) & set(partition["calibration_group_ids"]):
                    raise ValueError("Selection/calibration source groups overlap")
                if set(partition["selection_ids"]) & set(partition["calibration_ids"]):
                    raise ValueError("Selection/calibration IDs overlap")
                if validation_partition is None:
                    validation_partition = partition
                if partition != validation_partition:
                    raise ValueError("Matrix conditions use different validation halves")
                val_rows = read_jsonl(training_dir / "validation_predictions.jsonl")
                cal_ids = set(partition["calibration_ids"])
                cal_rows = [row for row in val_rows if row["item_id"] in cal_ids]
                if {row["item_id"] for row in cal_rows} != cal_ids:
                    raise ValueError("Missing calibration predictions")
                calibration[name] = fit_calibration(cal_rows, rows)
    contrasts = bootstrap_contrasts(records, conditions, CONTRASTS[args.kind])
    for name, coefficients in CONTRASTS[args.kind].items():
        differences = [100 * sum(
            coefficient * metrics[condition][seed]["macro_f1"]
            for condition, coefficient in coefficients.items()) for seed in SEEDS]
        contrasts[name].update(per_seed_pp=differences, mean_pp=statistics.mean(differences),
                               sample_sd_pp=statistics.stdev(differences))
    args.output.mkdir(parents=True)
    write_json(args.output / "verification.json", {
        "status": "passed", "prediction_files": len(expected_runs),
        "unique_test_records_per_file": 3194, "ordered_records_identical": True,
        "checkpoints_verified": len(expected_runs), "prediction_hashes": hashes,
        "summary_sha256": file_hash(args.input / "summary.json"),
    })
    write_json(args.output / "contrasts.json", contrasts)
    write_json(args.output / "domain_metrics.json", domain_metrics)
    write_json(args.output / "all_metrics.json", metrics)
    if calibration:
        write_json(args.output / "calibration_and_rejection.json", calibration)
    with (args.output / "costs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(costs[0]))
        writer.writeheader()
        writer.writerows(costs)
    lines = ["# Matched fusion matrix", "", f"Kind: {args.kind}. All predeclared conditions retained.", "",
             "| Condition | Test Macro-F1 mean | Sample SD, pp |", "| --- | ---: | ---: |"]
    for condition in conditions:
        values = [100 * metrics[condition][seed]["macro_f1"] for seed in SEEDS]
        lines.append(f"| {condition} | {statistics.mean(values):.4f}% | {statistics.stdev(values):.4f} |")
    lines += ["", "## Paired contrasts", ""]
    for name, values in contrasts.items():
        interval = values["conditional_group_bootstrap_95ci_pp"]
        lines.append(f"- {name}: {values['mean_pp']:+.4f} pp; fixed-model group interval "
                     f"[{interval[0]:+.4f}, {interval[1]:+.4f}] pp.")
    lines += [
        "", "SD is training-seed variability, not a confidence interval. Group intervals",
        "are conditional on these fitted models, omit selection/training uncertainty,",
        "and are not corrected for multiple contrasts. This is a repeatedly inspected",
        "HART holdout, not independent external confirmation. Negative results are retained.",
        "", "Per-class/domain results, checkpoint verification and resource costs are saved separately.",
    ]
    if calibration:
        lines += ["", "Calibration/rejection used the separately frozen calibration half,",
                  "not the checkpoint-selection half or test. An empirical 1% calibration",
                  "FPR does not guarantee 1% test FPR. Actual counts, rates and coverage are reported.",
                  "The new development split differs from the earlier fusion/scale batches;",
                  "use within-matrix comparisons for method claims."]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "verified", "kind": args.kind, "runs": len(expected_runs),
                      "contrasts_mean_pp": {name: row["mean_pp"] for name, row in contrasts.items()}}), flush=True)


if __name__ == "__main__":
    main()
