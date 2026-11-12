"""Verify downloaded fusion predictions and produce the personal result report."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import textwrap

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import LABELS, classification_metrics, file_hash, write_json


def load(path):
    return json.loads(path.read_text())


def metrics_close(actual, expected):
    if isinstance(actual, dict):
        return (isinstance(expected, dict) and actual.keys() == expected.keys()
                and all(metrics_close(value, expected[key]) for key, value in actual.items()))
    if isinstance(actual, list):
        return (isinstance(expected, list) and len(actual) == len(expected)
                and all(metrics_close(left, right) for left, right in zip(actual, expected)))
    if isinstance(actual, (float, int)) and isinstance(expected, (float, int)):
        return abs(actual - expected) <= 1e-12
    return actual == expected


def render_validation_screen(selection, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    screen = selection["all_validation_macro_f1"]
    figure, axis = plt.subplots(figsize=(9, 4.6), layout="constrained")
    bars = axis.bar([name.removeprefix("fusion_") for name in screen],
                    [100 * value for value in screen.values()],
                    color=["#19846B" if name == selection["selected"] else "#546E7A"
                           for name in screen])
    axis.bar_label(bars, fmt="%.2f%%", padding=4)
    axis.set(ylabel="Validation Macro-F1 (%)", ylim=(0, 104),
             title="Single-seed validation screening (seed 3407), not test performance")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Downloaded edu_fusion_experiment_v1 directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous reports; choose a new output directory")
    summary = load(args.input / "summary.json")
    selection = load(args.input / "selection.json")
    if summary["status"] != "completed" or selection["test_used_for_selection"]:
        raise ValueError("A completed validation-selected experiment is required")
    baseline, candidate = selection["baseline"], selection["selected"]
    seeds = selection["seeds"]
    if (seeds != [3407, 42, 2026] or baseline != "fusion_cls"
            or candidate not in ("fusion_mean", "fusion_attention", "fusion_gated")):
        raise ValueError("Expected a distinct candidate and a three-seed paired baseline")
    if summary["selection"] != selection:
        raise ValueError("Summary differs from the frozen selection")
    metrics, records, hashes, domains, costs = {}, {}, {}, {}, []
    for condition in (baseline, candidate):
        metrics[condition], records[condition], domains[condition] = {}, {}, {}
        for seed in seeds:
            directory = args.input / f"evaluation/{condition}_s{seed}"
            saved = load(directory / "result.json")
            predictions = directory / "test_predictions.jsonl"
            rows = [json.loads(line) for line in predictions.read_text().splitlines()]
            ids = [row["item_id"] for row in rows]
            if len(rows) != 3194 or len(set(ids)) != len(ids):
                raise ValueError("Expected all 3,194 unique held-out IDs")
            probabilities = np.array([
                [row["probabilities"][label] for label in LABELS] for row in rows
            ])
            if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                    or (probabilities > 1).any()
                    or not np.allclose(probabilities.sum(1), 1, atol=1e-5, rtol=0)):
                raise ValueError("Invalid test probability distribution")
            labels = [LABELS.index(row["true_label"]) for row in rows]
            recalculated = classification_metrics(labels, probabilities)
            expected = saved["splits"]["test"]["metrics"]
            if not metrics_close(recalculated, expected):
                raise ValueError("Saved and recomputed test metrics differ")
            if not metrics_close(recalculated, summary["test"][condition][str(seed)]):
                raise ValueError("Summary and prediction metrics differ")
            metrics[condition][seed] = recalculated
            domains[condition][seed] = {}
            for domain in sorted({row["domain"] for row in rows}):
                mask = np.array([row["domain"] == domain for row in rows])
                domains[condition][seed][domain] = classification_metrics(
                    np.asarray(labels)[mask], probabilities[mask],
                )
            records[condition][seed] = [
                (row["item_id"], row["true_label"], row["group_id"], row["domain"]) for row in rows
            ]
            hashes[str(predictions.relative_to(args.input))] = file_hash(predictions)
            source = args.input / ("screen" if seed == 3407 else "confirmation")
            training = load(source / f"{condition}_s{seed}/result.json")
            costs.append({
                "condition": condition, "seed": seed,
                "elapsed_seconds": training["elapsed_seconds"],
                "peak_cuda_allocated_mib": training["peak_cuda_allocated_mib"],
                "best_epoch": training["best_epoch"],
                "trainable_parameters": training["parameters"]["trainable"],
            })
    anchor = records[baseline][seeds[0]]
    if any(rows != anchor for by_seed in records.values() for rows in by_seed.values()):
        raise ValueError("Conditions or seeds do not share the same ordered test records")
    differences = [
        100 * (metrics[candidate][seed]["macro_f1"] - metrics[baseline][seed]["macro_f1"])
        for seed in seeds
    ]
    means = {
        condition: 100 * statistics.mean(row["macro_f1"] for row in by_seed.values())
        for condition, by_seed in metrics.items()
    }
    deviations = {
        condition: 100 * statistics.stdev(row["macro_f1"] for row in by_seed.values())
        for condition, by_seed in metrics.items()
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "verification.json", {
        "status": "passed", "prediction_files": 6, "unique_test_records_per_file": 3194,
        "all_metrics_recomputed": True, "paired_records_identical": True,
        "test_predictions_sha256": hashes,
        "selection_sha256": file_hash(args.input / "selection.json"),
        "summary_sha256": file_hash(args.input / "summary.json"),
    })
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "seed", "macro_f1", "accuracy", "macro_auroc_ovr"])
        for condition, by_seed in metrics.items():
            for seed, result in by_seed.items():
                writer.writerow([condition, seed, result["macro_f1"], result["accuracy"],
                                 result["macro_auroc_ovr"]])
    with (args.output / "resource_costs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(costs[0]))
        writer.writeheader()
        writer.writerows(costs)
    write_json(args.output / "domain_metrics.json", domains)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#546E7A", "#19846B"]
    render_validation_screen(selection, args.output / "01_validation_screen.png")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")
    bars = axes[0].bar(["CLS baseline", candidate.removeprefix("fusion_")],
                       [means[baseline], means[candidate]],
                       yerr=[deviations[baseline], deviations[candidate]], capsize=5, color=colors)
    axes[0].bar_label(bars, fmt="%.2f", padding=9)
    axes[0].set(ylabel="Test Macro-F1 (%)", ylim=(0, 105),
                title="Three seeds: mean +/- sample SD")
    axes[1].axhline(0, color="#444444", linewidth=1)
    bars = axes[1].bar([str(seed) for seed in seeds], differences,
                       color=["#19846B" if value >= 0 else "#B64A4A" for value in differences])
    axes[1].bar_label(bars, fmt="%+.2f", padding=4)
    axes[1].margins(y=0.3)
    axes[1].set(ylabel="Candidate - CLS (percentage points)", title="Matched-seed differences")
    figure.savefig(args.output / "02_paired_test.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4.8), layout="constrained")
    positions = np.arange(4)
    for index, condition in enumerate((baseline, candidate)):
        values = [100 * statistics.mean(metrics[condition][seed]["per_class"][label]["f1"]
                                         for seed in seeds) for label in LABELS]
        axis.bar(positions + (index - 0.5) * 0.35, values, width=0.35,
                 label=condition, color=colors[index])
    axis.set_xticks(positions, ["Human\noriginal", "Human\nAI-polished",
                               "AI\noriginal", "AI\nhumanized"])
    axis.set(ylabel="Test class F1 (%)", ylim=(0, 104),
             title="Per-class F1 averaged over three seeds")
    axis.legend(frameon=False)
    figure.savefig(args.output / "03_class_f1.png", dpi=180)
    plt.close(figure)

    examples_path = args.input / f"evaluation/{candidate}_s3407/validation_explanations.local.json"
    if examples_path.exists():
        examples = load(examples_path)["items"]
        example = next(row for row in examples if row["true_label"] == "ai_humanized")
        selected_edus = sorted(
            enumerate(example["edus"]), key=lambda pair: -pair[1]["attention_weight"],
        )[:8]
        if selected_edus:
            labels = [f"EDU {index + 1}: " + textwrap.shorten(
                row["decoded_excerpt"].replace("\n", " "), width=85, placeholder="...")
                for index, row in selected_edus]
            figure, axis = plt.subplots(figsize=(12, 6), layout="constrained")
            axis.barh(np.arange(len(labels)),
                      [row["attention_weight"] for _, row in selected_edus], color=colors[1])
            axis.set_yticks(np.arange(len(labels)), labels, fontsize=8)
            axis.invert_yaxis()
            axis.set(xlabel="EDU pooling weight (not causal attribution)",
                     title=f"{candidate}: fixed-rule validation example; residual multiplier = "
                           f"{example['residual_gate']:.3f}")
            figure.savefig(args.output / "04_validation_attention.local.png", dpi=180)
            plt.close(figure)
            write_json(args.output / "validation_example.local.json", example)

    lines = [
        "# Text-EDU fusion: personal experiment report", "",
        "## Actual result", "",
        "| Method | Test Macro-F1 mean +/- sample SD |",
        "| --- | ---: |",
        f"| Matched CLS baseline | {means[baseline]:.4f}% +/- {deviations[baseline]:.4f} pp |",
        f"| {candidate} | {means[candidate]:.4f}% +/- {deviations[candidate]:.4f} pp |", "",
        f"Mean matched-seed change: **{statistics.mean(differences):+.4f} percentage points**.",
        f"Per-seed changes (3407/42/2026): {', '.join(f'{value:+.4f}' for value in differences)} pp.", "",
        ("The candidate improved the measured mean on this holdout; three seeds alone"
         " do not establish general superiority."
         if statistics.mean(differences) > 0 else
         "The candidate did not improve the measured mean; this is a negative result,"
         " not evidence of a stronger detector."), "",
        "## Resource cost", "",
        "| Method | Mean run minutes | Peak allocated CUDA MiB (max) | Trainable parameters |",
        "| --- | ---: | ---: | ---: |",
        *[
            f"| {condition} | "
            f"{statistics.mean(row['elapsed_seconds'] for row in costs if row['condition'] == condition) / 60:.2f} | "
            f"{max(row['peak_cuda_allocated_mib'] for row in costs if row['condition'] == condition):.1f} | "
            f"{next(row['trainable_parameters'] for row in costs if row['condition'] == condition):,} |"
            for condition in (baseline, candidate)
        ], "",
        "Run time includes training and validation, not full cloud setup or final evaluation.",
        "These six runs exclude the two nonselected screening runs; they are not total project cost.",
        "Any interrupted attempt and continuation setup are additional cost, not included in this table.",
        "Per-domain metrics are in `domain_metrics.json`; they are descriptive, not selection criteria.", "",
        "## Protocol", "",
        "Four single-seed validation screens, then two additional seeds for the matched",
        "CLS baseline and validation-selected fusion candidate: eight formal training runs.",
        "The candidate was selected before opening test. All six final prediction files",
        "were independently recomputed and matched the saved metrics and ordered IDs.",
        "The existing HART holdout had been inspected in earlier work; this is follow-up",
        "analysis, not independent new-dataset confirmation. SD is not a confidence interval.", "",
        "## Actual reuse", "",
        "Reused: the group's frozen HART split, base-pretrained RoBERTa, upstream classifier",
        "components, same-window EDU cache/loader, balanced sampling, metrics/checkpoint",
        "utilities, and private Kaggle resource package/discovery code.",
        "No teammate's task checkpoint initialized the fusion model.",
        "Exact hashes are in the downloaded per-run provenance and code manifests.", "",
        "## New work", "",
        "CLS-preserving residual EDU fusion, attention, scalar gating, matched-head variants,",
        "validation-only screening/selection, new training runs and verified results.", "",
        "## Presentation assets", "",
        "- `01_validation_screen.png`: all candidates, not only the selected model.",
        "- `02_paired_test.png`: three-seed results and paired differences.",
        "- `03_class_f1.png`: all four classes.",
        "- `04_validation_attention.local.png`: fixed-rule validation example, if available.",
        "- `metrics.csv` and `verification.json`: values and verification evidence.", "",
        "Local attention assets contain decoded text; check permissions before redistribution.",
        "Attention is model behavior, not proof of authorship or causal attribution.", "",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
