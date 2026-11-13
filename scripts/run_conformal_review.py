"""Reproduce a retrospective, source-group-isolated conformal pilot; no GPU."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.group_conformal import load_rows, run, split_groups, write_json, write_text

RUNS = ["tfidf_lr_v1"] + [f"{model}_s{seed}" for model in (
    "roberta", "roberta_supcon", "race_no_supcon", "race") for seed in (3407, 42, 2026)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    reference = None
    manifest = None
    for name in RUNS:
        path = ROOT / "experiments" / "reports" / name / "test_predictions.jsonl"
        rows = load_rows(path)
        identities = {r["item_id"]: tuple(r[k] for k in ("group_id", "domain", "true_label")) for r in rows}
        if reference is None:
            reference = identities
        elif identities != reference:
            raise ValueError("Models do not share the same items/labels/groups")
        calibration, evaluation = split_groups(rows)
        current_manifest = {"calibration": sorted({r["group_id"] for r in calibration}),
                            "evaluation": sorted({r["group_id"] for r in evaluation})}
        if manifest is None:
            manifest = current_manifest
            write_json(args.output / "split_manifest.json", manifest)
        elif current_manifest != manifest:
            raise ValueError("Partition mismatch")
        artifact, metrics, predictions = run(calibration, evaluation, alpha=0.1)
        artifact.update(model_run=name, source_prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        retrospective=True, split_seed="6405-conformal-v1")
        folder = args.output / name
        folder.mkdir()
        write_json(folder / "calibrator.json", artifact)
        write_json(folder / "metrics.json", metrics)
        write_text(folder / "prediction_sets.jsonl", "".join(json.dumps(p) + "\n" for p in predictions))
        summaries.append({"model": name, **metrics["overall"]})
    write_json(args.output / "summary.json", {"protocol": "retrospective_group_max_split_conformal",
               "alpha": 0.1, "model_selection": "none; all 13 published runs included",
               "caution": "Previously inspected test predictions; no independent confirmation or deployment guarantee.",
               "calibration_groups": len(manifest["calibration"]), "evaluation_groups": len(manifest["evaluation"]),
               "runs": summaries})
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    lines = ["# Group conformal review: retrospective pilot", "",
             "Fixed alpha=0.10; all 13 published runs; deterministic SHA-256 group split.",
             f"Calibration groups: {len(manifest['calibration'])}; evaluation groups: {len(manifest['evaluation'])}.", "",
             "Coverage below refers to label-set coverage, not classification accuracy.",
             "Singleton accuracy is conditional on returning one label; review cases remain unresolved.",
             "These are previously inspected test predictions. Do not replace the original benchmark table.", "",
             "| Model | Top-1 accuracy | Set coverage | All labels covered within group | Mean set size | Singleton rate | Singleton accuracy | Review rate |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        rate = lambda key: f"{r[key]:.2%}" if r[key] is not None else "N/A"
        lines.append(f"| {r['model']} | {rate('top1_accuracy')} | {rate('set_coverage')} | {rate('group_all_covered')} | {r['average_set_size']:.3f} | {rate('singleton_rate')} | {rate('singleton_accuracy')} | {rate('review_rate')} |")
    lines += ["", "## Evidence boundaries", "",
              "The quantile uses one maximum nonconformity score per source group. Under exchangeable groups and a fixed model, the target is simultaneous coverage for a fresh group. The retrospective pilot cannot establish these assumptions or new-test validity.",
              "Per-domain and per-class metrics are descriptive; there is no conditional coverage guarantee. Singleton accuracy has no 90% guarantee. HC3 is not evaluated: it has a different label space and domain distribution.",
              "No model, alpha, partition or seed was selected using these results. The procedure does not change original argmax labels, use test groups to constrain output composition, or retrain checkpoints.",
              "For prospective use, reserve source-disjoint calibration data before inspecting a fresh evaluation set; freeze model and alpha first. Review flags identify ambiguity under this score rule, not verified authorship.", ""]
    write_text(args.output / "REPORT.md", "\n".join(lines))
    print("Completed", len(summaries), "runs:", args.output.resolve())


if __name__ == "__main__":
    main()
