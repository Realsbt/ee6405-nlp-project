"""Finish the frozen root, then dispatch only validation-gated descendants."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import classification_metrics, file_hash, write_json
from experiments.fusion_dfs_policy import SEEDS, continuation_gate
from scripts.prepare_fusion_followup import prepare
from scripts.report_fusion_matrix import arrays, read_jsonl


def root_decision(root):
    selection = json.loads((root / "selection.json").read_text())
    candidate = selection["selected"]
    reference = "fusion_cls"
    scores = {candidate: {}, reference: {}}
    runs, identities = [], {}
    for condition in (reference, candidate):
        for seed in SEEDS:
            folder = root / ("screen" if seed == 3407 else "confirmation") / f"{condition}_s{seed}"
            result = json.loads((folder / "result.json").read_text())
            if result["test_evaluated"] or result["test"] is not None:
                raise ValueError("Root training must be validation-only")
            if result["condition"] != condition or result["seed"] != seed:
                raise ValueError("Root condition/seed differs")
            rows = read_jsonl(folder / "validation_predictions.jsonl")
            labels, probabilities = arrays(rows)
            identity = [(row["item_id"], row["true_label"], row["group_id"]) for row in rows]
            if identities and identity != identities["reference"]:
                raise ValueError("Root validation rows differ")
            identities["reference"] = identity
            actual = classification_metrics(labels, probabilities)["macro_f1"]
            if abs(actual - result["validation"]["macro_f1"]) > 1e-12:
                raise ValueError("Root validation score differs from predictions")
            scores[condition][seed] = actual
            runs.append({
                "condition": condition, "seed": seed, "validation_macro_f1": actual,
                "predictions_sha256": file_hash(folder / "validation_predictions.jsonl"),
                "elapsed_seconds": result["elapsed_seconds"],
                "checkpoint_sha256": result["best_checkpoint_sha256"],
            })
    decision = continuation_gate(scores[candidate], scores[reference])
    screens = {
        condition: json.loads((root / "screen" / f"{condition}_s3407" / "result.json").read_text())
        for condition in ("fusion_cls", "fusion_mean", "fusion_attention", "fusion_gated")
    }
    return {
        "root_candidate": candidate, "reference": reference, "decision": decision,
        "parent": candidate if decision["decision"] == "continue" else reference,
        "confirmed_runs": runs,
        "screen_results": {
            name: {
                "validation_macro_f1": row["validation"]["macro_f1"],
                "elapsed_seconds": row["elapsed_seconds"],
                "trainable_parameters": row["parameters"]["trainable"],
                "disposition": "confirmed" if name in scores else "screen_only_not_expanded",
            } for name, row in screens.items()
        },
        "test_used_for_decision": False,
        "scope": "Adaptive development decision, not an independent statistical claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/fusion_dfs_supervision_v1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run = lambda *command: subprocess.run(list(map(str, command)), check=True)
    run(sys.executable, "-B", "scripts/supervise_fusion_stages.py",
        "--plan", "docs/fusion_supervision_main_only.json",
        "--output", "outputs/fusion_supervision_v1", "--interval", "120")
    prior = json.loads(Path("outputs/fusion_supervision_v1/journal.json").read_text())
    if prior["status"] == "gpu_quota_exhausted":
        write_json(args.output / "journal.json", prior)
        return
    decision = root_decision(Path("outputs/kaggle_fusion_experiment_v1/edu_fusion_experiment_v1"))
    decision_path = args.output / "root_decision.json"
    if decision_path.exists() and json.loads(decision_path.read_text()) != decision:
        raise ValueError("Preserve previous root decision; inputs changed")
    write_json(decision_path, decision)
    print(json.dumps({"event": "root_decision", **decision}), flush=True)
    package = Path("outputs/fusion_followup_package_v3")
    frozen_plan = {
        "parent": decision["parent"], "root_decision": decision["decision"],
        "root_candidate": decision["root_candidate"],
        "priority": ["gate_mechanism_if_supported", "consistency", "segmentation"],
        "pilot_minimum_validation_gain_pp": 0.2,
        "confirmation": "mean >= 0.2pp, positive in >=2/3 seeds, worst >= -0.2pp",
        "test_used_for_decision": False,
        "reason": "Follow supported contributions depth-first; prune unsupported descendants",
        "calibration_history": (
            "Parent chosen from historical full-validation results; calibration is held out "
            "only from current-run checkpoint selection and descendant gates"
        ),
    }
    if not package.exists():
        prepare(package, "dfs")
        write_json(package / "dataset/fusion_dfs_plan.json", frozen_plan)
    elif json.loads((package / "dataset/fusion_dfs_plan.json").read_text()) != frozen_plan:
        raise ValueError("Existing DFS package has a different frozen plan")
    uploaded = args.output / "dataset_upload.json"
    if not uploaded.exists():
        run("kaggle", "datasets", "create", "-p", package / "dataset", "--dir-mode", "zip")
        write_json(uploaded, {"dataset": "zhengyingqi/ee6405-fusion-followup-code-v3",
                              "plan_sha256": file_hash(package / "dataset/fusion_dfs_plan.json")})
    while True:
        status = subprocess.check_output(
            ["kaggle", "datasets", "status", "zhengyingqi/ee6405-fusion-followup-code-v3"],
            text=True,
        ).strip()
        if status.lower() == "ready":
            break
        if "error" in status.lower():
            raise RuntimeError(f"DFS dataset processing failed: {status}")
        time.sleep(120)
    plan_path = args.output / "dispatch_plan.json"
    write_json(plan_path, {"stages": [{
        "kernel": "zhengyingqi/ee6405-fusion-dfs-experiment-v1", "submit": True,
        "package": str(package / "kernel"), "prefix": "fusion_dfs_experiment_v1",
        "output": "outputs/kaggle_dfs_experiment_v1",
        "postprocess": [
            "-B", "scripts/report_fusion_dfs.py",
            "--input", "outputs/kaggle_dfs_experiment_v1/fusion_dfs_experiment_v1",
            "--output", "outputs/supervised_reports_v1/dfs",
        ],
    }]})
    run(sys.executable, "-B", "scripts/supervise_fusion_stages.py",
        "--plan", plan_path, "--output", args.output / "cloud", "--interval", "120")
    write_json(args.output / "journal.json", json.loads((args.output / "cloud/journal.json").read_text()))


if __name__ == "__main__":
    main()
