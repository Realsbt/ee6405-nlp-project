"""Complete frozen RACE/ensemble transfer evaluation; no model fitting or tuning."""

import json
import platform
import sys
import time

import joblib
import numpy as np
import torch

sys.dont_write_bytecode = True

from improve_cpu import (
    LABELS, ROOT, bootstrap_deltas, digest, probability_array, rows,
    save_predictions, write,
)
from evaluation_metrics import negative_only_threshold
from race_cpu_replay import build_model, infer, temperature_probabilities
from run_transfer_evaluation import summarize
from stack_cpu import stack_features

OUT = ROOT / "external_completed_cpu_v1"
MIX = ROOT / "resources/mixset_conservative_transfer_v1"


def check_parsed(original, parsed):
    """Require exact source preservation and full ordered nonspace coverage."""
    from experiments.rst_worker import leaves

    expected = {r["item_id"]: r for r in original}
    if len(expected) != len(original) or len(parsed) != len(original):
        raise ValueError("Duplicate IDs or wrong parsed count")
    seen = set()
    edu_counts = []
    for row in parsed:
        item = row["item_id"]
        if item in seen or item not in expected:
            raise ValueError("Duplicate or unexpected parsed ID")
        seen.add(item)
        if any(row.get(k) != v for k, v in expected[item].items()):
            raise ValueError("Parser changed source metadata")
        text = row["article"]
        units = leaves(row["rst_structure"])
        if not units:
            raise ValueError("No EDU leaves")
        end = 0
        for unit in units:
            start, stop = unit["start"], unit["end"]
            if not end <= start < stop <= len(text):
                raise ValueError("Unordered or invalid EDU span")
            if unit["text"] != text[start:stop]:
                raise ValueError("EDU text mismatch")
            if any(not c.isspace() for c in text[end:start]):
                raise ValueError("Uncovered nonspace characters")
            end = stop
        if any(not c.isspace() for c in text[end:]):
            raise ValueError("Uncovered trailing nonspace characters")
        edu_counts.append(len(units))
    if seen != set(expected):
        raise ValueError("Missing parsed IDs")
    return {"status": "passed", "records": len(parsed),
            "uncovered_nonspace_characters": 0,
            "edu_count_min": min(edu_counts), "edu_count_max": max(edu_counts)}


def evaluate_frozen(original, p_race, p_race_cal, output):
    """Apply only parameters/meta-models frozen in preceding CPU experiments."""
    p_ce = probability_array(original, ROOT / "transfer_roberta_v1/mixset_predictions.jsonl")
    p_tf = probability_array(original, ROOT / "transfer_tfidf_v1/mixset_predictions.jsonl")
    selection = json.loads((ROOT / "cpu_improvement_v1/selection.json").read_text())
    calibration = json.loads((ROOT / "race_cpu_v1/calibration.json").read_text())
    stack_selection = json.loads((ROOT / "cpu_stacking_v1/selection.json").read_text())
    val = rows(ROOT / "resources/experiments/hart_grouped_v1/val.jsonl")
    val_negative = np.array([r["label_id"] < 2 for r in val])
    alpha = selection["selected"]["race_blend"]["alpha"]
    p_stack = joblib.load(ROOT / "cpu_stacking_v1/meta_model.joblib").predict_proba(
        stack_features([p_ce, p_race, p_tf]))
    p_ce_race = joblib.load(ROOT / "cpu_stacking_controls_v1/ce_race.joblib").predict_proba(
        stack_features([p_ce, p_race]))
    oof_ce_race = probability_array(
        val, ROOT / "cpu_stacking_controls_v1/ce_race_oof_predictions.jsonl")
    predictions = {
        "roberta_ce_s3407": p_ce, "tfidf_lr": p_tf,
        "race_s3407": p_race, "race_temperature": p_race_cal,
        "race_blend": (1 - alpha) * p_ce + alpha * p_race,
        "stack_all_original": p_stack, "stack_ce_race": p_ce_race,
    }
    thresholds = {
        "roberta_ce_s3407": selection["thresholds"]["roberta_ce_s3407"]["threshold"],
        "tfidf_lr": json.loads((ROOT / "transfer_tfidf_v1/threshold.json").read_text())["threshold"],
        "race_s3407": calibration["uncalibrated_threshold"]["threshold"],
        "race_temperature": calibration["temperature_scaled_threshold"]["threshold"],
        "race_blend": selection["thresholds"]["race_blend"]["threshold"],
        "stack_all_original": stack_selection["threshold"]["threshold"],
        "stack_ce_race": negative_only_threshold(oof_ce_race[val_negative, 2:].sum(1))["threshold"],
    }
    # These controls were already fitted and assessed, not selected on this set.
    for name in ("ce_only", "ce_tfidf", "ce_newlex"):
        p = probability_array(
            original, ROOT / f"cpu_stacking_controls_v1/{name}_mixset_predictions.jsonl")
        oof = probability_array(
            val, ROOT / f"cpu_stacking_controls_v1/{name}_oof_predictions.jsonl")
        predictions[f"stack_{name}"] = p
        thresholds[f"stack_{name}"] = negative_only_threshold(oof[val_negative, 2:].sum(1))["threshold"]
    scores = {name: summarize(original, p, thresholds[name]) for name, p in predictions.items()}
    y = np.array([LABELS.index(r["label"]) for r in original])
    uncertainty = bootstrap_deltas(
        original, y, {k: p for k, p in predictions.items()
                      if k not in ("roberta_ce_s3407", "tfidf_lr", "race_temperature")},
        {"roberta_ce_s3407": p_ce, "tfidf_lr": p_tf})
    for name, p in predictions.items():
        save_predictions(output / f"{name}_predictions.jsonl", original, p)
    return {"metrics": scores, "macro_f1_paired_uncertainty": uncertainty,
            "thresholds": thresholds,
            "uncertainty_scope": "2000 source-group fixed-model percentile bootstrap replicates; "
                                 "no training, meta-selection or threshold-fit uncertainty; no multiplicity correction.",
            "race_blend_alpha": alpha}


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    manifest = json.loads((ROOT / "repository/resources/manifest.json").read_text())
    asset = next(r for r in manifest["files"] if r.get("run") == "race_s3407")
    checkpoint_path = ROOT / "resources/checkpoints" / asset["asset"]
    if digest(checkpoint_path) != asset["sha256"]:
        raise ValueError("RACE checkpoint mismatch")
    import subprocess
    upstream = ROOT / "resources/RACE"
    if (subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
            != manifest["upstream"]["commit"]
            or subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip()):
        raise ValueError("Upstream source changed")
    OUT.mkdir()  # Preserve previous and partial evidence; do not silently rerun.
    frozen = [
        ROOT / "cpu_improvement_v1/selection.json",
        ROOT / "race_cpu_v1/calibration.json",
        ROOT / "cpu_stacking_v1/selection.json",
        ROOT / "cpu_stacking_v1/meta_model.joblib",
        ROOT / "cpu_stacking_controls_v1/frozen_models.json",
        ROOT / "cpu_stacking_controls_v1/ce_race.joblib",
        ROOT / "cpu_stacking_controls_v1/ce_race_oof_predictions.jsonl",
        ROOT / "transfer_tfidf_v1/threshold.json",
    ]
    for name in ("ce_only", "ce_tfidf", "ce_newlex"):
        frozen += [ROOT / f"cpu_stacking_controls_v1/{name}_oof_predictions.jsonl",
                   ROOT / f"cpu_stacking_controls_v1/{name}_mixset_predictions.jsonl"]
    write(OUT / "protocol.json", {
        "date": "2026-11-09", "device": "cpu", "training": False,
        "model": "race_s3407 CE+SupCon; original full-text graph protocol",
        "evaluation": "Complete missing external results of previously frozen methods, no new selection.",
        "frozen_artifact_sha256": {str(p.relative_to(ROOT)): digest(p) for p in frozen},
        "checkpoint_sha256": asset["sha256"], "script_sha256": digest(__file__),
        "limitations": [
            "HART, HC3 and some MixSet results previously inspected; exploratory completion, not independent confirmation.",
            "Stacking OOF threshold may not calibrate final all-development fitted model.",
            "The base checkpoints used full development labels for checkpoint selection.",
            "MixSet conservative mapped subset is not the original complete benchmark; source license unresolved.",
            "Not a same-window structural ablation; no new neural training.",
        ]})
    protocol = json.loads((MIX / "protocol.json").read_text())
    if digest(MIX / "test.jsonl") != protocol["data_sha256"]:
        raise ValueError("Frozen external subset changed")
    parsed_path = MIX / "test_parsed_aligned_cpu.jsonl"
    audit = json.loads(parsed_path.with_suffix(".audit.json").read_text())
    if audit["status"] != "passed" or audit["output_sha256"] != digest(parsed_path):
        raise ValueError("Invalid parser audit")
    original, parsed = rows(MIX / "test.jsonl"), rows(parsed_path)
    coverage = check_parsed(original, parsed)
    by_id = {r["item_id"]: r for r in parsed}
    parsed = [by_id[r["item_id"]] for r in original]
    coverage.update({
        "parsed_sha256": digest(parsed_path), "parse_seconds": sum(audit["parse_seconds"]),
        "parser_revision": audit["parser_revision"], "parser_load_seconds": audit["model_load_seconds"],
        "peak_rss_mib_corrected": audit["peak_process_rss_mib"] / (1024 if platform.system() == "Darwin" else 1),
    })
    write(OUT / "parse_coverage.json", coverage)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(saved["config"], saved.get("metadata")).cpu().eval()
    model.load_state_dict(saved["model_state_dict"], strict=True)
    del saved
    z, seconds = infer(model, parsed, OUT / "race_logits.jsonl")
    del model
    t = json.loads((ROOT / "race_cpu_v1/calibration.json").read_text())["temperature"]
    result = evaluate_frozen(original, temperature_probabilities(z, 1),
                             temperature_probabilities(z, t), OUT)
    result.update({
        "status": "completed", "n": len(original), "groups": len({r["group_id"] for r in original}),
        "inference_seconds": seconds, "evaluation_seconds": time.perf_counter() - started,
        "parse_audit_sha256": digest(parsed_path.with_suffix(".audit.json")),
        "external_data_sha256": protocol["data_sha256"],
        "protocol_sha256": digest(OUT / "protocol.json"),
    })
    write(OUT / "results.json", result)
    lines = [
        "# RACE与冻结融合的外部CPU评测", "",
        "2026-11-09；MixSet保守映射359条/91来源组。无新增训练、选模或外部调参。",
        "此前已经看到部分外部结果；本次是补齐缺失对照，不是全新独立确认。", "",
        "| 方法 | 四分类Macro-F1 | creator AUROC | 锁阈值FPR | 锁阈值TPR |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in result["metrics"].items():
        locked = m["creator"]["at_threshold"]["hart_validation_locked"]
        lines.append(f"| {name} | {m['four_class']['macro_f1']:.2%} | "
                     f"{m['creator']['auroc']:.2%} | {locked['fpr']:.2%} | {locked['tpr']:.2%} |")
    lines += [
        "", "## 相对RoBERTa CE的固定模型来源组区间", "",
        "| 方法 | Macro-F1差，百分点 | 95%条件区间 |", "|---|---:|---|",
    ]
    for name, comparisons in result["macro_f1_paired_uncertainty"].items():
        u = comparisons["roberta_ce_s3407"]
        lo, hi = u["paired_95_percentile"]
        lines.append(f"| {name} | {100*u['delta']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += [
        "", "## 边界", "",
        "- RACE使用全文图、词汇首512 tokens；不是新的同窗口消融。",
        "- 温度与概率融合系数来自原HART开发集；Stacking使用先前保存的元模型，没有在MixSet重拟合。",
        "- Stacking阈值来自开发集OOF概率，最终全开发拟合可能改变校准；应看外部实际误报，不能保证1%。",
        "- 温度缩放保留四分类argmax，但不保证creator概率排序保持。",
        "- 区间只对固定模型的来源组抽样，不包含训练、选择和校准不确定性，也未校正多重比较。",
        "- 解析字符覆盖通过不等于RST语义关系正确。数据映射、规模和许可限制仍存在。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:18]), flush=True)


if __name__ == "__main__":
    main()
