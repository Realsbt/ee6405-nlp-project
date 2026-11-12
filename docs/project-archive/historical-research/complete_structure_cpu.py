"""Evaluate previously selected full-text structural CPU models without refitting."""

import json
import time

import joblib
import numpy as np

from complete_external_cpu import check_parsed
from improve_cpu import bootstrap_deltas, probability_array, save_predictions
from screen_structure_cpu import ROOT, LABELS, digest, features, rows, write
from evaluation_metrics import negative_only_threshold
from run_transfer_evaluation import summarize

OUT = ROOT / "structure_completed_cpu_v1"
SCREEN = ROOT / "structure_screen_v1"
DATA = ROOT / "resources/experiments/hart_grouped_v1"
KINDS = ("length", "structure", "combined")


def evaluate(records, extracted, models, selection, thresholds, neural, output, split):
    predictions = {"roberta_ce_s3407": neural}
    for kind in KINDS:
        p = models[kind].predict_proba([extracted[r["item_id"]][kind] for r in records])
        alpha = selection["models"][kind]["selected_fusion"]["alpha"]
        predictions[kind] = p
        predictions[f"{kind}_fusion"] = (1 - alpha) * neural + alpha * p
    result = {name: summarize(records, p, thresholds[name]) for name, p in predictions.items()}
    for name, p in predictions.items():
        save_predictions(output / f"{split}_{name}_predictions.jsonl", records, p)
    y = np.array([LABELS.index(r["label"]) for r in records])
    uncertainty = bootstrap_deltas(
        records, y, {k: p for k, p in predictions.items() if k not in ("length", "roberta_ce_s3407")},
        {"length": predictions["length"], "roberta_ce_s3407": neural})
    return {"metrics": result, "paired_uncertainty": uncertainty}


def main():
    OUT.mkdir()
    started = time.perf_counter()
    selection = json.loads((SCREEN / "development_results.json").read_text())
    data_protocol = json.loads((DATA / "protocol.json").read_text())
    frozen = [SCREEN / "development_results.json", SCREEN / "protocol.json"]
    for kind in KINDS:
        frozen += [SCREEN / f"{kind}.joblib", SCREEN / f"{kind}_development_predictions.jsonl"]
    write(OUT / "protocol.json", {
        "date": "2026-11-09", "fit": "none; reuse train-fitted models and earlier dev selection",
        "selection": "No candidate selection in this evaluation; all three feature sets and frozen blends reported.",
        "frozen_sha256": {str(p.relative_to(ROOT)): digest(p) for p in frozen},
        "script_sha256": digest(__file__),
        "graph_scope": "Original released full-text HART trees; pinned parser full-text MixSet trees. "
                       "Not same-window and not a controlled parser alignment comparison.",
        "limitations": [
            "HART and MixSet previously inspected: post-hoc evaluation.",
            "Features include length proxies; structural statistics are not pure relation semantics.",
            "No domain/label/ID is provided as a feature.",
            "Dev checkpoint selection was not independent; fixed-model bootstrap excludes selection uncertainty.",
            "All groups/results reported; no external fitting or threshold tuning.",
        ]})
    if digest(DATA / "val.jsonl") != data_protocol["partitions"]["val"]["sha256"]:
        raise ValueError("Development partition changed")
    val = rows(DATA / "val.jsonl")
    mask = np.array([r["label_id"] < 2 for r in val])
    base_val = probability_array(val, ROOT / "transfer_roberta_v1/validation_predictions.jsonl")
    thresholds = {"roberta_ce_s3407": negative_only_threshold(base_val[mask, 2:].sum(1))["threshold"]}
    models = {}
    for kind in KINDS:
        models[kind] = joblib.load(SCREEN / f"{kind}.joblib")
        if models[kind].classes_.tolist() != list(range(4)):
            raise ValueError("Unexpected structural class order")
        p = probability_array(val, SCREEN / f"{kind}_development_predictions.jsonl")
        alpha = selection["models"][kind]["selected_fusion"]["alpha"]
        thresholds[kind] = negative_only_threshold(p[mask, 2:].sum(1))["threshold"]
        fusion = (1 - alpha) * base_val + alpha * p
        thresholds[f"{kind}_fusion"] = negative_only_threshold(fusion[mask, 2:].sum(1))["threshold"]
    write(OUT / "thresholds.json", thresholds)
    if digest(DATA / "test.jsonl") != data_protocol["partitions"]["test"]["sha256"]:
        raise ValueError("Test partition changed")
    test = rows(DATA / "test.jsonl")
    if {r["split_block_id"] for r in test} & {r["split_block_id"] for r in val}:
        raise ValueError("Development/test source overlap")
    wanted = {r["item_id"]: r for r in test}
    extracted, graph_hashes = {}, {}
    for path in sorted((ROOT / "resources/assets/hart_official").glob("*_graph.jsonl")):
        graph_hashes[str(path.relative_to(ROOT))] = digest(path)
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                item = row["item_id"]
                if item not in wanted:
                    continue
                if item in extracted or any(row[k] != wanted[item][k] for k in ("article", "label")):
                    raise ValueError("Original/frozen graph mismatch")
                extracted[item] = features(row)
    if set(extracted) != set(wanted):
        raise ValueError("Missing HART graph features")
    base = probability_array(test, ROOT / "repository/experiments/reports/roberta_s3407/test_predictions.jsonl")
    result = {"hart": evaluate(test, extracted, models, selection, thresholds, base, OUT, "hart")}
    mix_root = ROOT / "resources/mixset_conservative_transfer_v1"
    mix_protocol = json.loads((mix_root / "protocol.json").read_text())
    if digest(mix_root / "test.jsonl") != mix_protocol["data_sha256"]:
        raise ValueError("MixSet subset changed")
    mix, parsed = rows(mix_root / "test.jsonl"), rows(mix_root / "test_parsed_aligned_cpu.jsonl")
    check_parsed(mix, parsed)
    mix_features = {r["item_id"]: features(r) for r in parsed}
    base = probability_array(mix, ROOT / "transfer_roberta_v1/mixset_predictions.jsonl")
    result["mixset"] = evaluate(mix, mix_features, models, selection, thresholds, base, OUT, "mixset")
    result.update({
        "status": "completed", "training": False, "elapsed_seconds": time.perf_counter() - started,
        "protocol_sha256": digest(OUT / "protocol.json"), "graph_sha256": graph_hashes,
        "mixset_parsed_sha256": digest(mix_root / "test_parsed_aligned_cpu.jsonl"),
        "thresholds_sha256": digest(OUT / "thresholds.json"),
        "uncertainty_scope": "2000 paired source-group percentile bootstrap replicates, fixed models; "
                             "no multiplicity correction or model-selection uncertainty.",
    })
    write(OUT / "results.json", result)
    lines = [
        "# 冻结轻量结构模型的CPU测试", "", "2026-11-09。仅评测先前训练/选择的模型，不再拟合参数。",
        "全部为全文结构探索，不是同512-token窗口消融。MixSet为359条/91来源组的保守映射子集。", "",
        "| 方法 | 开发Macro-F1 | HART测试Macro-F1 | MixSet子集Macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for name in result["hart"]["metrics"]:
        if name == "roberta_ce_s3407":
            dev = selection["baseline_roberta_ce_s3407"]["macro_f1"]
        elif name.endswith("_fusion"):
            dev = selection["models"][name.removesuffix("_fusion")]["selected_fusion"]["development"]["macro_f1"]
        else:
            dev = selection["models"][name]["development"]["macro_f1"]
        lines.append(f"| {name} | {dev:.2%} | "
                     f"{result['hart']['metrics'][name]['four_class']['macro_f1']:.2%} | "
                     f"{result['mixset']['metrics'][name]['four_class']['macro_f1']:.2%} |")
    lines += [
        "", "## 解释边界", "",
        "- 结构特征中EDU数、深度和长度也可能编码文体/领域，不能直接归因为关系语义。",
        "- 字符覆盖通过不表示RST关系正确；HART发布图与本机MixSet解析仍可能存在解析管线差异。",
        "- fusion系数只沿用此前开发集选择；结构模型权重为0时即原文本模型，不是新提升。",
        "- 低误报阈值只由HART开发集负类锁定；所有实际误报/召回与条件区间见results.json。",
        "- 本轮没有外部训练或调参；此前已见测试，不能称为独立确认。解析成本未包括在本轮评测耗时。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]), flush=True)


if __name__ == "__main__":
    main()
