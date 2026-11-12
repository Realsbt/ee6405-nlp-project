"""Source-balanced paired editing sensitivity of fixed external predictions."""

import json
from collections import defaultdict

import numpy as np

from improve_cpu import ROOT, LABELS, digest, probability_array, rows, write

OUT = ROOT / "edit_pair_diagnostics_cpu_v1"


def paired_summary(groups, p, index, cutoff):
    """Average variants within source first; sources receive equal weight."""
    deltas, before, after, rate_before, rate_after = [], [], [], [], []
    for original, variants in groups:
        base = p[index[original["item_id"]], 2:].sum()
        edited = np.array([p[index[r["item_id"]], 2:].sum() for r in variants])
        before.append(base)
        after.append(float(edited.mean()))
        deltas.append(float(edited.mean() - base))
        rate_before.append(float(base >= cutoff))
        rate_after.append(float((edited >= cutoff).mean()))
    rng = np.random.default_rng(20261109)
    values = np.array(deltas)
    draws = rng.integers(0, len(groups), size=(2000, len(groups)))
    return {
        "source_groups": len(groups), "edited_variants": sum(len(v) for o, v in groups),
        "source_balanced_mean_ai_score_original": float(np.mean(before)),
        "source_balanced_mean_ai_score_edited": float(np.mean(after)),
        "paired_mean_ai_score_delta": float(values.mean()),
        "source_bootstrap_95_percentile": np.quantile(values[draws].mean(1), [.025, .975]).tolist(),
        "fraction_ai_original_at_threshold": float(np.mean(rate_before)),
        "source_balanced_fraction_ai_edited_at_threshold": float(np.mean(rate_after)),
        "threshold": cutoff,
    }


def pair_groups(records):
    families = defaultdict(list)
    for row in records:
        families[row["group_id"]].append(row)
    result = []
    for group_id, family in sorted(families.items()):
        originals = [r for r in family if r["operation"] == "original"]
        edited = [r for r in family if r["operation"] != "original"]
        if len(originals) != 1 or not edited:
            raise ValueError("Need exactly one original and edited variants per family")
        original = originals[0]
        expected = "human_ai_polished" if original["label"] == "human_written" else "ai_humanized"
        if original["label"] not in ("human_written", "ai_generated") or any(r["label"] != expected for r in edited):
            raise ValueError("Creator is not constant within family")
        result.append((original, edited))
    return result


def main():
    OUT.mkdir()
    path = ROOT / "resources/mixset_conservative_transfer_v1/test.jsonl"
    protocol = json.loads((path.parent / "protocol.json").read_text())
    if digest(path) != protocol["data_sha256"]:
        raise ValueError("Frozen external set changed")
    records = rows(path)
    groups = pair_groups(records)
    index = {r["item_id"]: i for i, r in enumerate(records)}
    if len(index) != len(records):
        raise ValueError("Duplicate item IDs")
    source = ROOT / "external_completed_cpu_v1"
    previous = json.loads((source / "results.json").read_text())
    result = {
        "scope": "Exploratory paired editing sensitivity, equal source weight; not randomized causal effects. "
                 "Fixed models, no retraining/selection uncertainty or multiplicity correction.",
        "data_sha256": digest(path), "script_sha256": digest(__file__), "models": {},
    }
    for name in ("roberta_ce_s3407", "race_s3407", "stack_all_original"):
        predictions = source / f"{name}_predictions.jsonl"
        p = probability_array(records, predictions)
        cutoff = previous["thresholds"][name]
        entry = {"prediction_sha256": digest(predictions), "by_creator": {}}
        for creator, label in (("human", "human_written"), ("ai", "ai_generated")):
            subset = [(o, v) for o, v in groups if o["label"] == label]
            entry["by_creator"][creator] = {
                "fixed_0_5": paired_summary(subset, p, index, .5),
                "hart_locked": paired_summary(subset, p, index, cutoff),
            }
        entry["by_operation"] = {}
        for operation in sorted({r["operation"] for o, variants in groups for r in variants}):
            subset = [(o, [v for v in variants if v["operation"] == operation])
                      for o, variants in groups if any(v["operation"] == operation for v in variants)]
            entry["by_operation"][operation] = paired_summary(subset, p, index, .5)
        result["models"][name] = entry
    write(OUT / "results.json", result)
    lines = [
        "# 同来源原文—编辑版配对诊断", "",
        "2026-11-09。每个来源先平均其编辑版本，再对来源等权平均；区间为2,000次来源bootstrap。",
        "人类来源50组/200编辑版；AI来源41组/68编辑版。creator真值在来源内保持不变。", "",
        "| 模型 | 来源 | 原文AI分数 | 编辑后AI分数 | 配对差及95%条件区间 | 原文→编辑后判AI比例，阈值0.5 |",
        "|---|---|---:|---:|---|---|",
    ]
    for name, entry in result["models"].items():
        for creator, summaries in entry["by_creator"].items():
            s = summaries["fixed_0_5"]
            lo, hi = s["source_bootstrap_95_percentile"]
            lines.append(
                f"| {name} | {creator} | {s['source_balanced_mean_ai_score_original']:.3f} | "
                f"{s['source_balanced_mean_ai_score_edited']:.3f} | "
                f"{s['paired_mean_ai_score_delta']:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                f"{s['fraction_ai_original_at_threshold']:.2%} → "
                f"{s['source_balanced_fraction_ai_edited_at_threshold']:.2%} |")
    lines += [
        "", "## 解释边界", "",
        "- 人类来源行的判AI比例为误报，AI来源行为召回；编辑版比例先在来源内平均，不是简单逐行比例。",
        "- 原文和编辑版来自同一发布来源，减少题材差异，但不是随机试验，不能证明特定模型组件导致错误。",
        "- AI原文若本身就低分，则不能把编辑后失败全部归因为编辑；应同时看原文基线。",
        "- 所有模型已见部分外部数据结果，本分析仅定位问题，不反转分数、不重选阈值。",
        "- 每种编辑操作及HART锁定阈值下的配对统计见results.json；无新增标注或LLM裁判。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:14]), flush=True)


if __name__ == "__main__":
    main()
