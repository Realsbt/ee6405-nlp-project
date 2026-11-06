"""Paired, per-seed contrasts for the fixed four-condition experiment."""

import statistics

from experiments.ablation import CONDITIONS, SEEDS


CONTRASTS = {
    "supcon_with_roberta": {"roberta_supcon": 1, "roberta": -1},
    "supcon_with_race": {"race": 1, "race_no_supcon": -1},
    "race_minus_roberta_with_ce": {"race_no_supcon": 1, "roberta": -1},
    "race_minus_roberta_with_supcon": {"race": 1, "roberta_supcon": -1},
    "interaction": {"race": 1, "race_no_supcon": -1, "roberta_supcon": -1, "roberta": 1},
}
CONTRAST_NAMES = {
    "supcon_with_roberta": "RoBERTa 内 SupCon 增益",
    "supcon_with_race": "RACE 内 SupCon 增益",
    "race_minus_roberta_with_ce": "仅 CE 下 RACE − RoBERTa",
    "race_minus_roberta_with_supcon": "CE＋SupCon 下 RACE − RoBERTa",
    "interaction": "交互：RACE 的 SupCon 增益 − RoBERTa 的 SupCon 增益",
}


def summarize_factorial(analyses):
    effects = {}
    for name, coefficients in CONTRASTS.items():
        seeds = [seed for seed in SEEDS if all(f"{kind}_s{seed}" in analyses for kind in coefficients)]
        metrics = {}
        for metric in ("accuracy", "macro_f1", "macro_auroc_ovr"):
            values = [sum(weight * analyses[f"{kind}_s{seed}"]["test"][metric]
                          for kind, weight in coefficients.items()) for seed in seeds]
            metrics[metric] = {"values": values, "mean": statistics.mean(values) if values else None,
                               "sample_sd": statistics.stdev(values) if len(values) > 1 else None, "n": len(values)}
        effects[name] = {"coefficients": coefficients, "seeds": seeds, "metrics": metrics}
    missing = [f"{kind}_s{seed}" for kind in CONDITIONS for seed in SEEDS if f"{kind}_s{seed}" not in analyses]
    return {"status": "completed" if not missing else "partial", "missing_runs": missing,
            "planned_seeds": list(SEEDS), "effects": effects, "metric_unit": "fraction; multiply by 100 for percentage points",
            "scope": "Paired training seeds on one frozen HART split; sample SD is not a confidence interval. "
                     "Architecture effects also include pooling, parameter-count and information-coverage differences. "
                     "This extension follows inspection of the original results; no new independent test set."}


def markdown_factorial(summary):
    lines = ["## 2×2 消融：同种子成对差异", "",
             "每个差值先在相同种子内计算，再报告均值和样本 SD；单位为百分点。交互项为两种架构的 SupCon 增益之差。", "",
             "| 比较 | 配对种子数 | 3407 | 42 | 2026 | Macro-F1 差（均值 ± SD） |", "|---|---:|---:|---:|---:|---:|"]
    for name, effect in summary["effects"].items():
        metric = effect["metrics"]["macro_f1"]
        values = dict(zip(effect["seeds"], metric["values"]))
        cells = [f"{100*values[s]:+.2f}" if s in values else "未完成" for s in SEEDS]
        mean = f"{100*metric['mean']:+.2f}" if metric["mean"] is not None else "未完成"
        if metric["sample_sd"] is not None:
            mean += f" ± {100*metric['sample_sd']:.2f}"
        lines.append(f"| {CONTRAST_NAMES[name]} | {metric['n']} | " + " | ".join(cells) + f" | {mean} |")
    lines += ["", "SD 不是置信区间，3 个训练种子也不足以建立跨划分、跨数据集或未见生成模型的普遍结论。"
              "本轮补充在查看原有结果后确定，使用同一保留测试集，不构成新增独立测试。所有训练仍只用验证集选 checkpoint。",
              "", "相同损失下的架构差异仍同时包含 pooling、参数量及结构信息覆盖差异；不能全部解释为篇章图本身的因果收益。"]
    return lines
