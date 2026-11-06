"""Recompute unified metrics and analysis from verified, saved predictions."""

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from experiments.common import DATA_ROOT, DOMAINS, LABELS, classification_metrics, file_hash, read_jsonl, write_json, write_jsonl
from experiments.neural import PROJECT_ROOT, load_records
from experiments.ablation import CONDITIONS, NEURAL_RUNS, NAMES as NEURAL_NAMES
from experiments.factorial_summary import markdown_factorial, summarize_factorial


REPORTS = PROJECT_ROOT / "experiments/reports"
PRIMARY = ["tfidf_lr_v1", "roberta_s3407", "race_s3407", "race_no_supcon_s3407"]
DISPLAY_RUNS = ["tfidf_lr_v1"] + [f"{kind}_s3407" for kind in CONDITIONS]
EXPECTED = ["tfidf_lr_v1", *NEURAL_RUNS]
NAMES = {"tfidf_lr_v1": "TF-IDF + LR", **NEURAL_NAMES}
SHORT_LABELS = ["Human", "AI-polished", "AI-generated", "AI-humanized"]


def family(run):
    return run.rsplit("_s", 1)[0] if "_s" in run else run


def f1_from_confusion(matrices):
    diagonal = np.diagonal(matrices, axis1=-2, axis2=-1)
    denominator = matrices.sum(-1) + matrices.sum(-2)
    return np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float), where=denominator != 0).mean(-1)


def paired_block_bootstrap(rows, y, left, right):
    blocks = sorted({row["split_block_id"] for row in rows})
    indices = {block: index for index, block in enumerate(blocks)}
    matrices = []
    for predictions in (left, right):
        counts = np.zeros((len(blocks), 4, 4), dtype=int)
        for row, target, predicted in zip(rows, y, predictions):
            counts[indices[row["split_block_id"]], target, predicted] += 1
        matrices.append(counts)
    rng = np.random.default_rng(2026)
    deltas = []
    for _ in range(1000):
        selected = rng.integers(len(blocks), size=len(blocks))
        deltas.append(float(f1_from_confusion(matrices[1][selected].sum(0)) - f1_from_confusion(matrices[0][selected].sum(0))))
    return {"delta_macro_f1_right_minus_left": float(f1_from_confusion(matrices[1].sum(0)) - f1_from_confusion(matrices[0].sum(0))),
            "percentile_95_interval": np.quantile(deltas, [.025, .975]).tolist(),
            "resamples": 1000, "seed": 2026, "resampling_units": len(blocks),
            "unit": "frozen source/duplicate-connected split block",
            "scope": "Conditional on this split and the two fitted seed-3407 models; does not measure training-seed or cross-dataset uncertainty"}


def figures(out, analyses, summaries):
    colors = ["#728f86", "#4f7f91", "#8b72a3", "#c18b54", "#137f73"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.titleweight": "bold"})
    keys = [key for key in DISPLAY_RUNS if key in analyses]
    if not keys:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, title in zip(axes, ("macro_f1", "macro_auroc_ovr"), ("Macro-F1", "Macro-AUROC (OvR)")):
        values, deviations = [], []
        for key in keys:
            summary = summaries[family(key)][metric]
            values.append(100 * summary["mean"])
            deviations.append(100 * (summary["sample_sd"] or 0))
        ax.bar(np.arange(len(keys)), values, yerr=deviations, capsize=4, color=colors[:len(keys)], width=.65)
        ax.set_xticks(np.arange(len(keys)), [NAMES[family(key)] for key in keys], rotation=15, ha="right")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Percent")
        ax.set_title(title)
        for i, value in enumerate(values):
            ax.text(i, value + deviations[i] + 1.8, f"{value:.2f}", ha="center", fontsize=9)
    fig.suptitle("HART grouped holdout · mean ± training-seed SD where available", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "main_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 3.9), squeeze=False)
    for ax, key in zip(axes[0], keys):
        counts = np.asarray(analyses[key]["test"]["confusion_matrix"])
        fractions = counts / counts.sum(1, keepdims=True)
        ax.imshow(fractions, vmin=0, vmax=1, cmap="Greens")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{counts[i,j]}\n{fractions[i,j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if fractions[i,j] > .6 else "#18342a")
        ax.set_xticks(range(4), SHORT_LABELS, rotation=45, ha="right")
        ax.set_yticks(range(4), SHORT_LABELS)
        ax.set_title(NAMES[family(key)], fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle("Seed 3407 (TF-IDF deterministic) · counts and row percentages", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.3))
    width = .78 / len(keys)
    x = np.arange(4)
    for index, key in enumerate(keys):
        values = [analyses[key]["by_domain"][domain]["macro_f1"] * 100 for domain in DOMAINS]
        ax.bar(x - .39 + width * (index + .5), values, width=width, label=NAMES[family(key)], color=colors[index])
    ax.set_xticks(x, DOMAINS)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Mixed-domain training, same-benchmark domain slices · seed 3407")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "domain_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    seed_colors = {"3407": "#137f73", "42": "#4f7f91", "2026": "#8b72a3"}
    all_validation_values = []
    for ax, kind in zip(axes.flat, CONDITIONS):
        for run in analyses:
            if family(run) != kind:
                continue
            events = read_jsonl(DATA_ROOT.parent / "neural_runs" / run / "events.jsonl")
            epochs = [event for event in events if event["event"] == "epoch_complete"]
            if not epochs:
                continue
            x = [event["epoch"] for event in epochs]
            y = [100 * event["validation"]["macro_f1"] for event in epochs]
            all_validation_values.extend(y)
            seed = run.rsplit("_s", 1)[1]
            color = seed_colors[seed]
            ax.plot(x, y, marker=".", color=color, label=f"Seed {seed}")
            best = analyses[run]["best_epoch"]
            selected = next(event for event in epochs if event["epoch"] == best)
            ax.scatter([best], [100 * selected["validation"]["macro_f1"]], marker="*", s=105, color=color, zorder=5)
        ax.set_xlabel("Completed epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.set_title(NAMES[kind])
        if ax.lines:
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, "No completed, verified run yet", transform=ax.transAxes, ha="center")
        ax.grid(axis="y", alpha=.15)
    if all_validation_values:
        axes[0, 0].set_ylim(max(0, min(all_validation_values) - 3), min(100, max(all_validation_values) + 3))
    for ax in axes[:, 0]:
        ax.set_ylabel("Validation Macro-F1 (%)")
    fig.suptitle("Completed and reload-verified runs · stars mark validation-selected checkpoints\nNo test curve is used for selection", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    factorial = summarize_factorial(analyses)
    if factorial["status"] == "completed":
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.35]})
        for kinds, color, label in ((["roberta", "roberta_supcon"], "#4f7f91", "RoBERTa"),
                                    (["race_no_supcon", "race"], "#137f73", "RACE")):
            for seed in (3407, 42, 2026):
                axes[0].plot([0, 1], [100 * analyses[f"{kind}_s{seed}"]["test"]["macro_f1"] for kind in kinds],
                             color=color, alpha=.25, linewidth=1)
            means = [100 * summaries[kind]["macro_f1"]["mean"] for kind in kinds]
            sds = [100 * summaries[kind]["macro_f1"]["sample_sd"] for kind in kinds]
            axes[0].errorbar([0, 1], means, yerr=sds, color=color, marker="o", capsize=5, linewidth=2, label=label)
        axes[0].set_xticks([0, 1], ["CE", "CE + SupCon"])
        axes[0].set_xlim(-.2, 1.2)
        axes[0].set_ylabel("Test Macro-F1 (%)")
        axes[0].set_title("Four conditions · mean ± seed SD")
        axes[0].legend()
        labels = ["SupCon effect · RoBERTa", "SupCon effect · RACE", "RACE − RoBERTa · CE",
                  "RACE − RoBERTa · CE + SupCon", "Interaction (difference of effects)"]
        effects = list(factorial["effects"].values())
        for index, effect in enumerate(effects):
            metric = effect["metrics"]["macro_f1"]
            axes[1].scatter([100*x for x in metric["values"]], [index]*3, color="#8b72a3", alpha=.5, s=20)
            axes[1].errorbar(100 * metric["mean"], index, xerr=100 * metric["sample_sd"],
                             fmt="o", color="#18342a", capsize=4)
        axes[1].axvline(0, color="#aaaaaa", linestyle="--", linewidth=1)
        axes[1].set_yticks(range(5), labels)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Paired Macro-F1 difference (percentage points)")
        axes[1].set_title("Three matched seeds · mean ± SD, not CI")
        fig.suptitle("HART frozen grouped split · seeds 3407, 42, 2026", fontsize=12)
        fig.tight_layout()
        fig.savefig(out / "ablation_2x2.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    rows = load_records("test")
    y = np.asarray([row["label_id"] for row in rows])
    ids = [row["item_id"] for row in rows]
    quality = {}
    quality_report = REPORTS / "released_rst_span_audit.json"
    if quality_report.exists():
        q = json.loads(quality_report.read_text())
        quality_path = REPORTS / "released_rst_item_quality.jsonl"
        if quality_path.exists():
            assert q["data_protocol_sha256"] == file_hash(DATA_ROOT / "protocol.json")
            assert q["item_quality_sha256"] == file_hash(quality_path)
            quality = {row["item_id"]: row for row in read_jsonl(quality_path)}
            assert all(item_id in quality for item_id in ids)
    analyses, predictions, missing = {}, {}, []
    for run in EXPECTED:
        directory = REPORTS / run
        if not all((directory / name).exists() for name in ("result.json", "verification.json", "test_predictions.jsonl")):
            missing.append(run)
            continue
        result = json.loads((directory / "result.json").read_text())
        verification = json.loads((directory / "verification.json").read_text())
        assert result["status"] == "completed" and verification["status"] == "passed"
        assert result["data_protocol_sha256"] == file_hash(DATA_ROOT / "protocol.json")
        saved = read_jsonl(directory / "test_predictions.jsonl")
        assert [row["item_id"] for row in saved] == ids and [row["true_label"] for row in saved] == [row["label"] for row in rows]
        probabilities = np.asarray([[row["probabilities"][name] for name in LABELS] for row in saved])
        metrics = classification_metrics(y, probabilities)
        assert all(abs(metrics[key] - result["test"][key]) < 1e-8 for key in ("accuracy", "macro_f1", "macro_auroc_ovr"))
        cohorts = {}
        masks = {"lexical_window_complete": np.asarray([not row["input_truncated"] for row in rows]),
                 "lexical_window_truncated": np.asarray([row["input_truncated"] for row in rows])}
        if quality:
            clean = np.asarray([quality[row["item_id"]]["valid_and_ordered_spans"] and quality[row["item_id"]]["all_nonspace_covered"] for row in rows])
            masks.update(released_rst_coordinates_clean=clean, released_rst_coordinate_issues=~clean,
                         lexical_window_complete_and_rst_clean=clean & masks["lexical_window_complete"])
        for name, mask in masks.items():
            cohorts[name] = classification_metrics(y[mask], probabilities[mask])
        domains = {name: classification_metrics(y[mask], probabilities[mask]) for name in DOMAINS
                   for mask in [np.asarray([row["domain"] == name for row in rows]) ]}
        analyses[run] = {"test": metrics, "cohorts": cohorts, "by_domain": domains,
                         "validation_macro_f1": result["validation"]["macro_f1"],
                         "best_epoch": result.get("best_epoch"), "efficiency": result["efficiency"],
                         "result_sha256": file_hash(directory / "result.json"),
                         "predictions_sha256": file_hash(directory / "test_predictions.jsonl")}
        predictions[run] = probabilities.argmax(1)
    if args.require_complete and missing:
        raise RuntimeError(f"Required runs are not all verified: {missing}")
    summaries = {}
    for kind in sorted({family(run) for run in analyses}):
        runs = [run for run in analyses if family(run) == kind]
        summaries[kind] = {"runs": runs, "seeds": [int(run.rsplit("_s", 1)[1]) for run in runs if "_s" in run]}
        for key in ("accuracy", "macro_f1", "macro_auroc_ovr"):
            values = [analyses[run]["test"][key] for run in runs]
            summaries[kind][key] = {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                     "n": len(values), "values": values}
    bootstrap = {}
    for left, right in (("roberta_s3407", "race_s3407"), ("race_no_supcon_s3407", "race_s3407")):
        if left in predictions and right in predictions:
            bootstrap[f"{right}_minus_{left}"] = paired_block_bootstrap(rows, y, predictions[left], predictions[right])
    factorial = summarize_factorial(analyses)
    auxiliary = {}
    for run in PRIMARY:
        path = REPORTS / run / "hc3_aux_v1_1_result.json"
        if path.exists():
            result = json.loads(path.read_text())
            assert result["status"] == "completed" and result["protocol_id"] == "hc3_aux_v1_1"
            auxiliary[run] = result
    missing_auxiliary = [run for run in PRIMARY if run not in auxiliary]
    if args.require_complete and missing_auxiliary:
        raise RuntimeError(f"Required auxiliary evaluations are missing: {missing_auxiliary}")
    out = REPORTS / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    errors = []
    if "race_s3407" in predictions:
        for index, row in enumerate(rows):
            wrong = predictions["race_s3407"][index] != y[index]
            different = "roberta_s3407" in predictions and predictions["race_s3407"][index] != predictions["roberta_s3407"][index]
            if wrong or different:
                errors.append({"item_id": row["item_id"], "domain": row["domain"], "true_label": row["label"],
                               "input_truncated": row["input_truncated"], "race_error": bool(wrong),
                               "predictions": {run: LABELS[int(values[index])] for run, values in predictions.items() if run in PRIMARY},
                               "lexical_excerpt_first_600_chars": row["model_text"][:600],
                               "case_selection": "all RACE errors or RACE/RoBERTa disagreements, ordered by item ID"})
    errors.sort(key=lambda row: row["item_id"])
    write_jsonl(out / "error_cases.jsonl", errors)
    from experiments.analyze_auxiliary import summarize
    from experiments.case_studies import build_case_studies
    auxiliary_analysis = summarize(args.require_complete)
    qualitative_analysis = build_case_studies(args.require_complete)
    report = {"status": "completed" if not missing and not missing_auxiliary else "partial", "missing_verified_runs": missing,
              "missing_auxiliary_runs": missing_auxiliary,
              "data_protocol_sha256": file_hash(DATA_ROOT / "protocol.json"),
              "test_samples": len(rows), "analyses": analyses, "training_seed_summaries": summaries,
              "paired_block_bootstrap": bootstrap, "hc3_binary_auxiliary": auxiliary,
              "factorial_ablation": factorial,
              "auxiliary_analysis_sha256": file_hash(REPORTS / "auxiliary/summary.json"),
              "case_studies_sha256": file_hash(out / "case_studies.json"),
              "information_boundary": "All lexical inputs use the first 512 RoBERTa tokens. RACE drops out-of-window EDU nodes but retains full-document relation nodes and their connections; use untruncated cohort comparisons to examine this difference. Cohort differences are descriptive, not randomized controls.",
              "statistical_boundary": "The 2x2 design has four neural conditions and three planned seeds per condition on one frozen grouped split; completed runs are listed separately. The added ablations follow inspection of existing results, not an independent holdout. Bootstrap is conditional on fitted seed-3407 models; HC3 retains its original single-seed scope.",
              "analysis_script_sha256": file_hash(__file__)}
    report["analysis_support_sha256"] = {name: file_hash(PROJECT_ROOT / name) for name in
                                         ("experiments/ablation.py", "experiments/factorial_summary.py")}
    write_json(out / "summary.json", report)
    figures(out, analyses, summaries)
    lines = ["# EE6405 RACE 实验结果", "", f"状态：{'完整实验已复核' if report['status'] == 'completed' else '阶段结果，仍有实验未完成'}。固定 HART 分组测试集 {len(rows):,} 条；验证集选模。", "",
             "| 模型 | 训练次数 | Macro-F1（均值 ± SD） | Macro-AUROC（均值 ± SD） |", "|---|---:|---:|---:|"]
    for key in ("tfidf_lr_v1", *CONDITIONS):
        if key not in summaries:
            continue
        summary = summaries[key]
        def percent(name):
            item = summary[name]
            return f"{100*item['mean']:.2f}%" + (f" ± {100*item['sample_sd']:.2f}" if item["sample_sd"] is not None else "")
        lines.append(f"| {NAMES[key]} | {len(summary['runs'])} | {percent('macro_f1')} | {percent('macro_auroc_ovr')} |")
    lines += ["", "SD 表示同一冻结划分下训练随机种子的样本标准差；TF-IDF 为确定性拟合。四个神经配置均计划运行种子 3407、42、2026，缺失时明确标记。", ""]
    lines += markdown_factorial(factorial)
    if all(kind in summaries for kind in ("roberta", "race")) and summaries["roberta"]["seeds"] == summaries["race"]["seeds"]:
        delta = 100 * (summaries["race"]["macro_f1"]["mean"] - summaries["roberta"]["macro_f1"]["mean"])
        lines += ["", f"原方案比较（RACE＋SupCon 对 RoBERTa＋CE）：在相同的 {len(summaries['race']['seeds'])} 个训练种子上，平均 Macro-F1 差为 {delta:+.2f} 个百分点。该比较同时改变架构与训练目标。"
                  + ("RACE 的平均分数较低，本组实验不支持其总体优于 RoBERTa。" if delta < 0 else "该均值差仍需结合种子波动和对比条件解释，不能单独证明稳定优势。"),
                  "这是一个固定来源分组划分上的复核，没有覆盖重新划分、独立四分类数据或未见生成模型的不确定性。"]
    lines += ["", "| 主方法 | 种子 | 验证 Macro-F1 | 选中轮次 | 测试 Macro-F1 | 测试 Macro-AUROC |", "|---|---:|---:|---:|---:|---:|"]
    for run in EXPECTED:
        if run in analyses and family(run) in CONDITIONS:
            item = analyses[run]
            seed = int(run.rsplit("_s", 1)[1])
            lines.append(f"| {NAMES[family(run)]} | {seed} | {100*item['validation_macro_f1']:.2f}% | {item['best_epoch']} | {100*item['test']['macro_f1']:.2f}% | {100*item['test']['macro_auroc_ovr']:.2f}% |")
    lines += ["", "## 输入窗口与对比条件", "",
              "词汇输入均限制为首 512 个 RoBERTa token。RACE 会删除截断外的 EDU 节点，但保留全篇关系节点与连接，因此主表存在可用结构信息差异。下面分别报告词汇窗口完整与截断样本，属于描述性分组分析。", "",
              "| 单次模型（种子 3407） | 完整正文子集 n / Macro-F1 | 截断子集 n / Macro-F1 |", "|---|---:|---:|"]
    for run in DISPLAY_RUNS:
        if run in analyses:
            cohorts = analyses[run]["cohorts"]
            a, b = cohorts["lexical_window_complete"], cohorts["lexical_window_truncated"]
            lines.append(f"| {NAMES[family(run)]} | {a['samples']} / {100*a['macro_f1']:.2f}% | {b['samples']} / {100*b['macro_f1']:.2f}% |")
    if quality:
        q = json.loads(quality_report.read_text())
        test_quality = q["counts"]["test"]
        lines += ["", f"事后只读审计：作者发布的测试 RST 中有 {test_quality['records_with_invalid_spans']} 条含无效字符跨度；其余合法跨度记录中，{test_quality['records_with_uncovered_nonspace']} 条未覆盖全部非空白字符。发布数据及模型保持冻结。", "",
                  "下表由解析问题排查引出，只按坐标质量分组，不依据模型分数筛选；它属于事后描述性分析，不能作为重新训练的受控消融。坐标干净也不代表篇章关系标注正确。", "",
                  "| 模型 | 坐标干净 n / Macro-F1 | 坐标有问题 n / Macro-F1 | 词汇完整且坐标干净 n / Macro-F1 |", "|---|---:|---:|---:|"]
        for run in DISPLAY_RUNS:
            if run in analyses:
                cohorts = analyses[run]["cohorts"]
                values = [f"{cohorts[key]['samples']} / {100*cohorts[key]['macro_f1']:.2f}%" for key in
                          ("released_rst_coordinates_clean", "released_rst_coordinate_issues", "lexical_window_complete_and_rst_clean")]
                lines.append("| " + NAMES[family(run)] + " | " + " | ".join(values) + " |")
    lines += ["", "## 类别与子域", "", "以下使用预先指定的种子 3407，不从测试结果选择展示种子。", "",
              "| 模型 | 人工 F1 | AI 润色 F1 | AI 生成 F1 | 改写 AI F1 |", "|---|---:|---:|---:|---:|"]
    for run in DISPLAY_RUNS:
        if run in analyses:
            values = [f"{100*analyses[run]['test']['per_class'][label]['f1']:.2f}%" for label in LABELS]
            lines.append("| " + NAMES[family(run)] + " | " + " | ".join(values) + " |")
    lines += ["", "| 模型 | arxiv Macro-F1 | essay | news | writing |", "|---|---:|---:|---:|---:|"]
    for run in DISPLAY_RUNS:
        if run in analyses:
            values = [f"{100*analyses[run]['by_domain'][domain]['macro_f1']:.2f}%" for domain in DOMAINS]
            lines.append("| " + NAMES[family(run)] + " | " + " | ".join(values) + " |")
    pair = bootstrap.get("race_s3407_minus_roberta_s3407")
    if pair:
        low, high = pair["percentile_95_interval"]
        lines += ["", f"原方案种子 3407 的 RACE＋SupCon − RoBERTa＋CE Macro-F1 差为 {100*pair['delta_macro_f1_right_minus_left']:+.2f} 个百分点；按 {pair['resampling_units']} 个来源关联分组抽样的 95% 条件 bootstrap 区间为 [{100*low:+.2f}, {100*high:+.2f}] 个百分点。"
                  + ("该区间包含零，单次拟合没有明确的总体优劣证据。" if low <= 0 <= high else "该区间只描述这两个已拟合模型及当前测试来源组的抽样波动。"),
                  "它不涵盖重新训练和跨数据集的不确定性；主表中的训练种子波动另行报告。"]
    ablation_pair = bootstrap.get("race_s3407_minus_race_no_supcon_s3407")
    if ablation_pair:
        low, high = ablation_pair["percentile_95_interval"]
        lines += ["", f"种子 3407 的条件消融比较：完整 RACE 相对关闭监督对比损失的 Macro-F1 差为 {100*ablation_pair['delta_macro_f1_right_minus_left']:+.2f} 个百分点；来源组 bootstrap 95% 条件区间为 [{100*low:+.2f}, {100*high:+.2f}]。该条件区间不包含重新训练的不确定性；三种子配对差异另见 2×2 表，不能把损失收益当成图结构本身的独立收益。"]
    if "race_s3407" in analyses:
        matrix = np.asarray(analyses["race_s3407"]["test"]["confusion_matrix"])
        off_diagonal = matrix.copy()
        np.fill_diagonal(off_diagonal, 0)
        lines += ["", "主 RACE 单次模型的主要错误（按数量排序）：", ""]
        for flat_index in np.argsort(off_diagonal.ravel())[::-1][:3]:
            true, predicted = np.unravel_index(flat_index, (4, 4))
            count = int(matrix[true, predicted])
            if count:
                lines.append(f"- {LABELS[true]} → {LABELS[predicted]}：{count} 条，占该真实类别 {100*count/matrix[true].sum():.2f}%。")
        if "roberta_s3407" in predictions:
            race_correct = predictions["race_s3407"] == y
            roberta_correct = predictions["roberta_s3407"] == y
            lines += ["", f"RACE 分对而 RoBERTa 分错 {int((race_correct & ~roberta_correct).sum())} 条；反方向 {int((~race_correct & roberta_correct).sum())} 条。需要结合具体文本解释互补与失败，不能将所有分歧都归因为篇章关系。"]
            old = analyses["roberta_s3407"]["test"]["per_class"]["ai_humanized"]
            new = analyses["race_s3407"]["test"]["per_class"]["ai_humanized"]
            lines += ["", f"少数类 ai_humanized 的召回率由 RoBERTa 的 {100*old['recall']:.2f}% 变为 RACE 的 {100*new['recall']:.2f}%，精确率由 {100*old['precision']:.2f}% 变为 {100*new['precision']:.2f}%。这体现了具体的精确率/召回率权衡；至于原因是否来自篇章关系，需要进一步控制实验。"]
    lines += ["", "同一个基础 RoBERTa 初始化、同一冻结划分、相同平衡 batch、学习率及早停规则，仅解冻最后一个编码层。分类头与 RACE 图结构按官方实现使用；模型容量及可用结构信息并不完全相同。本项目是课程条件下的复现实验，不是论文原始分数的严格重现。", "", "## 独立 HC3 二分类辅助评估", "",
              "1000 条回答（500 个人类 / ChatGPT 题目对）。固定 AI 创作者分数为 P(ai_generated)+P(ai_humanized)，阈值 0.5；不在 HC3 上训练或调阈值。", "",
              "| 模型（种子 3407） | 二分类 Macro-F1 | AUROC | 人工误报率 | AI 召回率 |", "|---|---:|---:|---:|---:|"]
    for run in PRIMARY:
        if run in auxiliary:
            m = auxiliary[run]["metrics"]
            lines.append(f"| {NAMES[family(run)]} | {100*m['macro_f1']:.2f}% | {100*m['auroc']:.2f}% | {100*m['human_false_positive_rate_at_0_5']:.2f}% | {100*m['ai_recall']:.2f}% |")
    lines += ["", "HC3 缺少润色和人类化四分类标签，只支持上述二分类辅助结论。正文长度、来源与生成条件的变化仍是混杂因素；HART 与 HC3 的分数不能直接作为相同任务排序。HC3 原始 index 存在空值；v1.1 仅修复 ID，文本、标签与顺序均保持冻结版本不变。", "", "## 审核与复现", "",
              "summary.json 保存统一重算指标、分域和类别指标、资源记录、多种子摘要、成组 bootstrap 及证据边界。error_cases.jsonl 保存全部主 RACE 错误及其与 RoBERTa 的分歧，不按表现筛选案例。", "",
              "具体四个事后说明案例见 [CASE_STUDIES.md](CASE_STUDIES.md)，包含选择规则、原始样本位置、预测分数、输入/坐标限制和解释候选。HC3 输出分配、固定分数分布及来源/长度分析见 [辅助分析](../auxiliary/REPORT.md)，其中也将 HART 按相同创作者定义折叠为二分类，沿用同一分数/阈值观察跨基准差距；这是事后描述性参考。四个模型输出的分配不能称为 HC3 四分类准确率。", "",
              "图表：main_metrics.png、ablation_2x2.png（矩阵完整后生成）、confusion_matrices.png、domain_metrics.png、learning_curves.png。收敛图展示已完成且重载复核的四个神经配置的全部已完成种子，仅画验证集指标及其选中的 checkpoint；未完成训练不混入结果。不同设备、是否包含解析或模型加载的计时范围不同，不能直接相除解释加速倍数。"]
    lines += ["", "## 计算成本（种子 3407）", "",
              "| 模型 | 训练/验证/保存（秒） | 3194 条测试前向（秒） | CUDA 分配峰值 MiB |", "|---|---:|---:|---:|"]
    for run in DISPLAY_RUNS:
        if run not in analyses:
            continue
        e = analyses[run]["efficiency"]
        if run == "tfidf_lr_v1":
            fit = e["vectorizer_train_fit_plus_val_transform_seconds"] + e["all_candidate_fit_and_validation_seconds"]
            forward, memory = e["test_vectorize_and_predict_seconds"], "CPU"
        else:
            fit, forward, memory = e["training_validation_checkpoint_seconds"], e["test_forward_seconds_excluding_tokenization"], f"{e['peak_cuda_allocated_mib']:.1f}"
        lines.append(f"| {NAMES[family(run)]} | {fit:.2f} | {forward:.2f} | {memory} |")
    lines += ["", "TF-IDF 使用 CPU，测试时间含向量化；神经模型使用 RTX 5070 Ti，测试前向不含 tokenization 或 RST。图缓存构建和原始文本解析另计；表中 CUDA 分配峰值也不包含桌面或独立 parser 进程。"]
    demo_path = REPORTS / "demo_verification_all.json"
    if not demo_path.exists():
        demo_path = REPORTS / "demo_verification_tfidf_roberta_race.json"
    if demo_path.exists():
        demo = json.loads(demo_path.read_text())
        lines += ["", "本机 HTTP Demo 实测（CPU，运行时可能并行训练/解析）：", "",
                  "| 模型 | 该轮首请求（秒） | 再次请求（秒） | 首请求加载 detector / parser（秒） |", "|---|---:|---:|---:|"]
        for name, check in demo["checks"].items():
            first = check.get("first_request", check.get("cold_request"))
            repeat = check.get("repeat_request", check.get("warm_request"))
            a, b = first["timing"], repeat["timing"]
            parser_load = (a["rst_parser_initialization"] or {}).get("model_load_seconds", 0)
            lines.append(f"| {name} | {a['end_to_end_seconds']:.3f} | {b['end_to_end_seconds']:.3f} | {a['detector_model_load_seconds']:.3f} / {parser_load:.3f} |")
        lines += ["", "首请求可能复用更早操作已加载的模型，加载为零时属于缓存请求。这里的端到端时间包含实际需要的模型加载、文本预处理、解析和分类；样例无真实标签，延迟也不是独占资源下的基准。"]
        original_demo_path = REPORTS / "demo_verification_tfidf_roberta_race.json"
        if original_demo_path.exists():
            original_demo = json.loads(original_demo_path.read_text())
            original_first = original_demo["checks"]["race"].get("first_request", original_demo["checks"]["race"].get("cold_request"))
            t = original_first["timing"]
            if t["rst_parser_initialization"]:
                lines += ["", f"较早的真实冷启动记录保存在 {original_demo_path.name}：RACE 短例端到端 {t['end_to_end_seconds']:.3f} 秒，其中 parser 初始化 {t['rst_parser_initialization']['model_load_seconds']:.3f} 秒。后续缓存请求较快，不能据此省略首次加载成本。"]
    if missing:
        lines += ["", "尚未完成并复核：" + "、".join(missing) + "。"]
    if missing_auxiliary:
        lines += ["", "尚未完成的 HC3 辅助评估：" + "、".join(missing_auxiliary) + "。"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "verified_runs": list(analyses), "missing": missing, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
