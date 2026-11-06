"""Describe frozen HC3 predictions without fitting or changing thresholds."""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import DATA_ROOT, LABELS, file_hash, read_jsonl, write_json
from experiments.evaluate_external import binary_metrics
from experiments.neural import PROJECT_ROOT, load_records


RUNS = ["tfidf_lr_v1", "roberta_s3407", "race_s3407", "race_no_supcon_s3407"]
NAMES = ["TF-IDF + LR", "RoBERTa", "RACE", "RACE w/o SupCon"]


def summarize(require_complete=False):
    reports = PROJECT_ROOT / "experiments/reports"
    source = DATA_ROOT.parent / "hc3_aux_v1_1"
    protocol = json.loads((source / "protocol.json").read_text())
    assert file_hash(source / "test.jsonl") == protocol["answer_sha256"]
    rows = read_jsonl(source / "test.jsonl")
    y = np.asarray([row["binary_label"] for row in rows])
    lengths = np.asarray([len(row["article"].split()) for row in rows])
    hart_rows = load_records("test")
    hart_y = np.asarray([int(row["label_id"] >= 2) for row in hart_rows])
    analyses, missing = {}, []
    score_arrays = {}
    for run in RUNS:
        directory = reports / run
        path = directory / "hc3_aux_v1_1_predictions.jsonl"
        result_path = directory / "hc3_aux_v1_1_result.json"
        if not path.exists() or not result_path.exists():
            missing.append(run)
            continue
        original = json.loads(result_path.read_text())
        assert original["status"] == "completed" and original["protocol_sha256"] == file_hash(source / "protocol.json")
        saved = read_jsonl(path)
        assert [row["item_id"] for row in saved] == [row["item_id"] for row in rows]
        assert [row["true_binary_label"] for row in saved] == y.tolist()
        probabilities = np.asarray([[row["four_class_scores"][label] for label in LABELS] for row in saved])
        assert probabilities.shape == (1000, 4) and np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(1), 1)
        scores = probabilities[:, 2] + probabilities[:, 3]
        assert np.allclose(scores, [row["ai_creator_score"] for row in saved])
        metrics = binary_metrics(y, scores)
        assert all(abs(metrics[key] - original["metrics"][key]) < 1e-10 for key in ("macro_f1", "auroc", "accuracy"))
        trained = json.loads((directory / "result.json").read_text())
        assert trained["status"] == "completed"
        assert original["detector_sha256"] == trained.get("checkpoint_sha256", trained.get("model_sha256"))
        hart_saved = read_jsonl(directory / "test_predictions.jsonl")
        assert [row["item_id"] for row in hart_saved] == [row["item_id"] for row in hart_rows]
        assert [row["true_label"] for row in hart_saved] == [row["label"] for row in hart_rows]
        hart_probabilities = np.asarray([[row["probabilities"][label] for label in LABELS] for row in hart_saved])
        assert hart_probabilities.shape == (3194, 4) and np.isfinite(hart_probabilities).all()
        assert np.allclose(hart_probabilities.sum(1), 1)
        hart_binary = binary_metrics(hart_y, hart_probabilities[:, 2] + hart_probabilities[:, 3])
        allocation = np.asarray([[int(((probabilities.argmax(1) == label) & (y == target)).sum()) for label in range(4)] for target in (0, 1)])
        assert allocation.sum(1).tolist() == [500, 500]
        by_length = {}
        for name, mask in (("at_most_100_words", lengths <= 100), ("101_to_200_words", (lengths > 100) & (lengths <= 200)), ("over_200_words", lengths > 200)):
            by_length[name] = {"human_answers": int((y[mask] == 0).sum()), "chatgpt_answers": int((y[mask] == 1).sum()),
                               "metrics": binary_metrics(y[mask], scores[mask]) if len(np.unique(y[mask])) == 2 else None}
        analyses[run] = {"binary_metrics": metrics, "hart_aligned_binary_reference": hart_binary,
                         "binary_label_to_predicted_class_counts": allocation.tolist(),
                         "predicted_class_order": LABELS, "binary_label_order": ["human", "ChatGPT"],
                         "score_quantiles_by_binary_label": {name: np.quantile(scores[y == value], [0, .25, .5, .75, 1]).tolist() for value, name in enumerate(("human", "ChatGPT"))},
                         "by_answer_length": by_length, "per_source": original["per_source"],
                         "hart_predictions_sha256": file_hash(directory / "test_predictions.jsonl"),
                         "predictions_sha256": file_hash(path)}
        score_arrays[run] = scores
    if require_complete and missing:
        raise RuntimeError(f"HC3 methods still missing: {missing}")
    out = reports / "auxiliary"
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "completed" if not missing else "partial", "missing_runs": missing,
              "protocol_id": protocol["protocol_id"], "protocol_sha256": file_hash(source / "protocol.json"),
              "threshold_unchanged": .5, "score_unchanged": "P(ai_generated)+P(ai_humanized)",
              "descriptive_analysis_only": True, "analyses": analyses,
              "hart_reference": {"samples": len(hart_rows), "human_creator": int((hart_y == 0).sum()), "ai_creator": int((hart_y == 1).sum()),
                                 "test_file_sha256": file_hash(DATA_ROOT / "test.jsonl"),
                                 "binary_label_collapse": "human_written/human_ai_polished -> 0; ai_generated/ai_humanized -> 1",
                                 "post_hoc_descriptive_reference_not_selection": True},
              "interpretation_boundary": "Rows have binary authorship labels only. Allocation to four model outputs is not a four-class confusion matrix or four-class accuracy. Length/source slices are confounded descriptive groups, not controlled interventions.",
              "analysis_script_sha256": file_hash(__file__)}
    write_json(out / "summary.json", report)
    available = [run for run in RUNS if run in analyses]
    if available:
        fig, axes = plt.subplots(len(available), 2, figsize=(10, 3.05 * len(available)), squeeze=False)
        for index, run in enumerate(available):
            counts = np.asarray(analyses[run]["binary_label_to_predicted_class_counts"])
            ax = axes[index, 0]
            ax.imshow(counts / 500, vmin=0, vmax=1, cmap="Greens", aspect="auto")
            for a in range(2):
                for b in range(4):
                    ax.text(b, a, f"{counts[a,b]}\n{counts[a,b]/500:.1%}", ha="center", va="center", fontsize=9,
                            color="white" if counts[a,b] > 300 else "#173b2f")
            ax.set_xticks(range(4), ["Human", "AI-polished", "AI-generated", "AI-humanized"], rotation=15, ha="right")
            ax.set_yticks([0, 1], ["Human answer", "ChatGPT answer"])
            ax.set_title(NAMES[RUNS.index(run)] + " · prediction allocation", fontsize=10)
            ax = axes[index, 1]
            for label, color, name in ((0, "#4f7f91", "Human answer"), (1, "#c18b54", "ChatGPT answer")):
                ax.hist(score_arrays[run][y == label], bins=np.linspace(0, 1, 21), alpha=.65, color=color, label=name)
            ax.axvline(.5, linestyle="--", color="#333333", linewidth=1, label="Fixed threshold 0.5")
            ax.set_xlabel("P(AI creator) model score (not calibrated)")
            ax.set_ylabel("Answer count")
            ax.set_title("Binary score distribution", fontsize=10)
            ax.legend(fontsize=8)
        fig.suptitle("HC3 auxiliary binary labels · four-output allocation is not four-class ground truth", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(out / "hc3_prediction_analysis.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
        x = np.arange(len(available))
        for ax, key, title in zip(axes, ("macro_f1", "auroc"), ("Binary Macro-F1", "Binary AUROC")):
            for offset, field, color, name in ((-.18, "hart_aligned_binary_reference", "#137f73", "HART grouped holdout"),
                                               (.18, "binary_metrics", "#c18b54", "HC3 auxiliary")):
                values = [analyses[run][field][key] * 100 for run in available]
                ax.bar(x + offset, values, width=.35, color=color, label=name)
                for i, value in enumerate(values):
                    ax.text(i + offset, value + 1.3, f"{value:.1f}", ha="center", fontsize=8)
            ax.set_xticks(x, [NAMES[RUNS.index(run)] for run in available], rotation=15, ha="right")
            ax.set_ylim(0, 106)
            ax.set_title(title)
            ax.set_ylabel("Percent")
            ax.legend(fontsize=8, loc="lower left")
        fig.suptitle("Same AI-creator score; classification threshold 0.5 · seed 3407", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "aligned_binary_metrics.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
    lines = ["# HC3 辅助评估的输出与错误分析", "", f"状态：{report['status']}。同一冻结 1000 条回答、同一分数映射和 0.5 阈值，不重新拟合或选阈值。", "",
             "下面只分析模型输出。HC3 没有四类创作/编辑标签，因此不能把四个输出的分配称为四分类准确率或混淆矩阵。", "",
             "| 模型 | ChatGPT 回答被判为人工原文 | 被判为 AI 润色人工 | 被判为 AI 生成 | 被判为改写 AI |", "|---|---:|---:|---:|---:|"]
    for run in available:
        counts = analyses[run]["binary_label_to_predicted_class_counts"][1]
        lines.append("| " + NAMES[RUNS.index(run)] + " | " + " | ".join(f"{value}/500 ({value/5:.1f}%)" for value in counts) + " |")
    if "roberta_s3407" in analyses:
        a = analyses["roberta_s3407"]
        count = a["binary_label_to_predicted_class_counts"][1][1]
        lines += ["", f"观察：RoBERTa 将 {count}/500 条 ChatGPT 回答分到 human_ai_polished。当前固定二分类分数将该输出归在人类创作者一侧，因此大量 AI 回答没有跨过 0.5。其 AUROC 为 {100*a['binary_metrics']['auroc']:.2f}%，排序本身也较弱，不能将低召回率完全归为阈值问题。", "",
                  "解释候选：HART 的创作/编辑操作标签、文体及生成条件与 HC3 问答不同，模型可能把某些 AI 问答表面模式映射到训练中的润色类别。这不是已证实的真实创作流程，也没有证明某一种表面特征导致错误。将 AI-polished 改算为 AI 会改变所评子任务，本项目不借此重定义主要结果。"]
    if "race_s3407" in analyses and "race_no_supcon_s3407" in analyses:
        full, ablated = analyses["race_s3407"]["binary_metrics"], analyses["race_no_supcon_s3407"]["binary_metrics"]
        lines += ["", f"本次辅助样本中，完整 RACE AUROC 为 {100*full['auroc']:.2f}%，关闭监督对比损失为 {100*ablated['auroc']:.2f}%；固定阈值 Macro-F1 分别为 {100*full['macro_f1']:.2f}% / {100*ablated['macro_f1']:.2f}%。辅助排序方向与 HART 单次四分类消融不同，不能由 HART 的损失收益推出独立数据上也有同样收益。这里仍只有一个拟合种子和一份辅助抽样。", "",
                  f"完整 RACE 的固定阈值 AI 召回率 {100*full['ai_recall']:.2f}%，人工误报率 {100*full['human_false_positive_rate_at_0_5']:.2f}%。即使 AUROC 高于某些对照，当前阈值行为也不足以支持可靠的实际作者归因。"]
    lines += ["", "## 使用相同二分类定义的 HART 参考", "",
              f"把完整 HART 保留测试的四类仅按创作者折叠为人类/AI（{int((hart_y == 0).sum())}/{int((hart_y == 1).sum())} 条），仍用同一已拟合模型、P(ai_generated)+P(ai_humanized) 和阈值 0.5。以下为事后描述性重算，未训练、调阈值或选 checkpoint；不替代主要四分类指标。", "",
              "| 模型（3407） | HART 二分类 Macro-F1 | HC3 二分类 Macro-F1 | HART 二分类 AUROC | HC3 二分类 AUROC |", "|---|---:|---:|---:|---:|"]
    for run in available:
        a, b = analyses[run]["hart_aligned_binary_reference"], analyses[run]["binary_metrics"]
        lines.append(f"| {NAMES[RUNS.index(run)]} | {100*a['macro_f1']:.2f}% | {100*b['macro_f1']:.2f}% | {100*a['auroc']:.2f}% | {100*b['auroc']:.2f}% |")
    lines += ["", "aligned_binary_metrics.png 将评分定义对齐。AUROC 使用分数的排序，0.5 只用于分类指标。两个基准的文体、生成条件和编辑类别混合仍不同：HART 包含润色和改写过程，HC3 只有人类/ChatGPT 回答；HART 使用发布的 RST，HC3 使用新解析的 RST。差距说明当前跨基准行为变化，不能单独归因于领域、生成模型、parser 或某一种因素。"]
    lines += ["", "## 长度与来源", "",
              "summary.json 保存不超过 100 词、101–200 词、超过 200 词的分组计数和固定二分类指标，以及五个来源的指标。长度组的人类/AI 比例不同，来源也同时变化，所以这些分组只用于描述数据差异，不能把分数差直接解释为长度的因果影响。", "",
              "原始冻结抽样按问题配对、每来源 100 对。人工与 AI 回答的中位词数不同，没有通过选择更容易的样本或截断某一类来平衡表现。近重复、底座预训练暴露及未标明编辑过程仍未排除。", "",
              "图表 hc3_prediction_analysis.png 左列显示在二分类真值条件下的四个输出分配，右列显示固定 AI 创作者分数分布。所有主要模型仍应以 HART 四分类和 HC3 二分类分别报告。"]
    if missing:
        lines += ["", "待完成：" + "、".join(missing) + "。"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = summarize(args.require_complete)
    print(json.dumps({"status": report["status"], "missing": report["missing_runs"]}, indent=2))


if __name__ == "__main__":
    main()
