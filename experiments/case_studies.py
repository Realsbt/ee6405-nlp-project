"""Build traceable qualitative cases using a declared post-hoc selection rule."""

import json
from collections import Counter

import numpy as np

from experiments.common import DATA_ROOT, LABELS, file_hash, read_jsonl, write_json
from experiments.neural import PROJECT_ROOT, load_records


RUNS = ["tfidf_lr_v1", "roberta_s3407", "race_s3407", "race_no_supcon_s3407"]
TITLES = ["AI 到 AI 的改写链被判为人工内容润色", "改写 AI 文本仍被判为 AI 原生文本", "RACE 与 RoBERTa 的一次反向分歧", "RACE 纠正 RoBERTa 的一次错误"]
COMMENTS = [
    "文本讨论 few-shot prompt-tuning，并保留学术论述和引文的表达形式。发布标签把该 AI→AI 改写链归为 ai_generated；RoBERTa 和 RACE 均高分选择 human_ai_polished。该例正文完整、坐标覆盖干净，因此不能简单归因于 512-token 截断或已知跨度漏字。保留的论述形式可能与训练中的润色样本相似，这是待验证的解释候选；高输出分数不等于真实创作流程的可靠概率。",
    "文本讨论 Fokker–Planck 方程的多领域应用，发布标签为工具改写 AI。TF-IDF 在该例正确，两个神经模型均选择 ai_generated，说明总体较强的模型也存在互补错误。词汇窗口被截断，因此只能说明本窗口与所用结构信息下的失败；不能据此判断完整正文有没有编辑线索，或把所有工具改写视为可检测。",
    "文本讨论社交网络对新闻传播与信任机制的影响。RoBERTa 正确选择 human_ai_polished，RACE 更偏向 ai_generated，其两类分数相对接近。RACE 使用更多结构信息仍可能出错；模型容量、pooling、损失和训练随机性也不同，单个分歧不足以证明某条篇章关系导致错误。",
    "文本讨论非人类解释者与人工智能语境下的符号学。该工具改写 AI 样本中，RACE 的标签正确而 RoBERTa、TF-IDF 选择 ai_generated。它展示当前模型的互补性；单例不能把成功归因于图结构机制，也不能抵消整体成对区间包含零的事实。",
]


def build_case_studies(require_complete=False):
    reports = PROJECT_ROOT / "experiments/reports"
    records = {row["item_id"]: row for row in load_records("test")}
    errors = read_jsonl(reports / "comparison/error_cases.jsonl")
    pairs = [("ai_generated", "human_ai_polished"), ("ai_humanized", "ai_generated"), ("human_ai_polished", "ai_generated")]
    counts = Counter((row["true_label"], row["predictions"]["race_s3407"]) for row in errors if row["race_error"])
    assert [pair for pair, _ in counts.most_common(3)] == pairs, "Declared illustration rule must match the actual error counts"
    selected = [next(row for row in errors if row["true_label"] == a and row["predictions"]["race_s3407"] == b) for a, b in pairs]
    selected.append(next(row for row in errors if row["predictions"]["race_s3407"] == row["true_label"] and row["predictions"]["roberta_s3407"] != row["true_label"]))
    ids = [row["item_id"] for row in selected]
    quality = {row["item_id"]: row for row in read_jsonl(reports / "released_rst_item_quality.jsonl") if row["item_id"] in ids}
    predictions, missing = {}, []
    for run in RUNS:
        folder = reports / run
        if not (folder / "result.json").exists() or not (folder / "verification.json").exists():
            missing.append(run)
            continue
        result = json.loads((folder / "result.json").read_text())
        verification = json.loads((folder / "verification.json").read_text())
        assert result["status"] == "completed" and verification["status"] == "passed"
        assert result["data_protocol_sha256"] == file_hash(DATA_ROOT / "protocol.json")
        predictions[run] = {row["item_id"]: row for row in read_jsonl(folder / "test_predictions.jsonl") if row["item_id"] in ids}
        assert set(predictions[run]) == set(ids)
    if require_complete and missing:
        raise RuntimeError(f"Case-study methods still missing: {missing}")
    cases = []
    for index, selected_row in enumerate(selected):
        item_id = selected_row["item_id"]
        row = records[item_id]
        outputs = {}
        for run, saved in predictions.items():
            record = saved[item_id]
            scores = np.asarray([record["probabilities"][label] for label in LABELS])
            assert record["true_label"] == row["label"] and record["prediction"] == LABELS[int(scores.argmax())]
            outputs[run] = {"prediction": record["prediction"], "four_class_scores": record["probabilities"],
                            "true_label_score": float(record["probabilities"][row["label"]])}
        cases.append({"item_id": item_id, "title": TITLES[index], "true_label": row["label"],
                      "input_truncated": row["input_truncated"], "lexical_tokens_with_specials": row["input_token_count_with_specials"],
                      "article_characters": len(row["article"]), "rst_coordinate_quality": quality[item_id],
                      "source_file": row["source_file"], "source_line": row["source_line"],
                      "normalized_text_sha256": row["normalized_text_sha256"], "outputs": outputs,
                      "comment": COMMENTS[index]})
    report = {"status": "completed" if not missing else "partial", "missing_runs": missing,
              "selection": "Post-hoc illustration: first item ID in each of the three largest RACE seed-3407 error pairs; then first item ID among RACE-correct/RoBERTa-wrong cases. Does not select models, thresholds, partitions or primary metrics.",
              "cases": cases, "full_error_list_sha256": file_hash(reports / "comparison/error_cases.jsonl"),
              "data_protocol_sha256": file_hash(DATA_ROOT / "protocol.json"), "script_sha256": file_hash(__file__)}
    write_json(reports / "comparison/case_studies.json", report)
    lines = ["# 可追溯的错误与互补案例", "", f"状态：{report['status']}。案例来自种子 3407 的固定测试结果。", "",
             "选择规则属于事后说明：在三个最多的 RACE 错误标签对中各取 item ID 排序第一条，再取 RACE 分对且 RoBERTa 分错的第一条。它们用于理解，不是随机代表性样本或新的准确率估计，不参与选模、调阈值、改划分。全文可按 case_studies.json 中的原文件和行号核查。"]
    for index, case in enumerate(cases, 1):
        q = case["rst_coordinate_quality"]
        clean = q["valid_and_ordered_spans"] and q["all_nonspace_covered"]
        lines += ["", f"## {index}. {case['title']}", "", f"样本：`{case['item_id']}`。", "",
                  f"发布真值 `{case['true_label']}`；词汇窗口 {case['lexical_tokens_with_specials']} tokens，{'已截断' if case['input_truncated'] else '覆盖完整正文'}；坐标覆盖{'干净' if clean else '存在问题'}。坐标干净不证明篇章关系语义正确。", "",
                  "| 模型 | 预测标签 | 对发布真值的分数 |", "|---|---|---:|"]
        for run, output in case["outputs"].items():
            lines.append(f"| {run} | {output['prediction']} | {100*output['true_label_score']:.2f}% |")
        lines += ["", case["comment"]]
    lines += ["", "这些分数未经概率校准。真值来自数据集的操作标签，案例阅读不重新判定作者身份。关于可能原因的文字是解释候选；验证原因需要新设计的控制实验。"]
    (reports / "comparison/CASE_STUDIES.md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    result = build_case_studies()
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]), "missing": result["missing_runs"]}, indent=2))
