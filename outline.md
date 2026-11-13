# Presentation Outline: Group Conformal Prediction and Uncertainty Review

## 1. Contribution scope / 本次贡献

- Add a post-processing module to the existing four-class text-provenance classifiers.
- Return a set of candidate labels and a `needs_review` flag, alongside the original argmax prediction.
- Evaluate the coverage–automation trade-off using saved predictions from all 13 published model runs.
- No classifier retraining, new checkpoint inference, or integration into the existing web interface was performed.

中文：在原模型基础上增加“候选类别集合＋不确定时人工复核”。本次做的是后处理模块与回顾性评估，不是重新训练分类器。

## 2. Motivation / 为什么做

- A highest-scoring label does not tell us whether other labels are also plausible.
- Texts with uncertain provenance should be allowed to receive multiple candidate labels.
- Evaluate both prediction quality and how often the system can return a single candidate.

中文：原模型每次都选一个最高分类；新增模块让模型在不确定时保留多个可能类别，供后续复核。

## 3. Method / 如何实现

1. Keep the model and its four-class scores fixed.
2. Split source groups into disjoint calibration and evaluation partitions using a fixed hash-based rule, independent of labels and scores.
3. Calculate each calibration group's score as the maximum `1 - p(true_label)` across its related texts.
4. Set alpha to 0.10 and compute a finite-sample-corrected quantile threshold.
5. Include every label whose `1 - p(label)` is at or below that threshold.
6. Return a single candidate when the set size is one; flag multiple-label and empty sets for review.

中文：同源文章的多个改写版本具有相关性，因此按来源组划分，并让每组只贡献一个最大不符合度。预测新文章时只需模型分数，不需要真值或同组其他文章。

## 4. Experiment / 实验设置

- Input: 3,194 saved HART test predictions per model.
- Calibration: 399 source groups, 1,594 texts.
- Evaluation: 400 source groups, 1,600 texts.
- Identical split and alpha for all 13 model runs; no model or split selection based on these results.
- This is a retrospective pilot on previously inspected test predictions, not a fresh independent test.

## 5. Main example / 展示结果

Fixed example: original RACE, seed 3407.

| Metric | Result | Interpretation |
|---|---:|---|
| Original top-1 accuracy on all evaluation texts | 93.06% | All 1,600 evaluation texts |
| Original top-1 Macro-F1 on all evaluation texts | 90.34% | Separately recomputed from saved predictions |
| Singleton rate | 88.06% | 1,409 texts receive one candidate |
| Singleton accuracy | 96.17% | Accuracy on those 1,409 texts only |
| Singleton Macro-F1 | 93.84% | Separately recomputed on that selected subset |
| Review rate | 11.94% | 191 texts require review |
| Label-set coverage | 96.31% | True label is included in the candidate set |
| Simultaneous group coverage | 86.50% | All true labels in a source group are covered |

中文：约 88% 的文本返回唯一候选，这部分准确率约 96%；约 12% 需要复核。筛选前后的分母不同，不能宣称整体准确率或整体 Macro-F1 因此提高。

Note: The generated REPORT.md lists accuracy and coverage metrics. The two Macro-F1 values above were calculated separately and are not columns in that report.

## 6. Validation and limitations / 验证与局限

- Seven new unit tests pass; they cover group isolation, finite-sample quantiles, invalid inputs and edge cases.
- The user also ran the tests and the 13-run analysis successfully on their local Windows machine.
- Original saved-result consistency checks pass for 13 HART runs and four HC3 evaluations in the delivered package.
- Actual RACE group coverage is 86.50%, below the nominal 90% target; this result must be reported.
- The theoretical group-coverage target relies on exchangeable groups and a model fixed before calibration. The retrospective experiment cannot establish prospective validity.
- There is no per-class, per-domain, or singleton-accuracy guarantee. HC3 generalization is not established.
- Future validation requires independent source-disjoint calibration data and a fresh evaluation set.

## 7. Code and evidence / 演示文件

- `experiments/group_conformal.py`: calibration, prediction-set API and evaluation CLI.
- `scripts/run_conformal_review.py`: reproducible 13-run retrospective analysis.
- `tests/test_group_conformal.py`: seven unit tests.
- `experiments/reports/my_conformal_run/REPORT.md`: locally reproduced result table.
- `experiments/reports/my_conformal_run/race_s3407/prediction_sets.jsonl`: per-text candidates and review flags.
- `docs/CONFORMAL_REVIEW.zh-CN.md`: detailed Chinese explanation and reproducibility instructions.

## 8. Suggested spoken script / 英文讲稿参考

My contribution adds uncertainty handling to the existing text-provenance classifiers. Instead of always presenting one label, the module returns a set of plausible labels and flags ambiguous cases for review.

I used group-based split conformal prediction, because multiple rewritten texts can share the same source. The classifier stays fixed, and calibration uses one maximum nonconformity score per source group.

For the original RACE model with seed 3407, about 88 percent of the 1,600 evaluation texts received one candidate. Accuracy within that subset was about 96 percent, while about 12 percent required review. These are different denominators, so this is not an increase in overall classification accuracy.

All seven unit tests and the analysis of thirteen published runs completed successfully. However, this is a retrospective experiment using previously inspected predictions. RACE's observed group coverage was below the nominal target. Independent calibration data and a fresh evaluation set are still needed before making deployment claims.

中文提示：根据小组分配时长精简讲稿，确保自己理解并能解释分母差异、分组校准以及为什么不需要训练。贡献说明与 GAI declaration 应如实反映 AI 辅助。

## Reference

Angelopoulos, A. N., & Bates, S. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. https://arxiv.org/abs/2107.07511
