# CPU 执行结果

2026-11-06；本轮没有使用GPU，没有新训练，没有修改或推送原仓库。

## 已实际完成

| 工作 | 验收结果 | 产物 |
|---|---|---|
| 原有预测一致性复核 | 13组HART、4组HC3、配对消融全部通过 | 原仓库 `scripts/check_results.py` 执行输出 |
| 概率误差与可靠性分析 | 完成，原预测未修改 | `saved_probability_audit_20261106.json` |
| HC3问题对分层bootstrap | 500问题对、五来源、5000次 | `hc3_brier_bootstrap_20261106.json` |
| TF-IDF checkpoint CPU重放 | 1000条，四分类argmax零变化，最大概率差5.55e-16 | `tfidf_hc3_cpu_replay.json`及对应预测 |
| RoBERTa checkpoint CPU重放 | 1000条，四分类argmax零变化，最大概率差1.08e-5 | `roberta_hc3_cpu_replay.json`及对应预测 |
| 原仓库单元测试 | 15/15通过 | ablation、protocol、neural_data、rst_alignment |
| 新分析单元测试 | 6/6通过 | `test_saved_probabilities.py` |
| Bootstrap自检 | 零差异、恒定差异、确定性均通过 | `bootstrap_hc3_brier.py --self-test` |

TF-IDF的向量化与推理约0.125秒，RoBERTa的分词与推理约151.868秒；两者计时均不含模型加载，不含下载。它们是本机此次运行记录，不是独占资源的性能基准。环境为macOS arm64、Python3.12、PyTorch2.7.0 CPU，与原Linux Python3.11环境不同。

## 研究上新增的信息

在预先固定的AI创作者二分类定义下，四种现有模型在HC3上的Brier误差都高于恒定0.5概率基线。问题对bootstrap也支持该固定样本与固定模型下的差距；不能将之推广为所有数据上的结论。

RACE CE+SupCon相对RoBERTa CE的Brier差为+0.05478，95%条件区间[+0.04037,+0.06989]。RACE CE相对RoBERTa CE的区间包含零。因此“分类性能退化”和“概率输出不可靠”应同时讨论，但不要把二者混为同一种指标。

## 本轮新增：来源审计与外部评测

恢复并校验了HART的train/val/test原始图数据，重建后的分区哈希与归档一致。15,984条实际使用样本全部能与原始来源元数据匹配；原始发布16,000条另有12条元数据歧义，未出现在保留分区。ai_humanized的1,015条中，765条记录为工具改写、250条记录为真人改写。

MixSet保守映射子集在预测前冻结：359条、91来源组；删除9个标签冲突组，排除不兼容编辑操作和重复文本。与原始英文HART及冻结HC3无规范化精确文本重叠；未检查语义近重复。不能将此子集称为原始MixSet标准四分类benchmark。

TF-IDF与RoBERTa CE3407分别对HART验证集1,600条、MixSet359条做新CPU推理。HART测试及HC3的以下指标读取已有归档概率，HC3此前已独立重放验证。阈值只用HART验证集800个人类创作者负类，目标经验FPR不超过1%，分别为0.71320667与0.91252685。

| 模型/数据 | 四分类Macro-F1 | AI creator AUROC | 锁阈值实际FPR | 锁阈值TPR |
|---|---:|---:|---:|---:|
| TF-IDF / HART | 0.7940 | 0.9841 | 1.19% | 87.73% |
| RoBERTa / HART | 0.9075 | 0.9931 | 1.50% | 92.80% |
| TF-IDF / HC3 | 不适用 | 0.6011 | 0.60% | 1.00% |
| RoBERTa / HC3 | 不适用 | 0.6109 | 0.60% | 5.80% |
| TF-IDF / MixSet子集 | 0.3163 | 0.3476 | 5.20% | 1.83% |
| RoBERTa / MixSet子集 | 0.3901 | 0.4109 | 16.40% | 6.42% |

MixSet按91来源组配对bootstrap，2,000次、固定种子20261106：RoBERTa Macro-F1条件95%区间[0.3450,0.4363]，TF-IDF为[0.2734,0.3629]；配对差值区间[0.0307,0.1148]。RoBERTa的FPR条件区间[10.54%,22.80%]。这些区间固定checkpoint和阈值，不含训练种子或校准选择不确定性，也未做多重比较校正。

完整输出：`transfer_tfidf_v1/`、`transfer_roberta_v1/`、`transfer_uncertainty_v1.json`。新增评测6项测试与旧分析6项测试合计12项通过。

在研究目录运行（输出已存在时使用新目录名，不覆盖证据）：

```bash
.venv-cpu/bin/python -B -m unittest -v test_research_evaluation.py test_saved_probabilities.py
.venv-cpu/bin/python -B run_transfer_evaluation.py --model tfidf --output transfer_tfidf_repeat
.venv-cpu/bin/python -B run_transfer_evaluation.py --model roberta --output transfer_roberta_repeat
```

`prepare_research_data.py`复现审计与数据冻结；`summarize_transfer.py`复现保留来源统计及成组区间，默认拒绝覆盖已有报告。新结果属于事后扩展；没有将MixSet用于训练、调参或阈值选择。

## 后续完成：RACE与校准

原先的parser资源缺口已经补齐。RACE主配置seed3407完成HART验证集1,600条推理、测试集3,194条重放，以及HC3的1,000条全文重新解析与端到端推理。两份重放均passed，argmax零变化，最大概率差分别2.4694e-5和6.2304e-5；HC3全部文本无未覆盖非空白字符。

TF-IDF、RoBERTa CE3407与RACE CE+SupCon3407均完成HART验证集温度校准和测试评测。RACE温度1.6718，HART/HC3 Brier分别由0.02862/0.48048变为0.02726/0.43802。域内概率质量有所改善，但HC3校准后锁阈值实际FPR 3.2%、TPR 1.2%，不能说已解决外部泛化。

详见 `08_RACE重放与验证集校准.md` 与 `cpu_calibration_completed_v1.json`。本轮31项单元测试通过，原仓库/上游代码未改动，未使用GPU。

## 未完成且不能冒充GPU阻塞的事项

- RACE其他种子/CE配置的重放，以及RACE的MixSet推理：尚未执行，不再以缺parser为理由。
- 独立校准划分：本轮复用了曾参与选模的HART验证集；下一轮训练需单独冻结开发/校准来源组。
- HART来源审核的人工抽查与语义近重复：元数据逐条匹配已完成，不能因此宣称所有污染均已排除。
- 其他神经配置重放：后补五个checkpoint未包含在已发布Release中；已发布的7个神经checkpoint中，已重放RoBERTa CE3407及RACE CE+SupCon3407，其余5个尚未重放。

这些任务属于CPU可执行但尚缺资源或尚未执行的队列；同窗口结构重训和留域训练才放到GPU实验队列，详见 `02_GPU与CPU任务清单.md`。
