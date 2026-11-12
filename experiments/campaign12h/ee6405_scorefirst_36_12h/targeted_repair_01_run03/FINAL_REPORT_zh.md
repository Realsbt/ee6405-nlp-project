# EE6405 targeted repair v1 最终报告

## 结论

本轮只执行了既定的 B0/E26 有限融合分析、E16 概率诊断和 E30
精确恢复检查。没有训练、下载、部署，也没有读取 HART test 或 HC3。

**结论：`INSUFFICIENT_GATE_SUPPORT`，保留 B0；不保存可部署规则。**

固定门控为 `B0 margin < 0.10 且 B0/E26 预测不同`。960 条 S 样本中
gate 数为 0、覆盖来源组数为 0，因此三折交叉拟合没有可执行的候选门。

## B0 与 E26

口径：`S_historical_dev_selected`，同一 S 的 960 条样本；这是历史开发集
复用描述，不是独立盲测。

| 方法 | Accuracy | Macro-F1 | J | 正确数 | C 纠错 | R 误伤 | 净变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 94.3750% | 92.5641% | 93.4696% | 906 | - | - | - |
| E26 | 94.4792% | 92.7021% | 93.5906% | 907 | 3 | 2 | +1 |

E26 与 B0 仅有 6 个预测分歧；3 个 wrong-to-right、2 个
right-to-wrong，另有 1 个 both-wrong 变化。完整 S 固定 lambda
诊断中 gate 为空，所以 lambda=0、0.25、0.5、1.0 的结果完全相同；
该曲线仅为诊断，不是 OOF 成绩。

## 三折 lambda / OOF

| Fold | 选择 lambda | gate 样本 | gate 组数 | 变化 | 状态 |
|---:|---:|---:|---:|---|---|
| 0 | N/A | 0 | 0 | 未预测 | BLOCKED |
| 1 | N/A | 0 | 0 | 未预测 | BLOCKED |
| 2 | N/A | 0 | 0 | 未预测 | BLOCKED |

合并 OOF 没有生成；`crossfit.json` 明确为
`INSUFFICIENT_GATE_SUPPORT`，使用 B0 baseline fallback。不能把完整
S 的诊断曲线或 B0 fallback 称为 OOF 提升。

## E16

本轮真实训练次数为 0。现有 E16 概率诊断：

- Accuracy：82.7083%
- Macro-F1：69.7720%
- `ai_humanized` 排名：第1/2/3/4分别为 9/41/8/1
- U Top-2 recall：84.7458%
- U OvR AUROC：87.9174%
- U vs G log-ratio AUROC：84.4760%
- 44 个真实 U 被预测为 G
- U 全部错误被 oracle 改正的理论 Accuracy：87.9167%

历史已知：4 epochs、2800 updates、best epoch 3、encoder LR
`1e-5`、head LR `5e-5`、weight decay `0.01`、gradient checkpointing。

历史未知：正式 scheduler 类和参数、10% warmup 是否执行、每个参数组
的名称/数量、参数重复或遗漏、训练时源码快照/hash、完整 LR 序列和
逐 epoch loss。`schedule_reference_NOT_HISTORY.json` 只是理论参考，
不能作为历史运行证据。

## E30

状态：`NO_HISTORICAL_RECIPE_NOT_REFIT`。本轮没有找到可验证的原始
成员、权重、lock 和拟合状态组合，因此没有重建 E30，也没有重新拟合。
历史 E30 仅作为未完整复现候选保留，不能当作性能失败。

## R00—R12 状态

| 任务 | 状态 | 实际耗时/证据 |
|---|---|---|
| R00 | PASS | 预算建立；workspace snapshot 已保存 |
| R01 | PASS | 1.09s；B0/E26/E16 均为 ID-bound |
| R02 | PASS | 规范化 1.09s；960 条、真值和 label order 独立校验 |
| R03 | PASS | 1.04s；指标/混淆矩阵/NLL 重算 |
| R04 | PASS | disagreement audit 已生成 |
| R05 | PASS | 0.01s 级 CPU 分析；四个固定 lambda 全部诊断 |
| R06 | BLOCKED | gate=0，未生成三折 OOF |
| R07 | PASS | 固定规则结论为保留 B0 |
| R08 | PASS | E16 rank/ROC/AP/U 倍数诊断已生成 |
| R09 | NOT_VERIFIED | 历史 optimizer/scheduler/source 记录缺失 |
| R10 | PASS | 精确恢复检查完成，结果为不可恢复且未 refit |
| R11 | PASS | 固定 S 描述榜与 OOF 口径分离 |
| R12 | PASS | 本报告、状态和无正文证据包已输出 |

## 证据路径

- 运行目录：[targeted_repair_01_run03](.)
- 规范化输入：[inputs](./inputs)
- 分析结果：[results](./results)
- 最终报告：[FINAL_REPORT_zh.md](./FINAL_REPORT_zh.md)
- 无正文证据包：[evidence_no_text.zip](./evidence_no_text.zip)

证据包只包含绑定元数据、概率、指标、日志和配置，不包含原始文章正文。
