# 架构 × 监督对比目标：2×2 消融

本轮在查看原有结果后补全实验矩阵。保留原先七次神经训练，增加五次训练；没有改数据划分、测试集、超参数或选模规则。补充实验不是新增独立测试，也没有预先注册显著性检验。

| 架构 | CE | CE + SupCon |
|---|---|---|
| RoBERTa | `roberta_s{seed}`，已有三个种子 | `roberta_supcon_s{seed}`，新增三个种子 |
| RACE | `race_no_supcon_s{seed}`，保留 3407，新增 42、2026 | `race_s{seed}`，已有三个种子 |

每格使用种子 3407、42、2026。当前完成状态、结果与逐种子差异以[统一报告](../experiments/reports/comparison/REPORT.md)和对应运行的 `result.json` / `verification.json` 为准；计划和 smoke 通过不代表正式实验完成。

## 固定条件

- 使用 `hart_grouped_v1` 的既有分组划分，训练 11,190、验证 1,600、测试 3,194 条；读入时核对分区哈希。
- 沿用 `neural_v1.json`：同一基础预训练权重，只解冻最后一层，batch 16、每类 4 条，学习率 2.9e-5、最多 20 轮、patience 5。
- 仅按验证 Macro-F1 选择 checkpoint。完成训练后才读取测试分区，再独立重载最佳 checkpoint 核对全部测试预测。
- SupCon 温度 0.07、权重 1.0，使用已有官方损失；RoBERTa 采用归一化的 CLS 特征，RACE 采用归一化的 root 特征。没有新增投影头或调整分类头。
- 原有完整运行禁止覆盖；新增运行保存配置、源码快照、损失分量、验证曲线及 checkpoint 哈希。
- 原有 HC3 辅助结果仍是原四种方法的单种子分析，本轮补跑范围为 HART 2×2 矩阵。

## 独立开关与运行

`--model-type` 只选择架构，`--supcon` / `--no-supcon` 只覆盖损失开关；未给 CLI 参数时尊重 JSON。默认 JSON 为 RoBERTa + CE。**现在 `--model-type race` 不再隐式开启 SupCon**，完整 RACE 须显式传 `--supcon` 或在 JSON 中设 `use_supcon: true`。旧的 `race_no_supcon` 架构别名仍可接受，保存新配置时规范化为 `model_type: race, use_supcon: false`；旧结果 JSON 保持原样。

```bash
bash experiments/run_neural.sh --model-type roberta --no-supcon --seed 3407 --run-name review_roberta_ce
bash experiments/run_neural.sh --model-type roberta --supcon --seed 3407 --run-name review_roberta_supcon
bash experiments/run_neural.sh --model-type race --no-supcon --seed 3407 --run-name review_race_ce
bash experiments/run_neural.sh --model-type race --supcon --seed 3407 --run-name review_race_supcon
```

独立队列只补五次缺失训练；先检查全部四种条件的 train/validation-only smoke，已有正式结果只校验，停止的运行仅在进程退出且配置、源码快照相同后恢复。

```bash
python3 -B -m experiments.run_ablation_queue --plan
python3 -B -m experiments.run_ablation_queue --status
source experiments/runtime_env.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$EE6405_PYTHON" -B -u -m experiments.run_ablation_queue
```

队列计划、状态、日志和原 checkpoint 哈希在 `$EE6405_STORAGE/experiments/ablation_2x2_v1/`。默认资源位置为 `/data/botao/ee6405-race-validation`。

## 报告与解释

四格分别报告均值和样本 SD。对每个相同种子计算以下 Macro-F1、accuracy 与 Macro-AUROC 差值，再汇总均值和样本 SD：

1. RoBERTa 内的 SupCon 增益：RoBERTa(CE + SupCon) − RoBERTa(CE)。
2. RACE 内的 SupCon 增益：RACE(CE + SupCon) − RACE(CE)。
3. CE 下的架构差异：RACE(CE) − RoBERTa(CE)。
4. CE + SupCon 下的架构差异：RACE(CE + SupCon) − RoBERTa(CE + SupCon)。
5. 交互项：RACE 的 SupCon 增益 − RoBERTa 的 SupCon 增益。

不把不同种子的模型任意配对，也不把训练种子 SD 当成置信区间。既有来源组 bootstrap 仍仅描述种子 3407 两个已拟合模型在本测试集上的条件抽样波动。

该矩阵能区分当前设置下训练目标的作用及架构与目标的交互。架构比较仍有 pooling、额外参数量、表示维度和结构信息覆盖差异，因此不能独立证明篇章图本身的因果收益。仅三个种子、一个冻结划分，不能声称跨基准或未见生成模型的普遍收益。

完成后重算并验收：

```bash
source experiments/runtime_env.sh
"$EE6405_PYTHON" -B -m experiments.verify_training_entrypoint
bash experiments/run_review.sh
python3 -B scripts/check_results.py
```
