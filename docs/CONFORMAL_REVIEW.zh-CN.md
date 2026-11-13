# 本次独立改进：分组保形预测与不确定性复核

## 改进的用途

现有模型总是输出一个最高分类。本次增加一个后处理模块：输出可能类别的集合；只有一个候选时给出单一候选，多个候选或空集合时标记 `needs_review=true`。原始分数、argmax、模型权重、训练流程和主页正式成绩表均保留。

这项工作的价值是增加不确定性处理能力，让演示与评估不必强行把每篇文章归到唯一类别。它不是新的文本检测模型，也不证明作者身份。人工复核结果没有被模拟或计入准确率。

## 与三条组员分支的区别

本地核对时间：2026-11-13。GitHub 账户不一定等于真实成员姓名，下面按分支说明，不自行认定人员身份。

| 主线或分支 | 核对版本 | 已有工作及本次避开的方向 |
|---|---|---|
| main（原始提交作者 Realsbt / Botao 的基础项目） | eacb68f32d77bd8e4b859aee6e0e6ccec45495d7 | RACE 复现、TF-IDF/RoBERTa 基线、固定分组划分、三种子 2×2 消融、HC3、原始演示 |
| yingqi | 9c6d80bbd117cf247e01df87943c8a63195fd90c | 同词汇窗口实验、结构与编码器诊断、CPU/云端实验、堆叠、温度校准等；归档的 saved_temperature_v1 已包含温度校准结果 |
| yuzhe | 71c21411d8705d270b460cb5ab5d562c6d0a6a21 | 编码器解冻、学习率衰减、训练与损失配置搜索、多种子确认及算力流程 |
| zsp | 1bf0a11a6f38f5e7982c71984c6f60b8df1f9b72 | Text Lab 界面、解释与编辑、堆叠、组约束解码、评估审计、误差和重加权分析 |
| 本次新增 | improvement/group-conformal-review（本地分支） | 按来源组校准的候选集合、复核标记、组级覆盖和自动判定比例的评估、可独立调用的推理接口 |

核对范围为这些公开分支的代码、报告和提交记录；未发现同类 group-max conformal 功能。不能排除组员尚未上传的工作。实现基于当前 main，并未把三条分支的研究成果合并成自己的新增贡献。

## 如何运行

解压完整代码包，在仓库根目录打开终端，使用 Python 3.9 或更新版本。新增核心模块、实验和测试均只用 Python 标准库，无需 GPU、模型权重或下载数据。

```powershell
python -m unittest discover -s tests -p test_group_conformal.py -v
python scripts/check_results.py
python scripts/run_conformal_review.py --output experiments/reports/conformal_review_rerun
```

预先计算的完整结果在 `experiments/reports/conformal_review_v1/`，包含英文报告、CSV 汇总、固定组划分清单，以及每个模型的校准参数、分层指标和逐条候选集合。命令要求输出目录不存在，以免覆盖旧证据。Windows 克隆时若 Git 自动转换换行符，原仓库的逐字节哈希核验可能失败；本交付包保留上游文件原始字节。

真实新实验应准备独立校准文件和评估文件：

```powershell
python -m experiments.group_conformal --calibration calibration_predictions.jsonl --evaluation evaluation_predictions.jsonl --output new_conformal_experiment --alpha 0.1
```

每行沿用原仓库字段 `item_id, group_id, domain, true_label, probabilities`。四类概率键必须齐全、数值有限且和为 1。程序拒绝重复 ID、来源组跨校准/评估重叠以及不合法概率。文本层面的重复检查仍需上游数据流程保证，单靠预测文件不能发现不同 ID 下的重复原文。

推理接入现有预测服务时，不需真值或来源组：

```python
import json
from pathlib import Path
from experiments.group_conformal import predict_set

artifact = json.loads(Path(
    'experiments/reports/conformal_review_v1/race_s3407/calibrator.json'
).read_text(encoding='utf-8'))
# probabilities 应来自同一个 race_s3407 checkpoint 和相同预处理。
# result = predict_set(probabilities, artifact)
# result 包括 prediction_set、needs_review、singleton_label、original_argmax。
```

本次交付的是可调用模块和评估结果；未改变现有网页和预测服务。回顾性实验的校准器仅供演示，正式部署前须重新划出独立校准数据，并核对模型/预处理一致性。

## 方法与实验边界

对校准组 g，以 `max(1 - p(y_true))` 汇总该组所有文本的不符合度，每个来源组只贡献一个分数。固定 alpha=0.10，取排序位置 `ceil((G+1)*(1-alpha))` 的值 q；位置超过校准组数时取分数上界 1，返回全标签集合。预测候选集合为所有满足 `1-p(y)<=q` 的标签。分位点使用有限样本修正，不做插值。

在组可交换、模型预先固定等条件下，该构造面向新来源组内的同时覆盖。它不依赖组内恰好几个类别，不用其他文章帮助解码当前文章。理论背景为 Angelopoulos & Bates 的 split conformal 方法，组最大值是本次面向相关改写样本的分数设计，不声称原创 conformal 算法。

参考论文：https://arxiv.org/abs/2107.07511

公开主线没有独立验证集逐条预测。本次将已经公开、此前已被查看的 3,194 条测试预测，按与标签和分数无关的 SHA-256 固定排序，拆成 399 个校准组（1,594 条）和 400 个评估组（1,600 条）。13 个模型使用相同划分，不按结果选模型、不调 alpha、不挑划分。这只是回顾性可行性分析，不能把原测试集重新称为未见测试集，也不能覆盖原论文/项目主表。

| 固定示例模型 | 原始 Top-1 准确率（1,600 条） | 唯一候选比例 | 唯一候选部分准确率 | 需复核比例 | 候选集合覆盖率 | 整组全部覆盖率 |
|---|---:|---:|---:|---:|---:|---:|
| RACE seed 3407 | 93.06% | 88.06% | 96.17% | 11.94% | 96.31% | 86.50% |
| RoBERTa + SupCon seed 3407 | 93.38% | 83.06% | 98.04% | 16.94% | 97.81% | 92.25% |

准确率的分母不同，不能把上表写成“整体准确率提高到 98%”。集合覆盖率指真标签是否包含在集合中，也不等于分类准确率。RACE 的实际组覆盖率 86.50% 低于 90% 目标；其他若干模型也未达到目标，完整报告全部列出。有限样本的观测值不是每次必达保证；同时，本回顾性流程也不能确认理论条件在现实部署中成立。

按领域和类别拆分的指标只是描述性结果，不保证每个领域、类别或唯一候选子集都达到 90%。HC3 为二分类且存在分布变化，本次未将四分类覆盖承诺扩展到 HC3。所有原模型结果未重新推理，计算使用已保存预测。

## 代码与验证

- `experiments/group_conformal.py`：校准、独立推理 API、分层评估和独立输入文件 CLI。
- `scripts/run_conformal_review.py`：重现全部 13 个模型的回顾性实验，保存哈希与来源组清单。
- `tests/test_group_conformal.py`：7 项测试覆盖有限样本分位点、组最大值、组隔离、空集合、无效输入、评估标签不影响校准器，并枚举组顺序检验小样本覆盖计算。
- 原仓库 `scripts/check_results.py`：13 次 HART、4 次 HC3、种子汇总和 5 个配对对比的一致性核验通过。它不等同于重新加载模型推理。

## 展示与个人理解

可用约 45 秒说明：“原系统给出唯一类别，我增加按来源组校准的候选集合。多候选交给人工复核。以原 RACE 为例，约 88% 文本返回唯一候选，这部分准确率约 96%；约 12% 需要复核。它改善的是不确定性处理，整体分类器没有重新训练。结果来自已公开预测的分组回顾性实验，仍需独立校准数据和新测试验证。”

展示前应自己运行代码、理解分位点计算和分母区别，并根据自己的理解修改；成员贡献说明和课程 GAI declaration 应如实反映 AI 辅助代码与分析。这份说明不代填成员签名或个人贡献声明。
