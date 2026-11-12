# 融合路线收尾与多层改进

日期：2026-11-11。本文是分阶段执行协议，不是新增效果结果。

执行更新：用户已改为验证集门控的 DFS 策略，见 `FUSION_SUPERVISION.zh-CN.md`。
下文三条件九次、四条件十二次仅为历史完整对照设计，不再无条件全部启动。

## 原则和当前边界

现有 fusion v1 已有独立 Kaggle 作业。保持其代码、四候选、筛选规则和资源包不变，
不把后续对照插入运行中的实验，也不覆盖旧产物。复用公共数据/框架与独立的方法
贡献并不矛盾；所有新增效果以正式训练、严格重载和真实预测为准。

当前没有可靠来源配对的训练资源，因此不采用原文/编辑版配对损失，不根据预测
补造配对真值。已有 MixSet 分析属于已看过的外部诊断，不能用来选超参数。

## 第一阶段：完成现有主线

1. 收集 fusion v1 的四条件 CUDA smoke、四候选开发筛选、匹配模型的三种子训练、
   selection.json、严格重载与最终预测。任何缺失都如实报告。
2. 保留 CLS/均值/注意力/门控四组筛选结果。主效果用匹配 CLS 与入选融合的
   逐种子差值、均值和样本 SD，不用最好种子冒充平均效果。
3. 单独做下述门控机制实验，再决定能否说“自适应”有效。旧 HART 测试已被查看，
   本实验属于后验补充，不是独立确认。

## 已实现的补充控制：残差强度与自适应

原 attention 直接加入系数 1，原 gated 初始系数约 0.119，二者差异不只有
“是否自适应”。新增模块不修改原 fusion v1：

| 条件 | 系数 | 要回答的问题 |
| --- | --- | --- |
| scale_fixed | 恒定 sigmoid(-2)，约 0.119 | 较小残差本身是否就够 |
| scale_global | 全局可学习标量，初始约 0.119 | 学一个统一强度是否就够 |
| scale_adaptive | 每个文本根据 CLS/EDU 产生系数，初始约 0.119 | 输入相关的调整是否额外有用 |

三组使用相同 CLS、EDU、注意力、分类头、CE、采样、初始化种子及训练预算。
仅门控参数不同；它们不是严格等参数。固定系数和全局标量不得解释为解析置信度。

模块：`experiments/fusion_scale_controls.py`。
训练：`experiments/train_fusion_scale.py`。
重载：`experiments/evaluate_fusion_scale.py`。

先做 CPU 和目标 GPU smoke。正式矩阵预定三条件 × 3407/42/2026：
不先用一个种子的结果淘汰某个机制条件，不因域内分数修改固定系数。
主要比较 adaptive-global 和 global-fixed 的 Macro-F1 配对差值；同时保留
每类、每域、时间、参数量与显存。此阶段保持 CE，不添加其他训练目标。

```bash
export EE6405_STORAGE=/absolute/path/to/resources
export EE6405_UPSTREAM="$EE6405_STORAGE/RACE"
export CACHE="$EE6405_STORAGE/experiments/same_window_full_v3"

python -m experiments.train_fusion_scale \
  --condition scale_fixed --seed 3407 --cache "$CACHE" \
  --output outputs/scale_fixed_cpu_v1 --smoke

# 正式训练默认仅开发集；分别运行三个条件和三个种子，输出目录不能重复。
python -m experiments.train_fusion_scale \
  --condition scale_fixed --seed 3407 --cache "$CACHE" \
  --output outputs/scale_fixed_s3407 --device cuda --validation-only

# 所有比较对象/配方冻结后，才显式打开测试。
python -m experiments.evaluate_fusion_scale \
  --run outputs/scale_fixed_s3407 --cache "$CACHE" \
  --output outputs/scale_fixed_final_s3407 --device cuda --evaluate-test
```

新模型需从基础预训练权重训练，不能用原 gated checkpoint 事后替换门控来冒充
重训消融。三个条件的初始输出一致，不保证训练时 dropout 随机轨迹完全一致。
没有新增断点续训，正式入口沿用原训练器的此项限制。

### 本轮验证

- `experiments` 下 45 项单元测试全部通过，其中本模块新增 6 项。初始化对齐、
  固定系数无可训练参数、全局标量梯度、注意力梯度及默认不打开测试均有覆盖。
- 三条件均用真实 RoBERTa 完成 CPU smoke：每组 4 条训练/4 条开发，
  一个训练 batch；有限梯度、分类头和最后编码层更新、冻结 embedding 检查通过。
- 可训练参数：fixed 为 7,845,892，global 为 7,845,893，
  adaptive 为 7,847,429。不能声称三组严格等容量。
- smoke 产物分别在 `outputs/fusion_scale_cpu_v1/<condition>/`，没有评测试集。
  小样本分数不作为性能证据。本轮没有新增云任务或完成正式 GPU 矩阵。
- 新增重载入口已有验证集隔离单元测试；尚未用正式 GPU checkpoint 验收。

复跑测试前配置上述资源路径，然后执行：

```bash
python -B -m unittest discover -s experiments -p 'test_*.py' -v
```

## 后续路线：分开检验，再考虑组合

以下都尚未实现，不计为当前贡献。基础融合无可靠增益时，先分析失败再扩展，
不默认所有模块都要加入最终模型。

| 层次 | 下一步候选 | 独立问题与对照 | 进入条件 |
| --- | --- | --- | --- |
| 架构/数据 | 固定 token 块或句子替代 EDU | 同一正文、同一聚合器，只改变切分；比较效果与完整预处理成本 | 完成门控对照后，检验 parser 依赖是否必要 |
| 训练 | 同一文本的 dropout 双前向一致性 | CE 对照 CE+一致性；不需要来源配对，不裁剪可能改变类别证据的正文 | 先完成训练接口、开发筛选和数值测试 |
| 泛化 | 领域平衡或领域对抗 | 先列 domain×class 支持数，再比较普通训练；留一域不使用目标域选模 | 领域字段可靠，训练组覆盖足够；不承诺消除域偏差 |
| 推理 | 预算受限的按需额外窗口 | 区别于组员始终头/中/尾三窗口；比较单窗口、始终多窗口与自适应预算 | 独立开发/校准规则，且实现每窗切分与端到端计时 |
| 评测/可靠性 | 独立校准与拒判 | 覆盖率-风险、人类 creator 误报、实际 FPR/TPR、组 bootstrap | 从训练/开发来源组内安排校准；外部集不参与阈值拟合 |

这些是本组任务上的方法扩展，不声称发明一致性、对抗训练或拒判。
领域对抗不等于 creator 辅助头，但也可能删除有用信号；必须报告回退与最差域，
不能只凭领域分类变难就认定泛化改善。按需多窗口使用更多信息，不能混入
同 512-token 架构主表；与组员路线的区别必须落在实际预算策略和实验上。

## 决策规则

只在训练/开发数据选型，冻结后再评测试。先报告各路线单独效果，再决定是否做
组合；若要证明组合收益，补齐“原模型/仅A/仅B/A+B”，不能堆完所有模块只报最好分。
当前旧验证集和测试集都已有使用历史，后续结果须披露持续开发的局限；新切种子
不产生新独立测试。正式云端计算需先 profile 和确认预算，本轮不自动启动新作业。
