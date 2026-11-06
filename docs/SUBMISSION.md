# 提交清单

截至 2026-11-06，五次补充训练、完整重载复核与三种子 2×2 分析已完成，原本机 Demo 保留。GitHub 中的技术材料供小组审核和整理提交使用；上传代码不代表课程平台已提交。

依据：[项目要求原文](Group%20project%20details.pdf)第 4 页、[课程大纲](EE6405%20course%20syllabus.pdf) Appendix 3–4。完整中文整理见[课程要求与评分标准](course-requirements.md)；平台如有后续通知，以最新要求为准。

## 已有技术材料

| 内容 | 入口 |
|---|---|
| Python 实现、配置、依赖与运行说明 | [experiments/README.md](../experiments/README.md) |
| 传统基线与四个神经条件的结果及逐条预测（共十三次） | [主报告](../experiments/reports/comparison/REPORT.md)；各运行的 `result.json`、`verification.json`、`test_predictions.jsonl` |
| 独立 HC3 辅助评估 | [辅助报告](../experiments/reports/auxiliary/REPORT.md) |
| 错误案例、结果图、学习曲线 | [案例](../experiments/reports/comparison/CASE_STUDIES.md)；`experiments/reports/comparison/` |
| 模型说明与引用来源 | [METHOD_NOTES.md](../experiments/METHOD_NOTES.md)、[SOURCES.md](../experiments/SOURCES.md) |
| 四模型交互 Demo 与本机验收记录 | [运行说明](../experiments/README.md#本机交互-demo)、[验收记录](../experiments/reports/project_verification.json) |
| 真人讲解提纲 | [VIDEO_OUTLINE.md](../experiments/VIDEO_OUTLINE.md)，目标时长 9 分 20 秒 |

## 仍需完成或确认

| 提交事项 | 当前状态 | 完成标准 |
|---|---|---|
| 理论 slides 与团队视频 | 尚无正式 slides 或成品视频，已有提纲与图表 | 视频不超过 10 分钟，展示 Python 代码、理论 slides、参考文献；使用真人讲解。最高档视频评分要求全员参与、衔接流畅。 |
| Python 代码提交件 | 实现和文档已有，需按平台要求选择最终文件或压缩包 | 包含 `.py` 或 `.ipynb`、运行说明及所需配置；说明外部数据和权重的获取安排。 |
| Group reflection | 尚无填写并签署的成品 | 按实际工作记录每位成员贡献，由全员确认并签名。 |
| GAI declaration | 仓库中尚无课程正式模板及填写成品 | 从课程平台取得模板，如实填写 AI 使用情况并按要求签署；本清单不替代正式表单。 |
| Peer evaluation | 未核验平台完成状态 | 每位成员分别完成对其他所有组员的评价。 |
| 课程平台提交 | 未核验提交记录 | 由小组按平台要求上传，检查可播放／可下载，并保存最终提交确认。 |

课程大纲明确：声明代码“完全由 AI 工具生成”会被扣分，视频不得使用 GAI voice。应保留真实使用记录与成员工作依据，不以代填贡献或修改声明来替代实际工作。

## 录制与交付前核对

- [ ] 主表展示四格消融的三种子均值与样本标准差，并报告同种子损失效应、相同损失下的架构差异和交互；不能将架构差异全部归因于篇章图。
- [ ] HC3 只说明辅助二分类表现，展示实际退化；Demo 的无标签输入不作为准确率证据。
- [ ] 讲解者能够解释数据划分、模型、选模方式、主要错误及自己的实际贡献。
- [ ] 用录制所用设备启动 Demo，检查模型切换、文字与图表可读性、音量和完整视频时长。
- [ ] 告知接收方：仅克隆 Git 仓库可审核保存结果；完整推理和重训还依赖[外部资源](../data/README.md)。独立 Linux CPU 环境与资源迁移已验证；全新操作系统、WSL2 和全新 CUDA 环境未验证，见 [迁移记录](../resources/portability_validation.json)。

现有 HART 与 HC3 是两个数据来源。三个以上数据集属于课程建议，不是明确的最低提交门槛；HART 的四个子域不算四个独立数据集。项目说明未要求另交一篇论文式 PDF 报告。
