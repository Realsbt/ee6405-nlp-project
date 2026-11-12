# EE6405 项目档案 · 2026-11-13（新加坡时间）

本档案汇总了本地可核对的小组项目代码、实验与交付记录。最新结论见[英文 README](../../README.md)和[中文 README](../../README.zh-CN.md)。历史笔记保留写作时的状态；其中“尚未完成”或“正在运行”可能已被后续结果取代。这里的 HART 分组为训练 11,190、验证 1,600、测试 3,194。除明确标为 Micro-F1 或其他指标的数字，以下分数均为 Macro-F1。

| 尝试 | 实际结果 | 证据 |
|---|---|---|
| 六个早期选题：提示词注入、诈骗、标题党、毒性改写、迷因、故事结局 | 方案比较，未找到训练或评估记录；最终项目为文本创作与编辑检测 | [选题记录](historical-research/EE6405_Group_Project_Fun_Ideas.md) |
| 复现 RACE；TF-IDF 与 RoBERTa；CE × SupCon 消融 | TF-IDF 79.40%；三种子均值 RoBERTa CE 90.72%、SupCon 91.15%，RACE CE 89.92%、SupCon 90.37% | [共享消融](../ABLATION_2X2.md) |
| 小组其他分支：六层解冻、层间学习率、投影头／弱 SupCon／辅助目标；长文多窗口、Text Lab、便携运行 | 六层锚点均值 91.1558%，三模型集成 91.6999% Macro / 94.36% Micro；其他训练候选未替换锚点。长文首／中／尾推理已实现，未找到可确认的正式分数；Text Lab 与便携运行有实现和功能检查 | [小组归属与证据索引](presentation/EE6405_实验与贡献总表.html) |
| `zsp` 的 12 小时研究与 targeted repair | 末轮重放使用 960 条历史开发记录，B0 92.56%、E26 92.70%；固定门控命中为 0，未生成三折 OOF，保留 B0。E16 仅作概率诊断，E30 因缺少历史配方未重建；本轮没有训练或读取 HART 测试／HC3 | [zsp 原报告](https://github.com/Realsbt/ee6405-nlp-project/blob/1bf0a11/experiments/campaign12h/ee6405_scorefirst_36_12h/targeted_repair_01_run03/FINAL_REPORT_zh.md) |
| 标签来源、概率复核、重放、Bootstrap、温度校准与误报 | 复核 13 组 HART / 4 组 HC3。HART 保留的 1,015 条 AI 改写中 765 条标为工具改写；RACE 重放 HART/HC3 argmax 无变化；RACE 温度校准改善 Brier，未解决外部误报与漏报 | [CPU 汇总](historical-research/03_CPU执行结果.md)、[校准](historical-research/08_RACE重放与验证集校准.md) |
| 词汇模型、少数类专家、类别偏置、长度与篇章统计 | 词汇单模 83.05%，与 CE 融合 91.96%；专家 90.84%、类别偏置 91.01%。长度／结构／合用：HART 42.41／44.27／52.67%，MixSet 子集 35.06／23.95／29.16% | [CPU 改进](historical-research/cpu_improvement_v1/REPORT.md)、[轻量对照](historical-research/12_CPU续跑结果与下一步.md) |
| 同来源原文与改写配对、外部迁移 | 359 条／91 组的保守 MixSet 子集；误差定位到创作者与编辑操作的混淆。HC3 为另一项二分类辅助评估，两者都不能与 HART 四分类直接比较 | [配对诊断](historical-research/edit_pair_diagnostics_cpu_v1/REPORT.md)、[外部结果](historical-research/external_completed_cpu_v1/REPORT.md) |
| 冻结概率融合、Stacking、同标签预算与换底层种子 | CE+RACE 92.10%；CE+TF-IDF Stacking 92.91%；三模型 Stacking 93.61%（95.27% Micro），MixSet 仅 36.22%。固定元模型换底层种子不算独立重训 | [Stacking 对照](historical-research/cpu_stacking_controls_v1/REPORT.md)、[预测与指标](../../experiments/reports/yingqi_snapshot/) |
| 同窗口：正文、篇章图、打乱关系、EDU；NFKC 与缓存修复 | 缓存覆盖 15,984 条。T4 seed 3407 测试 91.28／89.69／90.07／90.89%；尚缺另外两种子的八次训练 | [实现与解析修复](historical-research/13_同窗口消融实现与CPU验收.md)、[正式结果](../SAME_WINDOW_ABLATION.md) |
| Text–EDU 四条件、Kaggle 启动及续跑 | seed 3407 验证 90.64／90.85／90.90／91.36%。首版入口误找 ZIP，修复后任务被取消；直接挂载取消任务的输出被拒，改用私有中转包续跑。保留 5 次、补完 3 次；三种子测试 CLS 91.01%，门控 90.72%，平均差 −0.30 个百分点 | [过程](historical-research/14_文本篇章融合改进与GPU交接.md)、[续跑证据](latest/fusion/summary.json) |
| 把门控预测加入 Stacking，或替换 RACE | 开发选中三种子门控均值替换 RACE；历史测试由 93.61% 降至 93.17%。其余候选仅报告开发结果 | [六个对照](historical-research/gated_stacking.json) |
| 固定、全局、自适应门控，EDU 与等长切分，少数类一致性、校准及 DFS | 实现和 CPU smoke 已保存；未找到正式 GPU 增益证据 | [执行方案](../FUSION_SUPERVISION.zh-CN.md)、[CPU 证据](latest/follow-ups/) |
| Kaggle 配额与 Colab v1–v3 编码器试验 | Kaggle 因配额拒绝；Colab v1 TensorFlow/protobuf 导入失败，v2 AMP 更新跳过，v3 T4 两组各完成 4 轮。DeBERTa 89.87%，全层 RoBERTa 91.48%，改写类各命中 70/110、69/110，门控为 75/110。重连受 reCAPTCHA 阻断，未取回权重；未评估测试集 | [运行历史](../ENCODER_PILOT.md)、[保存的证据](latest/encoder-pilot/) |
| 门控 + 新编码器等权概率融合、全竞争类调权、仅 AI 生成时调权 | 与 DeBERTa／全层 RoBERTa 融合的验证分数 91.99／91.58%；门控+DeBERTa 全竞争类／定向五折 92.77／93.27%，定向命中 85/110。既有 Stacking 加同类规则的历史测试由 93.61% 降至 93.54%。无 DeBERTa 测试结果 | [阈值协议](../ENCODER_PILOT.md)、[逐折指标](latest/threshold/) |

其他分支的材料保留原提交归属：[主线与发布历史](https://github.com/Realsbt/ee6405-nlp-project/tree/5485383)、[Yuzhe 的训练对照](https://github.com/Realsbt/ee6405-nlp-project/blob/71c2141/experiments/research_results/2026-11-09_confirmed_anchor/JOINT_REPORT.zh-CN.md)、[长文推理协议](https://github.com/Realsbt/ee6405-nlp-project/blob/13141ec/docs/MULTIVIEW.md)。主线已发布静态 Text Lab 和浏览器端 TF-IDF；另做的 VPS 服务提交随后被回退。

[展示大纲](presentation/Outline.md)、[实验流程图](presentation/Yingqi_实验延伸流程图.html)、[贡献总表](presentation/EE6405_实验与贡献总表.html)和[结果图页面](presentation/Yingqi_Text_Lab_Experiments.html)保留了项目展示材料。HTML 可下载后在浏览器打开。小组贡献按原归属记录。

清点范围包括主工作区的 `EE6405_paper_research`、`EE6405_Presentation`、幻灯片构建目录和两份顶层方案，也检查了课程目录 `EE6405_Natural_Language_Processing`、桌面旧 `EE6405` 课程目录及一个同名空工作目录。公开分支只收录项目原创资料、可公开的实验代码及指标／标签／概率等证据。授课讲义、课程名册、录播、题库、原始样本正文、含样本截图的幻灯片、凭证、模型权重、大型缓存与环境副本保留在本地。历史研究脚本和产物按来源归档；部分脚本仍依赖原工作目录布局及未发布资源，公开复现优先用根目录 README 的命令。
