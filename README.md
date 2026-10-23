# EE6405 NLP Group Project

EE6405 小组项目协作仓库，用于维护开题、任务分工、代码和实验结果。

当前阶段：开题准备。正式题目、成员、数据集和课程提交要求待小组确认；当前尚无实验结果。

## 开始协作

1. 在 [开题文档](docs/proposal.md) 确定研究问题、数据来源、对比方法和评估方式。
2. 在 [团队计划](docs/team-plan.md) 填写成员、负责人和真实截止日期。
3. 使用 Issues 拆分任务，每项写清负责人、交付物和验收条件。
4. 从 `main` 创建个人分支，修改完成后提交 Pull Request，由另一名成员检查后合并。
5. 使用 [实验记录](docs/experiments.md) 保存配置、数据划分、结果和局限。

## 目录

| 路径 | 用途 |
| --- | --- |
| `docs/proposal.md` | 开题模板 |
| `docs/team-plan.md` | 分工与阶段计划 |
| `docs/experiments.md` | 实验记录模板 |
| `src/` | 项目代码 |
| `notebooks/` | 数据探索与分析 |
| `configs/` | 可复现实验配置 |
| `data/README.md` | 数据获取与存储约定 |
| `.github/` | Issue 和 Pull Request 模板 |

## 协作约定

- `main` 保持可用；建议分支名为 `feat/<task>`、`docs/<task>` 或 `fix/<task>`。
- 提交信息说明具体变化，例如 `docs: draft research question`。
- 数据、模型权重、密钥和个人信息不进入版本控制。
- 引用论文、数据集和第三方代码时记录来源与许可，保留必要署名。
- 课件和课程平台材料保留在原位置，本仓库不复制课程材料。

现有金融情绪分析原型尚未纳入本仓库。是否复用及复用范围，由选题和许可核对结果决定。
