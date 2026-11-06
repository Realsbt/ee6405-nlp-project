# EE6405 本机实验主线

2026-11-06：已完成数据审计、固定分组划分、TF-IDF、RoBERTa/RACE 各三种子与关闭对比损失的单次消融，并逐一重载复核完整测试集。四方法 HC3 与原始文本 Demo 也已验证。仓库保存正式代码与评估证据；数据、环境及模型放在原主机 `/data`。统一结果页见 [reports/comparison/REPORT.md](reports/comparison/REPORT.md)，验收记录见 [project_verification.json](reports/project_verification.json)，最终交付状态见 [提交准备](../docs/SUBMISSION.md)。

仅克隆仓库时，可运行 `python3 scripts/check_results.py` 复核保存预测（在仓库根目录执行，无第三方依赖）。下文的训练、Demo 和完整验收需要原主机资源；新机器前提见 [数据与外部资源](../data/README.md)。报告 JSON 中的绝对路径与哈希保留原实验取证信息，不代表这些文件随 Git 提供。一般日志、smoke 输出及含原文摘录的错误清单仅保留在原工作区；`run_review.sh` 会在完整环境下重生成分析用错误清单。

## 当前研究问题与状态

比较传统词汇特征、RoBERTa 和 RACE 在四类英文文本来源检测上的表现，分析篇章结构与监督对比损失的作用。主模型预定种子为 3407、42、2026；关闭对比损失的消融先做种子 3407。所有神经模型只按验证集 Macro-F1 选 checkpoint，测试分数不用于改变实验参数。

- `human_written`：人工原文。
- `human_ai_polished`：人工内容经 AI 润色。
- `ai_generated`：AI 内容，包含发布版本中的 AI 到 AI 改写链。
- `ai_humanized`：AI 内容经人工或工具改写。

早期的 `local_validation/` 记录官方 checkpoint 的小样本推理，仅保留在原工作区并从 Git 排除。这里的 `experiments/` 记录在新划分上从基础预训练权重重新微调的课程实验；早期 smoke 不参与正式结果。

## 数据版本：hart_grouped_v1

使用 RACE 作者整理的 HART 英文四分类发布版本，共 16,000 条、4,000 个原文组，包含 arxiv、essay、news、writing 四个子域。这属于一个基准的四个子域，不是四个独立数据集。

原发布划分的原文组重叠为：训练/验证 1,328，训练/测试 2,334，验证/测试 695。相应的标准化文本重叠数为 1、2、1。数字表示集合交集大小，不能相加解释为不同组的总数。

清理规则在基线训练前固定：

1. 用 NFKC、空白合并及大小写折叠计算重复文本指纹。
2. 标准化文本相同但标签冲突的 14 条记录全部排除；同标签重复额外排除 2 条。
3. 所有模型共用有 RST 结构的样本池；本发布版本没有缺失 RST 的记录。
4. 原文组以及通过重复文本关联的不同原文组归入同一分组单元；防止只去掉重复本身后，其余同源变体仍跨分区。
5. 在分组约束下按“子域 × 标签”近似分层；预先固定第一折和随机种子，约 70%/10%/20%。划分不使用任何模型分数。

| 分区 | 样本数 | 原文组数 | 人工 | AI 润色人工 | AI | 人工/工具改写 AI |
|---|---:|---:|---:|---:|---:|---:|
| 训练 | 11,190 | 2,801 | 2,795 | 2,796 | 4,921 | 678 |
| 验证 | 1,600 | 400 | 400 | 400 | 690 | 110 |
| 测试 | 3,194 | 799 | 798 | 798 | 1,371 | 227 |

三个分区之间的 item ID、原文组、标准化重复文本和合并分组均无交集。未进行语义近重复检索；这是同基准、同子域分组保留测试，不代表跨基准、未见领域或未见生成模型的泛化验证。

词汇输入限制为 RoBERTa 首 512 个 token（含特殊 token）所覆盖的原文字符前缀，保存为 `model_text`。TF-IDF 只使用该字段；标签、ID、子域、生成模型名称不进入特征。RACE 的全篇 RST 结构提供了超出该词汇窗口的信息，统一报告明确披露这一差异，并按词汇窗口是否完整作描述性分组；它不是完全匹配结构信息的受控对照。

数据目录：`/data/botao/ee6405-race-validation/experiments/hart_grouped_v1/`。其中有固定文本分区、排除记录及 `protocol.json`。原始 RST 文件完整保留，文本分区保存来源文件和行号，供后续按同一 ID 清单恢复图数据。

**作者微调 checkpoint 仅用于早期可运行性检查。新测试集包含原发布训练分区的样本，因此不能用该 checkpoint 声称得到新划分上的独立测试结果。这里的全部正式神经模型均从基础预训练模型和新训练分区开始。**

## 第一项结果：TF-IDF + 逻辑回归

词 1–2 gram、最多 50,000 特征、最小文档频次 2、sublinear TF、类别权重 balanced。词表和 IDF 只在训练集拟合。C 在预定的 `[0.1, 1, 10]` 中按验证集 Macro-F1 选择，选中 10；训练集之外没有参与拟合，未合并训练和验证集重新训练。

| 指标 | 保留测试集结果 |
|---|---:|
| Accuracy | 85.32% |
| Macro-F1 | 79.40% |
| Macro-AUROC（OvR） | 96.24% |
| 人工文本 F1 | 84.20% |
| AI 润色人工文本 F1 | 80.63% |
| AI 文本 F1 | 91.49% |
| 人工/工具改写 AI 文本 F1 | 61.27% |

最后一类召回率为 46.70%；其中 94/227 被预测为 AI 类。后续神经模型的类别结果及其精确率/召回率权衡见统一报告；不能仅由传统基线的错误推断篇章结构必然有益。

子域 Macro-F1：arxiv 81.31%、essay 91.46%、news 71.49%、writing 70.18%。这些是同一次混合子域训练后的分域结果，不能称为跨域迁移实验。

多数类参照的 Accuracy 为 42.92%、Macro-F1 为 15.02%。当前基线用了一个固定划分，没有多划分或多随机种子的统计结论，也不能与论文不同划分下的分数直接作公平排名。

拟合与验证三个 C 的总时间约 4.09 秒，词表拟合及验证向量化约 3.84 秒，测试向量化和预测约 0.45 秒；这些是本机 CPU 单次记录，线程上限为 4，测时范围在结果 JSON 中明确记录。

## 结果文件与复核

- `reports/data_v1.json`：数据来源 SHA-256、清理规则、划分参数及交集审计。
- `reports/tfidf_lr_v1/selection.json`：全部验证候选和选择依据，在加载测试记录前写入。
- `reports/tfidf_lr_v1/result.json`：主指标、类别与子域分析、计时、模型和数据哈希。
- `reports/tfidf_lr_v1/test_predictions.jsonl`、`errors.jsonl`：逐条预测和错误记录。
- `reports/tfidf_lr_v1/top_training_features.json`：模型从训练数据学习的各类正权重特征。
- `reports/tfidf_lr_v1/confusion_matrix.png`：计数及行百分比混淆矩阵。
- `reports/tfidf_lr_v1/verification.json`：重新加载模型后的预测与原保存结果一致性检查。

模型保存在 `/data/botao/ee6405-race-validation/experiments/models/tfidf_lr_v1.joblib`，约 3.6MB。

在任意工作目录验证现有结果，无需重新训练：

```bash
bash '/home/botao/文档/ChatGPT/小组作业/experiments/run_baseline.sh'
```

传入新的运行名称会在同一冻结协议上重新训练，保留已有结果：

```bash
bash '/home/botao/文档/ChatGPT/小组作业/experiments/run_baseline.sh' tfidf_lr_recheck
```

四项有针对性的单元检查已通过：同源/重复关联分组、冲突标签排除、文本标准化、缺失类别的 AUROC 表示。完整数据上的交集断言和模型重新加载复核也已通过。已有数据版本禁止直接覆盖，调整协议时应另建版本。

技术依据：[StratifiedGroupKFold](https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)、[逻辑回归](https://scikit-learn.org/1.6/modules/linear_model.html#logistic-regression)。源数据来自 [RACE 作者共享目录](https://drive.google.com/drive/folders/1goJ9F5uXhBTabURr9q23KhbPxiYNFK-8)。

## 神经模型、复核与顺序队列

环境 `/data/botao/ee6405-race-validation/.venv`，Python 3.11、PyTorch 2.7.0+cu128。它以 overlay 方式复用现有 `rl_robotlab` 的 PyTorch，额外依赖装在自己的环境中，没有修改机器人环境。关键版本见 `requirements-core.txt`。当前主机复跑已验证；全新系统的从零安装尚未验证。

路径默认以仓库根目录为存储根：数据在 `experiments/hart_grouped_v1/`，下载资源在 `assets/`，训练输出在 `experiments/neural_runs/`。迁移到其他机器时可设置 `EE6405_STORAGE` 覆盖存储根，或分别设置 `EE6405_ASSETS`、`EE6405_DATA_ROOT`；训练和图缓存脚本会读取这些环境变量，不依赖 Linux `/data` 路径。

`configs/neural_v1.json` 固定 batch 16、最多 20 轮、学习率 2.9e-5、warmup 10%、patience 5、四类平衡 batch；仅解冻 RoBERTa 最后一层。训练用 BF16，选模与测试用 FP32。RACE 使用官方 RGCN、root pooling、MLP 和监督对比损失；`race_no_supcon` 只关闭该损失。RoBERTa 与 RACE 的分类头维度和模型容量不同，不能把两者之差完全归因为篇章结构。

架构和监督对比损失现在是独立开关。历史命令在 `--model-type race` 时仍默认启用 SupCon；可用 `--use-supcon` 或 `--no-supcon` 覆盖任意架构，例如：

```bash
bash experiments/run_neural.sh --model-type roberta --use-supcon --seed 3407 --run-name roberta_supcon_s3407
bash experiments/run_neural.sh --model-type race --no-supcon --seed 42 --run-name race_no_supcon_s42
```

首个 RoBERTa（3407）测试 Macro-F1 90.75%、Macro-AUROC 98.86%，最佳验证轮次 8。完整 3,194 条重载预测完全一致。这是一个种子、同基准分组测试的结果。

首个 RACE（3407）测试 Macro-F1 90.84%、Macro-AUROC 98.77%，最佳验证轮次同为 8。完整测试集重载后的类别预测一致，最大分数差约 1.19e-7。相对 RoBERTa 的 Macro-F1 差约 +0.10 个百分点；按来源组 bootstrap 的条件区间包含零，目前没有明确的总体优劣证据。

关闭对比损失的 RACE（3407）Macro-F1 88.89%、Macro-AUROC 98.36%，同样验证选第 8 轮并通过完整重载复核。完整 RACE 比它高约 1.95 个百分点，来源组条件区间约 [+0.86,+3.07]；这是单次拟合的消融证据，没有消融多种子稳定性结论。

完整三种子 Macro-F1：RoBERTa **90.72% ± 0.25**，RACE **90.37% ± 0.60**（均值 ± 样本 SD，SD 单位为百分点）。平均差为 RACE−RoBERTa **−0.35 个百分点**，本组实验不支持 RACE 总体优于 RoBERTa。每个种子的验证指标、选中轮次和测试结果都在统一报告中，主表没有挑选测试最好的种子。

```bash
# 在工作区根目录逐条运行；使用新名称保留已有正式结果。
bash experiments/run_neural.sh --model-type roberta --seed 42 --run-name review_roberta_s42
bash experiments/run_neural.sh --model-type race --seed 42 --run-name review_race_s42
bash experiments/run_neural.sh --model-type race_no_supcon --seed 3407 --run-name review_no_supcon_s3407

# 仅查看预定队列，不启动训练。
/data/botao/ee6405-race-validation/.venv/bin/python -m experiments.run_comparison_queue --plan
```

当前队列补齐 2×2 对照中缺少的五个运行：`roberta_supcon_s3407`、`roberta_supcon_s42`、`roberta_supcon_s2026`、`race_no_supcon_s42`、`race_no_supcon_s2026`。每个作业显式传入 `--use-supcon` 或 `--no-supcon`，并在训练后执行完整测试集重载验证；已有运行不会覆盖。实际队列状态：`/data/botao/ee6405-race-validation/experiments/comparison_queue_v1/status.json`；队列完成只说明训练与重载检查完成，整项目还需汇总与交付验收。文件锁阻止重复队列。中断的实验需先确认进程已退出，再用原命令加 `--resume`；恢复从最后一个完整 epoch 开始，配置和训练源码必须与该次快照一致。

checkpoint、optimizer、事件日志、源码快照在 `/data/botao/ee6405-race-validation/experiments/neural_runs/<run>/`；本目录 `reports/<run>/` 保存配置、结果、逐条预测和验证。现有完成运行禁止覆盖。图缓存经官方实时预处理和同一初始化模型输出对照，差异为零，见 `reports/graph_cache_verification.json`。

`--run-name` 只改变输出目录，不改变训练配置、数据或种子。新名称使用 1–80 个英文字母、数字、下划线或连字符，并以字母或数字开头。默认仍是 `<model>_s<seed>`；已有目录会报错并保留。复跑别名不自动加入预定主表，完成后可用 `verify_neural --run review_race_s42` 对其做完整重载检查。恢复中断的别名需带相同 `--run-name` 和 `--resume`。

仅检查训练入口可加 `--smoke`：每类 8 条训练、2 条验证，只跑一轮，不打开测试分区，输出名称自动加 `smoke_` 前缀。正式训练与 smoke 的结果分开；固定种子不承诺跨设备的逐位一致。

`reports/retraining_entrypoint.json` 记录了从 `/tmp` 实际运行两种模型的新名称 smoke：32 条训练、8 条验证，梯度有限且分类头确有更新，未读取测试集。非法目录名称和重复运行分别被拒绝，现有七个正式神经 checkpoint 哈希未变。这些 smoke 不参与主指标。

图模型的词汇输入仍为首 512 tokens。官方 graph builder 删除截断外 EDU 节点，但保留全篇关系节点及连接；因此结构信息可以超出词汇窗口。统一分析分别报告未截断与截断子集，并明确其只是描述性比较。

## 独立 HC3 辅助评估

数据：`/data/botao/ee6405-race-validation/experiments/hc3_aux_v1_1/`。固定 500 对题目、1000 条回答、五个来源均衡；文本选择不使用预测结果，与 HART 无标准化精确文本重叠。v1.1 只修复原始 index 空值导致的重复 ID，不改变任何文本、标签或顺序。

固定 AI 创作者分数为四分类输出中 `P(ai_generated)+P(ai_humanized)`，阈值 0.5，不在 HC3 上拟合或调阈值。这里只有二分类监督，不能作为四分类泛化证据。TF-IDF 的 HC3 AUROC 为 60.11%、AI 召回率为 5.80%，说明 HART 高分不能自动推广到独立数据。

四方法评估均已完成：RoBERTa / RACE / RACE-no-SupCon 的 HC3 AUROC 分别为 61.09% / 68.44% / 70.31%。完整 RACE 在固定阈值下 AI 召回仅 8.40%、人工误报 11.20%，不能因排序指标较高就宣称可靠检测。辅助数据上的单次消融方向也与 HART 不同，详见 `reports/auxiliary/REPORT.md`。

批量解析使用独立 CPU 环境 `.venv-rst`，与神经训练环境分开。源码与模型来源见 [SOURCES.md](SOURCES.md)，parser 依赖见 `requirements-rst.txt`。单句和多句原文解析已通过，字符区间合法，峰值进程 RSS 约 5.8 GiB。批量解析完成的 `.audit.json` 是评估前置条件，半写入的 JSONL 不算完成。

## 本机交互 Demo

```bash
bash '/home/botao/文档/ChatGPT/小组作业/experiments/run_demo.sh'
```

打开 <http://127.0.0.1:8765/>。默认 CPU 推理，首次使用按需加载模型；RACE 会按需启动独立的常驻 CPU RST parser。输入上限 12000 字符，支持输入英文、选择已完成并复核的模型、查看四类分数、截断信息、真实篇章树及冷/热启动耗时。文本不发送到远程服务，也不保存到推理日志。

四个模型的 HTTP 检查已通过，见 `reports/demo_verification_all.json`；实际浏览器交互记录及截图见 `reports/demo_ui_verification.json`、`reports/demo_ablation.jpg`。RACE 原始文本完整链路含真实解析器；185 字符短样例真实冷启动记录在较早的 `demo_verification_tfidf_roberta_race.json`，首次约 9.23 秒、再次约 0.29 秒。后续检查会复用缓存，不能把“该轮首请求”都称为冷启动。记录时同时有其他本机工作，不能当成独占资源下的延迟基准。示例文本无真实来源标签，不能作为准确率证据。训练队列活动时默认保持 Demo 在 CPU；`--device cuda` 仅在 GPU 空闲时使用。

更严格的字符覆盖检查在原 parser 的 1000 条 HC3 输出中发现 34 条漏字符区间。`rst_alignment.py` 根据预测 EDU 内容与原文的非空白字符进行精确对齐，拒绝字符不匹配，不改变预测分段或树关系；完整修复版已完成，并通过全部 1000 条覆盖检查，漏非空白字符记录为 0，见 `reports/hc3_rst_aligned_span_audit.json`。旧输出与审计保留；两种图模型的 HC3 评估均已使用这份完整修复版。

## 统一分析与复跑入口

一键重算保存预测的统一指标、图表，并检查完整实验的配置、checkpoint 哈希及重载证据：

```bash
bash '/home/botao/文档/ChatGPT/小组作业/experiments/run_review.sh'
```

此入口先等待分析命令成功，再执行验收；预定训练或辅助评估缺一项会报错。它读取已保存预测，逐条重新推理可使用下面的 `verify_neural` 命令。

设置 `HF_HOME=/data/botao/ee6405-race-validation/cache/huggingface`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`PYTHONDONTWRITEBYTECODE=1`，在工作区根目录运行：

```bash
# 重算当前已验证结果，未齐全时标记 partial。
/data/botao/ee6405-race-validation/.venv/bin/python -m experiments.analyze_results
# 最终验收要求预定主实验全部完成；缺一项会报错。
/data/botao/ee6405-race-validation/.venv/bin/python -m experiments.analyze_results --require-complete
# 对完成模型重新加载，复核完整测试集。
/data/botao/ee6405-race-validation/.venv/bin/python -m experiments.verify_neural --run race_s3407
# 验证网页服务到四个真实模型的接口链路。
/data/botao/ee6405-race-validation/.venv/bin/python -m experiments.verify_demo
# 检查完整项目的证据；先确认上面的完整分析成功。
/data/botao/ee6405-race-validation/.venv/bin/python -m experiments.verify_project
```

测试/验证脚本不会训练模型或根据测试结果调参。统一分析保存每类、子域、截断子集、混淆矩阵、资源记录及按来源关联分组 bootstrap 的成对比较。三种子标准差与单次模型的条件 bootstrap 衡量不同的不确定性；只有一个冻结划分，不能宣称已覆盖跨划分、跨数据集或未见生成模型泛化。

统一分析还会自动刷新 `reports/comparison/CASE_STUDIES.md` 的四个可追溯案例和 `reports/auxiliary/REPORT.md` 的 HC3 输出分配/来源/长度分析。案例按说明的事后规则选取，不是代表性抽样；HC3 四个输出的分配不是四分类真值。`--require-complete` 同时要求预定训练和四个方法的辅助评估全部完成。

`reports/protocol_and_alignment_checks.log` 记录 9 项协议、采样/padding 与 RST 对齐检查通过。`reports/parser_assets.json` 对 7 个 parser/tokenizer 文件做离线内容校验，匹配保存的 HF 固定 revision 与下载 etag；其中 2.545GB parser checkpoint 的 SHA-256 与 LFS 记录一致。最终 `verify_project` 核对每个训练配置、预训练初始化、验证最佳 epoch、源码快照、完整逐条预测及 Demo/解析证据，结果写入 `reports/project_verification.json`；任何必需证据缺失或不符都会返回 incomplete。
