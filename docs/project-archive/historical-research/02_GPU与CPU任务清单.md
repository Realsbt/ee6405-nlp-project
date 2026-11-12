# GPU 与 CPU 任务清单

不按人数分工；按资源与依赖推进。更新：2026-11-09，历史结果保留原日期。续跑结果详见 `12_CPU续跑结果与下一步.md`。

## GPU 队列：待实现或待运行，不代表已经开始训练

神经网络在理论上也能用CPU训练；这里的“GPU任务”指现有CUDA/BF16入口及可接受耗时下建议交给GPU执行的实验。

| 优先级 | 任务 | 预计运行数 | 启动前依赖 | 验收 |
|---|---|---:|---|---|
| G1 | 同窗口结构机制实验：RoBERTa、正常图、关系扰乱图、无图EDU pooling；固定CE | 4条件×3种子=12 | 新数据协议、可见文本重新解析、对照实现及CPU单元测试、train/val-only smoke | 保存配置、来源哈希、验证选模、完整测试预测和重载记录 |
| G2 | 留一领域：RoBERTa+SupCon和RACE+SupCon | 2模型×4领域×3种子=24 | 目标域不参与训练/验证、近重复和同源隔离、按领域重建协议 | 分域样本数、每类指标、训练/解析成本、全部失败运行 |
| G3 | 微调预算敏感性 | 未定，不能当成已冻结队列 | 先完成G1；原论文也只微调最后层，因此本项是扩展，不是纠错 | 预先确定预算、平等调参预算、相同选模规则 |

G1和G2的数量是研究设计预算。G1四条件入口及CPU smoke已实现，完整缓存仍未就绪；G2入口尚未实现。G1的条件启动命令及门槛见13号文档，不能直接视为可开始正式训练。不覆盖原13次正式结果，不把新数据或图缓存写入旧的 `hart_grouped_v1/graph_cache_v1`。

## 不需要重训、应优先执行

| 任务 | 资源 | 状态 |
|---|---|---|
| 保存预测重新计算原指标 | Python标准库 | 完成，13组HART及4组HC3通过 |
| Brier、概率可靠性分箱、固定阈值行为 | Python标准库 | 完成，见 `saved_probability_audit_20261106.json` |
| HC3的分来源、问题对成组bootstrap | CPU、NumPy | 完成，5000次，见 `hc3_brier_bootstrap_20261106.json` |
| 原论文与本组协议差异表 | 原论文和代码 | 已完成可核实部分，未知项明确保留 |
| 标签来源规则与代码链审计 | 原论文和代码 | 规则及原始metadata匹配完成 |
| 原始HART逐条provenance和近重复审计 | 原始数据、CPU | 15,984条保留样本全部标签匹配；语义近重复仍未做 |
| TF-IDF checkpoint的HC3重放 | CPU、权重、冻结HC3正文 | 完成1000条，类别零变化，最大概率差5.55e-16 |
| RoBERTa checkpoint的HC3重放 | CPU、权重、冻结HC3正文 | 完成1000条，类别零变化，最大概率差1.08e-5；预处理与推理约152秒 |
| RACE checkpoint原图推理、现有条件外部测试 | CPU、图数据及parser | 主配置seed3407已完成：HART测试3194条、HC3重新解析并推理1000条，重放通过 |
| 同窗口重新解析与图缓存 | CPU | 完成：v3共15,984条，train/dev/test=11,190/1,600/3,194，失败0、待处理0，见13 |
| 验证集阈值选择、温度缩放 | CPU和验证集预测 | TF-IDF、RoBERTa CE3407、RACE CE+SupCon3407全部完成验证集温度拟合、锁阈值和测试评测，见08 |
| 新数据零样本外部测试 | checkpoint、兼容标签及文本 | MixSet保守子集359条：TF-IDF、RoBERTa、RACE及固定融合均完成；来源组bootstrap已保存，见12 |
| 已冻结CPU融合候选 | CPU、开发预测与保存权重 | 概率融合、Stacking及相同开发标签预算对照已有结果；域内收益未迁移到MixSet，不等于三次独立重训 |
| 轻量结构基线 | CPU、已发布图与新解析 | 长度/结构/组合模型的HART与MixSet测试完成；组合52.67%→29.16%，见12 |
| 同来源编辑敏感性 | CPU、固定外部预测 | 完成50人类来源与41AI来源配对；定位原文就失效和润色后误报增加，非因果结论 |

### 同窗口准备补充

同窗口v3全量缓存及四条件CPU反向已完成；全量关系扰乱审计覆盖15,984张图。完整GPU训练仍未运行，因此G1是“数据与入口就绪”，不是“实验完成”。

## checkpoint可用性

实际查询私有仓库 `resources-v1`：有8个权重（TF-IDF、RoBERTa三种子、RACE+SupCon三种子、RACE CE的3407）。

发布包缺5个后补实验权重：RoBERTa+SupCon的三个种子、RACE CE的42/2026。它们的保存预测可分析，但重新推理需从原训练资源目录取得权重。不能用另一种配置的checkpoint替代。

## 本机边界与执行入口

- 本机是Apple Silicon macOS，系统Python3.13没有模型依赖，原 `/data/botao/ee6405-race-validation` 不存在。
- 新建项目隔离环境 `.venv-cpu`，采用现有可用Python3.12；不同于原Python3.11环境，因此CPU重放用于兼容性/数值验证，不宣称逐位环境复现。
- 原作者代码固定为 `61ce528430a8bb84fd1bfb3285ee34827182ef59`，不修改作者代码。
- 数据与权重按仓库manifest校验SHA-256；checkpoint加载前验证来源。
- TF-IDF、RoBERTa及RACE主配置3407均实际加载checkpoint，在完整冻结HC3的1000条正文上重新推理；三者重放通过。RACE另完成HART测试3194条重放；其HART/HC3最大概率差分别2.4694e-5、6.2304e-5，argmax均零变化。神经模型验收容限在运行前固定为1e-4且argmax零变化，不宣称跨平台逐位相同。
- 原仓库15项单元测试、新分析6项单元测试及bootstrap自检通过。本地原仓库快照与作者代码的Git工作区均保持干净。
- 本轮追加评测/校准测试后，研究目录共16项单元测试通过，连同原仓库15项共31项。
- 2026-11-09续跑后：研究目录43项、原仓库15项均通过，共58项；资源哈希和固定Stacking重放也已检查。
- 2026-11-09同窗口实现后：研究目录53项、原仓库15项通过，共68项；四条件CPU反向及v2索引缓存验收见13。
- 2026-11-11全量验收后：研究目录55项、原仓库15项通过，共70项；v3缓存与全量控制审计通过，GPU G1可以启动。
- 环境和资源位于本研究目录，已用 `.gitignore` 排除；本轮新增完整HART、原始来源与MixSet资源，未上传到GitHub。

在当前研究目录运行：

```bash
python3 -B -m unittest -v test_saved_probabilities.py
python3 -B analyze_saved_probabilities.py repository --output new_probability_audit.json
.venv-cpu/bin/python -B fetch_cpu_assets.py
.venv-cpu/bin/python -B replay_hc3_cpu.py --model tfidf --output tfidf_hc3_cpu_replay.json
.venv-cpu/bin/python -B replay_hc3_cpu.py --model roberta --output roberta_hc3_cpu_replay.json
```

输出存在时重放脚本拒绝覆盖。`fetch_cpu_assets.py`默认取backbone与HC3，新增 `--hart` 可仅取已固定HART资源；不自动下载checkpoint。本轮权重由GitHub CLI取得，放在 `resources/checkpoints/`。新增运行入口见03，具体GPU实验矩阵见06。
