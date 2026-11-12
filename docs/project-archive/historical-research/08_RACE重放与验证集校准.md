# RACE CPU 重放与验证集校准

日期：2026-11-06。原仓库与上游代码保持不变，新增脚本和结果保存在研究目录。本轮不训练模型、不改测试划分。

## 资源恢复

- 从项目 `resources-v1` Release取得 `race_s3407-best.pt`，514,550,389 bytes；SHA-256 `6797d46f3fb2f84734b0cf9e49d7d566eefc9f97ffa6d0db3219d8aa107098d1`。
- parser为Hugging Face `tchewik/isanlp_rst_v3`，revision `5141450ac3eb7decad0b8735fdbf5ec6928ec97a`，权重2,545,320,602 bytes；配置、关系表、XLM-R tokenizer与配置均按项目manifest逐文件校验。
- `isanlp-rst==3.2.2`，isanlp commit `2a102e59f9718acc7fe259dd8d83c66d5da39794`，其余推理依赖按原 `requirements-rst.txt` 安装在 `.venv-cpu`。
- HART原始图、分区正文已恢复；无需重新解析HART才能重放。HC3需要从冻结正文重新解析。
- 两条parser冒烟样本通过：非空树、合法有序字符跨度、18种关系清单。仅冒烟通过不代表完整1,000条均通过。

## 执行状态

| 项目 | 状态/结果 |
|---|---|
| RACE HART验证集1,600条 | 完成，T=1.6718143866；验证NLL 0.247779→0.202690 |
| RACE HART测试集3,194条重放 | 通过；argmax零变化，最大概率差2.4694e-5 |
| HC3全量1,000条CPU解析 | 完成；字符有序、跨度文本一致，未覆盖非空白字符为0 |
| HC3 RACE重放与校准后评测 | 通过；1,000条argmax零变化，最大概率差6.2304e-5；校准前后评测完成 |
| TF-IDF和RoBERTa校准 | 完成，见saved_temperature_v1 |
| 校准/重放单元测试 | 新增4项，连同已有分析共16项通过；原项目15项另行通过，共31项 |

## 校准协议

RACE使用此次验证集推理保存的原始logits。另两种模型复用已有验证集概率：检查所有概率大于零、有限且总和合法，归一化后取log作为等价logits；不做零值裁剪。

仅在HART验证集拟合一个正温度T，以四分类负对数似然为目标，固定搜索范围[0.05,20]；优化后验证损失不能高于T=1，并记录是否碰到边界。参数写盘后才加载测试预测做校准评估。没有用HC3或MixSet选T。

未校准与校准后的两套低误报阈值，分别只使用验证集的人类创作者负类，目标经验FPR不超过1%；测试仅报告锁定阈值的实际FPR与TPR。

温度缩放不改变四分类argmax。将后两类概率相加后的AI creator排序不一定严格不变，不能假定二分类所有指标都不变。

该验证集曾参与checkpoint选择，此次不是完全独立校准；已见过旧测试结果，因此是事后扩展。校准不是重新训练，域内校准也不保证外部可靠性。

## 已完成的概率校准对照

AI creator Brier越低越好：

| 模型 | T | HART校准前→后 | HC3校准前→后 | MixSet子集校准前→后 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.8410 | 0.04364→0.04148 | 0.37016→0.39074 | 0.29383→0.30405 |
| RoBERTa CE3407 | 1.5880 | 0.03007→0.02869 | 0.42570→0.39145 | 0.38524→0.35832 |
| RACE CE+SupCon3407 | 1.6718 | 0.02862→0.02726 | 0.48048→0.43802 | 本轮未评 |

RoBERTa的MixSet锁阈值实际FPR仍为16.4%、TPR仍为6.42%。这些是固定checkpoint的描述性结果，没有新增显著性检验。不能将校准后的概率称为已验证的真实作者身份置信度。

RACE的HART测试四分类Macro-F1保持0.908415，测试NLL从0.240596降至0.199754，AI creator ECE从0.018833降至0.008289。HC3校准后Brier仍高于恒定0.5概率输出的0.25，不能视为外部置信度已经可靠。

| RACE操作点 | HART实际FPR / TPR | HC3实际FPR / TPR |
|---|---:|---:|
| 未校准，验证集锁阈值0.99041946 | 0.313% / 85.106% | 3.0% / 1.2% |
| 校准后，验证集锁阈值0.94356021 | 0.313% / 84.793% | 3.2% / 1.2% |

两套阈值在验证集都为8/800假阳性，即经验1%；外部实际FPR不受此保证。HC3未校准的固定0.5结果与旧报告一致：Macro-F1 0.386925、AUROC 0.68436、AI召回8.4%。不要混用固定0.5与验证集低误报阈值下的指标。

完整汇总：`cpu_calibration_completed_v1.json`；逐条logits/概率与重放报告：`race_cpu_v1/`；其他模型校准结果：`saved_temperature_v1/`。本轮仅RACE主配置种子3407，不代表所有RACE种子都已CPU重放。

## 可复现入口

从研究目录运行；已有输出不覆盖，需使用新目录或保留原证据后明确安排重跑：

```bash
.venv-cpu/bin/python -B fetch_cpu_assets.py --parser
.venv-cpu/bin/python -B fetch_cpu_assets.py --race
.venv-cpu/bin/python -B race_cpu_replay.py --phase hart --output race_cpu_v1
.venv-cpu/bin/python -B race_cpu_replay.py --phase hc3 --output race_cpu_v1
.venv-cpu/bin/python -B audit_cpu_parses.py
.venv-cpu/bin/python -B calibrate_saved_models.py
.venv-cpu/bin/python -B -m unittest -v test_race_cpu.py test_research_evaluation.py test_saved_probabilities.py
```

HC3解析使用原仓库 `experiments.rst_worker`，设置 `EE6405_STORAGE` 和 `EE6405_UPSTREAM` 为本研究目录的resources及resources/RACE；输入冻结 `resources/experiments/hc3_aux_v1_1/test.jsonl`，输出 `test_parsed_aligned_cpu.jsonl`。推理时离线加载全部资源。

已发现原worker的RSS字段假设Linux的KiB单位，macOS实际为bytes，导致其 `peak_process_rss_mib` 数值放大1,024倍。新增 `audit_cpu_parses.py` 单独更正，不改原始审计文件。解析输出含逐条计时，不能以整个JSONL哈希不同就断言树结构不同。

HC3全量解析用时1,394.10秒，另有模型加载26.84秒；修正后进程峰值RSS为2,604.64 MiB。HART验证与测试的图预处理/检测推理分别488.88秒、998.36秒，不含读取发布图文件及加载检测器的时间。

HC3图预处理与检测推理157.90秒，不含此前单独的解析及检测器加载。原仓库及上游代码Git工作区均保持干净，未上传资源或推送GitHub。

重放验收预先固定：最大概率差不超过1e-4且四分类argmax无变化；若不通过，保存差异并标为review_required，不改容限“凑通过”。新解析下的预测一致性也不能单独证明树逐节点完全相同。

运行时间是本机并行执行parser和检测器时的记录，不是独占硬件的性能基准。HART沿用发布的全文图；本轮不属于同窗口结构消融，也没有重训。
