# Bounded encoder improvement pilot

Date: 2026-11-12. Goal: investigate a stronger candidate, not promise 97%
Macro-F1. The historical stacking result remains 93.61% test Macro-F1; this
experiment does not relabel it as a new single-model score.

## Colab v3 结项（2026-11-12）

两组在 Colab T4 上各完成 4 轮训练。本轮结束。两组均未达到替换现有门控融合模型的要求。门控融合会根据文本调整篇章特征在分类中的作用。Macro-F1 是四个类别 F1 的等权平均。

- DeBERTa-v3-base 验证 Macro-F1 为 **89.87%**。`ai_humanized`（经人工或工具改写的 AI 文本）召回率为 **63.64%（70/110）**。
- 全层微调 RoBERTa 验证 Macro-F1 为 **91.48%**，比门控融合高 **0.11 个百分点**。`ai_humanized` 召回率为 **62.73%（69/110）**，比门控融合低 **5.45 个百分点**。
- 现有门控融合的验证 Macro-F1 为 **91.36%**，`ai_humanized` 召回率为 **68.18%（75/110）**。三组结果均使用同一批 1,600 条验证样本和种子 3407。两组 Colab 模型均未评估测试集。

完整性校验通过的结果包保存在本地
`outputs/colab_pilot_monitor_v3/pilot_evidence.zip`，含逐条验证预测、指标和训练日志。
包内没有模型权重。浏览器等待下载事件曾超时，随后在本地下载目录找到了结果包。Colab 重连时提示无法连接 reCAPTCHA，权重未取回。

### 无需训练的概率融合对照

将门控融合分别与两组 Colab 模型按 1:1 平均四类预测概率，取概率最高的类别。用 `item_id` 对齐全部 1,600 条验证预测，并逐条核对真实标签、组 ID 和类别集合。门控融合的原始预测来自 [已保存的验证文件](../experiments/reports/yingqi_snapshot/fusion/screen/fusion_gated_s3407/validation_predictions.jsonl)。

- 门控融合 + DeBERTa 的验证 Macro-F1 为 **91.99%**，比门控融合高 **0.62 个百分点**。`ai_humanized` 召回率为 **67.27%（74/110）**。
- 门控融合 + 全层 RoBERTa 的验证 Macro-F1 为 **91.58%**，比门控融合高 **0.22 个百分点**。`ai_humanized` 召回率为 **62.73%（69/110）**。

两组等权概率融合都少识别了部分 `ai_humanized` 样本。

### 对 AI 生成与改写文本单独调整判定（2026-11-13）

冻结门控融合与 DeBERTa 的预测并等权平均。仅当原预测为
`ai_generated` 时，将 `ai_humanized` 的分数乘以权重，再重新选取得分最高的类别。
以 HART 的 `split_block_id` 做五折分组，同一来源及其关联重复文本留在同一折。
每折在另外四折从 1、1.25、1.5、2、3、4 中选权重，优先最大化目标类 F1，
并约束拟合折的 Macro-F1 至少持平、目标类精确率最多下降 5 个百分点。
表中的分数合并各折留出预测后计算。

| 判定方式 | Macro-F1 | 改写类 F1 | 改写类精确率 | 改写类召回率 |
|---|---:|---:|---:|---:|
| 门控融合原预测 | 91.36% | 78.12% | 91.46% | 68.18%（75/110） |
| 门控融合与 DeBERTa 等权平均 | 91.99% | 78.72% | 94.87% | 67.27%（74/110） |
| 等权平均，所有竞争类别均可触发调整 | 92.77% | 81.59% | 90.11% | 74.55%（82/110） |
| 等权平均，仅在原预测为 AI 生成时调整 | 93.27% | 83.33% | 90.43% | 77.27%（85/110） |

最后一组在五折均增加目标类命中数，分别增加 1、2、3、3、2 条，五折的
Macro-F1 也都上升。相比未调阈值的同一融合，目标类多命中 11 条，同时增加
5 个误报，Macro-F1 上升 1.28 个百分点。用完整验证集拟合时选得权重 1.5。

同样的流程用在三组单门控模型及现有 stacking 上，分组验证均未同时提高
Macro-F1 和改写类 F1。将完整验证集拟合的权重应用到项目历史测试集时，
只针对 AI 生成的判定规则使门控模型三个种子的 Macro-F1 分别变化
+0.27、0.00、+0.57 个百分点。现有 stacking 由 93.61% 降到 93.54%，
多命中 7 条改写文本，同时多出 9 个误报。

交叉验证仅隔离本轮权重拟合。基础模型曾用这批验证数据选择检查点，
当前融合及规则也在这批数据上探索。DeBERTa 的结果包仅提供验证预测，
尚无测试预测或模型权重，因此 93.27% 目前只有单种子的验证集证据。

从仓库根目录运行下列命令（需 NumPy、scikit-learn 和 joblib），使用[公开归档的预测和分组元数据](project-archive/latest/)复现，无需原文或权重。
结果写入 `outputs/minority_threshold_replay/`，包含分折指标和混淆矩阵。

```bash
python -m experiments.replay_minority_threshold
```

## Launch history

On 2026-11-12, the private Kaggle dataset upload succeeded, but the kernel push
was rejected with `Maximum weekly GPU quota of 30.00 hours reached`.
The CLI exited with code zero despite that error, so its exit code is not proof
of a successful submission. No encoder GPU training started on Kaggle.
Existing fusion training was not cancelled or modified.

The original local `paired_encoder_pilot_package_v1` is preserved. A subsequent
bootstrap fix supports nested, auto-unpacked Kaggle code archives; the updated
sources are frozen separately in `paired_encoder_pilot_package_v2`. The uploaded
private dataset still contains v1 and must be explicitly versioned before a
future submission uses v2. Do not repeatedly resubmit while quota is exhausted.
Colab allocated a Tesla T4 (15,360 MiB), verified with its preinstalled
`nvidia-smi` at 06:33:40 UTC. The user then approved private train/validation
upload, dependency installation and model execution. Both CUDA checks passed
with one verified update and zero AMP skips each; DeBERTa formal validation
training launched at **06:54:15 UTC / 14:54:15 Singapore**, followed by RoBERTa.
Both completed. The scores above use the full validation partition.

Colab v1 failed on TensorFlow/protobuf imports; v2 detected a skipped AMP
update and stopped. V3 disables unused TensorFlow/Flax imports and records an
initial AMP scale of 128 for both arms. All failed directories are preserved.
The v3 frozen sources/protocol are under
`experiments/reports/encoder_pilot/colab_v3/`; the old root-level manifest and
protocol still document the earlier Kaggle v2 package. Reproduction, archive
hashes and temporary-storage limitations are in `cloud/colab/README.md`.

## Why this experiment

Frozen validation predictions expose a minority-class bottleneck:

| Existing model | Validation Macro-F1 | AI-humanized F1 | AI-humanized recall |
|---|---:|---:|---:|
| RoBERTa CE | 91.0617% | 80.2030% | 71.8182% |
| RACE SupCon | 90.5578% | 78.4689% | 74.5455% |
| TF-IDF | 78.4321% | 55.5556% | 40.9091% |
| Matched fusion CLS | 90.6435% | 76.1905% | 65.4545% |
| Gated fusion | 91.3643% | 78.1250% | 68.1818% |

These are validation scores, not the historical test table. Of 1,600 validation
records, CE and gated both miss 71; gated rescues 31 CE errors, while CE rescues
17 gated errors. All five listed models miss 34 records. This overlap suggests
both remaining shared difficulty and some complementarity. It is not an oracle
Macro-F1 ceiling or an achievable ensemble score.

Gated has higher overall validation Macro-F1 but lower AI-humanized F1 than CE.
Thus continuing to add gates is not demonstrated to fix the weakest class.
The new pilot first asks whether all-layer fine-tuning and another pretrained
encoder provide better representations before porting the project's EDU module.

## Frozen pilot

| Factor | Both arms |
|---|---|
| Conditions | `deberta_full`, `roberta_full` |
| Encoder | Microsoft DeBERTa-v3-base versus existing local RoBERTa-base |
| Initialization | Base-pretrained only; no teammate's task checkpoint |
| Fine-tuning | All encoder layers, CLS + dropout 0.1 + linear 4-class head |
| Input | Existing audited visible character prefix; retokenize independently; max 512 tokens |
| Partition | 11,190 train / 1,600 validation; test partition absent from pilot package |
| Seed | 3407 only |
| Budget | At most 4 epochs per arm; validation patience 2 |
| Optimization | AdamW, LR 2e-5, weight decay 0.01, linear decay, warmup 10%, clip 1 |
| Sampling | Existing balanced sampler, effective batch 16; microbatch 4, accumulation 4 |
| Memory | Gradient checkpointing; FP16 on CUDA, AMP initial loss scale 128 |
| Hard limit | 14,400 training seconds per arm; hitting it is labeled incomplete |
| Selection | Validation Macro-F1; preserve both results; no automatic test evaluation |

At most eight **formal** epochs total, plus two one-update CUDA smokes.
Each smoke retries the same batch at most 12 times if AMP skips an overflowing
gradient, and stops after its first successful update. No successful update
means failure, not a passing smoke. Formal runs report overflow skips separately.
The training timer includes validation/checkpoint work after model loading, but
not dependency installation, downloads or initial tokenization. Total job time
can therefore exceed eight hours if both arms hit the training cap.
No paid compute, expanded hyperparameter grid, additional seed or duplicate
submission is enabled automatically.

The two architectures have different vocabularies/pretraining and parameter
counts. A common character-prefix input plus separate 512-token caps can still
truncate different suffixes; each run reports how many examples were truncated
again. DeBERTa uses its SentencePiece tokenizer without fast conversion so that
the fast tokenizer's unsupported byte-fallback conversion is avoided.
This is an encoder/recipe pilot, not an exactly parameter- or token-matched
architecture attribution. Both arms use the same maximum training budget and
head initialization seed; actual updates/time are recorded.

This recipe also differs from the older one-layer-fine-tuned fusion experiments.
Any gain versus those older numbers cannot all be attributed to DeBERTa. The
new all-layer RoBERTa arm is required for that reason.

## Attribution and sources

DeBERTa-v3-base is Microsoft's pretrained model, not Yingqi's invented encoder.
Pinned Hugging Face commit: `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`.
Primary source:
[Microsoft's model card](https://huggingface.co/microsoft/deberta-v3-base).
Its upstream evaluation tables are not expected scores for HART.

New project work is the bounded paired experiment, validation-only diagnostics,
portable runner and protocol. Shared HART partitions, cache, metrics and balanced
sampling retain their existing attribution. The independently added Text-EDU
module is still separate; neither new arm includes it yet. A promising encoder
would need a new matched encoder-only versus encoder-plus-EDU comparison before
claiming that the module adds value on that backbone.

## Code and reproduction

- `scripts/analyze_validation_errors.py`: validation-only confusion/domain/error
  overlap diagnostics, with no source-text export.
- `experiments/train_encoder_pilot.py`: paired recipe, accumulation, bounded
  training, best-checkpoint reload and validation predictions.
- `scripts/prepare_encoder_pilot.py`: immutable private code/data package.
- `cloud/kaggle/run_encoder_pilot.py`: source/data hash checks, both CUDA smokes,
  then serial formal runs; no effect on other kernels.

Local tests use real tiny RoBERTa/DeBERTa architectures for finite gradients,
matched head initialization, dynamic padding, partition integrity and a complete
synthetic smoke/reload. They are engineering checks, not pretrained performance.
The actual DeBERTa tokenizer was also loaded and exercised locally.

```bash
python -m pip install -r experiments/requirements-encoder-pilot.txt
python -B -m unittest experiments.test_encoder_pilot \
  scripts.test_encoder_packaging scripts.test_validation_diagnostics

python scripts/prepare_encoder_pilot.py \
  --cache /absolute/path/to/resources/experiments/same_window_full_v3 \
  --output outputs/paired_encoder_pilot_package_v2
```

The generated dataset contains private train/validation text and must stay
private and outside Git. Dataset creation defaults to private; do not pass
`--public`. Formal GPU execution is a separate explicit submission:

```bash
# Dataset v1 already exists privately. Version it only when ready to launch.
kaggle datasets version -p outputs/paired_encoder_pilot_package_v2/dataset \
  -m "Nested archive bootstrap fix" --dir-mode zip --keep-tabular
# Wait until Kaggle reports this dataset ready before pushing the kernel.
kaggle kernels push -p outputs/paired_encoder_pilot_package_v2/kernel
```

Kernel: `zhengyingqi/ee6405-paired-encoder-pilot-v1`.
Metrics and verified small evidence will be reported when available. A submitted
or running state is not a completed result and is not evidence of reaching 97%.

Validation has already been used extensively in this project; this is adaptive,
post-hoc development. Any promising pilot needs further matched-seed confirmation
before a frozen test comparison. No external evaluation improvement is implied.
