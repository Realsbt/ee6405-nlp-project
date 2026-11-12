# Bounded encoder improvement pilot

Date: 2026-11-12. Goal: investigate a stronger candidate, not promise 97%
Macro-F1. The historical stacking result remains 93.61% test Macro-F1; this
experiment does not relabel it as a new single-model score.

## Execution status

On 2026-11-12, the private Kaggle dataset upload succeeded, but the kernel push
was rejected with `Maximum weekly GPU quota of 30.00 hours reached`.
The CLI exited with code zero despite that error, so its exit code is not proof
of a successful submission. **No new encoder GPU smoke or formal training has
started, and no pilot performance result exists.** Existing fusion training was
not cancelled or modified.

The original local `paired_encoder_pilot_package_v1` is preserved. A subsequent
bootstrap fix supports nested, auto-unpacked Kaggle code archives; the updated
sources are frozen separately in `paired_encoder_pilot_package_v2`. The uploaded
private dataset still contains v1 and must be explicitly versioned before a
future submission uses v2. Do not repeatedly resubmit while quota is exhausted.
Colab allocated a Tesla T4 (15,360 MiB), verified with its preinstalled
`nvidia-smi` at 06:33:40 UTC. No private inputs were uploaded and no training
dependencies were installed there. Upload and software execution await user
confirmation. A connected runtime is not a training result.

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
| Memory | Gradient checkpointing; FP16 on CUDA |
| Hard limit | 14,400 training seconds per arm; hitting it is labeled incomplete |
| Selection | Validation Macro-F1; preserve both results; no automatic test evaluation |

At most eight **formal** epochs total, plus two one-batch CUDA smokes.
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
