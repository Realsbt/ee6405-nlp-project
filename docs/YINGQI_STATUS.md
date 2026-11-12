# Yingqi contribution and execution snapshot

**Later results (2026-11-13):** see the [full experiment ledger](project-archive/README.md). The Kaggle continuation and Colab pilot finished; the dated snapshot below preserves their earlier state.

Snapshot: **2026-11-12 13:22 Singapore / 05:22 UTC**. This is a dated record,
not a live training dashboard. Machine-readable API states and artifact hashes:
[`status.json`](../experiments/reports/yingqi_snapshot/status.json) and
[`manifest.json`](../experiments/reports/yingqi_snapshot/manifest.json).

## Contribution boundary

| Added work in this branch | Reused foundation | What the evidence establishes |
|---|---|---|
| Fair-window text/graph/shuffled/EDU controls; parser offset recovery; cache/control audit | Shared HART split, RoBERTa/RACE, parser and metric infrastructure | A controlled single-seed comparison and a verified implementation, not a universal claim about graphs |
| Text-EDU residual attention/gating and four matched-head variants | Earlier same-window cache/pooling scaffold; shared CE trainer, sampler and classifier components | Implemented module with real CPU/CUDA checks and completed validation screening |
| Validation-only candidate selection, checkpoint provenance, continuation packaging and reporting | Existing Kaggle resources and base-pretrained encoder | Completed runs can be preserved without treating partial training as complete |
| Frozen-model blending and source-group cross-fitted stacking | Existing CE/RACE/TF-IDF model predictions | Actual CPU-fitted ensemble score; no new GPU encoder training |
| External transfer, low-FPR and editing diagnostics | Existing checkpoints and conservatively mapped external records | In-domain gains fail to transfer to this external subset |
| Gate-strength, segmentation/consistency controls and validation-gated DFS runners | Main fusion module and shared training/evaluation utilities | Implemented follow-up code and CPU tests; formal GPU benefits remain unestablished |

The independently added project module is **RoBERTa + residual Text-EDU fusion**,
not RoBERTa or CLS alone. In plain language: CLS reads the whole visible passage,
EDUs represent its smaller discourse units, attention chooses which units matter,
and a gate controls how much local information supplements the global reading.
Segmenting into EDUs creates the units; equal averaging or attention/gating are
different ways to combine those units afterwards.

EDU pooling is not claimed as absent from all earlier group work: RACE already
uses EDU-related components. Likewise, stacking, attention and gates are known
methods. The contribution is their concrete project extension, controls,
execution and analysis. See [pinned reuse attribution](REUSE_ATTRIBUTION.md).
Implementation and documentation used AI assistance; this does not replace the
course GAI declaration or the group's verification of actual member roles.

## Completed results

All values below are **Macro-F1**, not accuracy. "Four conditions" does not mean
"four seeds". Planned same-window seeds are **3407, 42 and 2026**.

### Same-window GPU controls

One completed seed, 3407; CE; common visible 512-token prefix; pretrained
initialization; last encoder layer fine-tuned. Train/dev/test: 11,190/1,600/3,194.

| Condition | Validation | Test | Best epoch |
|---|---:|---:|---:|
| text | 91.3779% | 91.2798% | 10 |
| graph | 89.3164% | 89.6932% | 9 |
| shuffled | 89.3324% | 90.0694% | 9 |
| edu | 90.9089% | 90.8919% | 10 |

[`seed_3407/`](../experiments/reports/same_window/seed_3407/) includes all four
results, configs, parameter counts and complete dev/test predictions. This
single-seed ranking favors text and does not support a relation-label advantage;
the remaining eight planned trainings are not complete. It is not a three-seed
negative finding. Weights remain outside Git.

### CPU ensemble and transfer

| Frozen-model method | HART test | MixSet subset |
|---|---:|---:|
| RoBERTa CE, seed 3407 | 90.75% | 39.01% |
| RACE SupCon, seed 3407 | 90.84% | 33.38% |
| CE/RACE probability blend | 92.10% | 36.86% |
| CE + RACE + TF-IDF stacking | **93.61%** | **36.22%** |

Stacking is a separately trained small CPU meta-model: centered log
probabilities plus probabilities; StandardScaler + logistic regression, C=1.
The candidate grid was selected with four source-group development folds,
then refit on all development records. The meta-model is genuinely trained,
not an untrained average or a new MoE encoder.

These are reportable measured results. They use one underlying neural seed and
an already-inspected HART test set; the dev labels had also selected the neural
checkpoints. Consequently meta-level OOF is not an independent evaluation.
Do not describe 93.61% as a three-seed mean, a single-model score or independent
confirmation of outperforming a teammate. Teammate single-seed best scores
likewise need their seed/selection labels, rather than a different standard.

MixSet is a fixed **359-text / 91-source-group subset**, not the full official
benchmark. Its failure to preserve HART gains motivates reliability work.
At a dev-calibrated nominal 1% FPR threshold, actual external stacking FPR/TPR
is 18.80%/5.50%; calibration does not guarantee 1% FPR after distribution shift.
See [external report](../experiments/reports/yingqi_snapshot/external/REPORT.md).

### Fusion GPU screening, not final test

| Condition | Seed | Validation Macro-F1 | State |
|---|---:|---:|---|
| fusion_cls | 3407 | 90.6435% | Training complete |
| fusion_mean | 3407 | 90.8501% | Training complete |
| fusion_attention | 3407 | 90.8956% | Training complete |
| fusion_gated | 3407 | 91.3643% | Training complete; validation-selected |
| fusion_cls | 42 | 91.2243% | Training complete |

All four separate fusion CUDA smokes passed on Tesla T4. Gated's screening gap
is +0.7208 percentage points over matched CLS; this is a validation-selected gap,
not final test evidence. The selected candidate and five complete validation
prediction files are under [fusion evidence](../experiments/reports/yingqi_snapshot/fusion/).
No final three-seed fusion test result or external fusion result is established
by this snapshot.

## Incomplete GPU work and uploaded code

| Job / experiment | Snapshot state | What remains |
|---|---|---|
| `zhengyingqi/ee6405-same-window-seed-3407` | `COMPLETE` | Planned same-window seeds 42/2026 have no completed evidence |
| `zhengyingqi/ee6405-text-edu-fusion-experiment-v1` | `CANCEL_ACKNOWLEDGED` | Five complete trainings preserved; gated seed 42 was interrupted |
| `zhengyingqi/ee6405-text-edu-fusion-continuation-v1` | `RUNNING` | Complete/verify missing gated 42, CLS 2026, gated 2026 and final reload/test outputs |
| Gate-strength / segmentation-consistency / calibration and DFS | Code and CPU checks available | Formal GPU runs/results not complete |

The canceled job's API state does not establish why it stopped. The continuation
uses verified completed checkpoints for evaluation, but starts missing runs
from base pretraining. It is not interrupted-epoch optimizer resume. Runs may
advance after the snapshot; until their output is downloaded and verified,
they are not added to the completed count.

Code is uploaded even where results remain pending:

- Main model/training: `experiments/edu_fusion.py`, `train_edu_fusion.py`,
  `run_fusion_queue.py`, `evaluate_edu_fusion.py`.
- Kaggle launch/continuation: `cloud/kaggle/run_edu_fusion.py`,
  `resume_edu_fusion.py`; collection and packaging under `scripts/`.
- Gate strength: `experiments/fusion_scale_controls.py`,
  `train_fusion_scale.py`, `evaluate_fusion_scale.py`.
- Segmentation and consistency: `experiments/fusion_extensions.py`,
  `train_fusion_extensions.py`, `evaluate_fusion_extensions.py`.
- Budgeted validation search: `experiments/fusion_dfs_policy.py`,
  `cloud/kaggle/run_fusion_dfs.py`, `scripts/supervise_fusion_dfs.py`.
- CPU stacking: `experiments/train_stacking.py`, a portable adaptation of the
  original local fitting procedure, with unit tests. Historical results retain
  their original recorded script/input hashes rather than being relabeled as a
  run of this new entry point.

Full-matrix follow-up plans are historical; the current supervisor uses
validation-gated DFS. Do not count all planned runs as launched or completed.
Protocol and limitations: [fusion](EDU_FUSION.md),
[follow-up controls](FUSION_NEXT_STEPS.zh-CN.md),
[current supervision](FUSION_SUPERVISION.zh-CN.md).

## Reproduction

Saved-evidence checks require only standard-library Python:

```bash
python3 scripts/check_yingqi_results.py
python3 scripts/check_results.py
```

The first checks this contribution snapshot; the second checks the shared
historical results. Neither downloads weights or performs model inference.

To fit stacking, supply frozen development metadata and corresponding prediction
files in CE/RACE SupCon/TF-IDF order. Metadata must contain `item_id`, `label`,
`group_id` and the frozen duplicate/source `split_block_id`. Base prediction files
contain IDs, true labels and four label-keyed probabilities:

```bash
python -m experiments.train_stacking \
  --development /absolute/path/to/val.jsonl \
  --dev-predictions /absolute/path/to/ce_val.jsonl \
    /absolute/path/to/race_val.jsonl /absolute/path/to/tfidf_val.jsonl \
  --output outputs/stacking_reproduction
```

Optional `--test /absolute/path/to/test.jsonl --test-predictions ...` evaluates
the same ordered components only after saving selection. Shared GPU resources
and environment setup remain in [REPRODUCING.md](REPRODUCING.md). Fusion commands
and pending experiment commands are in their linked protocols above.

Publication checks passed: **64 experiment tests and 24 script tests**; both
saved-evidence checkers passed. The portable stacking entry point was also fitted
again on the frozen development inputs and evaluated on the existing test
predictions: it selected logistic C=1, reproduced Macro-F1 **0.9360969188130479**,
and matched every historical test probability exactly (maximum absolute
difference **0.0**). This is a reproducibility check, not a new independent seed.
Historical dev prediction files without labels are supported by aligning IDs to
the frozen labeled metadata; any supplied labels/groups are checked for agreement.

This snapshot includes code, tests, metrics, ID/label/probability predictions and
hashes. It excludes private source articles, raw-text explanation exports,
checkpoints, fitted joblib models and credentials. GPU checkpoints are retained
in the private runtime/resource storage, not silently promised in Git.
