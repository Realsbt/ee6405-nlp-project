# Reuse and contribution attribution

Audit date: 2026-11-11. Scope: code, data, weights and measured results actually
used by the Text-EDU fusion extension. Ideas reviewed but not implemented or
used as experimental evidence are intentionally excluded.

## Evidence sources

The following identifiers freeze the reviewed versions. Branch names alone may
move; use these full commit IDs when retrieving evidence with `git show`.

| ID | Source | Reviewed commit |
| --- | --- | --- |
| S1 | Shared project, `origin/main` snapshot | `b89517fb91178ef66919f3b8ab4b25371150c429` |
| S2 | Earlier same-window implementation on `yingqi` | `bbe7d262cec476ef6d44d49a7fac847e4e07fd2c` |
| S3 | Earlier Kaggle workflow on `yingqi` | `b298081d0df45196b62d2e62227e303fcc5a885e` |

## What the fusion extension actually reuses

| Reused item | Origin and evidence | Use in this extension | Credit boundary |
| --- | --- | --- | --- |
| Frozen HART source-group partitions, labels and metrics | S1; `experiments/prepare_data.py`, `common.py`, `neural.py` | Existing train/validation/test grouping, label order, balanced sampler, padding/collation and classification metrics | Shared project infrastructure, not newly designed for fusion |
| Base encoder and classifier components | S1; `experiments/neural.py`; upstream sources in `experiments/SOURCES.md` | `build_model` supplies the backbone; upstream `ClassificationHead` and initialization are reused | RoBERTa pretraining and upstream RACE components retain original attribution |
| Resource relocation | S1; `experiments/paths.py`, `docs/REPRODUCING.md` | Existing `EE6405_STORAGE`/`EE6405_UPSTREAM` and recorded-path mapping | Actual dependency is current shared code, not a new portability framework |
| Same-window cache and indexed loader | S2; `experiments/same_window_controls.py`; external `same_window_full_v3` manifest | Read existing token IDs/EDU spans without reparsing or changing the cache | Earlier same-window work is reused, not repeated as a new fusion achievement |
| EDU aggregation/head scaffold | S2; `EDUPoolingModel` in `experiments/same_window_controls.py` | Adapt its one-encoder/projection/classifier construction; replace the readout with CLS plus optional EDU residual | The scaffold is adapted; attention, gate and matched fusion variants are the new extension |
| CE training and checkpoint helpers | S2/S3; `experiments/train_same_window.py`, `train_neural.py` | Reuse training loop, sampler, validation selection, checkpoint writer and prediction format | New work is injectable factories, validation-only screening and additional provenance, not a new optimizer/training system |
| Earlier Kaggle resource package and runner | S3; `cloud/kaggle/`; private `ee6405-same-window-gpu-v1` inputs | Reuse resource archives, dataset discovery and extraction for actual fusion GPU execution | Earlier T4 smoke applies to `text/graph/shuffled/edu`, NOT `fusion_*`; new fusion validation has its own outputs |
| Existing main results | S1; `experiments/reports/comparison/REPORT.md` | Retain RoBERTa+SupCon as a historical strong comparator | Published/saved group results, not newly trained fusion results |
| This extension's completed first-stage GPU runs | `outputs/kaggle_fusion_experiment_v1/edu_fusion_experiment_v1`; checkpoint/source/cache hashes verified | Preserve five complete runs when continuing after the first job ended; rerun the incomplete gated seed 42 from base pretraining | Own earlier execution, not five new continuation trainings; these task checkpoints are reused for final evaluation, not to initialize missing runs |

## Weights and results

Fusion training starts from the shared base-pretrained RoBERTa weights, not a
teammate's task checkpoint. No task checkpoint from another research branch
has been used to initialize fusion. The exact pretrained-weight, upstream-file
and cache hashes are recorded in each run's `provenance.json` and `config.json`.

Historical main results are used only where explicitly labeled as historical
comparators. `fusion_cls` has a different head and must be newly trained for
the matched module comparison; old scores/checkpoints cannot substitute for it.
If a teammate's checkpoint or predictions are actually used later, record the
source, hash and role here before reporting the derived results.

## New work and current evidence

The fusion extension adds:

- The `CLS + g * EDU_context` residual readout with attention and a learned gate.
- Four matched-head fusion variants and common-initialization checks.
- Validation-only screening, a serial fusion queue and explicit final-test opt-in.
- Frozen-checkpoint reload checks and predefined validation-example export.
- Four real-backbone CPU smoke runs, their reload checks, and 14 initial unit tests.
- Three additional tests for the executable GPU candidate-selection protocol;
  these and the existing foundation form 39 relevant experiment tests, of which
  17 are new to this fusion module. Unrelated new modules are not counted here.

On 2026-11-12, all four fusion CUDA smokes passed on Tesla T4. Four complete
seed-3407 validation runs selected gated fusion at 91.3643% Macro-F1 versus
90.6435% for matched CLS (+0.7208 percentage points). This is the selection
validation set, not independent confirmation or test improvement. CLS seed 42
also completed; gated seed 42 was interrupted and is not counted as complete.
Five completed checkpoints and their source/cache hashes have been checked.
The first job ended with `CANCEL_ACKNOWLEDGED` after about 12 hours; the API
provided no failure explanation. Final test and external fusion results remain
unestablished. Earlier model scores, older GPU smoke and reused cache audits
must keep their original condition and phase labels in tables and figures.

## Suggested report statement

> This extension builds on the group's frozen HART protocol, RoBERTa/RACE
> components, sampling and evaluation utilities, the earlier same-window EDU
> cache, and the existing private Kaggle resource package. These reused assets
> are not newly created for this extension. The new contribution is the residual text-EDU attention/gating
> module, matched fusion variants, validation-only screening and its new
> implementation evidence. No teammate's task checkpoint was used to
> initialize this model.
