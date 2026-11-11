# Yingqi contribution report: fair-window mechanism ablation

Updated 2026-11-11. This report separates completed evidence from formal GPU
results that still need to run.

## Original problem

The existing HART table shows that RoBERTa and RACE both reach about 90% Macro-F1.
RACE does not beat RoBERTa on the three-seed mean, but the comparison changes
several factors at once: visible text range, EDU pooling, graph message passing
and parameter count. The old table therefore cannot identify why the complex
model fails to gain accuracy.

## Work completed

1. Built four CE-only controls over the exact same visible 512-token prefix:
   `text`, `graph`, `shuffled` and `edu`.
2. Completed and hashed the 15,984-record graph cache. No records failed or
   remain pending. Source groups remain disjoint and no visible EDU extends
   outside the shared prefix.
3. Added deterministic, label-independent relation reassignment. It preserves
   graph topology and each graph's relation histogram.
4. Recovered parser offsets for audited Unicode compatibility cases without
   changing model input text. Content changes and invalid EDU boundaries remain
   rejected.
5. Added a formal CUDA training entry point, saved-result protocol, unit tests
   and CPU forward/backward smoke checks for all four conditions.
6. Consolidated external transfer and editing-pair diagnostics from existing
   checkpoints to determine which claims need stronger evidence.
7. Packaged a private, minimal Kaggle dataset and passed all four training-only
   smoke conditions on a Tesla T4 before submitting the first formal seed.

## Data and engineering results

| Result | Value |
|---|---:|
| Verified cache records | 15,984 / 15,984 |
| Train / validation / test | 11,190 / 1,600 / 3,194 |
| Failed / pending | 0 / 0 |
| Cross-split exact visible-prefix duplicates | 0 |
| Hidden EDU nodes | 0 |
| Relation nodes changed | 467,369 / 638,038 (73.25%) |
| Directed edge types changed | 1,869,476 / 2,552,152 (73.25%) |
| Graphs unchanged by the permutation | 36 / 15,984 (0.23%) |
| Repository tests after integration | 22 passed |
| Four-condition CPU smoke | 4 / 4 passed; test split not evaluated |
| Four-condition Kaggle CUDA smoke | 4 / 4 passed on Tesla T4; test split not evaluated |

The parameter counts are 125.24M total / 7.68M trainable for `text`, 128.62M /
11.06M for `graph` and `shuffled`, and 125.31M / 7.75M for `edu`. Parameter
matching is improved but still not exact, so the paper must report the counts.
Peak allocated CUDA memory was 653.89 MiB for `text`, 838.98 MiB for `graph`,
838.99 MiB for `shuffled` and 655.39 MiB for `edu`. All four runs had finite
gradients, updated the classifier and last encoder layer, and left frozen
embeddings unchanged. The raw evidence is in `kaggle_gpu_smoke.json`.

## Existing model diagnostics

The frozen 359-text MixSet subset exposes poor transfer. RoBERTa CE falls from
90.75% HART Macro-F1 to 39.01%; RACE SupCon falls from 90.84% to 33.38%.
Probability blending and stacking improve the inspected HART test score but do
not retain the gain externally. The lightweight length + structure model reaches
52.67% on HART and 29.16% on the subset.

In 41 original AI texts, RACE creator recall is 9.76%. In paired human-origin
texts, creator false positives rise from 4% before AI polishing to 40% after
polishing. This is an observational result. It supports a hypothesis that the
detector relies on expression style, but it does not prove a causal mechanism.

## Work that should not be repeated

The main branch already completed the RoBERTa/RACE by CE/SupCon matrix over
three seeds. The `yuzhe` branch already tested unfreezing 1, 2, 4, 6 and 12
layers, a projection head, reduced SupCon weight, auxiliary heads, their
2-by-2-by-2 combinations, three-seed confirmation and an equal-weight
ensemble. Its six-layer seed 3407 result exceeded 92% Macro-F1, but the
three-seed mean was effectively unchanged from the existing RoBERTa SupCon
mean. These settings should not be rerun unchanged or presented from a single
seed as a stable improvement.

The useful part of `zsp` is the execution design: external resource roots,
environment variables, CUDA/BF16 checks, serial queues, resumable jobs and
checkpoint preservation. Current `main` already contains the path abstraction
and reproduction setup, while `cloud/kaggle/` now supplies the missing portable
GPU entry point. The older branch should therefore be reused selectively rather
than cherry-picked wholesale.

## Next experiments with distinct value

1. Finish the same-window `text`, `graph`, `shuffled` and `edu` matrix over
   three seeds. This is the only current experiment that can separate graph
   relation information from text range and EDU aggregation.
2. Reuse the existing head/middle/tail multiview inference implementation on a
   saved checkpoint. Select aggregation weights on validation data only and
   report the additional inference cost and larger information budget.
3. After the four-condition runs, test validation-selected probability
   ensembles only when component errors are measurably diverse. Compare against
   the existing 91.70% equal-weight ensemble rather than another single-seed
   maximum.
4. Add leave-one-domain-out evaluation and low-false-positive calibration.
   These address the observed external-transfer failure and are a stronger
   paper contribution than another small in-domain HART gain.
5. Treat 90% as a reproducibility floor, not the research target. Any claimed
   improvement should use three seeds, validation-only selection and paired
   comparisons; a single result above 92% is exploratory until it survives that
   protocol.

## Formal GPU work

Run four conditions with seeds 3407, 42 and 2026, for 12 formal runs. The primary
matched-seed comparisons are `graph - edu`, `graph - shuffled` and `graph - text`.
Each run must save the frozen configuration, development curve, best checkpoint,
validation and test probabilities, parameter counts, runtime and peak memory.

The seed 3407 four-condition job was submitted to Kaggle after the CUDA smoke
passed. It is not counted as completed evidence until its four `result.json`
files and predictions have been downloaded and checked.

The paper may currently claim that the fair-window implementation and data
integrity checks are complete. It may not claim that graph structure helps or
fails to help until all 12 GPU runs finish.

## Deliverables

- `experiments/reports/same_window/figures/`: six presentation-ready figures.
- `experiments/reports/same_window/figure_data.csv`: source values for all new charts.
- `experiments/reports/same_window/kaggle_gpu_smoke.json`: raw CUDA smoke evidence.
- `docs/SAME_WINDOW_ABLATION.md`: protocol and run commands.
- `experiments/same_window_controls.py` and `experiments/train_same_window.py`:
  implementation and formal training entry point.
