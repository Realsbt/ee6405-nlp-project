# Group conformal review: retrospective pilot

Fixed alpha=0.10; all 13 published runs; deterministic SHA-256 group split.
Calibration groups: 399; evaluation groups: 400.

Coverage below refers to label-set coverage, not classification accuracy.
Singleton accuracy is conditional on returning one label; review cases remain unresolved.
These are previously inspected test predictions. Do not replace the original benchmark table.

| Model | Top-1 accuracy | Set coverage | All labels covered within group | Mean set size | Singleton rate | Singleton accuracy | Review rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| tfidf_lr_v1 | 85.06% | 98.00% | 92.25% | 1.789 | 46.19% | 97.56% | 53.81% |
| roberta_s3407 | 92.69% | 97.12% | 89.00% | 1.169 | 84.62% | 97.34% | 15.38% |
| roberta_s42 | 93.38% | 97.25% | 89.75% | 1.151 | 86.44% | 97.40% | 13.56% |
| roberta_s2026 | 93.06% | 97.38% | 90.25% | 1.182 | 84.25% | 97.55% | 15.75% |
| roberta_supcon_s3407 | 93.38% | 97.81% | 92.25% | 1.181 | 83.06% | 98.04% | 16.94% |
| roberta_supcon_s42 | 93.44% | 97.81% | 92.00% | 1.174 | 83.69% | 97.98% | 16.31% |
| roberta_supcon_s2026 | 93.38% | 97.81% | 92.00% | 1.159 | 85.12% | 97.94% | 14.88% |
| race_no_supcon_s3407 | 92.25% | 97.75% | 91.50% | 1.228 | 80.44% | 97.82% | 19.56% |
| race_no_supcon_s42 | 93.56% | 97.06% | 88.75% | 1.164 | 85.75% | 97.23% | 14.25% |
| race_no_supcon_s2026 | 92.12% | 96.62% | 87.50% | 1.193 | 82.81% | 96.75% | 17.19% |
| race_s3407 | 93.06% | 96.31% | 86.50% | 1.126 | 88.06% | 96.17% | 11.94% |
| race_s42 | 93.88% | 97.62% | 90.75% | 1.207 | 81.38% | 97.70% | 18.63% |
| race_s2026 | 92.12% | 96.69% | 87.25% | 1.171 | 83.94% | 96.50% | 16.06% |

## Evidence boundaries

The quantile uses one maximum nonconformity score per source group. Under exchangeable groups and a fixed model, the target is simultaneous coverage for a fresh group. The retrospective pilot cannot establish these assumptions or new-test validity.
Per-domain and per-class metrics are descriptive; there is no conditional coverage guarantee. Singleton accuracy has no 90% guarantee. HC3 is not evaluated: it has a different label space and domain distribution.
No model, alpha, partition or seed was selected using these results. The procedure does not change original argmax labels, use test groups to constrain output composition, or retrain checkpoints.
For prospective use, reserve source-disjoint calibration data before inspecting a fresh evaluation set; freeze model and alpha first. Review flags identify ambiguity under this score rule, not verified authorship.
