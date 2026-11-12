# Information-theoretic and stylometric detectors vs the composited Top 3

**Scope.** Every number in this report is computed on this machine, on the same
published 3,194-item test set (798 / 798 / 1371 / 227). Nothing is copied from
the original host's result files.

| Block | How it was produced |
|---|---|
| Published five | `scripts/reinfer_published_local.py` reloads all 13 checkpoints and recomputes the probabilities |
| Composited Top 3 (E30, E26, E01) | `scripts/reconstruct_campaign_top3.py` re-runs the campaign recipes on the published test set |
| M1 / M2 / M3 | `scripts/eval_info_on_published_test.py`, trained on a pool that excludes every published test group |
| Fusion | `scripts/compare_info_vs_top3.py`, group-disjoint two-fold cross-fitted weight |

## Why the campaign's own `S` partition cannot be used

The campaign scored everything on its own `S` partition (960 items). That
partition is contaminated for a comparison against the published models:

| Partition | Items | Present in the published models' training data |
|---|---:|---:|
| `S` | 960 | 657 (68.4%) |
| `H` | 640 | 461 (72.0%) |

Any model that saw those items during training will look better than it is. The
published test set is the only set the published checkpoints never saw, so all
comparisons below use it.

## Local re-inference parity

Re-running each checkpoint here reproduces the archived predictions to floating
point, which is what makes the comparison fair rather than a re-statement of the
original numbers.

| Model family | Max abs difference vs archived probabilities |
|---|---:|
| TF-IDF + LR | 4.44e-16 |
| RoBERTa (3 seeds) | 6.66e-05 |
| RACE (6 runs) | 1.34e-04 |

The neural residuals are GPU non-determinism, not a code difference: the same
checkpoint, the same tokenizer, the same batching, and the same evaluation
function. TF-IDF is exact because it is deterministic on CPU.

## The three methods

| ID | Method | What it measures | Features |
|---|---|---|---|
| **M1** | Entropy + compressibility profile | How predictable the surface form is | Shannon entropy of characters, conditional entropy of character bigrams, zlib compression ratio, sentence-length burstiness, sentence density, character diversity |
| **M2** | Stylometric feature vector | Authorship-attribution style analysis | 100 function-word frequencies (Burrows), 32 punctuation frequencies, type-token ratio, hapax ratio, log token count, mean word length, sentence-length mean/SD/median/max |
| **M3** | Character n-gram cross-entropy | Generative information-theoretic fit | Smoothed character 4-gram model per class; classify by lowest cross-entropy |

M1 and M2 use logistic regression on standardized features. M3 is generative: it
assigns the class whose model gives the text the lowest cross-entropy.

## Standalone results

| Method | Accuracy | Macro-F1 | J |
|---|---:|---:|---:|
| M3 character n-gram cross-entropy | 78.62% | 60.95% | 69.78% |
| M2 stylometric feature vector | 69.47% | 64.35% | 66.91% |
| M1 entropy + compressibility | 40.48% | 39.72% | 40.10% |

None of the three is competitive alone. M1 is at chance on four classes, which is
the honest result: raw entropy and compression ratio do not separate these four
provenance classes.

## The composited Top 3, rebuilt on the published test set

The campaign's leaderboard ranked E30 first (93.61% J on `S`), E26 second
(93.59%), and E01 third (93.47%). Re-running those exact recipes on the published
test set gives:

| Rank | Method | Recipe | Accuracy | Macro-F1 | J |
|---|---|---|---:|---:|---:|
| 1 | **E30** | equal probability mean of E01 + E05 + E25 | **94.24%** | **92.02%** | **93.13%** |
| 2 | E01 | B0 RoBERTa+CE, first 512 tokens | 93.71% | 91.14% | 92.43% |
| 3 | E26 | head/middle/tail 512-token window mean | 93.68% | 91.17% | 92.42% |

The ordering is preserved: E30 still leads. E01 and E26 swap places, but they are
within 0.01 pp of each other, which is noise.

Component scores on the same set:

| Component | Accuracy | Macro-F1 | J |
|---|---:|---:|---:|
| E01 B0 single window | 93.71% | 91.14% | 92.43% |
| E05 TF-IDF word+char LR | 86.38% | 77.65% | 82.02% |
| E25 MC-Dropout ×5 | 92.39% | 89.66% | 91.03% |
| E26 head/middle/tail | 93.68% | 91.17% | 92.42% |
| E30 equal mean of E01+E05+E25 | 94.24% | 92.02% | 93.13% |
| E31 nonnegative fusion of E01+E05+E25 | 93.83% | 91.50% | 92.66% |

E30 beats every one of its own members, including the strongest (E01 at 93.71%).
That is the real finding: heterogeneous averaging helps, and it helps more than
the learned E31 weights do.

## 3 methods × composited Top 3

Fusion weight is chosen by group-disjoint two-fold cross-fitting: the test groups
are split in half, the weight is fitted on one half and applied to the other, then
the halves swap. The reported score is therefore out-of-fold and does not benefit
from tuning on the items it is scored on.

| Method | Partner | Accuracy | Macro-F1 | J | ΔJ vs partner |
|---|---|---:|---:|---:|---:|
| **M2 stylometric** | **E30** | **94.43%** | **92.42%** | **93.42%** | **+0.29** |
| M3 char n-gram | E30 | 94.40% | 92.28% | 93.34% | +0.21 |
| M1 entropy + compressibility | E30 | 94.24% | 92.02% | 93.13% | +0.00 |
| M2 stylometric | E01 | 93.80% | 91.27% | 92.53% | +0.10 |
| M2 stylometric | E26 | 93.71% | 91.20% | 92.45% | +0.03 |
| M3 char n-gram | E26 | 93.68% | 91.19% | 92.43% | +0.01 |
| M1 entropy + compressibility | E01 | 93.64% | 91.19% | 92.42% | −0.01 |
| M1 entropy + compressibility | E26 | 93.55% | 91.17% | 92.36% | −0.06 |
| M3 char n-gram | E01 | 93.55% | 90.84% | 92.19% | −0.24 |

Baselines for reference: E30 93.13%, E01 92.43%, E26 92.42%.

## What the best result actually supports

| Check | Result |
|---|---|
| Accuracy | 94.24% → 94.43% (**+0.19 pp**) |
| Macro-F1 | 92.02% → 92.42% (**+0.40 pp**) |
| J | 93.13% → 93.42% (**+0.29 pp**) |
| Paired bootstrap, 2000 resamples | mean **+0.19 pp**, 95% CI **[−0.09, +0.47]**, P(Δ>0) = **0.88** |

**The gain is not statistically distinguishable from noise.** The confidence
interval crosses zero. The direction is consistent — M2 helps all three partners,
and the two strongest fusions both pair with E30 — but the effect is a handful of
items.

## Comparison against the published five

| Model | Accuracy | Macro-F1 | J | Source |
|---|---:|---:|---:|---|
| TF-IDF + LR | 85.32% | 79.40% | 82.36% | published, re-inferred |
| RACE · CE | 92.83% | 89.92% | 91.38% | published, re-inferred |
| RACE · CE+SupCon | 93.08% | 90.37% | 91.73% | published, re-inferred |
| RoBERTa · CE | 93.45% | 90.72% | 92.08% | published, re-inferred |
| RoBERTa · CE+SupCon | 93.58% | 91.15% | 92.37% | published, re-inferred |
| E01 B0 single window | 93.71% | 91.14% | 92.43% | composited Top 3 |
| E26 head/middle/tail | 93.68% | 91.17% | 92.42% | composited Top 3 |
| **E30 equal-mean fusion** | **94.24%** | **92.02%** | **93.13%** | composited Top 3 |
| **M2 + E30** | **94.43%** | **92.42%** | **93.42%** | fusion |

**Yes, the composited models are better than all five published models.** E30
beats the strongest published model (RoBERTa · CE+SupCon, 93.58% / 91.15%) by
**+0.66 pp accuracy and +0.87 pp macro-F1**. M2 + E30 extends that to +0.85 pp and
+1.27 pp.

The margin is real but modest, and it comes from averaging rather than from a new
architecture: E30 is a plain mean of three existing predictors, two of which
(E05, E25) are individually weaker than RoBERTa · CE+SupCon.

## Conclusions

1. **The composited Top 3 does beat the published five.** E30 leads every
   published model on both metrics, computed locally on the same items.
2. **None of the three information-theoretic methods is a viable standalone
   detector.** The best, character n-gram cross-entropy, reaches 78.62% against
   94.24% for E30.
3. **M2 adds a small, unproven amount of complementary signal.** It is the only
   method that improves all three partners, and the best pairing gains +0.29 pp J.
   The bootstrap confidence interval still crosses zero.
4. **M1 adds nothing on the published test set.** Its earlier +0.21 pp on the
   campaign's `S` partition does not reproduce here — that partition was
   contaminated, and the apparent gain was an artifact of it.
5. **The effect size does not justify a new headline result.** A +0.19 pp accuracy
   change with a confidence interval spanning zero should be reported as
   "no reliable improvement", not as a win.

## Reproduce

```bash
python scripts/reinfer_published_local.py        # 13 checkpoints, local parity check
python scripts/reconstruct_campaign_top3.py      # E01/E05/E25/E26/E30/E31 on the published test set
python scripts/eval_info_on_published_test.py    # M1/M2/M3, group-disjoint training pool
python scripts/compare_info_vs_top3.py           # 3×3 fusion grid, bootstrap, figure
```

Outputs: `.local/local_reinference.json`, `.local/campaign_top3_published_test.json`,
`.local/info_methods_published_test.json`, `.local/info_vs_top3.json`,
`experiments/reports/comparison/info_vs_top3.png`.

## Sources

- Burrows, J. (2002). "Delta": a measure of stylistic difference and a guide to
  likely authorship. *Literary and Linguistic Computing* 17(3).
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System
  Technical Journal* 27.
- Stamatatos, E. (2009). A survey of modern authorship attribution methods.
  *JASIST* 60(3).
