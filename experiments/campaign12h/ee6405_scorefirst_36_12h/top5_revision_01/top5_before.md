# Top 5 before E16 (revised ID-bound verification)

- Scope: `S_dev_selected`, 960 samples.
- Ordered item IDs, labels, data hashes, and checkpoint-reinferred probabilities were verified.
- `S_oof` and `S_refit` are not available; this is not a test/HC3 result.
- E04 is an exact probability alias of E01 and is not a separate slot.

| Rank | Method | Accuracy | Macro-F1 | J |
|---:|---|---:|---:|---:|
| 1 | E26 | 94.4792% | 92.7021% | 93.5906% |
| 2 | E01 | 94.3750% | 92.5641% | 93.4696% |
| 3 | E03 | 94.3750% | 92.5641% | 93.4696% |
| 4 | E02 | 94.4792% | 92.4308% | 93.4550% |
| 5 | E25 | 93.0208% | 90.2467% | 91.6338% |

E01/E03 are tied on all reported classification metrics. E26 and E02 are tied on
accuracy but not Macro-F1. Per-class F1 and correction/impact counts are in
`top5_recomputed.json`; each row links to an ID-bound prediction file.
