# Top 5 after E16 (revised ID-bound verification)

E16 was the only newly authorized method. Its official resources were available,
and the fixed formal run completed with seed 3407, four epochs, 2800 optimizer
updates, and best epoch 3. It was independently re-inferred from `best.pt` with
ordered S `item_id` binding.

| Rank | Method | Accuracy | Macro-F1 | J | Status |
|---:|---|---:|---:|---:|---|
| 1 | E26 | 94.4792% | 92.7021% | 93.5906% | COMPLETE |
| 2 | E01 | 94.3750% | 92.5641% | 93.4696% | COMPLETE |
| 3 | E03 | 94.3750% | 92.5641% | 93.4696% | COMPLETE |
| 4 | E02 | 94.4792% | 92.4308% | 93.4550% | COMPLETE |
| 5 | E25 | 93.0208% | 90.2467% | 91.6338% | COMPLETE |

E16 ranked below the verified Top 5: Accuracy 82.7083%, Macro-F1 69.7720%,
J 76.2402%. Its per-class F1 was 89.0323%, 77.3389%, 86.6298%, and 26.0870%.
Relative to E01 it corrected 19 previously wrong items but made 131 previously
correct items wrong, so it is not a qualifying improvement.

`S_oof` and `S_refit` remain unavailable. H remains historical descriptive
evidence only. `test_partition_loaded=false` and `hc3_loaded=false`.
