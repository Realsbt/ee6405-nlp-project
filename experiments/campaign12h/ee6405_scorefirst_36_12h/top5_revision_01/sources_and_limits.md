# Sources and limits

- Ranking source: `top5_recomputed.json`, scope `S_dev_selected`.
- Every included candidate has an ordered 960-item S binding and normalized
  `(960, 4)` probabilities.
- E01/E02/E03/E04/E25/E26 were reconstructed from the locked B0 checkpoint;
  E04 is excluded as an exact alias of E01.
- E16 was trained once with the fixed DeBERTa-v3-base configuration, then
  re-inferred from its locked best checkpoint. Its result is included in the
  complete candidate list but does not enter Top 5.
- E05/E06/E07 remain excluded because their original fit state and generation
  time ID binding cannot be recovered without refitting.
- E08/E09/E11/E12/E13/E14 remain snapshot-only/undertrained and have no full
  ID-bound S prediction in this revision.
- E20/E21/E30/E31 remain excluded because their required generation-time
  binding evidence is unavailable.
- No HART test or HC3 data was loaded, and no new method, seed, fusion, or
  hyperparameter search was run.
