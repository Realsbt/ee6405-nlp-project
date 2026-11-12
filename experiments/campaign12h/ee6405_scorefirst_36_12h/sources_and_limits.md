# Sources and limits

- `top5_before.md` and `top5_recomputed.json` use the actual S prediction arrays from E01-E07, E25-E26 and E30-E31.
- All verified candidates have 960 identical S item IDs, four probability columns, finite non-negative probabilities, and row sums within 1e-5 of 1.
- Metrics were recomputed with the repository `classification_metrics` implementation and fixed label order.
- E08/E09/E11-E14/E20-E21 have no saved full S probability file in the ranking directory and were not inserted into the verified ranking.
- E15/E16 and other blocked/planned tasks do not participate in the ranking.
- E16 was checked first and not retrained: the existing record is RESOURCE_BLOCKED because `microsoft/deberta-v3-base` was unavailable locally/offline.
- H values in `top5_after.md` are historical descriptive reuse only; H was already exposed and was not used for selection.
- HART test and HC3 were not loaded. No claim of independent blind-test generalization is made.
- No second method, second seed, new fusion, or hyperparameter search was run.
