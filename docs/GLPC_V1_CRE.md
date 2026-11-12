# GLPC-v1 / CRE

CRE (Conditional Residual Experts) reuses the accepted B0 checkpoint and freezes
the entire B0 path. Each article is encoded once. The new adapter reads either
the CLS vector (`flat_cls`, `conditional_cls`) or fixed 128-dimensional
multi-scale evidence from visible tokens, sentence vectors, and adjacent
sentence changes (`conditional_multi`, `conditional_multi_pair`).

The three conditional residuals represent:

- `C`: human creation versus AI creation;
- `H`: human original versus AI-polished human text;
- `A`: AI-generated versus AI-humanized text.

They are combined with stable log-probabilities into the four fixed labels.
Only the final residual layers are zero initialized, so epoch 0 is exactly B0.
The pair objective uses same-group cross-label evidence only during training;
`group_id`, domain, model name, labels, and source names are never model inputs.

## Validation-only commands

Use the configured Python interpreter and run from the project root:

```powershell
$py = "C:\Users\danie\anaconda3\envs\pt311\python.exe"
& $py -m experiments.test_cre
& $py -m experiments.verify_cre --run-dir experiments\cre_v1_runs\cre_v1_conditional_multi_pair_smoke3
& $py -m experiments.evaluate_cre --run-dir experiments\cre_v1_runs\cre_v1_conditional_multi_pair_smoke3
```

Training reads only HART `train` and `val`. It writes to
`experiments\cre_v1_runs`, includes an epoch-0 `best_overall`, and marks
`ZERO_CORRECTION_SELECTED` when that candidate remains best. No HART test or
HC3 evaluation is part of this experiment.
