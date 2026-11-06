# GLPC-v0

GLPC-v0 is an independent train/validation-only prototype. It uses one local
RoBERTa-base encoder, the CLS vector for the global branch, and optional
sentence pooling from the same 512-token window. It does not use RST, GNN,
metadata, external APIs, or the HART test partition during development.

The four modes are:

| Mode | Creator head | Local branch | JS consistency |
|---|---:|---:|---:|
| `baseline` (B0) | no | no | no |
| `creator_aux` (B1) | yes | no | no |
| `creator_local` (B2) | yes | yes | no |
| `glpc` (B3) | yes | yes | yes |

Run from the repository root with the configured Python 3.11 environment:

```powershell
$py = "C:\Users\danie\anaconda3\envs\pt311\python.exe"
& $py -B -u -m experiments.train_glpc --mode baseline --seed 3407 --run-name b0_s3407
& $py -B -u -m experiments.train_glpc --mode creator_aux --seed 3407 --run-name b1_s3407
& $py -B -u -m experiments.train_glpc --mode creator_local --seed 3407 --run-name b2_s3407
& $py -B -u -m experiments.train_glpc --mode glpc --seed 3407 --run-name b3_s3407
```

Use `--smoke --run-name envcheck` for a small train/validation-only check.
Outputs are written below `experiments/glpc_runs/`, which is ignored by Git.
The trainer refuses to overwrite an existing run and records the configuration,
validation loss components, best validation macro-F1, and `test_partition_loaded:
false`. No GLPC test evaluator is provided until a mode is selected without
using test results.
