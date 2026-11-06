# GLPC-v1 residual pair experiments

GLPC-v1 keeps a trained B0 checkpoint fixed and trains only a zero-initialized
four-class residual head. The residual is added to the frozen B0 logits:

```text
final_logits = B0_logits + residual_logits
```

The training-only pair relation uses `group_id` and is never an inference
input. Only groups containing one of the following same-source cross-label
pairs are used:

- `human_ai_polished` (label 1) versus `ai_generated` (label 2)
- `ai_humanized` (label 3) versus `ai_generated` (label 2)

The relation is called a same-source cross-label pair, not a verified
edit-before/after pair.

| Run | Residual features | Pair ranking loss |
|---|---|---:|
| R1 | CLS MLP | no |
| R2 | CLS MLP | yes |
| R3 | sentence/local statistics | no |
| R4 | sentence/local statistics | yes |

All four runs start from the same saved B0 checkpoint. The base encoder and
classifier remain frozen; inference still receives one article only.

```powershell
$py = "C:\Users\danie\anaconda3\envs\pt311\python.exe"
Set-Location D:\6405_prj\ee6405-nlp-project

& $py -B -u -m experiments.train_glpc_v1 --mode r1_mlp --seed 3407 --run-name r1_s3407
& $py -B -u -m experiments.train_glpc_v1 --mode r2_mlp_pair --seed 3407 --run-name r2_s3407
& $py -B -u -m experiments.train_glpc_v1 --mode r3_local --seed 3407 --run-name r3_s3407
& $py -B -u -m experiments.train_glpc_v1 --mode r4_local_pair --seed 3407 --run-name r4_s3407
```

Each run reads only HART train and validation. Test evaluation is intentionally
not part of this exploratory trainer.
