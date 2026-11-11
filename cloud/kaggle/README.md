# Kaggle GPU execution

The private dataset `ee6405-same-window-gpu-v1` contains only the files needed
for training: repository modules, the indexed graph cache without `parsed.jsonl`,
the RoBERTa base weights and the pinned upstream RACE source. The runner accepts
either mounted directories or the `repository.zip` and `resources.zip` archives
produced by the Kaggle CLI.

`run_same_window_smoke.py` checks all four conditions on a real CUDA device. It
does not open the test split. Each output records the GPU name, peak allocated
memory, finite gradients, updated trainable layers and unchanged embeddings.

`run_same_window_seed.py` runs the first formal four-condition matrix for seed
3407 and updates a compact `summary.json` after each completed condition.

After smoke passes, formal runs should use separate outputs for each condition
and seed:

```bash
python -m experiments.train_same_window \
  --condition graph --seed 3407 \
  --cache /kaggle/input/ee6405-same-window-gpu-v1/resources/experiments/same_window_full_v3 \
  --output /kaggle/working/same_window_graph_s3407 \
  --device cuda
```

Run `text`, `graph`, `shuffled` and `edu` for seeds 3407, 42 and 2026. Preserve
all Kaggle outputs before a session ends. GPU runs must not change the frozen
cache, model recipe or validation-only checkpoint selection rule.
