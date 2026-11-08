# EE6405 — A4000 training and research

> **Current entry point:** `bash experiments/run_research.sh --max-hours 12 --push-branch Realsbt` automatically creates a UTC timestamp batch directory containing all checkpoints (`runs/`), logs (`logs/`), per-run summaries, `SUMMARY.md`, a detailed Chinese figure guide, and process/error logs. Only a positive final three-seed mean test Macro-F1 gain triggers a push, exclusively to `Realsbt`.

[中文训练与参数指南](README.zh-CN.md) · [Environment setup](docs/REPRODUCING.md) · [Historical results](experiments/reports/comparison/REPORT.md)

HART English four-class source detection, with a frozen 11,190 / 1,600 / 3,194 grouped split and a 512-token RoBERTa window. Select checkpoints and methods using validation Macro-F1 only.

## Start training on the prepared VM

```bash
cd /home/ubuntu/6405
source env-a4000.sh
cd ee6405-a4000-codex
export HYPERSTACK_API_KEY_FILE=/secure/mounted/hyperstack_api_key
export HYPERSTACK_VM_ID=YOUR_NUMERIC_VM_ID
"$EE6405_PYTHON" -m experiments.hyperstack --preflight
```

Supply the API key through a private external file (mode 600), never Git, JSON or command arguments. Preflight verifies API authentication and that the requested ACTIVE VM matches the local cloud UUID/hostname; it does not hibernate. Ensure the key's role also permits Hibernate.

Commit configuration/code changes on your own branch. Start inside tmux with a fresh batch name:

```bash
bash experiments/run_research.sh --name a4000_r01 \
  --config experiments/configs/a4000_research.json \
  --output "$EE6405_STORAGE/experiments/research_batches/a4000_r01" \
  --grace-seconds 600 --max-hours 8
```

The bounded queue runs 1/2/4/6/12-layer unfreezing, a SupCon projection ablation, and content/edit auxiliary-head ablations. It selects using validation only, completes seeds 42 and 2026, freezes a hash-checked manifest and equal ensemble weights, evaluates test independently, generates a report and five figures against the initial Git results, verifies artifacts, then requests VM hibernation.

An independent watchdog requests hibernation within a 10-minute cleanup deadline after training exits, even if evaluation/reporting or the parent fails. An eight-hour batch limit is also enforced. Provider/network failures are logged and retried; a local script cannot guarantee a remote state transition during outages. No ordinary shutdown/SHUTOFF fallback is used. Only run this queue on a VM dedicated to it.

To train one supplied recipe without the search grid, add `--single-config`; the supervised entry still completes three seeds, final evaluation, figures and hibernation.

## Change parameters

Copy `experiments/configs/a4000_research.json` to a new JSON and pass `--config`. See the [parameter table](README.zh-CN.md#4-如何改参数) for every supported control and queue overrides.

- Tune top-layer `learning_rate` (1.5e-5), `head_learning_rate` (3e-5), `layer_lr_decay` (.85), and SupCon temperature/weight in bounded validation experiments.
- `architecture.unfreeze_bert_layers` supports 1/2/4/6/12. The research queue overrides depth, batch, checkpointing, projection and auxiliary settings for its planned ablations.
- Real SupCon batch must be at least 8 and divisible by four. Accumulation changes update batch, not the contrastive set.
- `gradient_checkpointing`, worker/pinning options, fused AdamW, SDPA/eager and `torch_compile` are explicit performance controls.
- Embeddings stay frozen by default; only full-depth training may unfreeze them, with LR ≤2.5e-6.
- Keep the task, data hashes, window, validation selection metric and formal seeds fixed. The research queue requires seed 3407 and SupCon enabled before completing the other seeds.

For manual debugging, `bash experiments/run_neural.sh --config YOUR.json --run-name UNIQUE --smoke` runs a smoke check. Without `--smoke` it trains using train/validation by default; `--resume` requires the same configuration/source. **This bare entry point does not arm auto-hibernation; use the supervised research queue for unattended work.** Do not use `--evaluate-test` during method screening.

## Outputs

`$EE6405_STORAGE/experiments/research_batches/<name>/` contains configs, logs, the final manifest, evaluation metrics/predictions, `REPORT.md`, learning curves, historical test/class-F1 comparisons, a confusion matrix and artifact hashes. Original comparison data are read with `git show` from commit `877f12bb5b83a4825780f654309ecd9959f74f5a`. Model checkpoints and status/events remain under `experiments/neural_runs/<run_id>/` in storage.

The watchdog appends progress approximately every ten minutes. Its hibernation request/error files record the outcome; an accepted request is not proof that the VM reached HIBERNATED. Confirm the final state in Hyperstack. Storage and a retained IP may still bill. [Official API](https://docs.hyperstack.cloud/docs/virtual-machines/manage-vm-state/), [hibernation behavior](https://docs.hyperstack.cloud/docs/virtual-machines/hibernation/).

Historical RoBERTa+SupCon test Macro-F1 is **91.15% ± 0.24** across three seeds. The A4000 baseline reproduction scored **91.64% validation / 90.65% test** for seed 3407. New improvements are hypotheses until measured. Comparisons reuse the same fixed test set and are not fresh independent tests.

Research motivations: [discriminative fine-tuning](https://aclanthology.org/P18-1031/), [supervised contrastive learning](https://arxiv.org/abs/2004.11362/); creator/edit auxiliary targets are a task-specific hypothesis. Original methods, data and attribution remain in [METHOD_NOTES](experiments/METHOD_NOTES.md), [SOURCES](experiments/SOURCES.md), and [data documentation](data/README.md).
