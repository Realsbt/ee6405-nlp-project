# Text-EDU residual fusion

Implementation date: 2026-11-11. Status updated 2026-11-12: four CUDA smokes,
four validation-screening trainings and CLS seed 42 are complete. Gated was
selected at 91.3643% validation Macro-F1 versus matched CLS 90.6435%.
The original job ended in cancellation; a continuation is running. Final
three-seed test comparisons remain incomplete. See [current status](YINGQI_STATUS.md)
for saved evidence and contribution boundaries. The protocol below retains the
original design; its eight-run matrix is not a claim of eight completed runs.

## Research question

Can selected EDU content supplement a strong text representation without
requiring relation labels or graph message passing?

The original RACE reads a graph root; the existing EDU control averages EDUs
without retaining a direct CLS path. This candidate instead keeps CLS and adds
a local-content residual. All variants use exactly the frozen visible prefix,
the original audited EDU spans, one RoBERTa encoder and CE training.

## Architecture

For EDU i with token span [a_i, b_i), let e_i be the mean token representation.
Let c be the CLS representation.

```text
Visible text -> RoBERTa -> CLS -----------------------+
                        -> EDU token means            |
                           -> attention -> context    |
                           -> sigmoid gate ---------> + -> projection -> MLP
```

```text
score_i = v^T tanh(W e_i + b)
alpha_i = softmax(score)_i
u = sum_i alpha_i e_i
g = sigmoid(w^T [c; u] + b_g)
h = c + g * u
```

The scalar gate is initialized with zero weights and bias -2, giving a small
initial residual (about 0.119). It is not a probability that structure is
correct. Attention weights are not causal explanations or proof of AI authorship.

| Condition | Representation before shared projection |
| --- | --- |
| `fusion_cls` | c |
| `fusion_mean` | c + equal-weight mean of EDU representations |
| `fusion_attention` | c + attention-weighted EDU context |
| `fusion_gated` | c + gated attention-weighted EDU context |

Every variant uses the same 768-to-512 projection and 512-wide classifier,
initialized before variant-specific modules. Same-seed common encoder/head
initialization is matched. Attention and gating add parameters; report actual
counts rather than claiming exact parameter matching.

The new CLS baseline has a different head from the historical RoBERTa baseline.
Use it for matched module comparisons, and retain the original RoBERTa+SupCon
as the strong performance comparator. Beating only the new CE baseline is not
proof of beating the best existing model.

## Inputs and implementation

Reuse `same_window_full_v3` unchanged. No new parsing, graph-cache migration or
test-dependent preprocessing is needed. Among the 15,984 cached records, 405
have adjacent EDU spans sharing a boundary BPE token. Preserve that existing
character-to-token convention; do not discard these records or silently change
their spans. Pooling accepts monotonic shared boundaries but rejects out-of-range,
reversed, empty or padded spans. This is a descriptive cache check, not a measure
of parser quality.

Code:

- `experiments/edu_fusion.py`: pooling, residual gate, model and frozen defaults.
- `experiments/train_edu_fusion.py`: shared same-window training entry point.
- `experiments/run_fusion_queue.py`: CUDA smoke or validation-screening queue.
- `experiments/evaluate_edu_fusion.py`: checkpoint reload, domain metrics and
  validation examples; optional held-out test evaluation.
- `experiments/test_edu_fusion.py`: unit tests and test-partition exclusion check.

## Execution

From the repository root, use the environment described in `REPRODUCING.md`.
Set the resource paths for the actual host, not the author's machine:

```bash
export EE6405_STORAGE=/absolute/path/to/resources
export EE6405_UPSTREAM="$EE6405_STORAGE/RACE"
export CACHE="$EE6405_STORAGE/experiments/same_window_full_v3"
```

CPU engineering check, one variant at a time:

```bash
python -m experiments.train_edu_fusion \
  --condition fusion_gated --cache "$CACHE" \
  --output outputs/fusion_cpu_smoke --smoke
```

On the target GPU, smoke all variants first:

```bash
python -m experiments.run_fusion_queue \
  --cache "$CACHE" --output outputs/fusion_cuda_smoke --smoke
```

Then screen the four candidates with seed 3407:

```bash
python -m experiments.run_fusion_queue \
  --cache "$CACHE" --output outputs/fusion_screen_s3407 --seeds 3407
```

The queue never opens test. Select the best of the three fusion candidates by
validation Macro-F1, breaking exact ties by fewer trainable parameters, then
condition name. The CLS baseline is a fixed comparator, not a fusion candidate;
if it scores higher, retain and disclose that negative validation gap.
Before test evaluation, record the
selected condition, recipe, comparison models and seeds in a dated selection
record. Do not change them after inspecting test results.

Complete seeds 42 and 2026 for `fusion_cls` and the selected candidate using a
new output directory. For example, if validation selects `fusion_gated`:

```bash
python -m experiments.run_fusion_queue \
  --cache "$CACHE" --output outputs/fusion_confirmation \
  --conditions fusion_cls fusion_gated --seeds 42 2026
```

This is four initial runs plus four confirmation runs, not twelve mandatory
mechanism runs. The candidate's seed-3407 validation estimate was involved in
selection; do not treat it as independent evidence of an improvement.

`cloud/kaggle/run_edu_fusion.py` executes this complete protocol in one private
job: CUDA smoke, validation screening, a saved selection record, confirmation,
and frozen-checkpoint evaluation. It reuses the existing resource-discovery
and extraction functions from `run_same_window_smoke.py`. The dedicated code
archive is `edu_fusion_code_v3.zip`; the resource dataset is unchanged.

Reload each frozen baseline/candidate run for final testing:

```bash
python -m experiments.evaluate_edu_fusion \
  --run outputs/fusion_screen_s3407/fusion_gated_s3407 \
  --cache "$CACHE" --output outputs/fusion_gated_final_s3407 \
  --device cuda --evaluate-test
```

Omit `--evaluate-test` to reload only validation predictions and export examples.
The evaluator checks source, cache and checkpoint hashes and compares existing
saved probabilities with a maximum absolute tolerance of 1e-4.

Runs refuse to overwrite existing output. This entry point does not implement
interrupted-epoch resume; budget time accordingly and preserve completed outputs.
Do not modify code while a formal queue is running. The training format is FP32,
unlike historical BF16 runs, so discuss precision/head differences when comparing
historical scores. This iteration fixes CE; a SupCon comparison would be a
separate matched experiment, not an implemented toggle here.

## Evidence and presentation

Report all four screening results, including unsuccessful candidates. Final
comparisons should include:

- Three-seed Macro-F1 mean, sample SD and per-seed candidate-minus-baseline gaps.
- All four class recalls/F1s, domain metrics and minority-class support.
- Trainable parameters, training time and peak CUDA memory.
- External evaluation with the candidate frozen, if resource and data access allow.

No external inference adapter is included in this iteration. HART validation/test
and local validation examples are supported; new text requires compatible visible
window tokenization and EDU spans before prediction.

### Existing Kaggle resource dataset

The older private dataset contains an older repository snapshot without this
model. Use the new code-only archive, not that frozen repository, when running
fusion experiments. No source text or pretrained weights are in the code archive.
Keep using the private resource dataset for the cache, backbone and upstream RACE.

After placing the archive in your private notebook's accessible input files:

```python
from pathlib import Path
import os
import sys
import zipfile

# Replace these paths with the actual mounts shown by your notebook.
archive = Path("/kaggle/input/YOUR_PRIVATE_CODE_INPUT/edu_fusion_code_v3.zip")
resources = Path("/kaggle/input/YOUR_PRIVATE_RESOURCE_INPUT/resources")
code = Path("/kaggle/working/edu_fusion_code")
with zipfile.ZipFile(archive) as bundle:
    bundle.extractall(code)
os.environ["EE6405_STORAGE"] = str(resources)
os.environ["EE6405_UPSTREAM"] = str(resources / "RACE")
sys.path.insert(0, str(code))
os.chdir(code)
```

If resources are supplied as `resources.zip` instead of a directory, extract
them into a private working directory first and point `resources` there. Check
CUDA and install the project's pinned dependencies in the notebook before the
smoke queue. The archive does not upload data, create a notebook, install packages
or provision a GPU.

The explanation export selects the first two validation items per class in the
frozen order, rather than selecting flattering successes. It includes decoded
EDU excerpts, attention weights, residual gate and model probabilities. These
files contain text: keep them local and check redistribution permissions before
including screenshots. The CLS baseline has no EDU weights.

For a short presentation:

1. Motivation: preserve global text evidence while selecting useful local content.
2. Architecture: one encoder, weighted EDU pooling, a residual gate, no GNN.
3. Results: actual matched comparisons and one predefined validation example.

Do not label a smoke score as accuracy evidence or imply that training has
already improved the model. Old HART test diagnostics helped motivate this
candidate, so the work is post-hoc follow-up on the existing benchmark, not a
preregistered independent confirmation.

## Method attribution

Group reuse and phase attribution are recorded in
[`REUSE_ATTRIBUTION.md`](REUSE_ATTRIBUTION.md), including pinned source commits,
direct imports, adapted scaffolding, resource archives and any historical
results actually used for comparison. The shared encoder/data/trainer and
earlier EDU cache are reused; they are not new fusion work. The older
four-condition T4 smoke does not establish CUDA acceptance for the new fusion
variants. Pure idea references without a used artifact are not listed.

This implementation does not claim to invent attention or gating.
The task, base encoder, original graph model and data protocol retain
the attribution recorded in `experiments/SOURCES.md`.

## Collect and verify this run

The submitted private kernel is
`zhengyingqi/ee6405-text-edu-fusion-experiment-v1`.
Version 1 failed on the auto-unpacked code mount before training. Version 2
accepts both ZIP and unpacked code inputs; its bootstrap hash is recorded
separately. The archived model and training sources are unchanged.

Run the collector in an environment with configured Kaggle authentication:

```bash
python -B scripts/collect_fusion_run.py \
  --output outputs/kaggle_fusion_experiment_v1 --watch --checkpoints
```

The collector leaves other kernels alone, downloads only this experiment's
evidence, excludes smoke checkpoints, and records downloaded file hashes.
It does not turn a partial or failed job into a successful result.

After all formal runs finish, use the project Python environment:

```bash
python -B scripts/report_fusion_results.py \
  --input outputs/kaggle_fusion_experiment_v1/edu_fusion_experiment_v1 \
  --output outputs/fusion_report_v1
python -B scripts/render_fusion_architecture.py \
  --output outputs/fusion_presentation_v1/architecture.png
python -B -m unittest scripts.test_fusion_reporting
```

Local plotting additionally uses `matplotlib==3.9.4` and `contourpy==1.3.2`;
retain the project's `numpy==1.26.0`. These are report dependencies, not a
change to the submitted GPU training environment. Five local reporting tests
use synthetic fixtures only, never experimental evidence.

The report independently recomputes all metrics from six complete test
prediction files, checks paired IDs/labels/groups/domains and the frozen
selection, and exports class/domain metrics, training cost and figures.
A zero or negative mean change is explicitly reported as no improvement.
Attention figures contain decoded validation text and remain local.

### First-stage outcome and continuation

The first full job returned `CANCEL_ACKNOWLEDGED` after about 12 hours, without
an API failure explanation. Its four CUDA smokes and all four validation screens
completed, as did CLS seed 42. Gated seed 42 stopped during epoch 13 and has no
completed result. Test was not opened. All five complete checkpoints were
verified against their checkpoint, cache and training-source hashes.

The frozen screening selection is gated fusion (validation 91.3643% versus
matched CLS 90.6435%). The +0.7208-point difference is selection-set evidence,
not a test improvement claim. `scripts/report_fusion_screen.py` recomputes the
four full validation prediction files and renders this provisional comparison.

`cloud/kaggle/resume_edu_fusion.py` preserves the five complete runs, verifies
the original candidate selection, and trains only the three missing complete
runs with the original recipe. It does not resume an optimizer or initialize
missing runs from a partial task checkpoint. An attempted direct kernel-output
input was rejected by Kaggle; verified own outputs are therefore packaged in
the private `zhengyingqi/ee6405-fusion-stage1-v1` dataset instead.
`scripts/package_fusion_continuation.py` excludes the unused partial checkpoint.
Four synthetic continuation tests check these preservation boundaries.

Once the continuation is ready, collect its output into a separate directory:

```bash
python -B scripts/collect_fusion_run.py \
  --kernel zhengyingqi/ee6405-text-edu-fusion-continuation-v1 \
  --output outputs/kaggle_fusion_continuation_v1 \
  --watch --checkpoints --interval 1800
```

Thirty-minute polling is status monitoring, not a way to extend platform runtime.
The partial first attempt consumed time and must not be omitted when describing
overall compute cost; successful-run CSVs alone are not total project cost.
