# Free Colab encoder pilot

This is the same frozen, single-seed train/validation pilot described in
`docs/ENCODER_PILOT.md`, not a new test-set experiment. User approval was obtained
on 2026-11-12 for private train/validation upload, dependency installation,
pretrained-model download and execution on the free Colab T4. No Drive mounting,
API credentials, paid compute or test data are required.

## Package

```bash
python scripts/prepare_colab_encoder_pilot.py \
  --package outputs/paired_encoder_pilot_package_v3 \
  --backbone /absolute/path/to/resources/assets/roberta-base \
  --output outputs/ee6405_colab_pilot_v3.zip
```

Only allowlisted inputs are included. RoBERTa weights are downloaded from a
pinned official repository revision and checked against the SHA-256 of the
existing local backbone. Tokenizer/config files are copied from that backbone.
The runner checks all uploaded file hashes and the frozen train/validation
protocol before training. Do not upload the complete project resources archive,
which also contains the test partition.

The original v1 upload is 18,475,969 bytes with SHA-256
`02b919011fa750e966c5a14725300f0e9ea1e9491127f1447b8d65e9ba0d2999`.
Its CUDA check stopped before training because Transformers imported the
platform's TensorFlow, whose generated protobuf code requires a newer runtime.
The failed attempt and its logs remain at `/content/ee6405_encoder_pilot`.
V2 sets `USE_TF=0`, `USE_TORCH=1`, `USE_FLAX=0` before model imports, uses a new
output root, and preserves the unchanged train/validation recipe and hashes.
V2 SHA-256:
`9b422c7193526e76ec81dc91fcac16a7e9a857481b50c66fbe76ca24af927da8`.
Preserve both immutable packages; use a new filename if their sources change.

The v2 DeBERTa smoke detected an AMP overflow and correctly failed its
classifier-update check before formal training. V3 lowers the initial AMP loss
scale to 128 for both arms and permits at most 12 same-batch overflow attempts
to obtain the smoke's single real update. It does not ignore nonfinite gradients
or count skipped steps as updates. Failed v1/v2 outputs are preserved.
V3 SHA-256:
`e16c7b592a0b01d43b0d41cb11ff9d6022cd248f3213655e8e85bedd94f02ff7`.

## Launch

Select free T4 and upload the package to `/content` through Colab's Files pane.
Wait for upload completion. Install `transformers==4.46.3`,
`huggingface-hub==0.36.0`, `sentencepiece==0.2.1` and `protobuf==5.29.6`;
keep the platform's existing CUDA PyTorch. Verify the full uploaded archive
against its local hash, extract only `run_encoder_pilot.py` to a launcher file,
then execute it with `python -u`.

The runner performs both one-batch CUDA checks before starting either formal
arm, then runs DeBERTa followed by RoBERTa serially. Each formal arm is limited
to four epochs, patience two and 14,400 training seconds. Failure or incomplete
training stops the pipeline; there is no automatic duplicate or expanded grid.

Results and flushed logs are under
`/content/ee6405_encoder_pilot_v3/results`. The runner writes
`/content/ee6405_encoder_pilot_v3/pilot_evidence.zip` on normal completion or a
handled training failure. It contains metrics, IDs/probabilities and logs, not
source text or weights. Best weights are kept separately in each arm's directory.
Repeated launch is rejected to preserve existing outputs.

**Colab runtime storage is temporary.** A runtime termination can erase uploaded
inputs, checkpoints and evidence; notebook autosave is not a backup of runtime
files. There is no keep-alive bypass or automatic Drive persistence. Download
evidence and any needed checkpoints before ending the runtime. Free runtime
availability and duration are not guaranteed.
