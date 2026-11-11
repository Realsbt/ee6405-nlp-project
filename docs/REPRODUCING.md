# Run the project on Linux or Windows WSL2

**English** | [简体中文](REPRODUCING.zh-CN.md) | [Project overview](../README.md)

The code and saved predictions live in Git. Large resources live in a separate directory chosen with `EE6405_STORAGE`. You do not need the original machine's `/data/botao/ee6405-race-validation` directory or its Python environments.

| Goal | Resources / hardware |
|---|---|
| Recompute saved result tables | Python 3 standard library; no weights or GPU |
| Run all four English demo models | CPU, Python 3.11; about **4.28 GiB** of downloaded resources, plus the environment |
| Restore the original datasets and all 13 model files | `full` profile: about **9.21 GiB**, plus environment and generated data |
| Retrain neural models | NVIDIA CUDA GPU with BF16 support; substantially more time and checkpoint space |

For the CPU demo, allow about 15 GB of free disk space and preferably 16 GB of RAM. These are planning recommendations, not measured minimum requirements. For a full training campaign, reserve at least 40 GB of disk space and inspect the hardware measurements in the experiment reports before choosing a GPU.

## 1. Prepare the system and clone

Use Linux x86-64, or Ubuntu inside WSL2 on Windows. On Windows, install WSL from an administrator PowerShell with `wsl --install`, restart if prompted, and check `wsl -l -v` reports version 2. Run all commands below inside the Linux terminal. See [Microsoft's installation guide](https://learn.microsoft.com/en-us/windows/wsl/install).

Install **Python 3.11 with `venv`, Git, GitHub CLI and a C/C++ compiler**. On Ubuntu the compiler package is `build-essential`; `jsonnet` may need it during installation. If your distribution provides a different Python version, install a separate Python 3.11 interpreter rather than replacing the system Python. An existing Conda Python 3.11 interpreter can also be passed using `--python /path/to/python`.

The repository and checkpoints are private. Each teammate needs repository read access and must authenticate their own GitHub account:

```bash
gh auth login
gh repo clone Realsbt/ee6405-nlp-project
cd ee6405-nlp-project
python3 scripts/check_results.py
```

This first check recomputes metrics from the committed predictions. It does not run models or independently establish how the predictions were produced.

## 2. Install and run the CPU demo

Run from the repository root. Set `EE6405_STORAGE` in every new terminal, or save the export in your shell configuration. Choose a path with enough free space.

```bash
export EE6405_STORAGE="$HOME/ee6405-resources"

python3 scripts/setup_environment.py --python python3.11 --torch cpu
export EE6405_PYTHON="$EE6405_STORAGE/.venv/bin/python"

"$EE6405_PYTHON" scripts/fetch_resources.py --profile demo
"$EE6405_PYTHON" scripts/check_installation.py --predict
bash experiments/run_demo.sh
```

Open <http://127.0.0.1:8765/>. The English interface provides TF-IDF, RoBERTa, RACE and RACE without SupCon. RACE parses your raw text on the CPU; its first request also loads the parser. The service binds only to loopback. Stop it with `Ctrl+C`.

The installer creates a normal isolated environment without inheriting packages from another environment. It installs the pinned detector and exercised RST-DT parser dependencies together. It refuses to overwrite an existing environment. To rebuild, choose a new resource directory; to switch an existing environment to GPU PyTorch, see section 4.

The fetcher obtains pinned upstream code, pretrained weights and parser files from their original sources. Our trained weights come from the private [resources-v1 release](https://github.com/Realsbt/ee6405-nlp-project/releases/tag/resources-v1). Every resource is checked against the size and SHA-256 in [the manifest](../resources/manifest.json). Existing mismatched files are preserved and reported, not silently replaced. Downloads require internet access; inference uses the local files offline.

If your shell previously enabled offline Hugging Face mode, unset it before downloading:

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
```

## 3. Restore the frozen datasets and verify preprocessing

The `full` profile adds the original HART/HC3 files and downloads all 13 model files from the private release. That covers the TF-IDF classifier and the four neural configurations with three seeds each.

```bash
"$EE6405_PYTHON" scripts/fetch_resources.py --profile full
"$EE6405_PYTHON" scripts/restore_data.py

"$EE6405_PYTHON" -m experiments.prepare_graph_cache \
  --report "$EE6405_STORAGE/experiments/reproduction_checks/graph_cache.json"
"$EE6405_PYTHON" -m experiments.verify_graph_cache \
  --report "$EE6405_STORAGE/experiments/reproduction_checks/graph_check.json"
"$EE6405_PYTHON" -m experiments.verify_tfidf \
  --report "$EE6405_STORAGE/experiments/reproduction_checks/tfidf_reload.json"
```

`restore_data.py` reconstructs the **original byte-identical HART partitions** (11,190 / 1,600 / 3,194) and the **1,000-answer frozen HC3 partition**. It checks their archived hashes before accepting them. The HC3 selection recipe contains IDs, metadata and hashes, not answer text. It is needed because a newly sampled HC3 subset would not reproduce the published evaluation.

The graph-cache builder refuses to replace an existing cache. If it already exists, run its verification command directly. These report options keep new checks outside the archived repository reports. Use the original HART graph trees for benchmark evaluation; raw-text demo parsing is a separate path.

Google Drive can impose download limits on HART. If an automatic download fails, use the exact file's official URL in `resources/manifest.json`, place it at its listed path inside `EE6405_STORAGE`, and rerun the fetcher. The SHA-256 check still applies.

For an offline handoff from a verified existing resource directory, use:

```bash
"$EE6405_PYTHON" scripts/fetch_resources.py --profile full --local-source /path/to/resource-copy
```

This copies only selected resource files, not environments or caches. A pinned upstream checkout must already exist if there is no network. After installation, `fetch_resources.py --profile full --verify-only` checks resource bytes without writing or downloading anything.

## 4. Optional GPU training and neural checkpoint replay

CPU inference and TF-IDF training do not require CUDA. The current neural training and complete neural reload scripts use CUDA and BF16. They are not CPU training entry points.

For a new GPU environment, use `setup_environment.py --torch cu128`. To switch an already installed CPU environment, install the matching official wheel and check your GPU:

```bash
"$EE6405_PYTHON" -m pip install torch==2.7.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
"$EE6405_PYTHON" -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available()); print("BF16:", torch.cuda.is_bf16_supported())'
```

Both capability checks must be true. Use an NVIDIA driver compatible with that wheel; on WSL2 the GPU must be available inside the Linux guest. The version and wheel indexes follow the [official PyTorch 2.7.0 instructions](https://pytorch.org/get-started/previous-versions/).

After restoring full data and the graph cache, extract the archived recipes and train with **new run names** so original checkpoints and reports remain intact. Explicit recipes preserve the intended loss setting even when the training CLI is extended:

```bash
python3 - <<'PY'
import json, os
from pathlib import Path
target = Path(os.environ['EE6405_STORAGE']) / 'experiments/training_recipes'
target.mkdir(parents=True, exist_ok=True)
for kind in ('roberta', 'race', 'race_no_supcon'):
    run = f'{kind}_s3407'
    recipe = json.loads(Path(f'experiments/reports/{run}/result.json').read_text())['config']
    (target / f'{run}.json').write_text(json.dumps(recipe, indent=2) + '\n')
PY

bash experiments/run_baseline.sh teammate_tfidf
bash experiments/run_neural.sh --model-type roberta --seed 3407 --run-name teammate_roberta_s3407 \
  --config "$EE6405_STORAGE/experiments/training_recipes/roberta_s3407.json"
bash experiments/run_neural.sh --model-type race --seed 3407 --run-name teammate_race_s3407 \
  --config "$EE6405_STORAGE/experiments/training_recipes/race_s3407.json"
bash experiments/run_neural.sh --model-type race_no_supcon --seed 3407 --run-name teammate_no_supcon_s3407 \
  --config "$EE6405_STORAGE/experiments/training_recipes/race_no_supcon_s3407.json"

# Full held-out replay of an archived checkpoint; requires CUDA.
"$EE6405_PYTHON" -m experiments.verify_neural --run race_s3407 \
  --report "$EE6405_STORAGE/experiments/reproduction_checks/race_reload.json"
```

Use the run name from the archived report to replay another model, such as `roberta_supcon_s3407` or `race_no_supcon_s42`. Each run uses its own frozen recipe. To train from base initialization, add `--skip-checkpoints` to the full resource download. Training creates new logs, source snapshots and checkpoints in the selected resource directory and new reports in the repository.

HC3 RACE evaluation additionally needs parsing of the restored HC3 answers through `experiments.rst_worker`, followed by `experiments.evaluate_external`. See [the detailed experiment workflow](../experiments/README.md). Those commands write evaluation reports; use a separate working copy when independently replaying archived runs. The original host's complete provenance gate also inspects historical training events and source snapshots, which are not part of this compact release. It is not the teammate installation check.

Training on another GPU or software stack can change floating-point values and optimization trajectories. Identical data and checkpoint hashes are exact checks; a fresh training run is not promised to be bit-identical across hardware.

## Resource layout and validation boundary

```text
$EE6405_STORAGE/
  .venv/                           Newly installed Python environment
  assets/                          Original datasets, backbone and parser files
  experiments/hart_grouped_v1/      Restored HART splits and graph cache
  experiments/hc3_aux_v1_1/         Restored HC3 answers
  experiments/models/             TF-IDF classifier
  experiments/neural_runs/         Best checkpoints and any new training runs
  experiments/demo_runtime/        Local parser diagnostics
```

Upstream source is fetched to `local_validation/upstream/RACE/` inside the repository and is ignored by Git. `EE6405_UPSTREAM` can select another clean pinned checkout. Historical JSON records retain their original paths; `experiments/paths.py` resolves the two known historical prefixes at read time. Do not edit archived provenance to fit your machine. On the original host, the old resource directory remains the compatibility default; an explicit `EE6405_STORAGE` always takes precedence.

The [portability validation record](../resources/portability_validation.json) documents the checks completed in a separate Linux checkout, a new isolated CPU environment and a new resource directory. WSL2, a fresh GPU environment and a new full neural training campaign were not exercised in this portability check. The test reused hash-verified resource bytes from the existing store; it did not redownload every multi-gigabyte asset from the internet.
