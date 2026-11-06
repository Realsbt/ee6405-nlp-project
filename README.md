# EE6405 · Text Creation and Editing Detection

**English** | [简体中文](README.zh-CN.md)

A course reproduction and analysis of [RACE](https://github.com/ICTMCG/RACE), comparing **TF-IDF + logistic regression, RoBERTa, RACE**, with an architecture × objective 2×2 ablation (CE versus CE + supervised contrastive loss). The project adds a fixed grouped split, baseline comparisons, multiple training seeds, parser audits and an interactive raw-text demo.

**Status:** the five-run [2×2 extension](docs/ABLATION_2X2.md), full checkpoint verification and updated analysis are complete. The original local demo is retained. Final slides, a video with human narration and signed submission documents remain to be completed. See the [submission checklist](docs/SUBMISSION.md). This repository contains code and evaluation evidence; submission to the course platform is a separate step.

## Task and data

| Label | Operational definition in the released dataset |
|---|---|
| `human_written` | Human-written text |
| `human_ai_polished` | Human-written content polished by AI |
| `ai_generated` | AI-created content, including AI-to-AI rewriting chains |
| `ai_humanized` | AI-created content rewritten by a person or a tool; this does not confirm human editing |

The main benchmark is **HART**: 15,984 records after cleaning, with source-related and duplicate-linked texts kept in the same partition. The train / validation / test sizes are **11,190 / 1,600 / 3,194**. The arxiv, essay, news and writing subsets are domains within this benchmark.

An independent auxiliary evaluation uses 1,000 human / ChatGPT answers from **HC3**. HC3 provides binary labels only and cannot validate generalization of the four creation-and-editing classes. See [data and external resources](data/README.md) for versions, sources and runtime requirements.

### Method overview

```mermaid
flowchart TB
    T["English text"] --> V["Text covered by the first 512 tokens"]
    V --> F["TF-IDF + logistic regression"]
    V --> B["RoBERTa encoder"]
    B --> C["CLS + MLP"]
    T --> G["Full-text RST / EDU relation graph"]
    B --> R["EDU pooling + RGCN<br/>Root + MLP"]
    G --> R
    F --> Y["Four-class scores"]
    C --> Y
    R --> Y
```

Models are trained independently. RoBERTa and RACE use the same base-pretrained initialization. HART uses the released RST trees; the demo and HC3 use the local parser. Each neural architecture is trained both with and without SupCon; changing the objective leaves its inference architecture intact.

## Results

Results use the fixed HART grouped test set. Neural checkpoints are selected by validation Macro-F1. All twelve neural runs have complete test-set reload verification; the deterministic TF-IDF baseline is retained.

![HART results for the lexical baseline and four neural conditions](experiments/reports/comparison/main_metrics.png)

| Method | Runs | Macro-F1 | Macro-AUROC |
|---|---:|---:|---:|
| TF-IDF + logistic regression | 1 | 79.40% | 96.24% |
| RoBERTa · CE | 3 | 90.72% ± 0.25 | 98.89% ± 0.02 |
| RoBERTa · CE + SupCon | 3 | 91.15% ± 0.24 | 98.90% ± 0.06 |
| RACE · CE | 3 | 89.92% ± 0.90 | 98.34% ± 0.11 |
| RACE · CE + SupCon | 3 | 90.37% ± 0.60 | 98.67% ± 0.11 |

Neural results use seeds 3407, 42 and 2026. Values are means ± sample SD; SD is in percentage points. Each cell of the architecture × objective matrix has three runs.

- SupCon changes Macro-F1 by **+0.44 percentage points in RoBERTa** and **+0.45 in RACE**, using matched-seed mean differences.
- RACE − RoBERTa is **-0.80 points under CE** and **-0.78 under CE + SupCon**. The interaction (RACE's loss effect minus RoBERTa's) is **+0.01 points**. Per-seed differences and their SDs are in the report.
- Architecture comparisons also include pooling, representation dimension, parameter count and full-document structural information; they do not isolate graph structure alone.
- The five added runs extend an analysis whose original test results were already inspected. Three training seeds on the same split are not a new independent test or evidence of broad generalization.
- HC3 remains the original single-seed binary auxiliary comparison: full RACE has 38.69% Macro-F1 and 8.40% AI recall. This 2×2 extension concerns HART and does not establish an analogous HC3 improvement.

![Four-cell comparison and paired seed contrasts](experiments/reports/comparison/ablation_2x2.png)

See the [experimental report](experiments/reports/comparison/REPORT.md), [2×2 protocol and commands](docs/ABLATION_2X2.md), [error cases](experiments/reports/comparison/CASE_STUDIES.md), [HC3 analysis](experiments/reports/auxiliary/REPORT.md) and [method notes](experiments/METHOD_NOTES.md).

## Quick start

### Check saved results after cloning

Run from the repository root with the Python 3 standard library. No GPU, dataset or model weights are needed:

```bash
python3 scripts/check_results.py
```

The script recomputes metrics from individual predictions and checks thirteen formal runs, four HC3 evaluations, aggregate tables and five paired factorial contrasts. It checks internal consistency of the saved evidence; checkpoint inference still requires external resources.

### Run the English demo on Linux / Windows WSL2

<p align="center">
  <img src="experiments/reports/demo_race.jpg" alt="English RACE demo: text input, model selection, four-class scores and discourse parsing information" width="800">
</p>

*The interface and demonstration materials are in English. The example has no ground-truth authorship label. The displayed request time is an individual measurement, not a cold-start benchmark.*

```bash
# First install Python 3.11, Git, GitHub CLI and a C/C++ compiler.
# Each teammate needs access to this private repository and gh auth login.
export EE6405_STORAGE="$HOME/ee6405-resources"
python3 scripts/setup_environment.py --python python3.11 --torch cpu
export EE6405_PYTHON="$EE6405_STORAGE/.venv/bin/python"

"$EE6405_PYTHON" scripts/fetch_resources.py --profile demo
"$EE6405_PYTHON" scripts/check_installation.py --predict
bash experiments/run_demo.sh
```

Open <http://127.0.0.1:8765/>. The demo uses the CPU by default and displays scores, input truncation, the parsed discourse tree and timing. Free-form input without a ground-truth label demonstrates functionality only.

The demo downloads about **4.28 GiB** of resources plus its Python environment. Our checkpoints are distributed through the private [resources-v1 release](https://github.com/Realsbt/ee6405-nlp-project/releases/tag/resources-v1); upstream resources are fetched from pinned original sources. They are not stored in Git. A relocated Linux checkout and a fresh isolated CPU environment have passed all four demo models, exact dataset reconstruction and preprocessing checks. WSL2 instructions are provided but were not exercised in this check.

Follow the **[Linux / WSL2 reproduction guide](docs/REPRODUCING.md)** for prerequisites, full-data restoration, hardware requirements and GPU training. The original `/data/botao/ee6405-race-validation/` directory remains a local resource store; teammates choose their own `EE6405_STORAGE`. Historical result paths are resolved at read time without changing their provenance.

## Repository contents

```text
README.md                 English project overview, results and entry points
README.zh-CN.md           Chinese version of the project overview
docs/                     Original course documents and submission checklist
data/README.md            External data, weights and environment requirements
experiments/              Data preparation, training, evaluation, RST and demo
  configs/                Frozen experiment configurations
  reports/                Formal metrics, predictions, plots and audit records
  METHOD_NOTES.md          Implemented models and experimental methods
  SOURCES.md               Papers, code, data and version references
  VIDEO_OUTLINE.md         9-minute-20-second human presentation outline
scripts/check_results.py   Dependency-free saved-result consistency check
scripts/                  Environment setup, resource download and validation
resources/                Pinned resource manifest and frozen HC3 selection
```

Raw datasets, model weights, Python environments, official source copies and routine logs are excluded from Git. Historical smoke runs, superseded auxiliary outputs and error lists containing full text remain in the original workspace. Submitted prediction tables contain IDs, labels and scores, without the source articles. Video renders and their editable local project remain outside Git to keep the repository small.

## Before submission

- Prepare theory and results slides, and record a video of no more than 10 minutes with human narration, following the [presentation outline](experiments/VIDEO_OUTLINE.md).
- Complete the group reflection, actual member contributions, all required signatures and the official GAI declaration template.
- Package the code as required by the course platform and confirm how model resources and the demonstration environment will be provided.
- Each member must complete the individual peer evaluation. The group must upload the final materials and verify the submission receipt.

See the [submission checklist](docs/SUBMISSION.md) for status and acceptance criteria, the [course requirements](docs/course-requirements.md) for the brief and rubric, and [SOURCES](experiments/SOURCES.md) for third-party provenance and usage boundaries.
