# EE6405 · Text Creation and Editing Detection

**English** | [简体中文](README.zh-CN.md)

A course reproduction and analysis of [RACE](https://github.com/ICTMCG/RACE), comparing **TF-IDF + logistic regression, RoBERTa, RACE**, with an architecture × objective 2×2 ablation (CE versus CE + supervised contrastive loss). The project adds a fixed grouped split, baseline comparisons, multiple training seeds, parser audits and an interactive raw-text demo.

**Experiment status (2026-11-06): the [2×2 ablation](docs/ABLATION_2X2.md) is complete.** Four neural conditions have three seeds each: **12 neural runs plus one TF-IDF baseline**. All five added runs, full test-set checkpoint verification, paired analyses and figures are complete. The four-method HC3 auxiliary evaluation and raw-text demo have been verified. Course deliverables and platform submission status are tracked separately in the [submission checklist](docs/SUBMISSION.md).

## Text Lab demo

Paste an English document and choose TF-IDF + logistic regression, RoBERTa, RACE, or RACE without SupCon. Text Lab runs locally and displays a score for each of the four classes.

<p align="center">
  <img src="docs/images/textlab-structure.png" alt="Text Lab Structure view with the RACE discourse graph, the four document scores, and the measured effects of a selected unit" width="880">
</p>

*RACE running the demo's unlabeled example with the model already loaded.*

**Try it without installing anything:** the [static Text Lab demo](https://realsbt.github.io/ee6405-nlp-project/demo/) runs the real interface in the browser with frozen model outputs for the built-in example. It needs no backend, so it can only replay that example; run the local service below to analyze your own text.

- In **Explore & Edit**, use **Map** to jump between paragraphs and check explanation progress, and **Passages** to run a separate prediction on an excerpt. **Coverage** shows the lexical input window. TF-IDF reads the full input, and RoBERTa and RACE use up to 512 tokens for lexical input.
- In **Explain Prediction**, expand branches of the RACE discourse graph and request **Influence** for a selected unit or the visible units. The detail panel shows how suppressing a unit's features changes all four scores. Follow-up explanations can reuse preparation for the same text and model.
- With **What-if editing**, revise the text and choose **Reanalyze revised document** to run the same model again. Text Lab keeps the **Original** result and shows it alongside **Revised**, with the changed text and all four scores.
- **Compare models** runs independent predictions on the same text. **Export** saves a JSON report and includes the source text only if selected.

<details>
  <summary>See the score comparison after editing the example</summary>
  <p align="center">
    <img src="docs/images/textlab-revision.png" alt="Text Lab What-if editing panel showing the changed text and original and revised scores for all four classes" width="760">
  </p>
</details>

[Run Text Lab locally](#run-text-lab-on-linux-or-windows-wsl2) · [How the influence measurements work](docs/DOCUMENT_EXPLANATION.md)

## Task and data

| Label | Operational definition in the released dataset |
|---|---|
| `human_written` | Human-written text |
| `human_ai_polished` | Human-written content polished by AI |
| `ai_generated` | AI-created content, including AI-to-AI rewriting chains |
| `ai_humanized` | AI-created content rewritten by a person or a tool; this does not confirm human editing |

The main benchmark is **HART**: 15,984 records after cleaning, with source-related and duplicate-linked texts kept in the same partition. The train / validation / test sizes are **11,190 / 1,600 / 3,194**. The arxiv, essay, news and writing subsets are domains within this benchmark.

An independent auxiliary evaluation uses 1,000 human / ChatGPT answers from **HC3**. HC3 provides binary labels only and cannot validate generalization of the four creation-and-editing classes. See [data and external resources](data/README.md) for versions, sources and runtime requirements.

### Experiment pipeline

```mermaid
flowchart TB
    T["English text"] --> V["Text covered by the first 512 tokens"]
    T --> W["Full text"]
    W --> F["TF-IDF + logistic regression"]
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

### Run Text Lab on Linux or Windows WSL2

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

Open <http://127.0.0.1:8765/> after the server starts. Text Lab uses the CPU by default.

The private [resources-v1 release](https://github.com/Realsbt/ee6405-nlp-project/releases/tag/resources-v1) contains the TF-IDF classifier and all 12 neural checkpoints. Use `scripts/fetch_resources.py --profile full` to download the weights and data needed for checkpoint replay.

The demo profile downloads about **4.28 GiB** of resources plus its Python environment. Upstream resources come from pinned original sources. A relocated Linux checkout and a fresh isolated CPU environment have passed all four demo models, exact dataset reconstruction and preprocessing checks. WSL2 instructions are provided but were not exercised in this check.

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
