# Document prediction and internal feature interventions

## Integration

Base: `e3b2d6c6d1b9df8a923abd6a0730fe4ebc0230b2` in `Realsbt/ee6405-nlp-project`.
Upstream model contract: `ICTMCG/RACE` at `61ce528430a8bb84fd1bfb3285ee34827182ef59`.

Original inference, strict checkpoint loading and all portability fixes remain unchanged.
`/api/explain` is optional. If it fails a contract/parity check, the original prediction endpoint continues to exist. No optimizer, fit or training call is made.

The page is served by the same Python service; no npm, CDN, remote font or new production package is needed. The existing Reading Studio remains at `/reading`.

## RACE mathematics

Let `e_i` be the contextual EDU content mean, `t_E` the EDU type embedding, and `t_R` the relation type embedding. The pinned forward initializes:

`x_i = e_i + t_E`

`x_r = (sum(e_j for j in visible_descendants(r)) / n_r) + t_R`.

For a selected EDU k, the intervention sets:

`x'_k = t_E`

`x'_r = x_r - count(k in descendants(r)) * e_k / n_r`.

All other EDU contents, fixed mean denominators, graph edges and types remain unchanged. The existing `node_proj`, graph layers and classifier are run again. Every relation mean is checked against the pinned initialization equation before intervention. The model forward is not rewritten; a temporary `node_proj` input hook applies the edit and is removed in a finally block.

This addresses the direct path from the EDU into relation initialization, including the root when applicable. It does **not** remove information already mixed into other contextual EDU vectors, or remove the unit from the RST topology. It is a conditional internal feature probe, not a causal estimate of authorship or a full-text deletion.

We return actual graph nodes/edge types from the model's graph builder. Trace values are recorded at node projection output, each graph convolution output (before any subsequent activation), and classifier input. In root-pooling configurations, trace RMS is based on the root vector. The tensor shape is the actual full module output shape. ΔRMS is an activation change, not attention or a class contribution. Initial node types and repeated descendant shares are preserved correctly.

## Other models

RoBERTa uses CLS. Suppressing final non-CLS token representations would not in itself modify the existing CLS classifier, so the adapter instead zeros selected **word embedding outputs before encoding**. Positions, special tokens, masks and sequence length are fixed. The encoder is recomputed; this is not identical to the RACE intervention and its magnitudes should not be ranked against RACE as if they shared a baseline.

TF-IDF uses the existing normalized sparse vector. Occurrences of fixed-vocabulary words and bigrams are assigned to the segment containing the first word. For segment k, a feature with n total occurrences and m occurrences assigned to k is changed from x_j to x_j*(1-m/n). Repeated occurrences in other segments retain their shares; an ngram crossing a display boundary is assigned to its first-word segment. Vocabulary, IDF and original normalization remain fixed. No fresh vectorizer fit or new paragraph classification occurs. The adapter verifies that its occurrence counts equal the vectorizer's own analyzer output and refuses unsupported preprocessing.

## Scores and units

For every computed unit, all four `without_scores` are returned. `delta_pp[c] = 100*(original[c]-without[c])`. A positive value supports the selected document class under this reference; a negative value opposes it. The four differences across classes sum approximately to zero, but effects across units need not add up to the original class score.

Display colors are not four-way authorship labels. Uncomputed and out-of-window units remain explicitly unmeasured. A partially visible RST unit only has its lexical portion probed and highlighted.

RACE units are actual preprocessed RST EDUs. Non-RACE units are deterministic display segments. Offsets are emitted in Python code points and JavaScript UTF-16 units, validated against the original text. RACE truncation alignment must form the expected visible prefix; mismatches fail closed rather than labeling the wrong unit.

## API

`POST /api/explain`

```json
{"model":"race","text":"Full document…","max_units":8}
```

Optional `unit_ids` requests up to eight specific existing IDs. The response includes the original `document`, `method`, `units`, actual `graph`, captured `stages`, source hash, replay and restoration differences, and timing. It does not cache document text persistently. Requests are serialized under the existing service lock during hooks. Parsing is not repeated per unit.

The 90-second explanation-loop limit is cooperative between forwards; it does not kill an active operation. The normal prediction/parser startup is separate from this loop budget. UI timeout does not mean server cancellation.

## Verification

1. Original whole-document prediction is computed by unchanged `DetectorService.predict`.
2. Replayed original neural or sparse path must agree within absolute probability tolerance 2e-5.
3. Temporary interventions are removed even on errors.
4. Original inference is repeated after all interventions and must agree again.
5. The local verification script checks state digest, hook counts and predictions for all four actual checkpoints.

This package's included test report separately labels pure mathematical tests, lightweight synthetic-model contract tests, mock HTTP tests, and real Chromium with synthetic API fixtures. It is not a claim of real checkpoint validation on the user's machine.

## v5 navigation layer

The page now has two perspectives over one shared document, model and prediction.

**Explore & Edit** — `Edit`, `Map`, `Passages`, `Coverage`.
**Explain Prediction** — `Influence`, `Structure`.

Switching perspective never recomputes anything; the frozen prediction, selected unit and active version are preserved.

### Document Map

`Map` groups the source text by real blank-line paragraph breaks. Each row shows a preview, the number of intersecting units, how many are explained, how many are pending, how many fall outside the lexical window, and the paragraph's share of the text. Selecting a row focuses the first computed unit in that paragraph and keeps the Map open.

Paragraphs are navigation aids built from the actual text. They are never presented as RST nodes. An EDU that crosses a paragraph break keeps its real offsets and is not split.

### Focus graph

`Structure` renders only the visible subtree. The root is shown by default; relation nodes expand and collapse on click. Controls: `Collapse all`, `Expand selected branch`, `Show ancestors to root`, `Reset view`, `Full graph`. The counter reports `N of M nodes shown`.

Selecting a unit in `Structure` no longer jumps to `Influence`. The graph stays open, the selected node and its ancestor path are highlighted, and the shared detail panel shows the effect on all four document scores.

Three relationships stay visually distinct: parsed parent–child edges (solid), the selected unit's ancestor path (dashed blue), and node initializations directly changed by the intervention (highlighted outline). None of them is drawn as a weighted causal path.

### On-demand explanation

`Explain selected unit`, `Explain visible units` and `Continue explaining` replace the single `Explain first 8` action. The counter reports `N / M eligible units evaluated`. Unmeasured, out-of-window and failed units remain distinct states and never receive a fabricated number.

Merged results are only reused when `text_sha256`, `method` and `document.model` all match. Editing the text, changing the model or switching version invalidates the cache.

### What-if editing

`Original` is frozen at analysis time. Editing the textarea marks the document dirty without touching the frozen prediction. `Reanalyze revised document` runs a real whole-document prediction on the revised text with the same model and stores it as `Revised`. The What-if card shows the four class scores before and after.

A revised snapshot is only accepted when it belongs to the same model and a different text. Before/after prediction differences are labelled separately from feature-suppression effects; the two measurements are not interchangeable.

### Tests

`tests/test_document_map.py` covers paragraph segmentation, EDU/UTF-16 offset agreement, cross-paragraph units, unrequested units carrying no influence, batch limits, cache-key changes with text, and Original/Revised independence. Total suite: 46 tests.

### Onboarding tour

The tour is state-aware rather than a fixed four-step script. `TOUR_STEPS` holds fourteen entries; `tourAvailable()` keeps only those whose `when()` predicate passes and whose target exists in the DOM.

- **First visit** — five steps: text, model, analyze, the two perspectives, and the fixed prediction panel.
- **After the first analysis** — a `Show me the rest` button appears once and opens the full fourteen-step tour, which also covers Map, Passages, Coverage, Influence, on-demand controls, Structure, What-if, Compare and Export.
- **About dialog** — `Replay quick start` runs the state-filtered tour; `Show all steps` runs all fourteen regardless of state.

Each step may declare `app` and `mode`; `tourRender()` switches perspective and opens the relevant tab before positioning the highlight ring, so a step never points at a hidden element. Progress is shown as dots plus `N / M`, and ←/→/Esc work as navigation. Completion is stored under `ee6405.tour.v3` and `ee6405.tour.v3.deep`.

## Source references

- Project main inference.py: https://github.com/Realsbt/ee6405-nlp-project/blob/e3b2d6c6d1b9df8a923abd6a0730fe4ebc0230b2/experiments/inference.py
- Project neural.py and raw_graph.py at the same revision.
- Pinned upstream forward: https://github.com/ICTMCG/RACE/blob/61ce528430a8bb84fd1bfb3285ee34827182ef59/models/flexible_model.py
- Pinned graph builder: https://github.com/ICTMCG/RACE/blob/61ce528430a8bb84fd1bfb3285ee34827182ef59/utils/flexible_graph_builder.py
- Pinned dataset preprocessing: https://github.com/ICTMCG/RACE/blob/61ce528430a8bb84fd1bfb3285ee34827182ef59/utils/flexible_dataset.py
- Feature ablation general definition (conceptual reference, no Captum dependency): https://captum.ai/api/feature_ablation.html
