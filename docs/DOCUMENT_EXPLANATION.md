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

### Lexical window cap

Transformer models tokenize with `max_length=512, truncation=True`, so their lexical window is a hard **512-token** cap including special tokens — not a character limit. The 12,000-character limit is a separate request-size guard. Coverage is therefore bounded by tokens: a 9,000-character document typically exposes only ~2,900 characters to RoBERTa and RACE.

**TF-IDF is exempt.** It has no position embeddings, so the 512-token window is a transformer limit that does not apply to it. `DetectorService.predict` and the explanation preparation both use the full text for `tfidf` and report `input.lexical_window = "full_text"`; every display segment is then `coverage: "full"` and explainable. The other three models report `"512_tokens"`.

`Coverage` states this explicitly: a header with `N of M characters` and either `T / 512 tokens` or `T tokens · no position limit`, plus a sentence explaining that everything past the boundary is not classified, not explained and has no influence value. The score panel reports `32% of text` or `100% (full)` with the same explanation in its tooltip. Out-of-window units are reported as `Outside lexical window`, never as zero influence.

Measured on a 9,000-character document:

| Model | Window | Covered | Eligible units |
| --- | --- | --- | --- |
| TF-IDF | full text | 9,000 / 9,000 (100%) | 68 / 68 |
| RoBERTa | 512 tokens | 2,859 / 9,000 (32%) | 22 / 68 |
| RACE | 512 tokens | 2,859 / 9,000 (32%) | 57 / 179 |

Raising the cap for the transformer models is not a configuration change: `roberta-base` has `max_position_embeddings = 514`, so 512 is the architecture limit. The options are sliding-window aggregation (changes what "whole-document prediction" means), or a long-context encoder (requires retraining and invalidates the existing checkpoints).

RACE units are actual preprocessed RST EDUs. Non-RACE units are deterministic display segments. Offsets are emitted in Python code points and JavaScript UTF-16 units, validated against the original text. RACE truncation alignment must form the expected visible prefix; mismatches fail closed rather than labeling the wrong unit.

## API

`POST /api/explain`

```json
{"model":"race","text":"Full document…","max_units":8}
```

Optional `unit_ids` requests up to eight specific existing IDs. Optional `prepare_only: true` returns the reusable metadata (units, coverage, parsed graph, stages) without running any intervention. Optional `refresh: true` rebuilds the cached preparation instead of reusing it. The response includes the original `document`, `method`, `units`, actual `graph`, captured `stages`, source hash, replay and restoration differences, `timing` (`prepare_seconds`, `units_seconds`) and `cache` (`preparation: hit|miss` plus counters). It does not cache document text persistently. Requests are serialized under the existing service lock during hooks.

The 90-second explanation-loop limit is cooperative between forwards; it does not kill an active operation. The normal prediction/parser startup is separate from this loop budget. UI timeout does not mean server cancellation.

## v5.1 preparation cache

`explain_document()` is split into three stages so a new unit batch never repeats work that only depends on the document:

1. **Document prediction** — `DetectorService.predict`, unchanged. `/api/predict` hands its result to the explanation path through `remember_document()`, so the first explanation does not run a second full forward.
2. **Preparation** — tokenization with offsets, the RST parse, graph construction, unit derivation and coverage. Cached per `(preprocessing_version, model, service instance, text)` in a bounded LRU (`PREP_CACHE_ENTRIES = 4`). The service instance is part of the key because prepared state holds that instance's model objects and tensors.
3. **Baseline and interventions** — the traced baseline forward is deferred until influence is actually requested (`_ensure_baseline`), so `prepare_only` stays cheap. Each requested unit then runs exactly one intervened forward.

`PREPROCESSING_VERSION` is part of the key, so a change in tokenization, graph building or unit derivation can never reuse stale state. `cache_stats()` is exposed through `/api/telemetry` as `explanation_cache`; `clear_preparation_cache()` and `reset_cache_stats()` exist for tests and diagnostics.

Measured on the local CPU service with the real RACE checkpoint:

| Case | Before | After |
| --- | --- | --- |
| First analyze (395-char example) | ~17.8 s | ~17.8 s (prediction only) |
| Metadata for the same document | n/a | ~0.9 s, no intervention |
| First 3-unit Quick preview | ~2.9 s incl. preparation | ~2.9 s |
| Next 8 units (Continue) | full re-prepare | ~5.9 s, preparation cache hit |
| Re-clicking a fully evaluated document | re-ran inference | no request at all |

For an 11,861-character document the prediction took ~31 s, metadata preparation ~14 s (RST parse dominates), and 244 units were derived of which 57 were inside the lexical window and 187 were reported as outside it.

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

Before unit metadata exists a row reports `Not loaded` rather than `0 EDUs`. A passage prediction is labelled `Passage: …` because it covers only the passage span, not the whole paragraph.

Paragraphs are navigation aids built from the actual text. They are never presented as RST nodes. An EDU that crosses a paragraph break keeps its real offsets and is not split.

### Focus graph

`Structure` renders only the visible subtree. The root is shown by default; relation nodes expand and collapse on click. Controls: `Collapse all`, `Expand selected branch`, `Show ancestors to root`, `Reset view`, `Full graph`. The counter reports `N of M nodes shown`.

Selecting a unit in `Structure` no longer jumps to `Influence`. The graph stays open, the selected node and its ancestor path are highlighted, and the shared detail panel shows the effect on all four document scores.

Three relationships stay visually distinct: parsed parent–child edges (solid), the selected unit's ancestor path (dashed blue), and node initializations directly changed by the intervention (highlighted outline). None of them is drawn as a weighted causal path.

### On-demand explanation

`Explain selected unit`, `Explain visible units`, `Quick preview` and `Continue explaining` replace the single `Explain first 8` action. The counter reports `N / M eligible units evaluated`, then the number of units outside the lexical window and the total unit count, so the three numbers are never conflated.

- `Explain selected unit` measures the selected unit regardless of its position.
- `Explain visible units` resolves to the paragraph containing the selection, or to units actually inside the reader viewport. It never silently becomes "the first eight".
- `Quick preview` samples at most three pending units at the start, middle and end of the lexical window and is labelled as positional sampling, not an impact ranking.
- `Continue explaining` keeps the eight-unit request cap and skips units that already have a value.
- `Recompute` is the only action that re-runs an already measured unit (`refresh: true`).

Unit states are explicit: `Not loaded`, `Not evaluated`, `Queued`, `Computing`, `Evaluated`, `Outside lexical window`, `Failed`. A unit is never reported as failed or as zero influence merely because it has not been measured. The generic `Explain prediction` button only requests pending units and reports when nothing is left to measure.

Merged results are only reused when `text_sha256`, `method` and `document.model` all match. Editing the text, changing the model or switching version invalidates the cache.

`auto-explain` is off by default: `Analyze document` guarantees the real whole-document prediction, then loads metadata, and only runs influence when the checkbox is on.

### What-if editing

`Original` is frozen at analysis time. Editing the textarea marks the document dirty without touching the frozen prediction. `Reanalyze revised document` runs a real whole-document prediction on the revised text with the same model and stores it as `Revised`. The What-if card shows the four class scores before and after.

A revised snapshot is only accepted when it belongs to the same model and a different text. The card also shows the literal changed region (removed text struck through, added text highlighted) so the score change can be read against the actual edit. Before/after prediction differences are labelled separately from feature-suppression effects; the two measurements are not interchangeable.

### Tests

`tests/test_document_map.py` covers paragraph segmentation, EDU/UTF-16 offset agreement, cross-paragraph units, unrequested units carrying no influence, batch limits, cache-key changes with text, and Original/Revised independence. `tests/test_explanation_cache.py` covers `prepare_only` metadata, preparation reuse across batches, identical baselines after reuse, `refresh`, key separation by text/model/service, `remember_document` avoiding a second prediction, bounded capacity, hit/miss counters, cache clearing and hook cleanup. Total suite: 62 tests.

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
