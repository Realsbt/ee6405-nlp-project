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

## Source references

- Project main inference.py: https://github.com/Realsbt/ee6405-nlp-project/blob/e3b2d6c6d1b9df8a923abd6a0730fe4ebc0230b2/experiments/inference.py
- Project neural.py and raw_graph.py at the same revision.
- Pinned upstream forward: https://github.com/ICTMCG/RACE/blob/61ce528430a8bb84fd1bfb3285ee34827182ef59/models/flexible_model.py
- Pinned graph builder: https://github.com/ICTMCG/RACE/blob/61ce528430a8bb84fd1bfb3285ee34827182ef59/utils/flexible_graph_builder.py
- Pinned dataset preprocessing: https://github.com/ICTMCG/RACE/blob/61ce528430a8bb84fd1bfb3285ee34827182ef59/utils/flexible_dataset.py
- Feature ablation general definition (conceptual reference, no Captum dependency): https://captum.ai/api/feature_ablation.html
