# Same-window mechanism ablation

Status: implementation and CPU acceptance complete on 2026-11-11; formal GPU
training is pending.

## Why this ablation exists

The historical RACE and RoBERTa comparison does not isolate graph structure.
RACE can use full-document structure while the lexical model sees at most 512
tokens, and the systems also differ in pooling and parameter count. Similar
scores therefore cannot establish whether discourse relations are useful.

This protocol first freezes the RoBERTa-visible prefix, reparses exactly that
text, and then changes one structural mechanism at a time. All conditions use
cross-entropy, the same grouped partitions, seeds 3407/42/2026, validation
Macro-F1 selection and base-pretrained initialization.

| ID | Condition | Main comparison |
|---|---|---|
| A | `text`: CLS + MLP | Same-window text baseline |
| B | `graph`: RACE graph | Total effect of graph pipeline under equal input |
| C | `shuffled`: relation assignments permuted within each graph | B-C tests relation-type information |
| D | `edu`: equal-weight EDU pooling, no message passing | B-D tests graph processing beyond segmentation/pooling |

The relation permutation is deterministic from `item_id` and control seed
20261109, never reads the class label, preserves each graph's relation
histogram, and rebuilds relation-dependent edge types without changing topology.

## Data acceptance

The external cache is deliberately not stored in Git because it contains source
texts and is about 700 MB. Its committed manifest evidence records:

- 15,984 verified records; 11,190 train, 1,600 validation and 3,194 test.
- Zero failures, pending records, hidden EDU nodes or cross-split exact visible
  prefix duplicates.
- Strict offset alignment first. Fallbacks only restore offsets under verified
  NFKC equivalence, actual tokenizer offsets, or an audited projection of
  U+200B/U+FEFF/U+200D/U+2581. Model input text is never normalized or changed.
- Content changes, unapproved deleted characters, non-monotonic maps and EDU
  boundaries inside compatibility expansions are rejected.

See `experiments/reports/same_window/` for the protocol hashes, relation audit,
alignment summary and smoke invariants.

## Run commands

Set `EE6405_STORAGE` and `EE6405_UPSTREAM` as described in
`docs/REPRODUCING.md`. Place the verified cache at a local path, for example:

```bash
export CACHE="$EE6405_STORAGE/experiments/same_window_full_v3"
```

CPU smoke check, which never opens the test split:

```bash
python -m experiments.train_same_window \
  --condition text --cache "$CACHE" --output outputs/same_window_smoke_text --smoke
```

Formal CUDA runs are separate directories. Repeat for `text`, `graph`,
`shuffled`, `edu` and seeds `3407`, `42`, `2026`:

```bash
python -m experiments.train_same_window \
  --condition graph --seed 3407 --cache "$CACHE" \
  --output outputs/same_window_graph_s3407
```

Each run writes its frozen config, parameter inventory, epoch events, selected
checkpoint, validation/test probabilities and final metrics. Do not tune research
settings on the test partition. Report three-seed means, sample SDs and matched
seed differences for B-D, B-C and B-A.

## Current evidence boundary

CPU smoke proves data loading, finite gradients, frozen embeddings and parameter
updates. The full audit proves that the relation control materially changes edge
types while preserving topology. Neither result measures predictive quality.
No same-window test Macro-F1 should be added to the paper until all 12 formal
runs complete and their saved predictions are independently checked.
