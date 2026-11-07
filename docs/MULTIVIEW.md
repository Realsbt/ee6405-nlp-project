# Head/middle/tail inference comparison

`experiments/evaluate_multiview.py` compares the existing main single-window
inference path with an inference-only three-view strategy using the same
checkpoint, tokenizer, labels, records, evaluation mode, and batch size.

For each document, the implementation tokenizes the full article without
truncation, evaluates the first, centered, and last 512-token windows when the
article is longer than 512 tokens, and evaluates one window for shorter
articles. It averages the per-window **softmax probabilities**, not logits or
votes. Each document contributes one final prediction.

The window construction follows the existing main-side implementation in the
campaign code: the centered start is `(token_count - 512) // 2`; special
tokens are retained by the tokenizer; padding is masked and does not represent
document content. This strategy reads more article text and performs up to
three forward passes, so it is a same-weight inference comparison rather than
an equal-information or equal-cost architecture ablation.

## Run

From the repository root, provide paths for the checkpoint, its tokenizer,
the JSONL evaluation records, and an output file:

```powershell
python -m experiments.evaluate_multiview `
  --checkpoint <checkpoint.pt> `
  --config <matching-main-config.json> `
  --tokenizer <tokenizer-directory> `
  --data <evaluation-records.jsonl> `
  --batch-size 16 `
  --output <multiview-comparison.json>
```

The config is checked against the config embedded in the checkpoint, and model
weights are loaded with `strict=True`. No training, checkpoint selection,
automatic split, test/HC3 access, or seed/epoch choice is performed.

The output contains item-bound single-window and multiview probabilities,
Accuracy, Macro-F1, four-class F1, percentage-point deltas, correction and
regression counts, window specification, checkpoint hash, and timing.
