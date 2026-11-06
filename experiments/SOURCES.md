# Sources and local evidence

## Course context

The private group repository is [Realsbt/ee6405-nlp-project](https://github.com/Realsbt/ee6405-nlp-project), distinct from the RACE authors' repository. Its EE6405 syllabus and group-project brief are retained under `docs/` and were read before selecting the experimental scope. The repository contains the course harness and selected evaluation evidence; raw datasets, model weights, vendored upstream code and local runtime files are excluded. See [external resources](../data/README.md) and the [submission checklist](../docs/SUBMISSION.md).

The brief explicitly allows reproducing a recent model or studying variants. The rubric rewards relevant baselines and metrics, interpretation, documented/reusable code, efficiency and an intuitive demonstration. Multiple datasets are rewarded; the brief suggests three or more, while this scope includes HART and one independent HC3 auxiliary sample. The four HART domains count as slices of one benchmark. The video is at most ten minutes and uses human narration. This document does not replace the current course brief.

## RACE and HART

- Paper: [Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](https://aclanthology.org/2026.acl-long.235/).
- Authors' code: [ICTMCG/RACE](https://github.com/ICTMCG/RACE), pinned commit `61ce528430a8bb84fd1bfb3285ee34827182ef59` under `local_validation/upstream/RACE`.
- Author-supplied HART graph data and smoke checkpoint: [official shared folder](https://drive.google.com/drive/folders/1goJ9F5uXhBTabURr9q23KhbPxiYNFK-8). Local file hashes are in `reports/data_v1.json` and `/data/botao/ee6405-race-validation/assets/hart_official/download_manifest.json`.
- Official baseline/RACE graph builder and loss implementations are reused without tracked source changes. The course harness adds group-safe partitions, window alignment, correct RoBERTa padding, cached preprocessing, fixed experiments and explicit evaluation checks.
- Read-only coordinate audit of the frozen release found 50 records with invalid EDU spans and 3,355 otherwise valid records with uncovered non-whitespace characters across all partitions. The test portion contains 10 and 677 respectively. These counts concern records with any coordinate issue, not the proportion of text lost or discourse-relation accuracy. Existing trees, splits and trained models remain unchanged; coordinate-clean subsets are post-hoc descriptive analysis. See `reports/released_rst_span_audit.json` and `released_rst_item_quality.jsonl`.
- The author fine-tuned checkpoint is used only for the earlier smoke test. It is never an initialization or a fair evaluation checkpoint for the new grouped split.
- No separate LICENSE file was present in the pinned RACE tree. The upstream checkout and model/data assets are not redistributed in this repository; obtain them separately from the original sources. Public redistribution rights for those assets have not been established.

## Pretrained encoder

[FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base), revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`. All neural comparison runs initialize from these base-pretraining weights. The SHA-256 is stored in each result JSON. Fine-tuning updates only the final encoder layer and the task-specific components.

## Independent auxiliary data

[Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3), revision `4d0ff18143b5a7e1b1e79beb540c04549d1e59d3`. The original README and source manifest are retained in `/data/botao/ee6405-race-validation/assets/hc3/`.

The card specifies CC-BY-SA-4.0, with stricter terms of original source datasets retained. Local evaluation uses 100 question pairs from each of five sources, without training or threshold tuning on HC3. Normalized exact-answer overlap with all HART partitions and cleaning exclusions was excluded. This check does not establish absence of semantic overlap or base-pretraining exposure.

HC3 supplies human/ChatGPT labels, not creator/editor four-class labels. The predefined auxiliary score is `P(ai_generated)+P(ai_humanized)`, threshold `0.5`. The original `index` can be null/repeated; `repair_aux_ids.py` creates `hc3_aux_v1_1` using question fingerprints while preserving all selected text, labels, order and the original version. Old v1 prediction IDs are unsuitable for joining records.

## Raw-text RST parser

- [tchewik/isanlp_rst_v3](https://huggingface.co/tchewik/isanlp_rst_v3), RST-DT revision `5141450ac3eb7decad0b8735fdbf5ec6928ec97a`.
- `isanlp-rst==3.2.2`; [iinemo/isanlp](https://github.com/iinemo/isanlp), commit `2a102e59f9718acc7fe259dd8d83c66d5da39794`.
- Encoder config/tokenizer: [FacebookAI/xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large), revision `c23d21b0620b635a76227c604d44e43a9f0ee389`.
- `reports/parser_assets.json` verifies seven local parser/tokenizer files against preserved pinned-revision HF download etags. The 2,545,320,602-byte parser checkpoint SHA-256 is `c4008a42605d6fc2657be72ba1de50977b164f88bbc5f53b62a922f43fff3c32`, matching its LFS record. This is an offline check of original download metadata, not a new remote audit.
- The parser checkpoint already contains trained encoder weights; only the XLM-R config/tokenizer are loaded separately. Source config is preserved; a local config copy replaces the model-name field with the local tokenizer/config path.
- CPU initialization uses memory-mapped loading and disables unused mean initialization before the parser strictly loads its complete trained state. The original parser inference path and weights are retained.
- Newly parsed input uses validated original character spans and canonical relation names. Author-supplied HART trees remain unchanged; exact regeneration of those released trees has not been established. Parser/version differences may affect raw-text auxiliary and Demo outputs.
- A stricter HC3 coverage audit found 34/1000 released-parser outputs with dropped non-whitespace characters at EDU span boundaries. The decoded-subword and space-joined-word coordinate systems can differ. `rst_alignment.py` preserves decoded EDU content/order and predicted tree relations, then exactly aligns their non-whitespace characters to the original text. Non-whitespace mismatches raise an error. Corrected outputs are saved separately as `test_parsed_aligned.jsonl`; the original parse and its failed coverage audit remain for inspection. This is an offset repair, not a new segmentation or detection algorithm.

## Interpretation boundaries

This is a course reproduction and controlled comparison, not a claim of a new algorithm. A good result on a grouped HART holdout, a training-seed average, an HC3 binary result and an unlabeled Demo prediction support different conclusions. Neither raw model scores nor HART accuracy establish reliable authorship attribution on arbitrary English texts.
