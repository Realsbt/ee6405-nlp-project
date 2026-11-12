# Yingqi — Research Outline for the Group Presentation

**Working title** — What Discourse Adds to AI Text Detection, and Where the Gains Hold

**Evidence snapshot** — 12 November 2026. The core section takes about three minutes and follows slides 7–10 of the current group deck. The remaining material is for slide merging and Q&A.

## 1. My contribution in one sentence

I built same-window controls to test what discourse structure adds, implemented a gated Text–EDU fusion model, and evaluated whether combining frozen classifiers improves detection on HART and on external data.

## 2. Connection to the group project

The group compares text and discourse models on HART’s four creation and editing labels. The shared RoBERTa and RACE comparison motivates my question. With the text window held fixed, which discourse signals help classification, and do improvements on HART carry over to a different dataset? My work builds on the group’s split, model components, parser, and training infrastructure.

## 3. Core presentation — about three minutes

### Slide 7 — Same text, four discourse readouts (0:00–0:45)

**Handoff from Yuzhe** — “We’ve seen how changing the training recipe affects the baseline. I looked at what happens when we keep the visible text fixed and change how the model reads it.”

I trained four CE-only variants on the same RoBERTa-visible prefix of up to 512 tokens. The variants use a CLS text summary, an RST discourse graph, the same graph with relation labels shuffled, or pooled elementary discourse units (EDUs). The shuffle preserves graph topology and the per-document relation-label counts. I also verified the text-to-EDU alignment across all 15,984 records.

**HART test Macro-F1, seed 3407** — CLS text **91.28%**, real-relation graph **89.69%**, shuffled-relation graph **90.07%**, and EDU pooling **90.89%**. The text model led this run, and real relation labels showed no advantage over shuffled labels. The graph variants have more trainable parameters, and two additional seeds are still needed for the planned comparison.

### Slide 8 — Gated Text–EDU fusion (0:45–1:30)

I used the global CLS summary as the main representation and added a local path. The model pools tokens within each EDU, uses attention to combine the units, and learns a gate that controls how much of that context to add back to CLS.

**HART validation Macro-F1, seed 3407** — Matched CLS **90.64%**, equal-weight EDU fusion **90.85%**, EDU attention **90.90%**, and gated EDU attention **91.36%**. Gated fusion led the validation screen by **0.72 percentage points** over matched CLS. On the same validation set, AI-humanized recall moved from **65.45% to 68.18%**. Final multi-seed test evidence for fusion is still pending in the verified records.

### Slide 9 — Stacking and a paired editing example (1:30–2:15)

I trained a logistic-regression meta-model on development predictions from frozen RoBERTa CE, RACE SupCon, and TF-IDF models. I selected its configuration using four source-group development folds and refit it on all development records. Stacking reached **93.61% HART test Macro-F1**, compared with **92.10%** for the two-model probability blend. This uses one underlying neural seed and a previously inspected test split.

For the visual, show the two selected HART texts from the same source group. Stacking assigned **96.8%** to the dataset label for the original AI-generated text and **98.5%** to the AI-humanized label for the edited version. The pair illustrates how the scores change after editing. The aggregate result comes from the full test split.

### Slide 10 — External transfer (2:15–3:00)

On the fixed, mapped **359-text MixSet subset** from 91 source groups, **RoBERTa CE scored 39.01%**, the **probability blend 36.86%**, and **stacking 36.22%** Macro-F1. The HART ensemble gain did not carry over to this subset. At the stacking threshold calibrated on HART development data for a nominal 1% false-positive rate, external FPR was **18.80%** and TPR was **5.50%**.

**Closing line** — “Under the same text window, CLS led the completed graph comparison. Gated EDU fusion improved the validation score, and stacking improved the HART test score in the recorded setup. The MixSet result shows how sharply those gains can change with the dataset.”

## 4. Backup and Q&A

- **Same-window controls.** Built and audited the full graph cache, recovered parser offsets for verified Unicode compatibility cases, and checked that every visible EDU stayed inside the model’s input window. All four conditions passed CPU and Tesla T4 smoke checks. The complete seed-3407 run includes saved development and test predictions.
- **Fusion implementation.** Implemented the residual Text–EDU module, its four comparison variants, and validation-only selection and checkpoint handling. Seed 3407 has all four screening runs. The verified snapshot also includes a completed CLS validation run for seed 42. Gate-strength, segmentation, and consistency controls are implemented follow-ups with CPU checks.
- **Encoder pilot.** In a separate seed-3407 validation-only pilot, fully fine-tuned RoBERTa reached **91.48%** Macro-F1 with **62.73%** AI-humanized recall. Equal-weight blending of gated fusion and DeBERTa predictions reached **91.99%** Macro-F1 with **67.27%** AI-humanized recall. Gated fusion alone recorded **91.36%** and **68.18%**, respectively. Keep these exploratory comparisons in backup.
- **Reproducibility.** The saved-results checker verifies the same-window and fusion evidence, plus the historical stacking scores and prediction files. The shared repository supplies HART, RoBERTa and RACE components. My additions are the controlled comparisons, the project-specific fusion module, the stacking fit, and the external and editing diagnostics.

## 5. Notes for merging slides

- The shared HART baseline and Yuzhe’s training follow-up already appear before this section. Start with the common input window.
- Keep the split and seed on each figure. Slide 7 is a single-seed **test** comparison, slide 8 is a single-seed **validation** screen, and slide 9 shows selected examples alongside a single-base-seed ensemble result.
- Slide 10 can retain the shared HC3 result alongside MixSet if time permits. HC3 is binary and the mapped MixSet subset is four-class, so give each its own task label.
- The current deck’s runtime and per-class stacking slides can support Q&A.

## 6. Existing materials

- [Current group deck, slides 7–10](EE6405_Text_Creation_and_Editing_Detection_v3.pptx)
- [Yingqi contribution and verified results snapshot](../EE6405_paper_research/repository/docs/YINGQI_STATUS.md)
- [Same-window ablation report](../EE6405_paper_research/repository/experiments/reports/same_window/REPORT.md)
- [Text–EDU fusion protocol](../EE6405_paper_research/repository/docs/EDU_FUSION.md)
- [External transfer report](../EE6405_paper_research/repository/experiments/reports/yingqi_snapshot/external/REPORT.md)
- [Encoder pilot](../EE6405_paper_research/repository/docs/ENCODER_PILOT.md)
