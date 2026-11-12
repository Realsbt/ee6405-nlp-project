# 36-task score-first campaign

## Protocol
Only HART train and validation were loaded. S used selection; H was evaluated after candidate_lock.json. H is not a fresh external test because historical validation was previously reused. HART test and HC3 were not loaded.

## Locked candidates
E30 was the S Accuracy/J recommendation; E26 was the S Macro-F1 recommendation. The lock was written before H evaluation.

## Confirmation
- E01: H Accuracy=0.942187, H Macro-F1=0.913339
- E02: H Accuracy=0.935937, H Macro-F1=0.895315
- E03: H Accuracy=0.942187, H Macro-F1=0.913339
- E04: H Accuracy=0.942187, H Macro-F1=0.913339
- E05: H Accuracy=0.868750, H Macro-F1=0.741161
- E06: H Accuracy=0.248438, H Macro-F1=0.168755
- E07: H Accuracy=0.628125, H Macro-F1=0.464082
- E25: H Accuracy=0.921875, H Macro-F1=0.888985
- E26: H Accuracy=0.939063, H Macro-F1=0.908116
- E30: H Accuracy=0.943750, H Macro-F1=0.914626
- E31: H Accuracy=0.937500, H Macro-F1=0.906811

## Interpretation
E30 confirmed a joint H improvement over B0: Accuracy +0.156pp and Macro-F1 +0.129pp. E26 improved S but did not confirm on H. E02 traded a small Accuracy gain for lower Macro-F1. E08/E09 reached the 160-update screen budget and are UNDERTRAINED; E10 is RESOURCE_BLOCKED because PEFT is absent and the main environment was not reinstalled. E11-E24, E27-E29, E32-E36 remain PLANNED.

## Stage B continuation

The original `finalized` state was audited as an early export, not treated as
completion of the 36-task campaign. The original start and deadline were
preserved. After Stage A, selection used only train/S; H was not reopened.

- E14 completed its 320-update cold-start screen: S Accuracy=0.925000 and
  S Macro-F1=0.882201, below B0 S (0.943750/0.925641). This is a screen
  result, not a convergence claim.
- E11 and E13 completed 160-update screens; each best result fell back to B0
  (0.943750/0.925641), so neither was promoted.
- E12 completed the specified 100-update FGM-style screen; its best result
  also fell back to B0 (0.943750/0.925641), so it was not promoted.
- E20 (two-window average) reached S Accuracy=0.940625 and Macro-F1=0.917371;
  E21 (three-window average) reached 0.941667/0.920311. Both were below B0
  and were not promoted.
- E15 and E16 are `RESOURCE_BLOCKED`: the required DeBERTa weights are not
  present locally and offline downloads are disabled.
- E10 is `IMPLEMENTATION_BLOCKED`: `peft` is absent, but the project-local
  Linear-LoRA compatibility smoke passed for 24 query/value modules
  (zero-initialized B, nonzero gradients, parameter update). No E10 screen
  trainer was added or run, and the main environment was not modified.

The complete current S-only status table is
`leaderboard_stage_B.json`. The Stage A H table remains unchanged in
`leaderboard_confirm.json`. HART test and HC3 remain unopened.

## Status
This is a completed execution segment, not a claim that all 36 candidates
were trained. Unmeasured cells remain null; screen-budget results are not
reported as converged models.
