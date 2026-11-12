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

## Status
This is a completed execution segment, not a claim that all 36 candidates were trained. Unmeasured cells remain null.
