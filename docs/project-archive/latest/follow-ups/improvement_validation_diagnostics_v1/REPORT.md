# Validation error diagnosis

No test partition was opened.

| Model | Macro-F1 | AI-humanized F1 | AI-humanized recall |
|---|---:|---:|---:|
| roberta_ce | 91.0617% | 80.2030% | 71.8182% |
| race_supcon | 90.5578% | 78.4689% | 74.5455% |
| tfidf | 78.4321% | 55.5556% | 40.9091% |
| fusion_cls | 90.6435% | 76.1905% | 65.4545% |
| fusion_gated | 91.3643% | 78.1250% | 68.1818% |

All supplied models wrong: 34.
Pairwise overlaps are in diagnostics.json; they are not ensemble performance.
