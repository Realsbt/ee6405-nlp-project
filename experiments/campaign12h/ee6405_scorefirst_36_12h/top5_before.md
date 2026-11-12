# Top 5 before E16

- Scope: S selection split, 960 samples, same item IDs and labels for every candidate.
- Metrics recomputed from prediction `.npy` files; J=(Accuracy+Macro-F1)/2.
- Valid prediction candidates: 11. No unverified/blocked/planned task was filled to reach five.
- H is excluded from selection; test/HC3 were not loaded.

## J Top 5

|Rank|Task|Status|Accuracy|Macro-F1|J|wrong_to_right|right_to_wrong|
|---:|---|---|---:|---:|---:|---:|---:|
|1|E30|COMPLETE|94.5833%|92.6428%|93.6131%|7|5|
|2|E26|COMPLETE|94.4792%|92.7021%|93.5906%|3|2|
|3|E01|COMPLETE|94.3750%|92.5641%|93.4696%|0|0|
|4|E03|COMPLETE|94.3750%|92.5641%|93.4696%|0|0|
|5|E04|COMPLETE|94.3750%|92.5641%|93.4696%|0|0|

## Accuracy Top 5

|Rank|Task|Status|Accuracy|Macro-F1|J|wrong_to_right|right_to_wrong|
|---:|---|---|---:|---:|---:|---:|---:|
|1|E30|COMPLETE|94.5833%|92.6428%|93.6131%|7|5|
|2|E26|COMPLETE|94.4792%|92.7021%|93.5906%|3|2|
|3|E02|COMPLETE|94.4792%|92.4308%|93.4550%|5|4|
|4|E01|COMPLETE|94.3750%|92.5641%|93.4696%|0|0|
|5|E03|COMPLETE|94.3750%|92.5641%|93.4696%|0|0|

## Macro-F1 Top 5

|Rank|Task|Status|Accuracy|Macro-F1|J|wrong_to_right|right_to_wrong|
|---:|---|---|---:|---:|---:|---:|---:|
|1|E26|COMPLETE|94.4792%|92.7021%|93.5906%|3|2|
|2|E30|COMPLETE|94.5833%|92.6428%|93.6131%|7|5|
|3|E01|COMPLETE|94.3750%|92.5641%|93.4696%|0|0|
|4|E03|COMPLETE|94.3750%|92.5641%|93.4696%|0|0|
|5|E04|COMPLETE|94.3750%|92.5641%|93.4696%|0|0|

## Pareto non-dominated

E26, E30

## E16 status

E16 has no qualifying formal result. Existing record is RESOURCE_BLOCKED because DeBERTa-v3-base weights were unavailable locally/offline. No new model was trained.
