# 定向修正执行结果

本报告来自已绑定的 S 预测；S 曾参与历史开发。没有新训练、H/test/HC3 评价或部署。

## 结论

CV_BLOCKED_KEEP_B0；候选部署 lambda=0.0（不自动部署）。

## 现有固定方法：S 历史复用描述性榜

|方法|Accuracy|Macro-F1|J|纠正/误伤|
|---|---:|---:|---:|---:|
|E26|94.4792%|92.7021%|93.5906%|3/2|
|B0|94.3750%|92.5641%|93.4696%|0/0|
|E16|82.7083%|69.7720%|76.2402%|19/131|

## 唯一修正方法：分组交叉拟合

该过程的 OOF 分数单独报告，不与拟合内分数混排；全 S 重选 lambda 不算独立成绩。

```json
{
  "status": "CV_BLOCKED_KEEP_B0",
  "deploy_lambda": 0.0,
  "reason": "Cross-fitted selection requires audited groups for both models.",
  "crossfit_executed": false,
  "score_kind": "NO_OOF_BASELINE_FALLBACK",
  "baseline": {
    "n": 960,
    "correct": 906,
    "accuracy": 0.94375,
    "macro_f1": 0.9256410942828763,
    "J": 0.9346955471414382,
    "nll": 0.16193354968981197,
    "confusion_matrix": [
      [
        239,
        1,
        0,
        0
      ],
      [
        3,
        223,
        14,
        0
      ],
      [
        1,
        18,
        397,
        5
      ],
      [
        0,
        0,
        12,
        47
      ]
    ],
    "label_order": [
      "human_written",
      "human_ai_polished",
      "ai_generated",
      "ai_humanized"
    ],
    "per_class": {
      "human_written": {
        "support": 240,
        "predicted": 243,
        "precision": 0.9835390946502057,
        "recall": 0.9958333333333333,
        "f1": 0.989648033126294
      },
      "human_ai_polished": {
        "support": 240,
        "predicted": 242,
        "precision": 0.9214876033057852,
        "recall": 0.9291666666666667,
        "f1": 0.9253112033195021
      },
      "ai_generated": {
        "support": 421,
        "predicted": 423,
        "precision": 0.9385342789598109,
        "recall": 0.9429928741092637,
        "f1": 0.9407582938388626
      },
      "ai_humanized": {
        "support": 59,
        "predicted": 52,
        "precision": 0.9038461538461539,
        "recall": 0.7966101694915254,
        "f1": 0.8468468468468469
      }
    }
  }
}
```

## 需要交付的其他文件

`E16_diagnosis.json`、`E30_restoration.json`、`disagreement_audit.json`、`policy_candidate.json`。
若没有收益，最终建议保留 B0；不能因为生成了 policy 文件而声称已部署或性能提高。
