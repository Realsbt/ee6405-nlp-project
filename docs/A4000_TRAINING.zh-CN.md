# A4000 独立训练部署

从 `origin/main` 建立独立任务分支。资源、环境、上游源码、训练输出均放在 clone 外的持久目录；不要修改或覆盖历史报告。

```bash
export EE6405_STORAGE=/你的持久资源目录
export EE6405_UPSTREAM="$EE6405_STORAGE/upstream/RACE"
python3.11 scripts/setup_environment.py --python python3.11 --torch cu128
export EE6405_PYTHON="$EE6405_STORAGE/.venv/bin/python"
"$EE6405_PYTHON" scripts/fetch_resources.py --profile full --skip-checkpoints
"$EE6405_PYTHON" scripts/restore_data.py
"$EE6405_PYTHON" scripts/fetch_resources.py --profile full --skip-checkpoints --verify-only
"$EE6405_PYTHON" scripts/check_installation.py --training-only \
  --output "$EE6405_STORAGE/installation-training.json"
```

`--training-only` 检查基础资源、固定上游版本、冻结划分哈希和数量、必要导入、CUDA 与 BF16，不要求下载历史训练模型，也不进行模型测试评估。默认安装检查仍用于带历史 checkpoint 的 Demo，`--predict` 与 `--training-only` 不可组合。

原安装器使用 `--no-deps` 安装 RST 包，省略此项目未使用的 spaCy/Playwright 分支。如环境检查发现 NLTK/isanlp 运行依赖缺失，可补充 `click`、`grpcio`、`protobuf==5.29.6`，并保存实际 `pip freeze` 版本用于复现。

先运行原始模型 smoke，再启动单种子基线：

```bash
bash experiments/run_neural.sh --model-type roberta --supcon --seed 3407 \
  --run-name a4000_smoke_roberta_supcon_s3407 --smoke
bash experiments/run_neural.sh --model-type roberta --supcon --seed 3407 \
  --run-name a4000_baseline_roberta_supcon_s3407
```

训练报告现在与训练产物一同保存在资源目录：`experiments/neural_runs/reports/<run_id>/`。权重、事件、状态和源码快照仍保存在 `experiments/neural_runs/<run_id>/`。历史 `experiments/reports/` 保持不变。

本部署修复不改变模型、优化器或选模计算。当前完整基线依然会在选定 checkpoint 后评估 test；后续筛选之前必须先完成训练/测试分离及显式 final evaluator，不能把基线入口直接用于改进筛选。
