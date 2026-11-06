# Linux / Windows WSL2 复现指南

[English](REPRODUCING.md) | **简体中文** | [项目概况](../README.zh-CN.md)

组员不需要复制原机器的 `/data/botao/ee6405-race-validation`。这个约 17 GB 的目录保存数据、权重、环境、缓存和训练记录，并不是第二个代码项目。Git 保存代码与结果证据；模型单独放在私有仓库的 [resources-v1 Release](https://github.com/Realsbt/ee6405-nlp-project/releases/tag/resources-v1)，其他资源从固定版本的原始来源获取。

| 想做什么 | 要求 |
|---|---|
| 核对已保存结果 | Python 3 标准库，不需要模型或显卡 |
| 运行四模型英文 Demo | CPU、Python 3.11，下载资源约 4.28 GiB，另加环境 |
| 恢复原始数据划分和八个正式模型 | `full` 资源约 6.86 GiB，另加环境与生成数据 |
| 重新训练神经模型 | 支持 BF16 的 NVIDIA CUDA 显卡，以及更多时间和磁盘空间 |

Demo 建议预留约 15 GB 磁盘、16 GB 内存；这不是测定的最低配置。完整训练建议至少预留 40 GB 磁盘，显卡应结合实验报告的显存记录选择。没有显卡也能运行四模型 Demo 和 TF-IDF。

## 1. 系统准备

Linux 使用 x86-64 系统。Windows 先按 [微软说明](https://learn.microsoft.com/en-us/windows/wsl/install) 在管理员 PowerShell 运行 `wsl --install`，按提示重启，再用 `wsl -l -v` 确认版本为 2。之后所有命令都在 Ubuntu / Linux 终端运行。

准备 **Python 3.11（包含 venv）、Git、GitHub CLI 和 C/C++ 编译器**。Ubuntu 编译器包为 `build-essential`，安装 `jsonnet` 时可能需要。系统 Python 不是 3.11 时，单独安装一个 3.11，不要替换系统 Python；也可通过 `--python /实际路径/python` 指定现有 Conda Python 3.11。

仓库是私有的，每位组员需要仓库读取权限，并用自己的账号登录：

```bash
gh auth login
gh repo clone Realsbt/ee6405-nlp-project
cd ee6405-nlp-project
python3 scripts/check_results.py
```

最后一步只从保存的预测重算指标，不加载模型，也不替代模型推理验证。

## 2. 安装并启动英文 Demo

每次打开新终端都需要设置 `EE6405_STORAGE`，或把它加入自己的 shell 配置。路径可以自选，确保空间足够。

```bash
export EE6405_STORAGE="$HOME/ee6405-resources"
python3 scripts/setup_environment.py --python python3.11 --torch cpu
export EE6405_PYTHON="$EE6405_STORAGE/.venv/bin/python"

"$EE6405_PYTHON" scripts/fetch_resources.py --profile demo
"$EE6405_PYTHON" scripts/check_installation.py --predict
bash experiments/run_demo.sh
```

浏览器打开 <http://127.0.0.1:8765/>。界面为英文，包含 TF-IDF、RoBERTa、RACE 和无 SupCon 的 RACE。RACE 首次请求还需要加载真实篇章解析器；服务只监听本机地址，按 `Ctrl+C` 停止。

安装器创建独立环境，不继承原机器的 Python 包；已有环境不会被覆盖。下载器按[资源清单](../resources/manifest.json)逐个检查大小和 SHA-256，也不会覆盖已有但不匹配的文件。推理在本地离线运行，下载阶段需要联网。如果终端之前启用了离线变量，先运行 `unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE`。

## 3. 恢复固定数据和图缓存

```bash
"$EE6405_PYTHON" scripts/fetch_resources.py --profile full
"$EE6405_PYTHON" scripts/restore_data.py
"$EE6405_PYTHON" -m experiments.prepare_graph_cache \
  --report "$EE6405_STORAGE/experiments/reproduction_checks/graph_cache.json"
"$EE6405_PYTHON" -m experiments.verify_graph_cache \
  --report "$EE6405_STORAGE/experiments/reproduction_checks/graph_check.json"
"$EE6405_PYTHON" -m experiments.verify_tfidf \
  --report "$EE6405_STORAGE/experiments/reproduction_checks/tfidf_reload.json"
```

脚本恢复与原实验**字节及哈希完全一致**的 HART 三个分区（11,190／1,600／3,194）和 HC3 1,000 条回答。HC3 的选择清单只保存 ID、元数据、哈希和截断位置，不包含回答正文。重新抽一个 HC3 子集无法重现原来的评估，因此需要保留该选择清单。

图缓存存在时不会重建，直接运行校验即可。`--report` 将本次校验保存到资源目录，保留 Git 中的历史报告。HART 评估使用作者发布的图，Demo 使用实时解析，二者的输入流程有区别。

HART 的 Google Drive 下载若遇限流，可打开清单里的对应官方链接，将文件放到指定相对路径后重试，仍会检查哈希。也可从已有资源目录复制所需文件：

```bash
"$EE6405_PYTHON" scripts/fetch_resources.py --profile full --local-source /已有资源副本路径
```

该命令只复制选中的资源，不复制环境和缓存。完全离线时需要事先准备固定版本的上游源码。安装后可运行 `fetch_resources.py --profile full --verify-only` 只读检查资源。

## 4. 需要显卡的训练与重载

现有神经训练和完整神经 checkpoint 重载入口要求 CUDA 与 BF16；CPU 路径用于 Demo 和 TF-IDF。全新 GPU 环境使用 `setup_environment.py --torch cu128`。已有 CPU 环境可改装相同版本的 GPU wheel：

```bash
"$EE6405_PYTHON" -m pip install torch==2.7.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
"$EE6405_PYTHON" -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available()); print("BF16:", torch.cuda.is_bf16_supported())'
```

CUDA、BF16 都应为 `True`。驱动需要兼容该 wheel；WSL2 用户要确保 Linux 内能访问显卡。版本来源为 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/)。

完成 full 下载和数据恢复后，提取历史配置并用新的运行名训练，保留原始实验。显式配置也能在训练 CLI 扩展后保留原实验的损失函数设置：

```bash
python3 - <<'PY'
import json, os
from pathlib import Path
target = Path(os.environ['EE6405_STORAGE']) / 'experiments/training_recipes'
target.mkdir(parents=True, exist_ok=True)
for kind in ('roberta', 'race', 'race_no_supcon'):
    run = f'{kind}_s3407'
    recipe = json.loads(Path(f'experiments/reports/{run}/result.json').read_text())['config']
    (target / f'{run}.json').write_text(json.dumps(recipe, indent=2) + '\n')
PY

bash experiments/run_baseline.sh teammate_tfidf
bash experiments/run_neural.sh --model-type roberta --seed 3407 --run-name teammate_roberta_s3407 \
  --config "$EE6405_STORAGE/experiments/training_recipes/roberta_s3407.json"
bash experiments/run_neural.sh --model-type race --seed 3407 --run-name teammate_race_s3407 \
  --config "$EE6405_STORAGE/experiments/training_recipes/race_s3407.json"
bash experiments/run_neural.sh --model-type race_no_supcon --seed 3407 --run-name teammate_no_supcon_s3407 \
  --config "$EE6405_STORAGE/experiments/training_recipes/race_no_supcon_s3407.json"

"$EE6405_PYTHON" -m experiments.verify_neural --run race_s3407 \
  --report "$EE6405_STORAGE/experiments/reproduction_checks/race_reload.json"
```

Release 内完整 RACE 使用 SupCon，无 SupCon 版本不使用。比较原始结果时应保持冻结配置。仅从基础预训练初始化重新训练，可在 full 下载时加 `--skip-checkpoints`，无需下载我们的已训练模型。

HC3 的 RACE 复跑还需要对恢复的回答执行 `experiments.rst_worker`，再执行 `experiments.evaluate_external`，见[详细实验流程](../experiments/README.md)。这些入口会写评估结果，独立复跑历史模型时请使用单独工作副本。原主机的完整证据检查还依赖历史训练事件和源码快照，它们不属于这个精简 Release；组员安装验收请用 `check_installation.py`。

## 已验证范围

已经在单独的 Linux checkout、新建且不继承包的 Python 环境、新资源目录中进行迁移验证，见[校验记录](../resources/portability_validation.json)。历史 JSON 中的路径保持不变，由 `experiments/paths.py` 读取时映射到新目录，不需要手改来源记录。原主机仍兼容旧目录，显式设置 `EE6405_STORAGE` 总是优先。

本次未实际运行 WSL2、全新 GPU 环境或完整神经训练；不同硬件也不保证训练结果逐位一致。大资源通过本地原始文件的 SHA-256 校验后复制，并非全部重新联网下载。Release v1 覆盖原始八个正式模型，不包含后续新增实验、优化器状态、环境和历史日志。原来的约 17 GB 资源目录继续保留，不整体上传 Git。
