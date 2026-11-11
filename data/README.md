# 数据、模型与运行资源

Git 仓库保存代码、配置、结果、无正文的逐条预测和来源记录。原始数据、模型权重、环境、缓存及第三方 RACE 源码保存在仓库之外。仅克隆仓库不会自动下载这些资源，也不能直接启动已训练模型的 Demo。

## 仅用仓库可以做什么

从仓库根目录运行：

```bash
python3 scripts/check_results.py
```

此入口仅使用 Python 标准库，从已保存预测检查结果一致性，不加载模型、不训练，也不验证原始标签或模型是否产生了这些预测。组员安装、下载资源和重载检查见 [Linux / WSL2 复现指南](../docs/REPRODUCING.zh-CN.md)。

## 固定数据与来源

| 资源 | 项目使用版本与用途 | 来源／核对记录 |
|---|---|---|
| HART | 作者发布的英文四分类图数据；清理后 15,984 条，固定分组划分为 11,190／1,600／3,194 | [作者共享目录](https://drive.google.com/drive/folders/1goJ9F5uXhBTabURr9q23KhbPxiYNFK-8)、[data_v1.json](../experiments/reports/data_v1.json) |
| HC3 | revision `4d0ff18143b5a7e1b1e79beb540c04549d1e59d3`；冻结 500 个题目对、1,000 条回答，协议 `hc3_aux_v1_1` | [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)、[hc3_aux_v1_1.json](../experiments/reports/hc3_aux_v1_1.json) |
| RoBERTa-base | revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`；正式神经实验的基础预训练初始化 | [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base)；每次运行的 `result.json` 记录权重哈希 |
| RACE 源码 | commit `61ce528430a8bb84fd1bfb3285ee34827182ef59` | [ICTMCG/RACE](https://github.com/ICTMCG/RACE)；本机位置为 `local_validation/upstream/RACE/` |
| RST parser | `isanlp-rst==3.2.2`；RST-DT revision `5141450ac3eb7decad0b8735fdbf5ec6928ec97a` | [tchewik/isanlp_rst_v3](https://huggingface.co/tchewik/isanlp_rst_v3)、[parser_assets.json](../experiments/reports/parser_assets.json) |

HART 的原始图数据保留不变；同源变体和重复文本关联组不跨分区。HC3 不参与训练或阈值选择，且只有二分类标签。清理、分组与解析对齐过程见[实验说明](../experiments/README.md)。

HC3 数据卡注明 CC-BY-SA-4.0，并保留原始来源中更严格的条款。固定 RACE 源码树没有单独的 LICENSE 文件，尚未确立上游代码及资源的公开再分发权限，因此不将它们打包上传。全部版本、许可说明与解析器配套依赖见 [SOURCES.md](../experiments/SOURCES.md)。

## 当前已验证的本机布局

下表相对于 `/data/botao/ee6405-race-validation/`：

| 路径 | 内容 |
|---|---|
| `assets/hart_official/` | HART 原始图数据及 `download_manifest.json` |
| `assets/hc3/` | HC3 原始数据、数据卡和来源清单 |
| `assets/roberta-base/` | 固定版本的基础编码器、配置和 tokenizer |
| `assets/isanlp_rst_rstdt/`、`assets/isanlp_rst_local/`、`assets/xlm-roberta-large-config/` | RST 权重、适配后的本机配置及 tokenizer/config |
| `experiments/hart_grouped_v1/` | 冻结的训练／验证／测试分区、协议及图缓存 |
| `experiments/hc3_aux_v1_1/` | 冻结辅助数据、协议及已对齐的 RST 输出 |
| `experiments/models/tfidf_lr_v1.joblib` | 正式 TF-IDF 模型 |
| `experiments/neural_runs/<run>/` | 十二个正式神经模型的 checkpoint、训练事件及源码快照 |
| `.venv/`、`.venv-rst/` | 训练／推理与独立 RST 环境 |

正式神经运行是 `roberta_s{seed}`、`roberta_supcon_s{seed}`、`race_no_supcon_s{seed}`、`race_s{seed}`，每组种子为 3407、42、2026，共十二次。作者提供的已微调 checkpoint 只用于早期 smoke，不能替代这些正式实验的 checkpoint。

## 在另一台机器复跑

新的安装入口创建独立 Python 3.11 环境，不依赖原机器的 overlay 环境；用 `EE6405_STORAGE` 指定自己的资源目录。配置和历史结果中的原始绝对路径保持不变，由 `experiments/paths.py` 在读取时映射。不要为迁移修改历史来源记录。

`scripts/fetch_resources.py` 按 [manifest.json](../resources/manifest.json) 下载并检查资源的大小和 SHA-256。上游源码、数据和预训练资源来自原始来源。私有 [resources-v1 Release](https://github.com/Realsbt/ee6405-nlp-project/releases/tag/resources-v1) 提供全部 13 个模型文件。Demo profile 约 4.28 GiB，full profile 约 9.21 GiB，均不含 Python 环境和后续生成数据。

完整步骤和 CPU / GPU 区别见[复现指南](../docs/REPRODUCING.zh-CN.md)。最小下载入口为：

```bash
export EE6405_STORAGE="$HOME/ee6405-resources"
python3 scripts/setup_environment.py --python python3.11 --torch cpu
"$EE6405_STORAGE/.venv/bin/python" scripts/fetch_resources.py --profile demo
```

`scripts/restore_data.py` 可在 full 下载后恢复原始 HART / HC3 分区并逐一核对冻结哈希。HC3 选择清单不包含回答正文。完整历史证据检查所需的训练事件、源码快照和优化器状态未打入这个精简资源包。

四模型 CPU 推理、固定数据重建和图预处理已在单独 Linux checkout、新资源目录及新建环境中验证，详见[迁移验证记录](../resources/portability_validation.json)。WSL2、全新 GPU 安装和完整神经训练不在本次验证范围内。原约 17 GB 资源目录继续保留，不整体上传 Git，也不要求组员复制其中的环境和缓存。
