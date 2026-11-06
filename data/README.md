# 数据、模型与运行资源

Git 仓库保存代码、配置、结果、无正文的逐条预测和来源记录。原始数据、模型权重、环境、缓存及第三方 RACE 源码保存在仓库之外。仅克隆仓库不会自动下载这些资源，也不能直接启动已训练模型的 Demo。

## 仅用仓库可以做什么

从仓库根目录运行：

```bash
python3 scripts/check_results.py
```

此入口仅使用 Python 标准库，从已保存预测检查结果一致性，不加载模型、不训练，也不验证原始标签或模型是否产生了这些预测。完整 checkpoint 重载与数据哈希验收属于已有本机环境的检查，见 [experiments/README.md](../experiments/README.md)。

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
| `experiments/neural_runs/<run>/` | 七个正式神经模型的 checkpoint、训练事件及源码快照 |
| `.venv/`、`.venv-rst/` | 训练／推理与独立 RST 环境 |

正式神经运行名称为 `roberta_s3407`、`roberta_s42`、`roberta_s2026`、`race_s3407`、`race_s42`、`race_s2026`、`race_no_supcon_s3407`。作者提供的已微调 checkpoint 只用于早期 smoke，不能替代这些正式实验的 checkpoint。

## 在另一台机器复跑前

当前训练代码、配置和结果记录含上述绝对路径；环境通过 overlay 复用了本机已有 PyTorch。`requirements-core.txt` 与 `requirements-rst.txt` 记录已使用的依赖，**全新系统从零安装与完整迁移尚未验证**。

需要从上述来源取得固定版本资源，或由小组按适用条款传递现有资源，再核对清单和结果中的 SHA-256。仅下载上游基础模型不会得到本项目训练后的模型。Demo 至少需要基础编码器、正式 checkpoint、固定 RACE 源码及运行环境；RACE 原始文本推理还需要 RST 资源和独立环境。完整实验验收还需要冻结数据、训练事件与源码快照。

若尚无官方源码副本，可在仓库根目录单独获取固定版本：

```bash
git clone https://github.com/ICTMCG/RACE.git local_validation/upstream/RACE
git -C local_validation/upstream/RACE checkout 61ce528430a8bb84fd1bfb3285ee34827182ef59
```

这一步仅取得源码，不下载数据、基础模型、RST 权重或本项目训练后的 checkpoint。已经存在的官方 checkout 应先核对版本，不覆盖其中的本地修改。

迁移时需处理代码／配置中的存储路径，并验证原始协议与 checkpoint，不要为适配新机器覆盖历史结果中的来源记录。取得全部资源并验证迁移前，不将仓库描述为“一键全新安装”。
