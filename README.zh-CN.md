# EE6405：A4000 训练与模型改进

[English](README.md) · [环境部署](docs/REPRODUCING.zh-CN.md) · [原项目方法](experiments/METHOD_NOTES.md) · [历史结果](experiments/reports/comparison/REPORT.md)

本仓库用于 HART 英文文本来源四分类。固定 train/validation/test 为 11,190/1,600/3,194，输入最多 512 token，按 validation Macro-F1 选模。当前改进实现了多层解冻、分层学习率、SupCon 投影头、内容来源/后处理辅助头，以及训练收尾后的 Hyperstack 自动休眠。**实现与测试通过不等于提分；最终指标以实际批次报告为准。**

## 1. 加载现有 VM 环境

```bash
cd /home/ubuntu/6405
source ./env-a4000.sh
cd ee6405-a4000-codex
```

换机器需先按部署指南安装 Python 3.11、CUDA/BF16 环境，下载 `full --skip-checkpoints` 资源、恢复冻结数据。`EE6405_STORAGE` 必须指向仓库外的持久磁盘，`EE6405_UPSTREAM` 指向资源目录内固定版本的 RACE 源码。

## 2. 配置自动休眠（启动正式队列的前置条件）

从 Hyperstack 控制台取得 API key 和当前 VM 的数字 ID。将 key 通过安全 secret 注入为仓库外的只读/私有文件，权限 **600**，不要把内容写入 JSON、Git、命令参数或日志。

```bash
export HYPERSTACK_API_KEY_FILE=/你的安全挂载目录/hyperstack_api_key
export HYPERSTACK_VM_ID=你的数字VM_ID
"$EE6405_PYTHON" -m experiments.hyperstack --preflight
```

预检只读，检查鉴权、ACTIVE 状态，以及 API 返回的 VM 与本机 cloud UUID/hostname 是否一致；**不会执行休眠**。不匹配时停止，不能跳过身份检查或选择名称相似的机器。能读取 VM 不必然意味着角色有休眠权限：API key 对应角色还必须允许 Hibernate，该权限需在控制台确认。

## 3. 一条命令启动完整研究队列

使用新批次名；不覆盖已有运行。代码改动须先提交到自己的分支，然后在 tmux 中启动：

```bash
tmux new -s a4000-research
# 在 tmux 内重新加载第 1、2 节环境变量，然后：
bash experiments/run_research.sh \
  --name a4000_r01 \
  --config experiments/configs/a4000_research.json \
  --output "$EE6405_STORAGE/experiments/research_batches/a4000_r01" \
  --grace-seconds 600 --max-hours 8
```

先按 `Ctrl+B` 再按 `D` 离开，训练继续。重连：`tmux attach -t a4000-research`。队列只允许一个实例；运行期间不要在同一 VM 启动其他独立训练，因为 watchdog 会休眠整台 VM。

执行顺序：

1. 每项先 GPU smoke，再跑 seed 3407：解冻 1 → 2 → 4 → 6 → 12 层；12 层用真实 batch 8 和 gradient checkpointing；6 层 batch 16 OOM 时保留失败记录，另开 batch 8 运行。
2. 对验证集最佳深度比较投影头；仅 validation 改善时保留。
3. 在同一选定主干上比较内容头、后处理头、双头（权重各 0.1），不会在三个消融间偷偷更换主干。
4. 选择 validation Macro-F1 最高的配置，补齐 seed 42 和 2026；固定三种子等权概率集成。
5. 冻结含数据、源码、配置和 checkpoint 哈希的 final manifest，独立运行一次最终 test evaluator。
6. 检查权重与恢复点、评估输出、Markdown 报告、学习曲线、逐类 F1、混淆矩阵及**初始 Git 提交结果的对比图**，核验产物哈希后请求休眠。

只训练你指定的一套配置、跳过结构搜索时，在队列命令中加 `--single-config`；仍自动完成三个种子、最终评估、报告与休眠。该模式读取你的深度/batch/投影/辅助头设置，不覆盖为搜索网格。

此队列是有限的第一轮研究，不执行交接文档中所有 LR/速度搜索组合。胜者即使没有超过基线，也如实交付负结果，不扩大搜索来追逐 test 分数。

## 4. 如何改参数

复制配置，不修改历史冻结配置：

```bash
cp experiments/configs/a4000_research.json experiments/configs/my_research.json
# 用编辑器修改 JSON，保存并提交；启动时传 --config experiments/configs/my_research.json
```

| 参数 | 默认 | 何时修改 / 边界 |
| --- | --- | --- |
| `learning_rate` | 1.5e-5 | 编码器顶层 LR；后续只做有限验证集扫参 |
| `head_learning_rate` | 3e-5 | 分类/投影/辅助头 LR |
| `layer_lr_decay` | 0.85 | 向下一层逐层相乘，范围 (0,1] |
| `architecture.unfreeze_bert_layers` | 1 | 单次训练支持 1/2/4/6/12；完整队列自动遍历，覆盖此值 |
| `batch_size` | 16 | SupCon 必须 ≥8、可被4整除；队列覆盖为 16，12层为8 |
| `gradient_checkpointing` | false | 显存不足时开启；队列对12层自动开启 |
| `gradient_accumulation_steps` | 1 | 扩大更新 batch，**不扩大 SupCon 真实对比集合** |
| `max_epochs` / `early_stopping_patience` | 20 / 5 | 正式对照保持一致；提前停止只看 validation |
| `supcon_temperature` / `supcon_weight` | 0.07 / 1.0 | 控制对比目标；投影头消融时先不同时修改 |
| `supcon_projection` | false | 单次训练可开；完整队列在深度选择后自动消融 |
| `aux_content_weight` / `aux_edit_weight` | 0 / 0 | 0关闭对应头；队列自动做0.1的三种消融 |
| `dataloader_num_workers` | 4 | CPU/内存紧张时降为0；persistent_workers 只在 workers>0启用 |
| `pin_memory` / `persistent_workers` | true / true | DataLoader 性能开关 |
| `fused_adamw` | true | 融合 AdamW 开关 |
| `attention_implementation` | sdpa | 支持 sdpa/eager，速度对照须保持其他参数不变 |
| `torch_compile` | false | 可开，但包含首次编译成本；非默认提分方法 |
| `unfreeze_embeddings` | false | 仅12层允许；开启需单列方法 |
| `embedding_learning_rate` | 2e-6 | 必须 ≤2.5e-6，仅 embeddings 解冻时生效 |
| `seed` / `use_supcon` | 3407 / true | 本研究队列要求此值，自动补齐42/2026；单次入口可覆盖 |
| `weight_decay` / `warmup_ratio` / `gradient_clip` | 0.01 / 0.1 / 1.0 | 第一轮保持固定，避免混杂 |

保持标签、数据协议/哈希、512 token 窗口、validation 选模规则不变。不要修改 `architecture.backbone_model_path` 指向其他模型后仍称作同一 RoBERTa 基线；历史绝对路径由项目路径模块映射到当前资源目录。

**只跑一个实验（无自动休眠，调试入口）：**

```bash
bash experiments/run_neural.sh --config experiments/configs/my_research.json \
  --run-name my_debug_01 --smoke
# 去掉 --smoke 为完整训练；默认只读 train/validation。
# 中断恢复：使用相同 --config、--run-name，另加 --resume。
```

当前节的裸训练命令不会自动装 watchdog。要满足本项目的强制休眠要求，应使用第 3 节队列入口；不要在无人值守时单独启动裸命令。`--evaluate-test` 是显式一次性测试开关，**不得在配置筛选中使用**；正式队列使用冻结 manifest 的独立 evaluator。

## 5. 查看进度和结果

```bash
tail -n 60 /home/ubuntu/6405/PROGRESS.md
cat "$EE6405_STORAGE/experiments/research_batches/a4000_r01/pipeline.json"
cat "$EE6405_STORAGE/experiments/research_batches/a4000_r01/watchdog.json"
# 每个实验的 config、status、events、best.pt、last.pt：
ls "$EE6405_STORAGE/experiments/neural_runs/"
```

批次目录保存配置、逐运行日志、`final_manifest.json` 和 `.sha256`、`final_evaluation/`、`REPORT.md`、五张 PNG 图、`artifacts.json`、`hibernate-request.json`。`initial_git_results.json` 来自固定提交 `877f12bb5b83a4825780f654309ecd9959f74f5a`，不会读取后续修改的历史报告来充当原始基线。

测试报告包含三种子均值 ± sample SD、Accuracy、Macro-AUROC、TPR@1% FPR、逐类 precision/recall/F1、混淆矩阵及与原 RoBERTa+SupCon 的逐种子差异。历史结果是同一固定测试集上的持续比较；硬件/运行环境也有差异，不把配对差异全归因于模型改动。

## 6. 强制休眠与异常处理

- API：`GET /core/virtual-machines/{id}/hibernate?retain_ip=true`；保留公网 IP 便于恢复，IP/磁盘可能继续产生费用。[官方接口](https://docs.hyperstack.cloud/docs/virtual-machines/manage-vm-state/)、[休眠行为](https://docs.hyperstack.cloud/docs/virtual-machines/hibernation/)。
- 正常情况：确认所有必需产物后立即请求休眠。
- 异常情况：独立 watchdog 观察训练 PID；训练停止后最多 **600秒** 给评估/报告收尾，到期即请求休眠，未完成项会明确标记。主进程崩溃也不取消该期限；另有8小时全队列硬上限。
- API 每次请求有超时，失败每5秒重试，不会用 Linux shutdown 或 Stop 冒充休眠。`accepted` 仅代表 API 已接收请求；最终 `HIBERNATED` 状态须在控制台核实。本机遇断网、API 拒绝、watchdog 被杀或宿主故障时，不能绝对保证远端休眠；不能用“无条件保证成功”的措辞掩盖这些限制。
- 休眠是释放 GPU/RAM 的 VM 生命周期操作，不承诺保留进程内存。代码、数据与 checkpoint 必须在持久磁盘。失败后检查 `pipeline.json`、`hibernate-error.json`、日志；不要删除已有批次重跑。完整队列不支持自动续跑旧批次，以避免覆盖正式测试；单个中断实验可用相同源码/配置恢复。

## 研究依据

分层微调参考 [ULMFiT](https://aclanthology.org/P18-1031/)，训练投影空间参考 [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)。两项迁移到本任务的收益仍需实验验证。内容来源与后处理辅助头是本项目的可检验假设：尝试把四分类拆成两个相关因素，观察 ai_humanized 的 precision/recall/F1，不预设其必然有效。

历史三种子 RoBERTa+SupCon：test Macro-F1 **91.15% ± 0.24**。A4000 原始复跑单种子：validation **91.64%**、test **90.65%**。保留原始证据；新研究结果尚待训练，不在 README 填入推测分数。
