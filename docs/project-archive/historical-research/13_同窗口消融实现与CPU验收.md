# 同窗口消融实现与CPU验收

日期：2026-11-09，最终验收更新至2026-11-11。承接12号文档。四条件代码、全量同窗口缓存及CPU反向验收已完成；正式GPU神经训练和新消融效果评测尚未执行。原仓库与上游RACE源码未修改。

## 1. 原本的问题与本轮贡献

### 贡献一：把公平信息预算落到实际输入

- 原问题：旧RACE用全文结构，文本主干只看前512 tokens；差异混合了信息范围、聚合和网络容量。
- 为什么做：不固定可见正文，就无法解释结构分支是否必要。
- 做了什么：`same_window_cache.py`先为15,984条样本冻结可见正文，再仅解析该正文，包含特殊符号总长不超过512 tokens。来源划分沿用原冻结协议；检查各组模型token一致、字符覆盖、EDU spans和隐藏节点。
- 结果：可见输入和图缓存全部完成，train/dev/test分别11,190/1,600/3,194；共15,984条，失败0、待处理0。来源块无交集，跨分区完全相同可见正文为0，隐藏EDU为0。v3 manifest的`full_cache_ready=true`。
- 结论：形成了可执行的公平窗口数据管线。测试正文只做固定预处理，没有评模型或用于选参。
- 限制：精确重复为0不等于语义近重复为0；偏移恢复通过不等于parser理解了特殊字符语义。完整状态以`resources/experiments/same_window_full_v3/manifest.json`为准。v1/v2源码、失败历史、迁移哈希均保留，不以新结果覆盖旧证据。

### 贡献二：实现四组机制控制并做反向验收

| 条件 | 实际实现 | 总参数 | 可训练参数 |
|---|---|---:|---:|
| A `text` | 原RoBERTa CLS及原MLP | 125,239,300 | 7,681,540 |
| B `graph` | 同可见正文上的原RACE | 128,617,148 | 11,059,388 |
| C `shuffled` | B架构；逐图打乱关系类型的节点分配 | 128,617,148 | 11,059,388 |
| D `edu` | EDU内token均值，再对EDU等权均值；768→512投影及512宽MLP；无GNN | 125,305,092 | 7,747,332 |

- 原问题：只有消融计划，没有可运行的EDU基线或关系控制。
- 为什么做：分别检验总体结构模型收益、关系类型分配、切分聚合与消息传递的竞争解释。
- 做了什么：新增`window_controls.py`及`train_window_controls.py`，复用原模型、图构建器、采样器和指标，不另写一套RACE。从共同基础权重初始化，不使用旧任务最佳checkpoint热启动。
- 结果：四组每组使用4条训练、4条开发样本，完成1个训练batch及开发前向；有限梯度、分类头与最后一层编码器更新、冻结embedding未更新全部通过。四组训练token指纹完全相同，B/C完整初始state_dict指纹相同。
- 结论：输入、初始化和反向路径可用。**这些小样本开发分数没有模型效果解释价值，不放入论文结果表。**
- 限制：CPU验证不能证明CUDA训练、显存预算、正式checkpoint恢复或完整训练收敛。D与B匹配分类头形状，不匹配总容量；B-D不是严格单一机制因果对照。A沿用原CLS头，读出方式也不同。
- 证据：`window_controls_smoke_v1/summary.json`与`window_controls_indexed_smoke_v2/summary.json`。前者用32条原始解析缓存，后者从新建v2索引缓存读取，四组均通过。训练入口补存开发集概率后，另以`window_controls_indexed_smoke_v3/`复验当前代码；各条件均保存配置、代码快照和事件日志。

### 贡献三：证明关系控制真的改变了图类型

- 原问题：全局关系重命名可被模型重新学习，不能用于检验关系信息；随机置换也可能碰巧没有改变图。
- 为什么做：需要可复核、与标签无关的破坏方式，并量化破坏强度。
- 做了什么：固定控制种子20261109与item_id的SHA256决定逐图关系分配置换，所有训练种子共用同一控制图。保留每图关系频数、节点、EDU和拓扑；调用原构图器重新生成依赖关系类型的边。
- 结果：全量15,984张图共有638,038个关系节点，其中467,369个改变分配，比例73.25%；2,552,152条有向边中1,869,476条类型改变，比例同为73.25%。拓扑全部一致；36张图置换后未变化，占0.225%，其中17张只有0或1种关系。见`window_controls_full_audit_v3.json`。
- 结论：不是“改了关系名字但实际边类型没变”的空消融。
- 限制：它保留拓扑与关系频数，检验的是关系类型的正确位置分配，而非全部篇章结构价值。全量报告仍需包含单一关系图及不变图比例。

### 贡献四：找到全量解析的新失败原因并验证修复

- 原问题：NFKC兼容映射仍不能处理部分训练文本，不能静默删掉这些样本。
- 为什么做：解析失败可能与学术域、数学符号等内容相关，直接删除会引入选择偏差。
- 做了什么：`diagnose_window_alignment.py`截获原始EDU字符串并保存差异；发现三条训练样本中的U+225C、U+0202、U+2291被parser tokenizer转换成`<unk>`。这不是空白或Unicode等价差异。
- 修复方法：独立`repair_window_unknowns.py`重放parser实际的razdel及fast tokenizer步骤，要求原EDU文本与实际token pieces完全对应，再用offset_mapping映射回原字符。拒绝切开token、跨EDU重叠或非空白遗漏；不把`<unk>`当任意字符通配符，也不改变RoBERTa输入。
- 结果：三条训练样本全部通过，涉及2/2/3个unknown tokens；完整字符覆盖、模型token一致、EDU可见性及官方图预处理检查均通过。
- 结论：全量过程中18条曾失败；3条parser `<unk>` 已通过真实token offsets恢复，余下15条经字符差异审计后恢复。15条中13条只涉及parser删除`U+200B/U+FEFF/U+200D`或把字面`U+2581`当控制标记，2条为完整NFKC等价的跨字符组合。原文和RoBERTa输入不变，修复只恢复RST偏移。
- 限制：`U+2581`属于parser控制标记冲突，不是普通不可见字符；已单独列入审计白名单。适配拒绝未批准字符删除、内容变化、非单调映射及切开兼容展开的EDU边界。恢复跨度不证明parser利用了被忽略字符的语义。
- 证据：`window_alignment_failures_v3.json`、`window_alignment_resolution_v3.json`、`parse_nfkc_cpu.py`、`resources/experiments/same_window_full_v3/promotion.json`及manifest。

## 2. 应有的实验架构

```text
带creator/editor来源记录的原文
  -> 冻结来源组划分、领域和四分类标签
  -> 同一RoBERTa tokenizer确定可见正文（<=512 tokens）
       -> A: RoBERTa CLS -> MLP
       -> 只解析该可见正文 -> 审计后的EDU/RST缓存
            -> B: RoBERTa -> RACE -> root -> MLP
            -> C: 同B，关系类型分配逐图扰乱 -> MLP
            -> D: RoBERTa -> EDU均值 -> 等权EDU聚合 -> 投影 -> MLP
  -> 相同CE、采样预算、最后一层微调和开发集选模规则
  -> 冻结best checkpoint -> 测试逐样本概率
  -> Macro-F1 / 每类表现 / creator与editor / 校准 / 低误报
  -> 来源组配对区间 + 三种子差值 + 外部迁移 + 成本
```

这是研究比较架构，不要求最终部署同时保留四组。工程应用仍以简单强文本基线为起点；增加parser/GNN需要证明收益足以覆盖成本。

主要比较预设为B-C、B-D；B-A回答总体效果。若B≈C，不支持正确的关系类型位置带来稳定收益；若B≈D，不支持现有GNN相对该聚合对照有额外收益。二者都不能证明所有结构方法无用。若有增益，也需结合容量/读出残余差异和跨种子区间判断。

这里固定CE的A是机制对照，不是宣布性能基线只剩CE。最终性能表仍应保留TF-IDF及原RoBERTa+SupCon强基线；机制消融与“超过现有最好方法”是两种不同问题，不能用击败较弱CE替代后者。

## 3. 训练实现与GPU交接

新入口固定CE、均衡四分类batch、原学习率与早停，默认float32，避免旧入口写死BF16的兼容限制。新旧精度不同，因此不与历史BF16表直接拼接解释机制。

正式训练具备代码快照、配置/缓存/基础权重哈希、每epoch开发指标、best/last checkpoint、来源与概率记录。最佳checkpoint选定后保存开发集与测试集逐样本概率，开发概率供后续CPU校准和锁定阈值；主训练不在测试上拟合阈值。`--resume`校验配置和last checkpoint哈希，恢复优化器、scheduler和随机状态。恢复逻辑针对完整epoch，没有实现batch中途恢复；正式保存/恢复以及评测中断后的恢复路径尚未在GPU验证。已完成run禁止再次resume。

顺序如下：

1. CPU全量解析、失败清零、缓存完备、无隐藏EDU和全量控制审计均已完成。
2. GPU先运行单条件短训练，检查CUDA、显存、checkpoint保存/恢复和数值；不要拿测试集调显存配置。
3. 确认同一batch配置适用于四条件；若需调整，统一修改并重新冻结运行配置。
4. 四条件×3407/42/2026，共12次正式CE训练。各条件保留同样batch预算；显存不足需统一修改配置并重新冻结。
5. 统一分析预设比较；之后再决定SupCon、全文分块或留域训练。不要直接把24次留域队列全部启动。

当前CPU复核命令（在本研究目录执行）：

```bash
.venv-cpu/bin/python -B -m unittest discover -p 'test_*.py'
.venv-cpu/bin/python -B train_window_controls.py --mode smoke --condition all --cache resources/experiments/same_window_full_v3 --output window_controls_full_smoke_v4 --device cpu
.venv-cpu/bin/python -B audit_window_controls.py --cache resources/experiments/same_window_full_v3 --output window_controls_full_audit_v4.json
```

v3已经冻结完成，不应继续追加或重跑。若修改parser、适配器或缓存代码，应建立v4并显式迁移，不得改写v3 protocol/manifest。历史`failures.jsonl`是审计日志，manifest中的`failed=0`表示所有失败ID后来都有验证成功记录。

完整缓存门槛已通过。单次正式GPU命令形式：

```bash
.venv-cpu/bin/python -B train_window_controls.py --mode train --condition graph --cache resources/experiments/same_window_full_v3 --output window_graph_s3407 --device cuda --seed 3407
```

依次替换条件`text/graph/shuffled/edu`和种子3407/42/2026，共12次。云端需要同样依赖及完整本地资源目录，不包含自动下载、登录或上传步骤；还需为多组best/last权重及优化器状态预留磁盘。

## 4. 论文现在可以写什么

可以写：公平窗口控制的设计和实现、输入一致性验收、解析故障与显式修复、待完成实验及已知残余混杂。CPU smoke是工程验证，不是算法性能贡献。

还不能写：正式同窗口RACE优于文本基线、关系结构已证实有用/无用、三种子消融已完成、外部失效已解决。更强算法仍需以开发数据选型、独立评测与可重复收益证明，不能由代码复杂度推断。

## 5. 验证记录

研究目录55项测试通过，原仓库15项测试通过，共70项。新增测试覆盖确定性/标签独立关系扰乱、EDU聚合梯度、unknown及投影偏移、未批准删除拒绝、图缓存Tensor序列化/索引读取、部分缓存训练拦截、哈希篡改拦截和预测顺序/输出保护。测试通过不等于CUDA路径已验证。

四条件从完整v3缓存读取后的CPU反向smoke均通过，见`window_controls_full_smoke_v3/summary.json`。完整缓存、索引和控制图审计通过；正式运行前仍需在目标GPU环境验证CUDA保存/恢复与显存，不能只复制训练脚本。

原仓库与上游RACE Git工作区保持干净；没有使用GPU、提交、推送、登录云账号或上传数据。当前新增神经运行只有train/dev小样本smoke，没有新增正式测试集效果结果。
