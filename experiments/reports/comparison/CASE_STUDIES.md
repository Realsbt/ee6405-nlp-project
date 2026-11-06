# 可追溯的错误与互补案例

状态：completed。案例来自种子 3407 的固定测试结果。

选择规则属于事后说明：在三个最多的 RACE 错误标签对中各取 item ID 排序第一条，再取 RACE 分对且 RoBERTa 分错的第一条。它们用于理解，不是随机代表性样本或新的准确率估计，不参与选模、调阈值、改划分。全文可按 case_studies.json 中的原文件和行号核查。

## 1. AI 到 AI 的改写链被判为人工内容润色

样本：`arxiv-2101.03103v1_ai_generated_llama-3.3-70b-instruct_THEN_gemini-1.5-pro-002`。

发布真值 `ai_generated`；词汇窗口 479 tokens，覆盖完整正文；坐标覆盖干净。坐标干净不证明篇章关系语义正确。

| 模型 | 预测标签 | 对发布真值的分数 |
|---|---|---:|
| tfidf_lr_v1 | human_written | 1.19% |
| roberta_s3407 | human_ai_polished | 0.50% |
| race_s3407 | human_ai_polished | 0.23% |
| race_no_supcon_s3407 | human_ai_polished | 1.06% |

文本讨论 few-shot prompt-tuning，并保留学术论述和引文的表达形式。发布标签把该 AI→AI 改写链归为 ai_generated；RoBERTa 和 RACE 均高分选择 human_ai_polished。该例正文完整、坐标覆盖干净，因此不能简单归因于 512-token 截断或已知跨度漏字。保留的论述形式可能与训练中的润色样本相似，这是待验证的解释候选；高输出分数不等于真实创作流程的可靠概率。

## 2. 改写 AI 文本仍被判为 AI 原生文本

样本：`arxiv-2006.11038v2_ai_humanized_tool`。

发布真值 `ai_humanized`；词汇窗口 512 tokens，已截断；坐标覆盖干净。坐标干净不证明篇章关系语义正确。

| 模型 | 预测标签 | 对发布真值的分数 |
|---|---|---:|
| tfidf_lr_v1 | ai_humanized | 63.70% |
| roberta_s3407 | ai_generated | 6.07% |
| race_s3407 | ai_generated | 10.77% |
| race_no_supcon_s3407 | ai_generated | 0.13% |

文本讨论 Fokker–Planck 方程的多领域应用，发布标签为工具改写 AI。TF-IDF 在该例正确，两个神经模型均选择 ai_generated，说明总体较强的模型也存在互补错误。词汇窗口被截断，因此只能说明本窗口与所用结构信息下的失败；不能据此判断完整正文有没有编辑线索，或把所有工具改写视为可检测。

## 3. RACE 与 RoBERTa 的一次反向分歧

样本：`arxiv-2006.04959v1_human_ai_polished_gpt-3.5-turbo`。

发布真值 `human_ai_polished`；词汇窗口 512 tokens，已截断；坐标覆盖干净。坐标干净不证明篇章关系语义正确。

| 模型 | 预测标签 | 对发布真值的分数 |
|---|---|---:|
| tfidf_lr_v1 | human_written | 4.22% |
| roberta_s3407 | human_ai_polished | 85.32% |
| race_s3407 | ai_generated | 40.64% |
| race_no_supcon_s3407 | human_ai_polished | 76.75% |

文本讨论社交网络对新闻传播与信任机制的影响。RoBERTa 正确选择 human_ai_polished，RACE 更偏向 ai_generated，其两类分数相对接近。RACE 使用更多结构信息仍可能出错；模型容量、pooling、损失和训练随机性也不同，单个分歧不足以证明某条篇章关系导致错误。

## 4. RACE 纠正 RoBERTa 的一次错误

样本：`arxiv-2008.10522v2_ai_humanized_tool`。

发布真值 `ai_humanized`；词汇窗口 420 tokens，覆盖完整正文；坐标覆盖存在问题。坐标干净不证明篇章关系语义正确。

| 模型 | 预测标签 | 对发布真值的分数 |
|---|---|---:|
| tfidf_lr_v1 | ai_generated | 5.85% |
| roberta_s3407 | ai_generated | 8.10% |
| race_s3407 | ai_humanized | 99.79% |
| race_no_supcon_s3407 | human_ai_polished | 12.50% |

文本讨论非人类解释者与人工智能语境下的符号学。该工具改写 AI 样本中，RACE 的标签正确而 RoBERTa、TF-IDF 选择 ai_generated。它展示当前模型的互补性；单例不能把成功归因于图结构机制，也不能抵消整体成对区间包含零的事实。

这些分数未经概率校准。真值来自数据集的操作标签，案例阅读不重新判定作者身份。关于可能原因的文字是解释候选；验证原因需要新设计的控制实验。
