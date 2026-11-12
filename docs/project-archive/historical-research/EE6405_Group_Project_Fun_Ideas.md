# EE6405 Group Project
## 1. Prompt Injection 密室逃脱

### 题目示例

**PromptShield: An Interactive Prompt-Injection Detection Game**

### 核心玩法

用户扮演攻击者，尝试通过不同指令骗过 AI，例如让它忽略系统指令、泄露秘密口令或执行被禁止的任务。防御模型负责识别并拦截攻击。

前端可以设计成关卡制游戏：

- 每关提供一个需要保护的系统提示或秘密
- 用户输入攻击 prompt
- 防御模型实时给出风险分数
- 攻击成功则进入下一关
- 展示触发检测的文本片段
- 提供用户与不同模型的对战排行榜

### 模型比较

- 关键词与正则规则
- TF-IDF + Logistic Regression/SVM
- DistilBERT
- DeBERTa prompt-injection detector
- 现有 guardrail 方法

### 实验设计

- 直接指令攻击
- 角色扮演攻击
- 编码或字符混淆攻击
- 多语言攻击
- 未见过的攻击类型
- 正常讨论 prompt injection 的文本，测试模型是否误杀

### 评价指标

- Accuracy
- Macro-F1
- Attack Recall
- False Positive Rate
- 不同攻击类型的成功率
- 推理时间

### 数据与参考

- [PINT Benchmark](https://github.com/lakeraai/pint-benchmark)

### 综合评价

趣味性和新颖度最高，也能结合 RAG、LLM 和 Agentic AI 等课程内容。主要风险是范围容易过大，建议把核心任务限制为攻击检测，不需要实现完整的大模型安全平台。

---

## 2. Scam Detective 诈骗侦探游戏

### 题目示例

**Scam Detective: Cross-Channel Scam Message Detection and Explanation**

### 核心玩法

系统逐条展示 Email、SMS 或即时通信消息。用户先判断是否诈骗，再由模型公布结果、风险分数和可疑证据。

可设置以下类别：

- Legitimate
- Spam
- Phishing
- Fake job
- Financial scam
- Romance scam

### 模型比较

- Naive Bayes
- TF-IDF + Logistic Regression/SVM
- BiLSTM
- DistilBERT
- RoBERTa

### 实验设计

- 同一来源内训练和测试
- Email 训练、SMS 测试等跨渠道实验
- 检查紧迫语气、索要 OTP、可疑链接等语言特征
- 分析模型是否过度依赖 URL、电话号码或特定关键词
- 比较移除这些表面特征后的表现

### 前端设计

- 模拟收件箱
- 用户与 AI 判断比赛
- 高亮可疑词句
- 展示不同模型的判断分歧
- 给出安全建议，但不自动点击或访问可疑链接

### 评价指标

- Precision、Recall、F1
- Scam Recall
- False Alarm Rate
- Cross-domain F1
- Confusion Matrix
- 推理速度

### 数据与参考

- [Fraud and scam research datasets](https://github.com/RockENZO/data)

### 综合评价

实用性强、游戏感明显，数据和 baseline 相对容易获得。最终选择数据集时需要检查来源、隐私处理和许可证。

---

## 3. Clickbait Killer 标题党终结者

### 题目示例

**Clickbait Killer: Detection, Explanation and Automatic Spoiler Generation**

### 核心玩法

用户输入新闻标题，系统判断标题党程度，并直接给出标题刻意隐藏的关键信息。

示例：

> You won't believe what happened next...

系统输出：

> The airline introduced a new baggage fee.

该方向可以结合两个 NLP 任务：

1. Clickbait detection 或强度回归
2. Clickbait spoiler extraction/generation

### 模型比较

- TF-IDF + Logistic Regression/SVM
- BiLSTM
- BERT/DistilBERT
- Extractive QA
- 生成式摘要模型

### 实验设计

- 二分类与 clickbait 强度预测
- 短语型、段落型和多段型 spoiler
- 只使用标题与结合文章正文的性能差异
- 不同新闻领域的泛化能力
- 检查生成结果是否真的回答标题问题

### 前端设计

- 模拟新闻网站
- 自动标记标题党标题
- 一键显示 spoiler
- 将标题改写为信息完整的普通标题
- 展示 clickbait 强度和模型解释

### 评价指标

- Classification F1
- Mean Squared Error 或相关系数
- BLEU/ROUGE/BERTScore
- Answer relevance
- 人工可读性评价

### 数据与参考

- [Webis datasets](https://webis.de/data.html)
- [Clickbait Spoiling paper](https://arxiv.org/abs/2203.10282)

### 综合评价

产品感很强，同时包含分类和文本生成，理论与实验深度都比普通文本分类更丰富。

---

## 4. Toxicity Makeover 高情商改写器

### 题目示例

**Toxicity Makeover: Polite Rewriting with Meaning Preservation**

### 核心玩法

把攻击性、粗鲁或带有敌意的文本改写成礼貌表达，同时尽量保持原意。

示例：

> This idea is completely stupid.

改写为：

> I don't think this approach addresses the main issue yet.

### 模型比较

- 规则替换 baseline
- Retrieval-based rewriting
- T5
- BART
- Prompt-based LLM
- 不同 decoding 参数

### 实验设计

- 不同毒性等级的改写
- 英文与多语言改写
- 礼貌程度与语义保留之间的权衡
- 模型是否删除了关键信息
- 对讽刺、隐性攻击和直接辱骂分别评估

### 前端设计

- 毒性温度计
- 语气强度滑块
- 改写前后并排比较
- 高亮被修改的位置
- 提供多个候选改写

### 评价指标

- Toxicity reduction
- Semantic similarity
- Fluency
- BLEU/ROUGE/BERTScore
- 人工评价

### 数据与参考

- [TextDetox and ParaDetox](https://textdetox.github.io/)

### 综合评价

同时包含文本分类和生成，展示效果好。项目展示时需要妥善处理冒犯性样本，避免使用不必要的极端文本。

---

## 5. Meme Court 梗图审判庭

### 题目示例

**Meme Court: Multimodal Humor, Sarcasm and Offensiveness Analysis**

### 核心玩法

用户上传 Meme，系统像法庭一样展示证据，并判断它是否属于：

- Funny
- Sarcastic
- Offensive
- Motivational
- Harmful but disguised as humor

### 模型比较

- 只使用 OCR 文本
- 只使用图片特征
- BERT 文本模型
- 图像模型
- 文本与图片特征融合模型

### 实验设计

- 比较文字、图片和多模态模型
- 分析只有文字时无法识别的 Meme
- 分析讽刺和冒犯类别之间的混淆
- 检查模型是否依赖模板、字体或图片风格

### 前端设计

- 上传 Meme
- 展示 OCR 结果
- 显示多个模型的投票
- 给出最终“判决”和置信度
- 高亮影响判断的文本或图片区域

### 数据与参考

- [Memotion Analysis paper](https://arxiv.org/abs/2008.03781)

### 综合评价

视觉效果最好，但它属于 NLP 与 Computer Vision 的多模态项目，工作量和算力需求较高。开始前应向老师确认是否接受多模态方向。

---

## 6. Story Detective 故事结局侦探

### 题目示例

**Story Detective: Human versus AI Story-Ending Reasoning**

### 核心玩法

系统给出一段四句小故事，让用户和模型分别选择最合理的结局，然后解释另一个结局为什么不合理。

可以设计以下结局类型：

- 正常结局
- 喜剧结局
- 因果关系错误
- 人物行为不一致
- 与前文信息冲突

### 模型比较

- N-gram Language Model
- TF-IDF similarity
- FastText
- LSTM
- BERT/RoBERTa

### 实验设计

- 不同故事主题上的表现
- 正确结局与困难负样本比较
- 人类与模型准确率比较
- 分析模型是否真正理解因果关系
- 检查模型是否只依赖结局句子的表面特征

### 前端设计

- 互动小说界面
- 用户选择结局
- 用户与 AI 积分排行
- 展示模型解释
- 自动生成额外错误结局作为挑战

### 评价指标

- Accuracy
- Macro-F1
- 不同错误类型的准确率
- Explanation relevance
- 人类与模型胜率

### 数据与参考

- [ROCStories and Story Cloze Test](https://cs.rochester.edu/nlp/rocstories/)

### 综合评价

课堂展示很有互动性，但报告中必须有严格的分类和错误分析，不能只展示自动生成故事。

---

## 综合比较

| 项目 | 趣味性 | 实现难度 | 实验深度 | 展示效果 | 主要风险 |
|---|---:|---:|---:|---:|---|
| Prompt Injection 密室逃脱 | 5/5 | 3/5 | 5/5 | 5/5 | 攻击类型过多导致范围失控 |
| Scam Detective | 5/5 | 2/5 | 4/5 | 5/5 | 数据来源和隐私需要检查 |
| Clickbait Killer | 5/5 | 3/5 | 5/5 | 5/5 | 同时做检测和生成可能超出时间 |
| Toxicity Makeover | 4/5 | 3/5 | 5/5 | 4/5 | 语义保留不容易自动评价 |
| Meme Court | 5/5 | 5/5 | 5/5 | 5/5 | 多模态工作量与算力较大 |
| Story Detective | 5/5 | 3/5 | 4/5 | 5/5 | 需要证明模型不是利用表面线索 |

## 推荐顺序

### 第一推荐：Prompt Injection 密室逃脱

它比较新颖，互动性强，与 LLM、RAG 和 Agentic AI 相关，也可以完整比较规则方法、传统机器学习和 Transformer。

建议控制范围：

- 核心只做 prompt-injection 检测
- 选择 3-4 类攻击
- 比较 3-4 个模型
- 使用 Streamlit 制作 5 个互动关卡

### 第二推荐：Clickbait Killer

检测与 spoiler generation 的组合很有产品感，也容易展示传统模型与 Transformer 的差异。

如果时间有限，可以先完成 clickbait detection 和 extractive spoiler，再把生成式改写作为加分功能。

### 第三推荐：Scam Detective

这是最稳妥、最容易按时完成的趣味项目。它能够自然地加入跨数据集泛化、模型解释和用户对战，不需要依赖昂贵 API。

## 最终选题时需要确认

- 数据集是否允许课程项目使用和重新分发
- 是否能够准备至少两个算法和一个传统 baseline
- 是否有足够的数据支持训练、验证和独立测试
- 是否能在普通电脑或免费 GPU 上完成实验
- 前端是否服务于模型演示，而不是占用过多开发时间
- 项目是否能在 10 分钟视频内讲清问题、方法、实验、结果和贡献
- 所有参考代码和 AI 使用情况是否能够在 GAI declaration 中如实说明
