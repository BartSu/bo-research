# 遗忘数据如何关联知识腐蚀？在 Unlearning 之前预测知识腐蚀

## 动机与类比

### 参考论文

**A Curious Case of Searching for the Correlation between Training Data and Adversarial Robustness of Transformer Textual Models** (Findings of ACL 2024, arXiv:2402.11469) 提出：

> 训练数据如何关联对抗鲁棒性？能否在模型 fine-tune 之前、不生成对抗样本的情况下估计对抗鲁棒性？

作者采用 **data-first** 的方法：从训练语料中提取 4 类共 13 个特征（embedding 分布、标签分布、代理模型可学习性、token 统计），然后用 Random Forest 预测攻击成功率。该框架比传统对抗评估节省 30x–193x 计算量，并且在 BERT、RoBERTa、BART、ELECTRA、GPT-2 之间具有迁移性。

核心洞察：**训练数据本身的属性就能预测下游模型行为（对抗鲁棒性）**，且这种预测可以在 fine-tuning 发生之前完成。

### 我们的类比问题

我们为 LLM unlearning 提出平行的问题：

> **遗忘数据如何关联知识腐蚀？能否在模型 unlearning 之前估计知识腐蚀的严重程度？**

参考论文研究 `训练数据 → 对抗鲁棒性`，我们研究 `遗忘数据 → 无关知识腐蚀`。

| 维度 | 参考论文 | 我们的提案 |
|---|---|---|
| 输入数据 | 训练（fine-tuning）语料 | 遗忘数据（待 unlearn 的数据） |
| 模型操作 | Fine-tuning | Unlearning |
| 目标结果 | 对抗鲁棒性（攻击成功率） | 知识腐蚀（无关知识的退化程度） |
| 预测时机 | Fine-tuning 之前 | Unlearning 之前 |
| 方法路径 | 从数据提取特征 → 预测结果 | 从遗忘数据提取特征 → 预测腐蚀 |

更宏观的研究方向是：**LLM unlearning 会造成无关知识的腐蚀（irrelevant knowledge corruption）**，我们希望在 unlearning 之前，通过分析遗忘数据的特征来理解、预测并最终缓解这种腐蚀。

---

## 第一部分：框架设计

### 1. 整体框架概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pre-Unlearning 腐蚀预测框架                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入:  遗忘数据 F, 保留数据 R, 待 unlearn 模型 M               │
│                                                                 │
│  Stage 1: 特征提取                                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐ │
│  │Embedding │ │知识图谱  │ │Circuit/  │ │模型行为  │ │Token/ │ │
│  │分布特征  │ │关系特征  │ │机制特征  │ │探测特征  │ │表层特征│ │
│  └─────┬────┘ └─────┬────┘ └─────┬────┘ └────┬─────┘ └───┬───┘ │
│        └────────────┴───────────┴────────────┴───────────┘     │
│                              │                                  │
│  Stage 2: 腐蚀预测           ▼                                  │
│  ┌─────────────────────────────────────────────┐               │
│  │    轻量级预测器 (Random Forest / XGBoost)     │               │
│  │    输入: 特征向量    输出: 腐蚀指标预测值      │               │
│  └─────────────────────┬───────────────────────┘               │
│                        │                                        │
│  Stage 3: 决策与干预    ▼                                        │
│  ┌─────────────────────────────────────────────┐               │
│  │  预测腐蚀低 → 直接执行 unlearning             │               │
│  │  预测腐蚀高 → 数据干预后再 unlearning          │               │
│  │    · 分解高风险子集                            │               │
│  │    · 添加 retain anchors                      │               │
│  │    · 改写歧义样本                              │               │
│  │    · 调整 unlearning 超参数                    │               │
│  └─────────────────────────────────────────────┘               │
│                                                                 │
│  输出: 腐蚀预测值 + 干预建议 + (可选) 精炼后的遗忘数据集        │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 遗忘数据特征体系（Feature Taxonomy）

类比参考论文的 4 类 13 个特征，我们提出 5 类 18+ 个特征，覆盖从表层统计到深层机制的不同层次。

#### A. Embedding 分布特征

| 特征 | 定义 | 与腐蚀的预期关系 |
|---|---|---|
| 遗忘-保留 embedding 重叠度 | 遗忘数据 embedding 与保留/通用知识 embedding 的平均 cosine 相似度 | 重叠越高 → 腐蚀风险越大 |
| 遗忘聚类离散度 | 遗忘数据在 embedding 空间中的分散程度 | 越分散 → 影响的知识领域越多样 |
| 最近保留邻居距离分布 | 每个遗忘样本到最近保留样本的距离分布 | 距离越近 → 纠缠风险越高 |
| 遗忘子空间维度 | 遗忘数据 embedding 矩阵的有效秩（effective rank） | 维度越高 → 表征足迹越大 → 腐蚀潜力越高 |
| 共享子空间比例 | 遗忘数据主成分与保留数据主成分重叠的比例（通过 SVD 计算） | 比例越高 → 表征纠缠越严重 |

#### B. 知识图谱 / 关系特征

| 特征 | 定义 | 与腐蚀的预期关系 |
|---|---|---|
| 知识连通度 | 遗忘实体与非遗忘实体之间的知识图谱边数 | 连通度越高 → 腐蚀传播路径越多 |
| 多跳可达性 | 在知识图谱中，从遗忘知识出发 k 跳内可达的保留知识比例 | 可达比例越高 → 遗忘的"爆炸半径"越大 |
| 实体共现频率 | 遗忘实体与非遗忘实体在训练语料或模型生成文本中的共现频率 | 共现越频繁 → 知识关联越紧密 |

#### C. Circuit / 机制特征

| 特征 | 定义 | 与腐蚀的预期关系 |
|---|---|---|
| 遗忘知识的 circuit 深度 | 编码遗忘数据的 circuit 的平均深度（借鉴 CUD [16]） | 越深 → 与其他知识共享组件越多 |
| 遗忘-保留 circuit 重叠度 | 遗忘数据使用的 circuit 边中，同时也被保留数据使用的比例 | 重叠越高 → 修改时波及范围越大 |
| 层集中度 | 遗忘知识主要编码在哪些层。早期层（更共享） vs. 晚期层（更专用） | 集中在早期层 → 腐蚀更广泛 |

#### D. 模型行为探测特征（轻量级 probing）

| 特征 | 定义 | 与腐蚀的预期关系 |
|---|---|---|
| 代理遗忘成功率 | 少量梯度上升步骤（1–5 步）后保留数据性能的下降幅度 | 下降越大 → 实际腐蚀风险越高 |
| 遗忘数据困惑度 | 模型对遗忘数据的 perplexity | 越低（深度记忆）→ 越难干净移除 |
| 梯度对齐度 | 遗忘数据梯度与保留数据梯度的 cosine 相似度 | 对齐度越高 → 遗忘方向与保留方向冲突越大 |

#### E. Token / 表层统计特征

| 特征 | 定义 | 与腐蚀的预期关系 |
|---|---|---|
| 平均遗忘序列长度 | 遗忘样本的平均 token 数 | 辅助控制变量 |
| 遗忘-保留词汇重叠率 | 遗忘集和保留集的词汇交集占比 | 重叠越高 → 表层干扰越大 |
| 命名实体密度 | 遗忘数据中命名实体的密度 | 密度越高 → 结构化知识越多 |
| 主题多样性 | 遗忘数据覆盖的主题数（LDA 或其他主题模型度量） | 主题越多样 → 影响面越广 |

### 3. 轻量级腐蚀预测器设计

#### 核心思路

训练一个预测器：`遗忘数据特征向量 → 腐蚀指标预测值`

类比参考论文用 Random Forest 预测攻击成功率，我们用 Random Forest / XGBoost / 小型 MLP 预测知识腐蚀程度。

#### 预测目标（要预测的腐蚀指标）

| 指标 | 度量方式 | 意义 |
|---|---|---|
| 保留准确率下降 | Unlearning 前后在 MMLU / TriviaQA 等 benchmark 上的准确率差 | 整体模型能力退化 |
| 领域特定知识退化 | 与遗忘目标无关的特定知识领域的性能下降 | 无关领域的定向腐蚀 |
| 知识空洞范围 | 借鉴 Ko et al. [8] 的探测方法度量知识空洞的广度 | 静态 benchmark 无法检测到的隐性损伤 |
| 多跳推理退化 | 非遗忘主题上的多跳推理性能变化 | 深层知识关联的破坏程度 |

#### 训练流程

```
Phase 1: 数据收集
  · 选取 3–5 个模型 (Llama-2-7B, Llama-3-8B, Mistral-7B, Phi-3 等)
  · 使用现有 benchmark (TOFU, WMDP) + 构造受控属性的遗忘集
  · 对每个遗忘集提取上述 18+ 个特征
  · 对每个 (模型, 遗忘集, unlearning 方法) 组合执行 unlearning
  · 在保留集和通用 benchmark 上测量腐蚀指标
  → 产出: (特征向量, 腐蚀指标) 训练对

Phase 2: 预测器训练
  · 按模型或遗忘集划分 train / validation / test
  · 训练 Random Forest、XGBoost、小型 MLP
  · 评估: R², MAE, ranking correlation
  · 使用 SHAP values 分析特征重要性
  · 测试跨 unlearning 方法迁移性
  · 测试跨模型迁移性

Phase 3: 预测导向的干预
  · 用训练好的预测器识别高腐蚀风险的遗忘集
  · 应用特征导向的数据精炼
  · 对比有/无预测干预下的 unlearning 结果
  · 测量相比完整 unlearning + evaluation 节省的计算量
```

#### 计算节省分析

参考论文实现了 30x–193x 的运行时节省。对于 unlearning 场景：

- **不使用预测器**：对每个候选遗忘集，需要完整执行 unlearning + 全面评估 → 昂贵
- **使用预测器**：提取特征（几分钟）+ 预测（毫秒级）→ 筛选出高风险组合 → 仅对这些执行完整 unlearning
- 特别是在比较多个候选遗忘集或 unlearning 配置时，节省最为显著

### 4. 跨方法迁移性研究

#### 核心问题

遗忘数据特征与知识腐蚀之间的关联在不同 unlearning 方法间是否稳定？

#### 假设

- **通用特征**（如 embedding 重叠度、知识连通度）是方法无关的腐蚀预测因子
- **方法特异特征**（如梯度对齐度、circuit 深度）的预测力因方法而异

#### 对比方法

| 类别 | 方法 |
|---|---|
| 梯度方法 | Gradient Ascent, NPO |
| 表征方法 | RMU, LUNAR [24], FALCON [25] |
| 二阶方法 | SOUL [32], Gauss-Newton [33] |
| 数据增强方法 | ReLearn [28] |

如果存在通用特征，则单一预测器即可作为方法无关的腐蚀估计器；如果特征是方法特异的，则预测器需要方法条件分支。

### 5. 从相关性到因果机制的分析

#### 目标

不仅发现 "遗忘数据特征 X 与腐蚀程度 Y 相关"，还要解释 *为什么*。

#### 方法

1. 用 causal tracing 识别编码遗忘知识的 circuit 组件
2. 映射遗忘 circuit 与支持无关知识的 circuit 之间的重叠
3. 证明修改共享组件的 unlearning 方法导致更多腐蚀，而靶向遗忘特异组件的方法腐蚀更少
4. 证明 circuit 共享程度可以从数据层面的特征预测

这将为经验性预测提供因果机制解释，增强整个框架的可解释性。

### 6. 特征导向的 Pre-Unlearning 数据干预

当预测器判断腐蚀风险超过阈值时，执行以下干预：

| 干预类型 | 具体操作 | 依据 |
|---|---|---|
| 分解 | 将遗忘集分解为不同腐蚀风险的子集 | 按 embedding 重叠度和 circuit 重叠度分组 |
| 添加 retain anchors | 为高风险子集添加最近的保留数据样本作为锚点 | 保护距遗忘数据最近的良性知识 |
| 改写 | 重写歧义遗忘样本，减少与保留知识的重叠 | 降低共享子空间比例 |
| 调参 | 基于预测的困难度调整 unlearning 超参数（学习率、步数） | 高腐蚀风险 → 更保守的更新 |

这与 SURF 框架（见 `papers/llm-unlearning-research-gap-table.md`）形成互补，但增加了**基于预测的决策依据**，使数据操作从启发式变为证据驱动。

---

## 第二部分：研究假设

### H1 — 遗忘数据特征可以预测腐蚀
从遗忘数据中提取的特征可以预测无关知识腐蚀的严重程度，预测准确率达到与 TuneAhead 预测 fine-tuning 性能（~89%）相当的水平。

### H2 — Embedding 重叠度是最强预测因子
在所有特征类别中，遗忘-保留 embedding 重叠度（共享子空间比例、最近保留邻居距离）是知识腐蚀最具信息量的预测因子，因为它直接度量了表征纠缠。

### H3 — 预测可以跨 Unlearning 方法迁移
在一种 unlearning 方法（如 gradient ascent）上训练的腐蚀预测器，在另一种方法（如 LUNAR）上仍保持显著的预测能力，因为数据-知识纠缠的基本结构是方法无关的。

### H4 — 腐蚀预测能改善 Unlearning 效果
使用预测的腐蚀指标指导 pre-unlearning 数据精炼（添加 retain anchors、调整遗忘数据），相比盲目 unlearning，能显著减少实际腐蚀。

### H5 — 轻量级探测就足够了
少量代理 unlearning 步骤（1–5 次梯度更新）结合数据特征，比单独使用数据特征能提供显著更好的腐蚀预测，而额外计算成本极低。

### H6 — 知识图谱连通度预测腐蚀范围
从遗忘实体出发 k 跳内可达的非遗忘实体数量预测腐蚀的**广度**（受影响领域数），而 embedding 重叠度预测腐蚀的**深度**（各领域内的严重程度）。

---

## 第三部分：实验设计

### Phase 1: 特征工程与数据收集

1. 选取 3–5 个模型（Llama-2-7B, Llama-3-8B, Mistral-7B, Phi-3）
2. 使用现有 benchmark（TOFU, WMDP）+ 构造受控属性的遗忘集（变化重叠度、连通度、深度）
3. 对每个遗忘集提取全部 18+ 个特征
4. 在每个组合上运行 3–5 种 unlearning 方法
5. 腐蚀度量：
   - MMLU / TriviaQA / 领域特定 benchmark 上的保留准确率
   - Knowledge hole probing（Ko et al. [8]）
   - 非遗忘主题的多跳推理
   - 通用语言建模 perplexity

### Phase 2: 预测器训练与评估

1. 按模型或遗忘集划分 train / validation / test
2. 训练 Random Forest、XGBoost、小型 MLP
3. 评估：
   - 预测准确度（R², MAE, ranking correlation）
   - 特征重要性（SHAP values）
   - 跨 unlearning 方法迁移
   - 跨模型迁移

### Phase 3: 预测导向的干预实验

1. 用训练好的预测器识别高腐蚀风险的遗忘集
2. 应用特征导向的数据精炼
3. 对比有/无预测干预下的 unlearning 结果
4. 测量计算节省（对比完整 unlearning + evaluation 的开销）

### 评估基准

| Benchmark | 用途 |
|---|---|
| TOFU | 基础遗忘 + 效用评估 |
| WMDP | 安全敏感知识移除 |
| MUSE | 多维度评估 |
| BLUR | 真实遗忘-保留重叠 |
| MMLU | 通用知识保留 |
| TriviaQA | 事实知识保留 |
| 自定义领域探测 | 定向腐蚀测量 |

---

## 第四部分：定位与新颖性

### 最强新颖性声明

> 现有工作已广泛记录了 LLM unlearning 导致无关知识腐蚀的现象，近期研究也开始将知识纠缠刻画为腐蚀的驱动因素。然而，尚无工作提供一个系统的、以数据为中心的框架，在 unlearning 发生之前，基于遗忘数据的可测量属性来预测知识腐蚀的严重程度。我们提出第一个这样的框架——受 fine-tuning 前基于数据特征预测对抗鲁棒性的工作启发——从遗忘数据的 embedding 分布、知识图谱结构、circuit 属性和表层统计中提取特征，用于在任何 unlearning 更新执行之前预测腐蚀指标。

### 这个工作不是什么

- 不是又一个 unlearning 方法
- 不是又一个评估 benchmark
- 不是在 unlearning 过程中分析纠缠

### 这个工作是什么

- **预测框架**：给定遗忘数据，在 unlearning 之前估计腐蚀
- **特征体系**：系统分类与腐蚀相关的遗忘数据属性
- **实用工具**：通过预测结果代替完整 unlearning + evaluation，节省计算
- **data-centric AI 与 LLM unlearning 的桥梁**：将数据特征预测（成熟范式）应用于新的重要问题

### 与工作区其他 idea 的联系

- **与 SURF 框架的联系**（`papers/llm-unlearning-research-gap-table.md`）：SURF 提出 pre-unlearning 数据精炼。本提案增加了**预测层**——先预测腐蚀程度，再据此指导精炼决策，使 SURF 的数据操作从启发式变为证据驱动。
- **与不可遗忘安全知识的联系**（`idea/non-forgettable-safety-knowledge.md`）：该 idea 识别不可被腐蚀的安全关键知识。本提案提供机制来预测 unlearning 操作是否会腐蚀这些安全知识，实现主动保护而非事后检测。
- **与涌现能力的联系**（`idea/unlearning-emergent-capabilities.md`）：涌现能力 idea 研究 unlearning 的正面副效应。本提案研究负面（腐蚀），但特征框架可以扩展为同时预测正面和负面下游效应——统一的 "unlearning 影响预测器"。

---

## 第五部分：Research Gap 分析

### Gap 总结表

| 已有工作 | 缺失环节 |
|---|---|
| 事后诊断知识腐蚀 (TOFU [2], WMDP [3], MUSE [4], BLUR [5], knowledge holes [8], blind spots [9]) | **Pre-unlearning** 从遗忘数据特征预测腐蚀严重程度 |
| 预测单样本 unlearning 难度 (CUD [16]) | 预测**无关知识的腐蚀**（而非遗忘难度） |
| Unlearning 过程中的知识纠缠分析 (EGUP [11], CLReg [13], SKeB [12], UIPE [14]) | 从遗忘数据提取特征来**量化纠缠驱动的腐蚀风险** |
| 基于数据特征预测 fine-tuning 结果 (TuneAhead [20], Data2Behavior [21]) | 同类方法应用于 **unlearning 腐蚀结果预测** |
| 训练数据 → 对抗鲁棒性关联 (参考论文 [1]) | **遗忘数据 → 知识腐蚀关联** |
| 表征层 unlearning 方法 (LUNAR [24], FALCON [25], MRP [26], CIR [15]) | 将表征层特征用作**腐蚀预测的输入** |
| 基于知识图谱评估 unlearning 完整性 [23] | 将知识图谱结构用作**腐蚀的预测特征** |

### 核心 Research Gap

**没有任何现有工作系统地从遗忘数据（及其与模型现有知识的关系）中提取特征，来预测 unlearning 之前无关知识会被腐蚀多少。**

这个 gap 直接类比参考论文为对抗鲁棒性填补的空白：他们证明了训练数据特征能在 fine-tuning 前预测对抗脆弱性。我们要证明遗忘数据特征能在 unlearning 前预测知识腐蚀。

### 具体子 Gap

**Sub-Gap 1：没有遗忘数据属性的特征体系。**
参考论文定义了 4 类 13 个特征。Unlearning 场景中没有类似的遗忘数据特征体系。

**Sub-Gap 2：没有轻量级腐蚀预测器。**
参考论文用 Random Forest 预测攻击成功率。没有类似的轻量级预测器用于预测 unlearning 知识腐蚀。

**Sub-Gap 3：没有跨方法迁移性分析。**
参考论文展示了跨模型迁移性。没有工作研究过遗忘数据特征的腐蚀预测是否跨 unlearning 方法迁移。

**Sub-Gap 4：没有将数据驱动预测与缓解措施连接。**
即使研究纠缠的工作（EGUP [11], CLReg [13]）也是用纠缠来调整 unlearning 过程，而非先预测腐蚀再决定如何处理。

**Sub-Gap 5：没有成本收益分析框架。**
参考论文量化了 30x–193x 的计算节省。没有类似分析量化在 unlearning 前预测腐蚀能节省多少评估开销。

---

## 第六部分：文献综述

### 6.1 LLM Unlearning 基础与 Benchmark

LLM unlearning 旨在从训练好的模型中移除特定知识，同时保留通用能力。当前有以下主要 benchmark：

- **TOFU** [2]（Maini et al., 2024）— 使用虚构作者档案的基础 LLM unlearning benchmark；证明大多数方法无法同时满足遗忘和效用目标。
- **WMDP** [3]（Li et al., ICML 2024）— 面向安全的 benchmark（生物安全、网络安全）；证明在危险领域中保持效用的知识移除很困难。
- **MUSE** [4]（Shi et al., 2024）— 六维度评估（遗忘、隐私、效用、可扩展性、可持续性）；揭示单指标评估遗漏的盲点。
- **BLUR** [5]（Hu et al., 2025）— 具有真实遗忘-保留重叠的 benchmark；当遗忘集和保留集不可清晰分离时，现有方法显著退化。

**与本提案的关系**：这些 benchmark 记录了 unlearning 导致知识腐蚀，但没有系统分析遗忘数据的哪些属性可以预测腐蚀的严重程度。

### 6.2 LLM Unlearning 中的知识腐蚀与效用退化

LLM unlearning 的核心挑战是**效用-遗忘权衡**：激进的 unlearning 会对无关知识造成附带损伤。

- **Rethinking Machine Unlearning for LLMs** [6]（Nature Machine Intelligence 2025）— 全面综述，指出数据-模型交互动态和 unlearning 范围是常被忽视的要素。强调 unlearning 不应影响因果无关的信息。
- **Does Unlearning Truly Unlearn?** [7]（2024）— 黑盒评估显示 LLMU 和 RMU 导致通用能力显著退化。在无关数据上训练几乎可以完全恢复 unlearning 前的性能。
- **Probing Knowledge Holes in Unlearned LLMs** [8]（NeurIPS 2025）— 证明 unlearning 创建广泛的"知识空洞"，静态 benchmark 无法检测。
- **Unlearning's Blind Spots** [9]（2025）— 形式化了盲点失败模式，包括过度遗忘。
- **Multi-Turn Robustness Evaluation** [10]（2026）— 表明单次评估会遗漏交互轨迹上的失败恢复或泄漏。

**核心发现**：知识腐蚀作为 unlearning 的后果已被充分记录，但文献主要进行**事后**诊断。没有工作系统研究遗忘数据的哪些属性可以预测腐蚀的严重程度。

### 6.3 知识纠缠：遗忘数据属性为何重要

近期工作确立了知识纠缠——遗忘数据和保留数据共享重叠表征——是附带损伤的主要驱动因素。

- **EGUP** [11]（2025）— 使用样本间和样本内纠缠指标自适应调整 unlearning 强度。语义上更接近保留知识的遗忘样本得到更谨慎的处理。
- **SKeB** [12]（2025）— 通过领域图建模信息纠缠。证明通过有说服力的 prompting 可以从已 unlearn 的模型中召回知识。
- **CLReg** [13]（2026）— 证明 logit 层面操作无法消除潜空间中的遗忘-保留纠缠。提出对比表征塑形将遗忘和保留特征推开。
- **UIPE** [14]（EMNLP 2025）— 发现模型可以通过逻辑上相关的知识重建被遗忘的内容。通过移除与遗忘目标高度相关的知识来解决。
- **CIR** [15]（2025）— 认为现有方法擦除的是广泛的共享特征而非事实特异子空间。

**核心发现**：遗忘数据与其他知识之间的表征纠缠程度是腐蚀的关键驱动因素。这种纠缠正是一种可以在 unlearning 之前测量的属性。

### 6.4 预测 Unlearning 难度与效果

少量但增长中的工作试图理解为什么某些数据更难 unlearn，以及这是否可以预测。

- **CUD** [16]（Cheng et al., 2026）— **Pre-unlearning 指标**，使用 circuit 级信号为每个样本分配连续难度分数。简单样本关联较短、较浅的 circuit 交互（集中在早期到中间层）；困难样本依赖更长、更深的路径。**这是与本提案最接近的现有工作**，但它预测的是遗忘难度（遗忘一个样本有多难），而非知识腐蚀（多少无关知识受损）。
- **When to Forget?** [17]（ICML 2025）— 建立了 unlearning 效率的首个理论界限。
- **Mechanistic Unlearning** [18]（ICML 2025）— 使用机制级 circuit 定位靶向实际事实回忆路径。
- **KUDA** [19]（2026）— 使用 causal tracing 定位知识存储层。

**核心发现**：CUD 证明了 pre-unlearning 的样本级属性预测是可行的。但 CUD 预测的是遗忘难度，而非腐蚀程度。

### 6.5 以数据为中心的方法与训练前预测

多项工作从数据特征预测模型行为。

- **TuneAhead** [20]（2025）— 在训练开始前预测 LLM fine-tuning 性能，准确率达 89.4%，节省 58.4% 计算。使用 SHAP 分析特征重要性。
- **Data2Behavior** [21]（2025）— 不更新参数即可预测非预期模型行为，仅使用 ~20% GPU 资源。
- **PRISM** [22]（2025）— 在单次前向传播中将 LLM 预测追溯到训练数据原型。

**核心发现**：从数据特征预测模型结果的思路在 fine-tuning 场景已成熟，但**从未被应用于预测 unlearning 腐蚀**。

### 6.6 表征层面对 Unlearning 效果的理解

- **LUNAR** [24]（NeurIPS 2025）— 将表征重定向到表达"无法回答"的激活区域，达到 2.9–11.7× 的效用-效能综合改进。
- **FALCON** [25]（NeurIPS 2025）— 使用对比正交去对齐进行细粒度激活操作。
- **MRP** [26]（2025）— 旨在实现不可逆的隐藏状态变换。
- **Representation-Aware Unlearning via Activation Signatures** [27]（2026）— 通过激活签名研究抑制 vs. 擦除的区别。

**核心发现**：表征层特征包含丰富信息，可以作为 pre-unlearning 腐蚀预测器的输入基础。

### 6.7 知识关联与评估

- **Do LLMs Really Forget?** [23]（2025）— 使用带置信度评分的知识图谱捕捉潜在推理依赖关系。证明被假设遗忘的事实可以通过相关信息持续存在。
- **Dynamic Evaluation Framework**（COLM 2025）— 使用多跳推理和实体别名进行压力测试。单跳查询容易被 unlearning 破坏；多跳查询使用替代路径通常保持完整。

**核心发现**：LLM 中的知识通过关联结构和推理依赖相互连接。遗忘数据在这个知识图谱中的位置可能决定了附带腐蚀的程度——但这种联系尚未被形式化为预测框架。

---

## 参考文献

### 核心参考
[1] A Curious Case of Searching for the Correlation between Training Data and Adversarial Robustness of Transformer Textual Models. ACL Findings 2024. arXiv:2402.11469

### LLM Unlearning Benchmark
[2] TOFU: A Task of Fictitious Unlearning for LLMs. Maini et al., 2024. arXiv:2401.06121
[3] The WMDP Benchmark. Li et al., ICML 2024. arXiv:2403.03218
[4] MUSE: Machine Unlearning Six-Way Evaluation. Shi et al., 2024. arXiv:2407.06460
[5] BLUR: A Benchmark for LLM Unlearning Robust to Forget-Retain Overlap. Hu et al., 2025. arXiv:2506.15699

### 知识腐蚀与效用退化
[6] Rethinking Machine Unlearning for LLMs. Fan et al., Nature Machine Intelligence 2025. arXiv:2402.08787
[7] Does Unlearning Truly Unlearn? Doshi & Stickland, 2024. arXiv:2411.12103
[8] Probing Knowledge Holes in Unlearned LLMs. Ko et al., NeurIPS 2025. arXiv:2511.00030
[9] Unlearning's Blind Spots. Ha et al., 2025. arXiv:2506.01318
[10] Multi-Turn Robustness Evaluation. Pan & Wang, 2026. arXiv:2603.00823

### 知识纠缠
[11] EGUP: Entanglement-Guided Unlearning with Proxy Constraint. 2025. arXiv:2508.20443
[12] SKeB: Stimulus-Knowledge Entanglement-Behavior Framework. 2025. arXiv:2510.25732
[13] CLReg: From Logits to Latents. Tang & Khanna, 2026. arXiv:2601.22028
[14] UIPE: Enhancing LLM Unlearning by Removing Related Knowledge. EMNLP 2025. arXiv:2503.04693
[15] CIR: Collapse of Irrelevant Representations. Sondej & Yang, 2025. arXiv:2509.11816

### 预测 Unlearning 难度
[16] CUD: Circuit-Guided Unlearning Difficulty Metric. Cheng et al., 2026. arXiv:2601.09624
[17] When to Forget? Complexity Trade-offs in Machine Unlearning. van Waerebeke et al., ICML 2025
[18] Mechanistic Unlearning. Guo et al., ICML 2025. arXiv:2410.12949
[19] KUDA: Knowledge Unlearning by Deviating Representation. Fang et al., 2026. arXiv:2602.19275

### 以数据为中心的预测
[20] TuneAhead: Predicting Fine-tuning Performance Before Training. OpenReview, 2025
[21] Data2Behavior. 2025. arXiv:2602.04735
[22] PRISM: Training Data Prototypes for Language Models. GuideAI, 2025
[23] Do LLMs Really Forget? Knowledge Correlation and Confidence Awareness. 2025. arXiv:2506.05735

### 表征层 Unlearning
[24] LUNAR: Neural Activation Redirection. Shen et al., NeurIPS 2025. arXiv:2502.07218
[25] FALCON. Hu et al., NeurIPS 2025. arXiv:2502.01472
[26] MRP: Metamorphosis Representation Projection. Wu et al., 2025. arXiv:2508.15449
[27] Representation-Aware Unlearning via Activation Signatures. Mahmood et al., 2026. arXiv:2601.10566
[28] ReLearn: Unlearning via Learning. Xu et al., ACL 2025. arXiv:2502.11190

### 相关方法与综述
[29] A Comprehensive Survey of Machine Unlearning Techniques for LLMs. 2025. arXiv:2503.01854
[30] Unlearning in LLMs: Methods, Evaluation, and Open Challenges. 2026. arXiv:2601.13264
[31] Align-then-Unlearn. Spohn et al., 2025. arXiv:2506.13181
[32] SOUL: Second-Order Optimization for LLM Unlearning. Jia et al., 2024. arXiv:2404.18239
[33] Gauss-Newton Unlearning. McKinney et al., SaTML 2026. arXiv:2602.10568
[34] Unlearning That Lasts (JensUn). Singh et al., 2025. arXiv:2509.02820
