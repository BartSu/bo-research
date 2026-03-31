# 表征学习 (Representation Learning) 与 LLM Unlearning 中的 Representation Shift

## 目录

1. [Saining Xie 的表征学习研究](#1-saining-xie-的表征学习研究)
2. [表征学习在 LLM 中的含义](#2-表征学习在-llm-中的含义)
3. [表征学习与自监督学习的关系](#3-表征学习与自监督学习的关系)
4. [Unlearning 中的 Representation Shift](#4-unlearning-中的-representation-shift)
5. [研究 Unlearning Representation Shift 的可能方向](#5-研究-unlearning-representation-shift-的可能方向)
6. [关键论文索引](#6-关键论文索引)

---

## 1. Saining Xie 的表征学习研究

**Saining Xie**（谢赛宁），现任 NYU Courant Institute 助理教授，同时是 AMI Labs 联合创始人兼 CSO。他的研究核心主题是：**如何设计架构和训练方法，使模型学到更好的内部表征 (representation)**。

### 1.1 代表性工作

| 论文 | 年份/会议 | 核心贡献 |
|---|---|---|
| **ResNeXt**: Aggregated Residual Transformations | CVPR 2017 | 引入 cardinality 维度，证明增加并行变换路径比增加深度或宽度更有效地提升表征质量 |
| **MoCo** (Momentum Contrast) | CVPR 2020 | 自监督对比学习的里程碑——通过动量编码器和大型负样本字典，实现无监督视觉表征学习，首次让无监督预训练在 7 个下游任务上超越有监督预训练 |
| **MAE** (Masked Autoencoders) | CVPR 2022 | 自监督视觉学习——遮盖 75% 的图像 patch，让编码器学习重建，ViT-Huge 在 ImageNet-1K 上达到 87.8% |
| **ConvNeXt** | CVPR 2022 | 现代化 ConvNet 设计，纯卷积网络达到 87.8% ImageNet 准确率，与 Vision Transformer 竞争 |
| **DiT** (Diffusion Transformers) | 2023 | 将 Transformer 引入扩散模型框架，成为 Sora 等生成系统的基础架构 |
| **RAE** (Representation Autoencoders) | ICLR 2026 | 用预训练表征编码器（DINO, SigLIP, MAE）替代传统 VAE，提供语义丰富的隐空间，FID 达 1.13 (ImageNet 512×512) |
| **SANA 1.5** | ICML 2025 | 线性 Diffusion Transformer，引入深度增长范式和推理时缩放策略 |

### 1.2 Saining Xie 研究的核心哲学

Xie 的研究一贯围绕一个中心问题：**什么样的结构和训练信号能让模型学到最有用的内部表征？**

- **架构设计视角**（ResNeXt → ConvNeXt → DiT）：网络结构如何影响表征质量
- **训练范式视角**（MoCo → MAE）：自监督学习如何在无标签数据上学到好表征
- **表征作为生成基础**（RAE）：用预训练语义表征作为生成模型的隐空间，替代纯像素重建的 VAE

**关键洞察**：Xie 的工作表明，高质量的表征学习不依赖标签，而依赖于：(a) 合适的学习信号（对比、遮盖重建），(b) 合适的架构归纳偏置，(c) 足够的数据规模。

---

## 2. 表征学习在 LLM 中的含义

### 2.1 基本概念

**表征学习 (Representation Learning)** 在 LLM 语境中指的是：模型在训练过程中，将输入 token 序列转化为高维向量空间中有意义的内部表示。这些内部表示（激活值、隐藏状态）编码了语义、语法、事实知识、推理模式等多层次信息。

LLM 的每一层 Transformer 产生一组隐藏状态向量 $h_l \in \mathbb{R}^d$，其中：
- **浅层**倾向于编码表面语法和局部语义
- **中间层**编码更抽象的语义关系和事实知识
- **深层**编码面向输出的任务相关表征

### 2.2 线性表征假设 (Linear Representation Hypothesis, LRH)

2024 年以来，LLM 内部表征研究最重要的发现之一是 **线性表征假设**：

> 高层语义概念在 LLM 的表征空间中以**线性方向**编码。

**关键论文与发现**：

- **Park et al. (ICML 2024)**：形式化了 LRH，使用反事实定义了"线性表征"的精确含义，在 LLaMA-2 上实验证实概念（如情感、主题、真假）确实以线性方向存在。发现了非欧几里得的"因果内积"，统一了 probing 和 steering 两种线性表征操作。

- **Jiang et al. (2024)**：从理论角度解释了线性表征的**起源**——next token prediction + softmax cross-entropy loss + 梯度下降的隐式偏置共同促使线性表征自然涌现。

- **Garg et al. (2026 preprint)**：将 LRH 拆分为"线性表征"（特征线性嵌入）和"线性可达性"（特征可被线性探测解码），证明在 LRH 下神经元可以存储指数数量的特征，支持 superposition 假说。

### 2.3 表征工程 (Representation Engineering, RepE)

**Zou et al. (2023)** 提出的 Representation Engineering 是一个"自上而下"的 LLM 透明度方法，将**群体级别的表征**（而非单个神经元或回路）作为分析核心：

- **RepReading**：读取和监控模型内部表征
- **RepControl**：通过激活编辑控制模型输出行为（又称 activation steering / control vectors）

应用领域：诚实性、安全性、公平性、权力寻求行为、知识编辑、记忆控制等。

**2025 年综述** (arxiv 2502.17601) 进一步系统梳理了 RepE 方法体系：
- Activation steering（激活引导）
- Model editing（模型编辑）
- Activation analysis（激活分析）
- Linear probing（线性探测）

### 2.4 Superposition 与多义性

Anthropic 的 mechanistic interpretability 研究揭示：

- LLM 神经元是**多义的 (polysemantic)**：单个神经元同时编码多个不相关概念
- 表征存在**竞争**：当特征数量超过模型维度，稀疏概念被压缩到共享子空间，产生干扰
- **稀疏自编码器 (Sparse Autoencoders, SAEs)** 可以将多义表征分解为单义特征

这意味着 LLM 的表征空间不是"一对一"映射，而是高度纠缠和压缩的——这对理解 unlearning 中的 representation shift 至关重要。

---

## 3. 表征学习与自监督学习的关系

### 3.1 关系总结

```
自监督学习 (Self-Supervised Learning, SSL)
    └── 是一种训练范式（不使用人工标签）
    └── 目的：学到好的表征 (Representation Learning)
    └── 方法包括：
         ├── 对比学习 (Contrastive Learning): MoCo, SimCLR, CLIP
         ├── 遮盖预测 (Masked Prediction): BERT, MAE, GPT (next token)
         └── 其他：旋转预测、拼图、聚类等

表征学习 (Representation Learning)
    └── 是一个目标（学到有用的内部表示）
    └── 可以通过多种范式实现：
         ├── 自监督学习 ← 主流方法
         ├── 有监督学习
         └── 强化学习
```

**核心关系**：自监督学习是实现表征学习的最重要方法之一，而表征学习是自监督学习的核心目标。

### 3.2 在 LLM 中

LLM 的预训练本质上就是自监督表征学习：

| 模型类型 | SSL 任务 | 表征学习方式 |
|---|---|---|
| GPT 系列 (Decoder-only) | Next Token Prediction | 通过预测下一个 token，学习到包含语法、语义、事实知识的表征 |
| BERT 系列 (Encoder-only) | Masked Language Modeling | 通过填充被遮盖的 token，学习双向上下文表征 |
| T5 系列 (Encoder-Decoder) | Span Corruption | 通过修复被破坏的文本片段，学习序列到序列的表征 |

**关键区别**：在视觉领域（Xie 的研究方向），SSL 和表征学习的界限更清晰——先用 SSL 预训练，再用表征做下游任务。在 LLM 中，SSL 预训练和表征学习是**同一个过程的两面**——预训练就是在做表征学习，生成文本的能力直接来源于学到的表征质量。

### 3.3 Saining Xie 在这一交叉点上的贡献

Xie 的 MoCo 和 MAE 是视觉领域 SSL + 表征学习的典范：

- **MoCo** 证明了**对比学习**可以产生与有监督学习一样好（甚至更好）的表征
- **MAE** 证明了**遮盖重建**（与 BERT/GPT 同族的预测式 SSL）在视觉中同样有效
- **RAE** 进一步证明了 SSL 学到的表征可以作为生成模型的隐空间基础

这些工作构建了一条从 SSL → 表征学习 → 下游应用（分类、检测、生成）的完整链路。

---

## 4. Unlearning 中的 Representation Shift

### 4.1 问题定义

**Representation Shift** 在 unlearning 语境中指：当模型通过 unlearning 过程"遗忘"某些知识后，其内部表征空间发生的变化。这种变化可以从多个层次衡量：

- **逐层激活变化**：每层隐藏状态与原始模型的偏差
- **子空间漂移**：通过 PCA/SVD 提取的主要表征子空间的旋转或缩放
- **CKA (Centered Kernel Alignment)**：衡量两个模型表征的结构相似性
- **Fisher Information**：衡量参数对特定知识的敏感度变化

### 4.2 核心发现：抑制 vs. 删除

2025-2026 年的研究揭示了一个关键事实：

> **大多数 unlearning 方法只是在输出层抑制了信息，而非在表征层真正删除了知识。**

#### 4.2.1 "Suppression or Deletion" (arxiv 2602.18505, Feb 2026)

**方法**：使用 Sparse Autoencoders 识别中间层的类别特定专家特征，通过推理时 steering 进行定量分析。分析了 12 种主流 unlearning 方法。

**关键发现**：
- 大多数方法的 restoration rate（通过少量微调恢复遗忘行为的比率）非常高
- 即使从预训练 checkpoint 重新训练也表现出高恢复率——说明预训练阶段获得的鲁棒语义特征不会被 unlearning 移除
- 信息只是在决策边界层面被抑制，语义特征在中间层表征中被保留

**启示**：当前基于输出的评估指标不足以衡量真正的遗忘。需要表征级别的验证。

#### 4.2.2 "Unlearning Isn't Deletion" (OpenReview, ICLR 2026)

**关键发现**：
- 使用 PCA 相似性、CKA、Fisher Information 等指标分析表征漂移
- Unlearned 模型的内部表征与原始模型仍然高度相似
- 通过最少量的微调即可恢复原始行为

### 4.3 表征级别的 Unlearning 方法

针对上述问题，2025-2026 年涌现了多种在表征层面直接操作的 unlearning 方法：

#### 4.3.1 Erase at the Core (EC) — arxiv 2602.05375, Feb 2026

**核心思想**：在整个网络层级强制遗忘，而非仅修改最终分类器。

**方法**：
- 在中间层附加辅助模块
- 对遗忘集应用**多层对比 unlearning**
- 对保留集应用深度监督学习
- 使用逐层加权损失函数

**效果**：显著降低中间层表征与原始模型的相似度，同时保持保留集性能。

#### 4.3.2 KUDA — arxiv 2602.19275, Feb 2026

**核心思想**：通过因果追踪定位知识存储层，诱导表征偏离原始位置。

**方法**：
1. **Causal Tracing**：定位存储目标知识的具体层
2. **Representation Deviation**：设计 unlearning 目标函数，使目标知识的表征偏离原始位置
3. **Relaxation Null-Space Projection**：在偏离目标知识表征的同时，通过零空间投影保护保留知识的表征

#### 4.3.3 CIR (Collapse of Irrelevant Representations) — arxiv 2509.11816

**核心思想**：选择性地坍缩包含通用（非目标）表征的子空间，仅保留事实特定表征的 unlearning 更新。

**方法**：
- 对 MLP 输出（而非残差流）进行 PCA，识别通用表征子空间
- 坍缩通用子空间后再计算 unlearning 更新
- 使用 MLP breaking loss 使表征与原始表征正交

**效果**：
- 在 Llama-3.1-8B 上，有害知识消除效率是 Circuit Breakers 的 30 倍
- 通用性能退化仅 0.1%（WikiText loss 增加）
- 每个事实仅需不到 3 GPU 秒

#### 4.3.4 LUNAR — arxiv 2502.07218

**核心思想**：基于线性表征假设，将 unlearned 数据的表征重定向到表示"无法回答"的激活区域。

**效果**：综合效能和效用分数提升 2.9-11.7 倍。

#### 4.3.5 KIF (Knowledge Immunization Framework) — arxiv 2601.10566

**核心思想**：靶向内部激活签名（activation signatures），而非仅表面输出。

**方法**：
- 动态抑制主题特定表征
- 结合参数高效适应
- 在 3B 到 14B 参数模型上实现接近 oracle 的擦除效果

#### 4.3.6 Representation Unlearning via Information Compression — arxiv 2601.21564

**核心思想**：在模型的表征空间中直接执行 unlearning——学习变换以最大化与保留数据的互信息，同时抑制与遗忘数据的信息。

### 4.4 Representation Shift 的度量体系

| 指标 | 含义 | 用途 |
|---|---|---|
| **PCA Similarity** | 原始与 unlearned 模型的主成分对齐度 | 衡量子空间层面的表征漂移 |
| **CKA (Centered Kernel Alignment)** | 两组表征的结构相似性 | 跨层、跨模型的表征比较 |
| **Fisher Information** | 参数对特定输入的敏感度 | 识别哪些参数编码了目标知识 |
| **Restoration Rate** | 通过少量微调恢复遗忘行为的比率 | 区分抑制与真删除 |
| **SAE Feature Activation** | 稀疏自编码器解码出的特定特征激活强度 | 检测中间层是否保留了目标知识的特征 |
| **Probing Accuracy** | 线性探测器从中间层提取目标信息的准确率 | 衡量表征中残留的可解码信息 |
| **Cosine Similarity (Layer-wise)** | 逐层隐藏状态与原始模型的余弦相似度 | 细粒度的层级表征变化追踪 |

---

## 5. 研究 Unlearning Representation Shift 的可能方向

### 5.1 核心研究问题

基于以上文献综述，研究 unlearning 的 representation shift 可以围绕以下几个问题展开：

#### Q1：Representation Shift 的拓扑结构是什么？

现有工作大多使用单一指标（如 CKA）衡量 shift。但 representation shift 可能有更丰富的结构：

- 不同层的 shift 模式是否不同？（浅层 vs. 深层）
- Shift 是否有方向性？（向某个特定区域偏移，还是随机分散？）
- 遗忘集内部的表征是否坍缩到同一点（如 CIR 的设计），还是分散到不同区域？

#### Q2：Representation Shift 与 Unlearning 效果的因果关系是什么？

关键区分：
- **充分条件**：表征 shift 到什么程度，遗忘才是不可逆的？
- **必要条件**：是否存在不需要 representation shift 的有效 unlearning？（纯输出层修改是否可以在某些条件下真正删除？）

#### Q3：如何预测 Unlearning 会引起多大的 Representation Shift？

- 知识在表征空间中的"纠缠度"是否可以预测 shift 的幅度？
- 预训练表征的鲁棒性是否设定了 shift 的下界？（如 "Suppression or Deletion" 论文所暗示的）

### 5.2 与本仓库已有 idea 的连接

#### 连接到 `unlearning-emergent-capabilities.md`

Representation shift 是理解"遗忘是否能产生新涌现能力"的机制基础：

- 如果 unlearning 仅在输出层抑制（小 shift），涌现新能力不太可能
- 如果 unlearning 在表征层产生真正的结构性 shift，那么 superposition 中被释放的容量可能确实激活了被抑制的能力
- **可测试假设**：representation shift 的幅度与涌现能力变化的幅度正相关

#### 连接到 `forget-data-knowledge-corruption-correlation.md`

Representation shift 可以作为知识腐蚀的**机制性解释**：

- 知识腐蚀（模型在保留集上的性能退化）可能正比于该保留知识与遗忘知识之间的表征子空间重叠
- 预测腐蚀 = 预测 representation shift 的传播范围

#### 连接到 `non-forgettable-safety-knowledge.md`

安全知识的不可遗忘性可能根植于表征层面：

- 如果安全知识的表征与大量通用能力高度纠缠，那么遗忘安全知识必然导致大幅 representation shift，进而导致广泛的能力退化
- 这种"表征纠缠"本身就是安全知识不可遗忘性的机制来源

### 5.3 具体研究方向提议

#### 方向 A：Representation Shift 的几何刻画

**目标**：建立 unlearning 过程中 representation shift 的完整几何描述。

**方法**：
1. 对多种 unlearning 方法，在每一层提取 forget set 和 retain set 的隐藏状态
2. 使用 PCA、t-SNE、UMAP 可视化 shift 的方向和幅度
3. 计算每层的 CKA、余弦相似度、子空间角度
4. 将几何特征与 unlearning 效果（forget quality, retain quality, robustness to restoration）关联

**新颖性**：现有工作通常只报告单一聚合指标。系统的逐层几何分析尚未有。

#### 方向 B：Representation Shift 作为 Unlearning 质量的统一度量

**目标**：提出一个基于 representation shift 的 unlearning 质量评估框架，替代或补充现有的输出级别指标。

**核心观点**：如果 representation shift 不够大/不够"深"，unlearning 就是可逆的。

**方法**：
1. 定义 "Representation Erasure Score (RES)"：综合多层的 shift 幅度和方向性
2. 验证 RES 与抗微调恢复性的相关性
3. 将 RES 作为 unlearning 过程的正则项，引导方法产生更深层次的 shift

#### 方向 C：预训练表征的鲁棒性边界

**目标**：研究预训练阶段学到的表征对 unlearning 施加的限制。

**核心假设**：预训练表征是 unlearning 的"硬下界"——某些语义结构如此根深蒂固，以至于任何 unlearning 方法都无法在不破坏通用能力的前提下移除它们。

**方法**：
1. 分析预训练 checkpoint 上的特征鲁棒性（哪些特征最难移除？）
2. 与 "Suppression or Deletion" 论文的发现对接
3. 研究预训练数据组成对 representation shift 下界的影响

---

## 6. 关键论文索引

### 6.1 Saining Xie 表征学习核心论文

| 论文 | 链接 | 关键词 |
|---|---|---|
| ResNeXt | arxiv 1611.05431 | 架构设计, cardinality, 表征学习 |
| MoCo v1/v2 | arxiv 1911.05722 / 2003.04297 | 自监督对比学习, 动量编码器 |
| MAE | arxiv 2111.06377 | 自监督遮盖重建, ViT |
| ConvNeXt | arxiv 2201.03545 | 现代 ConvNet, 表征设计 |
| DiT | 2023 | Diffusion Transformers |
| RAE | arxiv 2510.11690 | 表征自编码器, 生成模型隐空间 |

### 6.2 LLM 表征学习与理解

| 论文 | 链接 | 关键词 |
|---|---|---|
| Linear Representation Hypothesis | ICML 2024 (Park et al.) | 线性表征, 因果内积, LLaMA-2 |
| Origins of Linear Representations | arxiv 2403.03867 | next token prediction, 隐式偏置 |
| Representation Engineering (RepE) | arxiv 2310.01405 | RepReading, RepControl, 激活引导 |
| RepE Survey | arxiv 2502.17601 | 表征工程综述, 激活分析 |
| LRH Theoretical Analysis | arxiv 2602.11246 | 线性表征 vs 线性可达性, superposition |

### 6.3 Unlearning 中的 Representation Shift 核心论文

| 论文 | 链接 | 核心发现/方法 |
|---|---|---|
| Suppression or Deletion | arxiv 2602.18505 | SAE 分析 12 种方法, 大多仅抑制非删除 |
| Unlearning Isn't Deletion | OpenReview (ICLR 2026) | PCA/CKA/Fisher 分析, unlearning 可逆性 |
| Erase at the Core (EC) | arxiv 2602.05375 | 多层对比 unlearning, 全层级表征擦除 |
| KUDA | arxiv 2602.19275 | 因果追踪定位 + 表征偏离 + 零空间投影 |
| CIR | arxiv 2509.11816 | PCA 选择性坍缩通用子空间, 30× 改进 |
| LUNAR | arxiv 2502.07218 | 基于 LRH 的激活重定向 |
| KIF | arxiv 2601.10566 | 激活签名靶向, 3B-14B 模型 |
| Representation Unlearning via Info Compression | arxiv 2601.21564 | 互信息优化, 表征空间变换 |
| MUNKEY | arxiv 2603.15033 | 记忆增强 Transformer, 零样本遗忘 |

---

## 总结

### 表征学习、自监督学习、与 Unlearning 的关系图

```
[自监督学习 SSL]
    │ 训练方式
    ▼
[表征学习 Representation Learning]
    │ 产出
    ▼
[LLM 内部表征空间]
    │
    ├── 线性表征假设 (LRH): 概念以线性方向编码
    ├── Superposition: 多个概念共享同一表征子空间
    ├── Polysemanticity: 单个神经元编码多个概念
    │
    │ 当执行 Unlearning 时
    ▼
[Representation Shift]
    │
    ├── 输出层抑制 (浅层 shift): 大多数现有方法 → 可逆, 知识仍在
    ├── 中间层表征偏离 (深层 shift): KUDA, EC, CIR → 更彻底的遗忘
    └── 全层级表征擦除 (结构性 shift): 理想目标 → 但可能损害通用能力
         │
         ├── 机会: 释放 superposition 中的容量 → 涌现新能力?
         └── 风险: 纠缠知识的连带腐蚀 → 通用能力退化
```

### 关键 takeaway

1. **Saining Xie** 的工作定义了现代表征学习的方法论：从架构设计（ResNeXt）到自监督学习（MoCo, MAE），再到将表征作为生成模型基础（RAE）。他的核心哲学是：好的表征是一切下游能力的基础。

2. **LLM 中的表征学习**本质上就是预训练过程——通过 next token prediction 这一自监督任务，模型学到编码语义、语法、事实、推理的高维内部表示。线性表征假设 (LRH) 和 superposition 理论为理解这些表征提供了框架。

3. **自监督学习与表征学习**是手段与目标的关系：SSL 是最有效的训练方式之一，表征学习是其核心目标。在 LLM 中两者几乎不可分割。

4. **Unlearning 中的 representation shift** 是当前该领域最活跃的研究前沿。核心发现是大多数方法仅在输出层面抑制信息而非真正删除表征中的知识。新方法（EC, KUDA, CIR, LUNAR）开始直接在表征空间操作，试图实现"深层遗忘"。

5. **研究 representation shift 的价值**在于：(a) 作为 unlearning 质量的更可靠度量；(b) 理解遗忘对通用能力的影响机制；(c) 探索遗忘是否能通过释放表征容量产生建设性效果（连接到本仓库的 emergent capabilities idea）。
