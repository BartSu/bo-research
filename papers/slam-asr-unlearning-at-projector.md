# SLAM-ASR / LLM-based ASR 中的 Unlearning：聚焦 Linear Projector 的文献梳理

这份笔记聚焦一个非常具体的交叉方向：

> **在 SLAM-ASR 类架构（speech encoder + linear projector + frozen LLM）下，能否通过对 linear projector 做 unlearning 来实现对 LLM-based ASR 系统的选择性遗忘？**

这个问题的核心结构是：

1. **SLAM-ASR 架构**的可训练部分几乎只有 linear projector（~18–21M 参数）；
2. **Unlearning 的目标**可以是隐私数据删除、说话人信息移除、或有害内容遗忘；
3. **在 projector 上做 unlearning** 既有成本优势（参数量小），也有对齐稳定性风险（projector 是唯一的跨模态桥梁）。

下面按三条线索组织：
- **A. 直接研究 LLM-based ASR unlearning 的论文**（极少，但已有开创性工作）
- **B. 研究多模态 LLM 中 projector 级 unlearning 的论文**（视觉-语言域，但架构逻辑高度类比）
- **C. 相关的参数高效 unlearning / 适配器级 unlearning 方法论**

---

## 0. 为什么 "在 projector 上做 unlearning" 是一个值得追的方向？

SLAM-ASR (arXiv:2402.08846) 的核心发现是：**冻结 speech encoder 和 LLM，只训练一个线性 projector，就可以做出竞争力很强的 LLM-based ASR。** 这意味着：

- Projector 是系统中唯一被训练的部分，所有任务特定知识都被压缩在这里；
- 如果 projector 记住了训练数据中的隐私信息（说话人身份、特定语句、个人数据），那么 unlearning 的自然操作点就是 projector；
- 但 projector 同时承担跨模态对齐的全部负担——对它做梯度上升或其他 unlearning 操作，极易破坏语音-文本的对齐几何。

这和 vision-language 模型中的 projector unlearning 问题完全同构。SineProject 的作者明确指出：

> "We trace this failure to the projector network during unlearning: its Jacobian becomes severely ill-conditioned, leading to unstable optimization and drift in cross-modal embeddings."

---

## A. 直接研究 LLM-based ASR / Speech Unlearning 的论文

这一块论文非常稀少，是一个刚刚起步的方向。

### A1) Unlearning LLM-Based Speech Recognition Models
- **作者：** Zhe Liu
- **会议：** Interspeech 2025, pp. 3214–3218
- **DOI：** 10.21437/Interspeech.2025-287
- **链接：** https://www.isca-archive.org/interspeech_2025/liu25b_interspeech.html
- **TL;DR：** **第一篇**直接研究 LLM-based ASR 模型中非预期记忆（unintended memorization）的检测与消除。提出了高效的 unlearning 方法，在 LibriSpeech 上实现了良好的 privacy-utility 平衡。
- **核心贡献：**
  1. 首次测量并检测 LLM-based ASR 模型中的训练数据记忆；
  2. 提出了计算高效的 unlearning 方案，无需从头重训；
  3. 实验证明可以在保持识别性能的同时有效遗忘目标数据。
- **局限 / 未回答的问题：**
  - 论文没有明确区分 "在 projector 层做 unlearning" vs. "在 LLM 层做 unlearning" 的效果差异；
  - 仅在 LibriSpeech 上验证，跨域和多语言场景未涉及；
  - 未讨论 unlearning 对跨模态对齐稳定性的影响。
- **与你的方向的关系：** 这是目前**最直接相关**的论文。它证明了 LLM-based ASR 确实存在记忆问题，也证明了 unlearning 是可行的。但它没有专门研究 projector 作为 unlearning 操作点的问题。

**Zhe Liu 的相关前序工作**（同一作者的隐私/记忆相关研究线）：
- **Forgetting Private Textual Sequences in Language Models via Leave-One-Out Ensemble** (ICASSP 2024)
  - https://ieeexplore.ieee.org/document/10446299
- **Mitigating Unintended Memorization in Language Models via Alternating Teaching** (ICASSP 2023)
  - https://ieeexplore.ieee.org/document/10096557

### A2) Speech Unlearning
- **作者：** Jiali Cheng, Hadi Amiri (University of Massachusetts Lowell)
- **发表：** arXiv preprint, 2025
- **链接：** https://arxiv.org/abs/2506.00848
- **TL;DR：** 首次系统定义 speech unlearning 问题。提出两种基本任务：**sample unlearning**（移除单条语音记录的影响）和 **class unlearning**（移除整个说话人的所有数据）。在 keyword spotting 和 speaker identification 上进行实验，发现语音数据的 unlearning 比图像和文本更困难。
- **核心发现：**
  - 语音数据的高维、序列化、说话人依赖特性使 unlearning 显著更难；
  - 提出了 feature-level unlearning 作为未来方向。
- **与你的方向的关系：** 提供了 speech unlearning 的问题定义和难度分析。但该论文研究的是传统语音模型（非 LLM-based ASR），不涉及 projector 结构。**Feature-level unlearning** 的提议与在 projector 做 representation-level unlearning 有方向性的一致。

### A3) OrthoGrad: Go Beyond Your Means — Unlearning with Per-Sample Gradient Orthogonalization
- **作者：** Aviv Shamsian, Eitan Shaar, Aviv Navon, Gal Chechik, Ethan Fetaya
- **发表：** arXiv preprint, 2025 (v2: 2026)
- **链接：** https://arxiv.org/abs/2503.02312
- **TL;DR：** 提出 OrthoGrad 方法：将 unlearn 集的梯度投影到与 retain 集所有梯度正交的子空间上，避免梯度干扰。在多个 benchmark 上有效，**包括 automatic speech recognition**。
- **方法核心：**
  - 逐样本计算 retain 集梯度，通过 QR 分解建立正交基；
  - 将 unlearn 梯度投影到正交补空间；
  - 特别适用于 retain 集很小的场景。
- **与你的方向的关系：** 目前唯一明确包含 ASR benchmark 的通用 unlearning 方法论文。其正交投影思想天然适合在低参数量的 projector 上操作——projector 参数空间小，retain/forget 方向更容易冲突，正交化约束可能特别有价值。

---

## B. 多模态 LLM 的 Projector 级 Unlearning（Vision-Language 域）

以下论文虽然来自视觉-语言领域，但架构逻辑与 SLAM-ASR 高度类比：

| SLAM-ASR | Vision-Language MLLM |
|---|---|
| Speech encoder (HuBERT/WavLM) | Vision encoder (CLIP/ViT) |
| Linear projector (~18M params) | Linear/MLP projector |
| Frozen LLM (Vicuna-7B) | Frozen LLM (LLaMA/Vicuna) |

因此 vision-language projector unlearning 的发现可以直接迁移到 speech-language 场景。

### B1) SineProject: Machine Unlearning for Stable Vision–Language Alignment ⭐ 最重要
- **作者：** Arpit Garg, Hemanth Saratchandran, Simon Lucey (University of Adelaide)
- **发表：** arXiv preprint, 2025 (v2: 2026)
- **链接：** https://arxiv.org/abs/2511.18444
- **代码：** https://github.com/arpit2412/SineProject
- **TL;DR：** **第一篇明确将 MLLM unlearning 失败归因于 projector 的论文。** 发现 unlearning 过程中 projector 的 Jacobian 矩阵变得严重病态（ill-conditioned），导致跨模态 embedding 漂移（alignment drift），模型同时拒绝有害和正常查询。提出 SineProject：在冻结的 projector 上增加正弦调制的可训练参数，改善 Jacobian 的谱条件数，稳定 unlearning 过程中的对齐。
- **核心发现：**
  1. **Projector 是 MLLM unlearning 的关键瓶颈**：它是唯一的跨模态桥梁，梯度上升对它的扰动会系统性破坏对齐几何；
  2. 标准 unlearning 方法（GA/NPO 等）在 MLLM 上失败的根本原因不是 LLM 部分，而是 projector 层；
  3. 通过控制 projector 的 Jacobian 谱特性，可以在完全遗忘目标信息的同时保持正常查询能力。
- **在 LLaVA-v1.5 7B/13B 上的结果：** SOTA forget-retain 平衡，计算开销可忽略。
- **对 SLAM-ASR 的启示：** 这个发现**直接适用**于 SLAM-ASR。SLAM-ASR 的 linear projector 结构更简单（纯线性），Jacobian 病态的风险可能更高也可能更容易控制。SineProject 的正弦调制方案可以直接迁移到 speech-LLM projector 上。

### B2) Stable Forgetting: Bounded Parameter-Efficient Unlearning in Foundation Models
- **作者：** Arpit Garg, Hemanth Saratchandran, Ravi Garg, Simon Lucey (University of Adelaide)
- **发表：** arXiv preprint, 2025 (v2: 2026)
- **链接：** https://arxiv.org/abs/2509.24166
- **TL;DR：** 理论分析了梯度差分法（gradient descent on retain + gradient ascent on forget）在 cross-entropy loss 下导致权重和梯度无界增长的机制。提出在 LoRA adapter 的 MLP 上施加有界函数（bounded functions）来稳定 unlearning。
- **核心理论：** 在 cross-entropy + gradient ascent 下，logits 会发散到无穷，导致最终层权重无界增长，不稳定性向前传播。有界激活函数可以控制这种发散。
- **覆盖范围：** ViT (CIFAR-100), LLM (TOFU/TDEC/MUSE), 参数量 22M–8B。
- **对 SLAM-ASR 的启示：** Projector 是低参数量的适配层，恰好是 LoRA-style parameter-efficient 操作的自然目标。有界函数策略可能在 SLAM-ASR 的 projector unlearning 中非常实用。

### B3) MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of MLLMs
- **作者：** Jiahao Huo, Yibo Yan, Xu Zheng, Yuanhuiyi Lyu, Xin Zou, Zhihua Wei, Xuming Hu
- **会议：** Findings of ACL 2025
- **链接：** https://arxiv.org/abs/2502.11051
- **代码：** https://github.com/Z1zs/MMUnlearner
- **TL;DR：** 提出几何约束的梯度上升方法，通过 weight saliency map 选择性擦除与特定实体关联的视觉模式，同时保留 LLM backbone 中的文本知识。
- **方法核心：** 使用 retain concept 和 textual knowledge 联合约束 weight saliency map，保护非目标知识所需的参数。
- **对 SLAM-ASR 的启示：** Saliency map 的思想可以迁移：在 projector 参数上计算 forget/retain 的 saliency，选择性更新。

### B4) MLLMEraser: Test-Time Unlearning via Activation Steering
- **作者：** 多位作者
- **发表：** arXiv preprint, 2025
- **链接：** https://arxiv.org/abs/2510.04217
- **TL;DR：** 提出 training-free 的测试时 unlearning 框架。通过对比对抗扰动的知识召回 vs. 知识擦除的 image-text pair，构造多模态擦除方向，在推理时动态 steering activation。
- **对 SLAM-ASR 的启示：** 如果不想修改 projector 参数，activation steering 可以在 projector 输出端操作——在推理时将特定知识的激活重定向。

### B5) SMFA: Sculpted Memory Forgetting Adapter for Selective MLLM Unlearning
- **作者：** Zhen Zeng, Leijiang Gu, Zhangling Duan, Feng Li, Zenglin Shi, Cees G. M. Snoek, Meng Wang
- **发表：** arXiv preprint, 2025
- **链接：** https://arxiv.org/abs/2511.20196
- **TL;DR：** 提出 "记忆遗忘适配器"（SMFA）：先微调模型用拒绝回复替换敏感响应（生成 forgetting adapter），再通过 retain anchor-guided masking 防止干扰无关知识和理解能力。同时提出 S-MLLMUn Bench 评估基准。
- **对 SLAM-ASR 的启示：** Adapter-based 遗忘 + masking 保护的策略可以直接应用于 projector 层：在 projector 上叠加一个 forgetting adapter，配合 masking 约束。

### B6) ViKeR: Visual-Guided Key-Token Regularization for MLLM Unlearning
- **作者：** Chengyi Cai, Zesheng Ye, Peike Li, Bo Han, Jianzhong Qi, Feng Liu
- **发表：** arXiv preprint, 2026
- **链接：** https://arxiv.org/abs/2601.22020
- **TL;DR：** 现有 MLLM unlearning 方法统一处理所有 answer token，忽略了不同 token 在遗忘过程中的重要性差异。ViKeR 利用不相关视觉输入来预测理想的 unlearning 后 token 分布，通过 information entropy 定义关键 token，进行 token 级梯度重加权。
- **对 SLAM-ASR 的启示：** 在 speech-LLM 中，不同的输出 token 对隐私泄露的贡献也不同。可以用 "不相关音频输入" 类比 "不相关视觉输入" 来指导 token 级 unlearning。

### B7) SafeEraser: Enhancing Safety in MLLMs through Multimodal Machine Unlearning
- **作者：** 多位作者
- **会议：** Findings of ACL 2025
- **链接：** https://arxiv.org/abs/2502.12520
- **TL;DR：** 构建了 3000 图片 + 28.8K VQA 对的安全 unlearning 基准。发现现有方法严重 over-forget（模型通用能力退化），提出 Prompt Decouple (PD) Loss 解耦 unlearning 目标与通用能力。
- **提出的指标：** Safe Answer Refusal Rate (SARR) 量化 over-forgetting。
- **对 SLAM-ASR 的启示：** Over-forgetting 在 speech 场景同样关键——unlearning 不应该让模型丧失正常识别能力。PD Loss 的解耦思路可迁移。

### B8) PEBench: Fictitious Dataset for MLLM Unlearning
- **作者：** Zhaopan Xu, Pengfei Zhou 等
- **发表：** arXiv preprint, 2025
- **链接：** https://arxiv.org/abs/2503.12545
- **代码：** https://pebench.github.io/
- **TL;DR：** 构建虚构人物+事件场景数据集评估 MLLM unlearning，发现 cross-concept interference：遗忘一个概念会意外退化关联概念在同一图像中的表现。
- **对 SLAM-ASR 的启示：** 在 speech 场景中，遗忘特定说话人信息可能干扰同一录音中的语言内容识别——类比的 cross-concept interference。

---

## C. 参数高效 Unlearning 方法论（与 Projector 操作相关）

### C1) ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning
- **链接：** https://arxiv.org/abs/2603.01792
- **TL;DR：** 非对称 LoRA 架构 + token entropy 指导的参数隔离。共享 LoRA 矩阵捕获高熵 token，在目标子域内做 token 级隔离。在 TOFU/WMDP/MUSE 上 forget quality >95%、utility >90%。
- **对 projector unlearning 的启示：** LoRA 级的非对称设计可以应用于 projector 的 adaptation。

### C2) Reducing Prompt Sensitivity in LLM-based ASR Through Learnable Projection *(相关但非 unlearning)*
- **作者：** Sergio Burdisso 等
- **发表：** arXiv preprint, 2026
- **链接：** https://arxiv.org/abs/2601.20898
- **TL;DR：** 发现 LLM-based ASR 的性能对 prompt 选择高度敏感。提出 prompt projector 模块——在不修改底层模型的情况下学习投影 prompt embedding 到 LLM 输入空间中更有效的区域。
- **与 unlearning 的关系：** 这篇本身不做 unlearning，但它揭示了 LLM-based ASR 中 projector 层对模型行为有决定性影响，从另一个角度证实了 projector 是 unlearning 的关键操作点。

---

## D. 综合对比表

| Paper | 领域 | Unlearning 在哪操作 | 涉及 Projector | 涉及 ASR/Speech | 年份 |
|---|---|---|---|---|---|
| **Liu (Interspeech 2025)** | LLM-based ASR | 模型级 | 未明确 | ✅ | 2025 |
| **Cheng & Amiri** | Speech (传统) | 模型级 | ❌ | ✅ | 2025 |
| **OrthoGrad** | 通用 (含 ASR) | 梯度空间 | 可应用 | ✅ | 2025 |
| **SineProject** ⭐ | Vision-Language | **Projector** | ✅ 核心贡献 | ❌ | 2025 |
| **Stable Forgetting** | Vision + Language | LoRA adapter | ✅ 间接 | ❌ | 2025 |
| **MMUnlearner** | Vision-Language | Weight saliency | ✅ 间接 | ❌ | 2025 |
| **MLLMEraser** | Vision-Language | Activation steering | ✅ 间接 | ❌ | 2025 |
| **SMFA** | Vision-Language | Adapter + masking | ✅ 适配器级 | ❌ | 2025 |
| **ViKeR** | Vision-Language | Token-level gradient | ❌ | ❌ | 2026 |
| **SafeEraser** | Vision-Language | Benchmark + PD Loss | ❌ | ❌ | 2025 |
| **PEBench** | Vision-Language | Benchmark | ❌ | ❌ | 2025 |
| **ALTER** | LLM | LoRA | 可应用 | ❌ | 2026 |

---

## E. Research Gap 分析：当前空白在哪里？

### 已经有答案的问题

1. **LLM-based ASR 有隐私记忆问题吗？** → 有（Liu, Interspeech 2025）
2. **语音数据的 unlearning 比文本/图像更难吗？** → 是（Cheng & Amiri, 2025）
3. **多模态 LLM 的 unlearning 会破坏 projector 对齐吗？** → 会（SineProject, 2025）
4. **有没有方法在 projector 级做稳定的 unlearning？** → 在视觉-语言域有初步解决方案（SineProject, Stable Forgetting）

### 明确的空白（defensible research gap）

| 空白 | 当前最近的工作 | 缺什么 |
|---|---|---|
| **在 SLAM-ASR 的 linear projector 上做 unlearning** | SineProject（视觉域）、Liu（ASR 但未聚焦 projector） | 没有任何工作专门研究 speech-LLM projector unlearning |
| **Speech projector 的 Jacobian 稳定性分析** | SineProject（视觉 projector） | 未迁移到 speech projector；speech projector 通常是纯线性的，动力学可能不同 |
| **说话人级 unlearning 在 LLM-based ASR 中** | Speech Unlearning（传统模型）| 未在 LLM-based ASR 架构中研究 |
| **Cross-modal interference：speech unlearning 对语言理解的影响** | PEBench（视觉 cross-concept interference）| 未在 speech-text 域研究 |
| **Projector unlearning 对流式 / 实时推理的影响** | 无 | 完全空白 |
| **多语言 / code-switching 场景下的 projector unlearning** | 无 | 完全空白 |

### 一句话定位

> **在 SLAM-ASR 类 speech-LLM 架构中，对 linear projector 做 unlearning 是一个架构上自然、成本上高效、但理论上未被研究的方向。视觉-语言域的 projector unlearning（尤其是 SineProject）提供了直接可迁移的方法论基础，但 speech projector 的特殊性（纯线性、语音的高维序列结构、说话人相关性）意味着不能简单照搬，需要独立研究。**

---

## F. 如果要做这个方向，建议的切入方式

### 方案 1: SineProject for Speech — 将 SineProject 迁移到 SLAM-ASR

**核心思路：** 在 SLAM-ASR 的 frozen linear projector 上增加正弦调制的可训练参数，在 unlearning 时保持语音-文本对齐的 Jacobian 谱条件数。

**实验设置：**
- 模型：SLAM-ASR（HuBERT + linear projector + Vicuna-7B）
- Forget set：特定说话人的语音数据 / 包含隐私信息的转写
- Retain set：其他说话人 / 通用测试集
- 评估：WER（utility）、membership inference attack（privacy）、对齐漂移（alignment stability）

**预期贡献：** 首次在 speech-LLM 中研究 projector-level unlearning 的对齐稳定性问题。

### 方案 2: 正交梯度投影（OrthoGrad）在 Projector 参数空间的应用

**核心思路：** SLAM-ASR 的 projector 参数量小（~18M），forget/retain 梯度在低维参数空间中冲突概率更高。OrthoGrad 的正交投影在这种设置下可能特别关键。

**优势：** 不需要修改 projector 架构，纯优化层面的方案。

### 方案 3: 基于 Activation Steering 的推理时 Unlearning

**核心思路：** 借鉴 MLLMEraser 的思路，在 projector 输出端构造 "擦除方向"（speech-text erasure direction），推理时动态 steering。

**优势：** 不修改任何参数，完全可逆；适合需要动态遗忘的部署场景。

### 方案 4: Adapter-Based Forgetting + Masking 保护

**核心思路：** 借鉴 SMFA，在 projector 上叠加一个 forgetting adapter，配合 retain anchor-guided masking 保护正常识别能力。

**优势：** 模块化设计，遗忘能力可以按需插拔。

---

## G. 推荐阅读顺序

### 第一层：理解问题存在性

1. **An Embarrassingly Simple Approach for LLM with Strong ASR Capacity** (SLAM-ASR 架构)
   https://arxiv.org/abs/2402.08846

2. **Unlearning LLM-Based Speech Recognition Models** (LLM-based ASR 的记忆 + unlearning)
   https://www.isca-archive.org/interspeech_2025/liu25b_interspeech.html

### 第二层：理解 projector 是 unlearning 的关键瓶颈

3. **SineProject: Machine Unlearning for Stable Vision–Language Alignment** ⭐
   https://arxiv.org/abs/2511.18444

4. **Stable Forgetting: Bounded Parameter-Efficient Unlearning in Foundation Models**
   https://arxiv.org/abs/2509.24166

### 第三层：理解 speech unlearning 的特殊难度

5. **Speech Unlearning** (语音 unlearning 问题定义)
   https://arxiv.org/abs/2506.00848

6. **OrthoGrad** (含 ASR benchmark 的通用 unlearning 方法)
   https://arxiv.org/abs/2503.02312

### 第四层：多模态 unlearning 的方法工具箱

7. **MMUnlearner** (几何约束 + saliency map)
   https://arxiv.org/abs/2502.11051

8. **MLLMEraser** (测试时 activation steering)
   https://arxiv.org/abs/2510.04217

9. **SMFA** (adapter-based forgetting + masking)
   https://arxiv.org/abs/2511.20196

10. **ViKeR** (token 级重要性引导)
    https://arxiv.org/abs/2601.22020

### 第五层：评估基准参考

11. **SafeEraser** (MLLM 安全 unlearning benchmark + over-forgetting 分析)
    https://arxiv.org/abs/2502.12520

12. **PEBench** (MLLM 人物实体 unlearning + cross-concept interference)
    https://arxiv.org/abs/2503.12545

---

## H. 总结判断

1. **"在 SLAM-ASR 的 linear projector 上做 unlearning" 是一个有明确 gap 的方向。** 目前没有任何论文专门研究这个问题。

2. **最重要的迁移来源是 SineProject。** 它首次证明了 projector 是多模态 LLM unlearning 的关键瓶颈，其 Jacobian 稳定性分析和正弦调制解决方案可以直接迁移到 speech-LLM。

3. **最重要的基线和动机来源是 Liu (Interspeech 2025)。** 它证明了 LLM-based ASR 的记忆和 unlearning 需求，但没有深入到 projector 层面。

4. **语音的特殊性创造了独立的研究空间：**
   - 语音是高维、序列化、说话人依赖的——unlearning 的几何结构可能不同于视觉；
   - SLAM-ASR 的 projector 是纯线性的（vs. 视觉 MLLM 常用 MLP projector）——理论分析更简洁但约束更强；
   - 语音数据的隐私要求（说话人身份、生物特征信息）比视觉和文本更紧迫。

5. **这个方向的 novelty 可以这样表述：**

> Prior work on multimodal LLM unlearning (SineProject, MMUnlearner, etc.) focuses on vision-language alignment, while LLM-based ASR unlearning (Liu, 2025) does not differentiate the projector as a distinct unlearning target. We study projector-level unlearning in speech-LLM architectures, where the linear projector is the sole trained component and the only cross-modal bridge, making it both the natural and the most fragile point for selective forgetting.
