# Introduction 写作逻辑：Plug-and-Play Audio-Embedding Corrupted Prompts for LLM-based ASR

这份文档整理 Introduction 的段落逻辑、每段该讲什么、段与段之间如何衔接，以及每段可以引哪些文献。

---

## 总体叙事主线（一句话）

> **LLM-based ASR 面临隐私遗忘需求 → 现有 ASR unlearning 方法依赖权重更新，代价大且伤 utility → 文本 LLM 侧已有 inference-time embedding corruption 方案 → 但没人把它迁移到 audio embedding 上 → 我们提出 classifier-gated audio prompt corruption，第一个 plug-and-play、inference-time 的 speech-LLM forgetting 方法。**

Introduction 的核心任务是让读者在 5 段之内走完这条线。

---

## 段落结构

### ¶1 — 背景：LLM-based ASR 的崛起与 speech-LLM 架构

**目标**：建立技术背景，让读者知道你在什么架构上做文章。

**逻辑**：
1. LLM-based ASR 已成为语音识别的重要新范式（区别于传统 E2E ASR）。
2. 其中一个特别有吸引力的架构路线是 **speech encoder + projector + frozen LLM**（即 SLAM-ASR 风格）：
   - 结构简单：只需要训练一个轻量 projector；
   - 效果已被证明可以很强；
   - projector 是一个天然的、可插拔的跨模态接口层。
3. 这类系统的部署正在加速，覆盖了 long-form transcription、contextual ASR、多语言识别等场景。

**关键信号词**：用 "recently" / "a growing body of work" 开头，点明趋势。

**可引文献**：
- SLAM-ASR / "An Embarrassingly Simple Approach for LLM with Strong ASR Capacity" (arXiv:2402.08846) — 证明 projector-only 就能做强 ASR
- Seed-ASR (arXiv:2407.04675) — 代表 context-aware LLM-based ASR
- "A Comprehensive Solution to Connect Speech Encoder and LLM for ASR" (arXiv:2406.17272) — 系统训练 recipe
- Speech ReaLLM (arXiv:2406.09569) — streaming speech-LLM
- Survey: "Recent Advances in Speech Language Models" (arXiv:2410.03751)

**段末过渡**：

> 然而，随着 LLM-based ASR 被部署到越来越多的场景中，一个关键问题浮现：**这些系统是否也继承了 LLM 的隐私风险——记忆了训练数据中的敏感语音内容？**

---

### ¶2 — 问题：LLM-based ASR 的记忆与隐私遗忘需求

**目标**：引出 forgetting / unlearning 需求，说明为什么这件事在 ASR 场景里特别重要且特别难。

**逻辑**：
1. LLM 会记忆训练数据，这在文本 LLM 上已被广泛研究。
2. 这种记忆风险在 LLM-based ASR 中同样存在——模型可能记住说话人身份、私密对话内容、电话号码、地址等敏感信息。
3. GDPR / RTBF（Right to Be Forgotten）等法规要求模型能够按需遗忘特定数据。
4. Machine unlearning 是当前解决此问题的主要技术路线，已在文本 LLM 上积累了大量研究。
5. **但 LLM-based ASR 的 unlearning 才刚刚起步**——目前仅有极少数工作直接探讨这个问题。

**关键技巧**：这一段要在"文本 LLM unlearning 研究很多"和"ASR unlearning 研究极少"之间制造反差（contrast），让读者感受到 gap 的存在。

**可引文献**：
- "Unlearning LLM-Based Speech Recognition Models" (Interspeech 2025) — 目前最直接的 LLM-based ASR unlearning
- "Rethinking Machine Unlearning for LLMs" (Nature Machine Intelligence 2025, arXiv:2402.08787) — 文本 LLM unlearning 综述
- TOFU (arXiv:2401.06121) — 代表 LLM unlearning benchmark
- WMDP (arXiv:2403.03218) — 安全导向 unlearning benchmark
- MUSE (arXiv:2407.06460) — 多维度评估

**段末过渡**：

> 现有的少量 ASR unlearning 工作采用的方案——如梯度上升、LoRA 微调、adapter 更新——**都需要对模型执行后训练更新**。这带来两个根本性问题……

---

### ¶3 — 现有方法的局限：权重更新 unlearning 的代价

**目标**：说明当前 ASR unlearning 方法（以及更广泛的 LLM unlearning 方法）的核心痛点，为你的方案创造动机。

**逻辑**：
1. **代价问题**：权重更新（无论是 full fine-tuning、LoRA、还是 adapter-only）都需要重新训练，对于大规模部署的 ASR 系统来说成本不可忽视。
2. **Utility 损伤问题**：更严重的是，后训练更新几乎不可避免地伤害模型对非遗忘数据的识别能力。文本 LLM unlearning 文献已大量记录了这种"知识腐蚀"（knowledge corruption / collateral damage）——遗忘操作会波及无关知识。
3. **不可逆问题**：权重一旦更新，就难以精确回退；如果遗忘请求后来被撤回，或判定有误，恢复原模型能力的成本很高。
4. **不可组合问题**：如果要处理多个独立的遗忘请求，权重更新方法通常需要依次执行或联合训练，难以实现"按需插拔"。

**关键技巧**：这段的核心修辞目标是让读者心里产生一个问题——**有没有一种方法可以不改原模型就实现 forgetting？** 你接下来就要回答这个问题。

**可引文献**：
- "Does Unlearning Truly Unlearn?" (arXiv:2411.12103) — 通用能力退化
- "Probing Knowledge Holes in Unlearned LLMs" (NeurIPS 2025, arXiv:2511.00030) — 知识空洞
- EGUP (arXiv:2508.20443) — 知识纠缠导致附带损伤
- CLReg (arXiv:2601.22028) — logit 操作无法消除潜空间纠缠
- CIR (arXiv:2509.11816) — unlearning 擦除的是共享特征而非事实特异子空间

**段末过渡**：

> 一个自然的问题是：**能否在不修改模型任何参数的前提下，仅通过 inference-time 的干预实现选择性遗忘？** 事实上，在文本 LLM 领域，已有工作给出了肯定的回答。

---

### ¶4 — 启发与 Gap：ECO 的成功与 audio 侧的空白

**目标**：这是 Introduction 最关键的一段。你需要做三件事：（1）引入 ECO 作为方法论灵感来源；（2）指出 audio 侧的空白；（3）说明为什么 SLAM-ASR 的 projector 接口让这种迁移是可行的。

**逻辑**：
1. **ECO 的成功**：Large Language Model Unlearning via Embedding-Corrupted Prompts (NeurIPS 2024) 提出了一种优雅的推理时遗忘方案：
   - 训练一个轻量 prompt classifier，判断当前输入是否属于 forget target；
   - 若命中，对 text prompt embeddings 注入学习得到的 corruption；
   - LLM 主体完全冻结，无需任何权重更新；
   - 实现了 plug-and-play、低副作用的 inference-time unlearning。
2. **但 ECO 只在文本 LLM 上做**：它操作的是 text prompt embeddings，完全没有涉及 audio embeddings 或 ASR 场景。
3. **SLAM-ASR 的 projector 提供了天然的迁移条件**：
   - 在 speech encoder + projector + frozen LLM 架构中，projected audio prompts 在形式上与 text prompt embeddings 高度类似——都是 LLM 输入空间中的 embedding 序列。
   - Acoustic Prompt Tuning 等工作已证明 audio embeddings 可以作为 soft prompts 注入 LLM。
   - 因此，projector 输出的 audio prompt embeddings 就是一个天然的 corruption intervention point。
4. **Gap 的精确表述**：

   > 截至目前，没有公开论文将 ECO 风格的 classifier-gated embedding corruption 从文本端迁移到 audio embedding 端，用于 LLM-based ASR 的 plug-and-play forgetting。

**关键技巧**：

这段的结构是经典的 **"三条证据线存在，但交叉点是空白"** 论证：

```
证据线 1：文本 LLM 有 inference-time embedding corruption unlearning (ECO)
证据线 2：LLM-based ASR 有 unlearning 需求和初步工作
证据线 3：Speech-LLM 架构中 audio embeddings / projector 是可控的接口层
               ↓
         交叉点 = 你的论文
```

这种"三线合一"的叙事是 novelty 说服力最强的 Introduction 结构之一，因为：
- 你不是在 claim "从零发明"；
- 而是在 claim "跨领域迁移 + 新干预位点 + 新任务形态"；
- Reviewer 会觉得这个故事合理、可信、好验证。

**可引文献**：
- **ECO**: "Large Language Model Unlearning via Embedding-Corrupted Prompts" (NeurIPS 2024, arXiv:2406.07933) — 核心灵感来源
- "Acoustic Prompt Tuning: Empowering LLMs with Audition Capabilities" (TASLP 2025, arXiv:2312.00249) — audio embeddings 作为 soft prompts
- SLAM-ASR (arXiv:2402.08846) — projector 是可控接口
- "Reducing Prompt Sensitivity in LLM-based Speech Recognition Through Learnable Projection" (ICASSP 2026, arXiv:2601.20898) — projector 可独立训练且稳定

**段末过渡**：

> 基于以上观察，我们提出 [方法名]，第一个将 classifier-gated embedding corruption 引入 LLM-based ASR 的 inference-time forgetting 方法。

---

### ¶5 — 我们的方案与贡献

**目标**：简洁、清晰地说明你做了什么，以及 contribution 是什么。

**逻辑**：
1. **一句话方法概述**：我们在 speech encoder + projector + frozen LLM 的 ASR 系统上，仅新增两个轻量模块——(1) audio embedding classifier 和 (2) corruption module——在推理时对命中 forget target 的 projected audio prompts 注入可学习的 corruption，使 LLM 输出"从未学过该知识"的行为，同时保持非 forget target 的 ASR 性能不受影响。
2. **核心卖点强调**：
   - 完全 plug-and-play：不修改 speech encoder、projector、LLM 的任何参数；
   - Inference-time：按需触发，可随时启用/禁用；
   - Utility-preserving：只在 classifier 命中时干预，非 forget 数据理论上不受扰动。
3. **Contributions 列表**（通常 3-4 条）：

> Our contributions are as follows:
>
> 1. **新问题/新视角**：We identify inference-time audio embedding corruption as a viable forgetting mechanism for LLM-based ASR, bridging text-side embedding-corrupted unlearning and speech-LLM architectures.
> 2. **方法设计**：We propose [方法名], a plug-and-play framework consisting of a classifier for forget-target detection at the audio embedding level and a learnable corruption module that operates on projected audio prompts without modifying any base model parameter.
> 3. **Utility 分析**：We demonstrate that classifier-gated corruption preserves retain-set ASR utility (WER/CER) significantly better than weight-update baselines, while achieving comparable forgetting effectiveness.
> 4. **系统评估**：We conduct comprehensive evaluation covering forgetting success, utility preservation, classifier reliability, and robustness to acoustic perturbations, establishing the first benchmark for inference-time forgetting in speech-LLM ASR.

---

## 整体流程图

```
¶1  LLM-based ASR 很重要，SLAM-ASR 架构是关键
       │
       ▼  （但是……）
¶2  这些系统有记忆/隐私风险，需要 unlearning，而 ASR unlearning 才刚起步
       │
       ▼  （现有方法怎么做的？）
¶3  现有方法靠权重更新 → 代价大 + 伤 utility + 不可逆 + 不可组合
       │
       ▼  （有没有更好的方式？）
¶4  文本侧已有 ECO (inference-time embedding corruption) → 但没人迁移到 audio 侧
    + SLAM-ASR 的 projector 提供了天然迁移条件 → GAP！
       │
       ▼  （所以我们做了什么？）
¶5  我们提出 classifier-gated audio prompt corruption → contributions
```

---

## 写作注意事项

### 1. Positioning 选择

**强烈建议**：不要把论文定位成 "strict machine unlearning"，而是定位成：

> **Inference-time forgetting** / **plug-and-play forgetting approximation**

原因：
- 你的方法不修改模型参数，底层知识仍存在；
- 如果关闭 classifier / corruption，知识可恢复；
- Reviewer 一定会追问 "这是真 unlearning 还是 suppression？"

用 "forgetting" 而非 "unlearning" 可以避免这场不必要的定义争论。如果实验结果特别强（比如能通过 membership inference 等更严格的测试），再往 "unlearning" 靠也不迟。

### 2. Novelty 表述方式

**不要说**：
- "我们首次在 ASR 上用了 embedding corruption" ← 太窄，像 trivial extension

**应该说**：
- "我们首次将 inference-time、classifier-gated 的 embedding corruption 机制从文本 LLM 迁移到 speech-LLM ASR，利用 projected audio prompts 作为干预位点，实现了无需任何权重更新的 plug-and-play forgetting"

区别在于后者同时包含了**迁移方向 + 干预位点 + 架构约束 + 核心性质**，这比单纯说 "用了 embedding corruption" 要有说服力得多。

### 3. 主动回应 "是否是真 unlearning" 的质疑

在 Introduction 的 ¶5（贡献部分）或紧随其后，可以加一句前置防御：

> We note that our method provides an inference-time forgetting approximation rather than parameter-level knowledge deletion. We view this as complementary to weight-update approaches: our method offers rapid, reversible, and composable forgetting with minimal utility degradation, while weight-update methods provide more permanent deletion guarantees.

这句话的好处：
- 诚实地承认了边界；
- 把 "缺点" 重新框定为 "互补性"；
- Reviewer 会更难以此为由 reject。

### 4. 标题与关键术语一致性

在 Introduction 中，统一使用以下术语：
- **projected audio prompts**（而非 "audio embeddings"，因为更精确地指向 projector 输出）
- **classifier-gated corruption**（强调不是盲目 corruption，而是有选择性的）
- **plug-and-play forgetting**（而非 "unlearning"，除非实验特别强）
- **speech-LLM ASR** 或 **LLM-based ASR**（而非 "SLAM-ASR"，后者是一个具体模型名）

### 5. 引文策略

Introduction 中引文量一般控制在 15-25 篇。按上述 5 段结构，分配大致如下：

| 段落 | 引文数 | 主要引文类型 |
|---|---|---|
| ¶1 背景 | 4-6 篇 | LLM-based ASR 代表作 + survey |
| ¶2 问题 | 4-5 篇 | LLM unlearning 综述 + ASR unlearning + benchmark |
| ¶3 局限 | 3-5 篇 | 知识腐蚀 + utility 退化 |
| ¶4 Gap | 3-4 篇 | ECO + acoustic prompt tuning + SLAM-ASR projector |
| ¶5 贡献 | 0-1 篇 | 通常不引文，或仅引自己方法的最近前置工作 |

---

## 一个可直接参考的 Introduction 骨架（英文）

以下是一个可以直接在上面改写的段落级骨架。方括号内是需要填充具体内容的位置。

---

**¶1**
Recent advances in large language models (LLMs) have catalyzed a new paradigm in automatic speech recognition (ASR), where a speech encoder is coupled with a decoder-only LLM to perform end-to-end transcription [refs]. Among the various architectures explored, the *speech encoder + linear projector + frozen LLM* pipeline—exemplified by SLAM-ASR [ref]—has emerged as a particularly attractive design due to its simplicity: by training only a lightweight projector while keeping both the speech encoder and the LLM frozen, it achieves competitive ASR performance with minimal computational overhead [refs]. This architectural simplicity also means that the projector output—a sequence of projected audio prompts in the LLM's embedding space—serves as a clean, modular interface between the acoustic and linguistic components.

**¶2**
As LLM-based ASR systems are increasingly deployed in real-world applications, they inherit a well-documented vulnerability of LLMs: unintended memorization of training data [refs]. In the speech domain, this raises acute privacy concerns—models may retain and reproduce sensitive information such as speaker identities, private conversations, phone numbers, or medical dictations. Regulatory frameworks including GDPR's Right to Be Forgotten mandate the ability to remove specific data influence from deployed models. While machine unlearning has received extensive attention for text LLMs [refs], its investigation in LLM-based ASR remains nascent, with only [one/a handful of] work(s) directly addressing this setting [ref: Interspeech 2025].

**¶3**
Existing approaches to LLM-based ASR unlearning—and LLM unlearning more broadly—rely on post-training weight updates such as gradient ascent, LoRA fine-tuning, or adapter retraining [refs]. While effective at reducing target memorization, these methods entail significant drawbacks: (i) *computational cost*, as each forget request requires a retraining pass; (ii) *utility degradation*, as weight updates inevitably perturb the model's performance on non-target data—a phenomenon extensively documented as knowledge corruption or collateral damage [refs]; (iii) *irreversibility*, as parameter modifications are difficult to undo precisely; and (iv) *non-composability*, as handling multiple independent forget requests typically requires sequential or joint retraining rather than modular insertion.

**¶4**
A natural question arises: *can selective forgetting be achieved at inference time, without modifying any model parameters?* For text LLMs, a recent affirmative answer is provided by Embedding-Corrupted Prompts (ECO) [ref], which trains a lightweight prompt classifier to detect forget-target inputs and injects learnable corruption into their text prompt embeddings, achieving plug-and-play unlearning with minimal side effects. However, ECO operates exclusively on text prompt embeddings and has not been explored in the speech domain. We observe that the projected audio prompts in speech-LLM ASR architectures occupy a structurally analogous role to text prompt embeddings—both are embedding sequences consumed by the same frozen LLM—and that prior work on acoustic prompt tuning [ref] has demonstrated the viability of treating audio embeddings as soft prompts. This structural parallel suggests that *classifier-gated corruption of projected audio prompts* could serve as an inference-time forgetting mechanism for LLM-based ASR. To the best of our knowledge, no prior work has explored this direction.

**¶5**
In this paper, we propose **[Method Name]**, the first inference-time, plug-and-play forgetting framework for LLM-based ASR. Our method introduces two lightweight modules on top of a frozen speech-LLM ASR system: (1) an *audio embedding classifier* that determines whether the current input matches a forget target, and (2) a *corruption module* that, upon classifier activation, injects learnable perturbations into the projected audio prompts to steer the LLM's output toward a "never-learned" behavior. Crucially, our approach requires no modification to the speech encoder, projector, or LLM parameters, and can be activated or deactivated on demand. Our contributions are:

- We identify projected audio prompts as a viable intervention point for inference-time forgetting in speech-LLM ASR, bridging the gap between text-side embedding-corrupted unlearning and speech-LLM architectures.
- We design a classifier-gated corruption framework that [具体方法细节].
- We demonstrate that our approach preserves retain-set utility (WER/CER) significantly better than weight-update baselines while achieving comparable forgetting effectiveness.
- We establish a comprehensive evaluation protocol covering forgetting success, utility preservation, classifier reliability, and robustness to acoustic perturbations.

---

## 写作顺序建议

Introduction 通常不是最先写的部分。建议的写作顺序：

1. **Method** — 先把方法确定下来
2. **Experiments** — 确定实验设置和 baseline
3. **Introduction** — 方法和实验定了之后，回头写 intro 更容易 claim 准确
4. **Related Work** — 最后写，因为它是 Introduction gap 论述的展开版

如果要先写 Introduction 来理清思路，可以先按上面的骨架写一个 draft，后续根据实验结果再回来调整 claim 的强度。
