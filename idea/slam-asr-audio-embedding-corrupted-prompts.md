# Idea: Plug-and-Play Audio-Embedding Corrupted Prompts for LLM-based ASR

这份笔记回答一个很具体的问题：

> **有没有人已经做过：在 LLM-based ASR 里，不改原模型主体，只额外加一个 audio embedding classifier，然后对命中的 audio embeddings / soft prompts 做 corruption，从而实现 forgetting，同时尽量不伤原始 ASR utility？**

我的当前判断是：

> **截至 2026-03-26，我还没有检索到公开论文直接把这整套组合完整做出来。**
>
> 但已经存在三条非常接近的证据链：
>
> 1. **文本 LLM unlearning** 里已经有 **Embedding-Corrupted Prompts (ECO)** 这类 **classifier-gated, inference-time, plug-and-play forgetting**；
> 2. **LLM-based ASR** 里已经有直接的 **unlearning** 论文；
> 3. **Speech encoder + LLM** 路线里已经证明 **audio embeddings / projector / soft audio prompts** 是可插拔、可单独训练、可稳定操控的接口层。
>
> **缺的正是把这三条线合起来的那一步。**

---

## 1. 一句话结论

如果把你的想法压缩成一句话，我会这样概括：

> **你这个 idea 很像把 ECO-style embedding corruption 从 text LLM 迁移到 SLAM-ASR / speech-LLM ASR，并用一个 audio embedding classifier 决定何时触发 corruption；我目前没有看到公开文献已经把这个组合完整做过。**

更细一点说：

- **有人做过 LLM unlearning via corrupted prompt embeddings**，但对象是 **text prompts**，不是 ASR 的 audio embeddings；
- **有人做过 LLM-based ASR unlearning**，但方法是 **post-training unlearning / gradient-based update / PEFT**，不是 **inference-time corruption + classifier gating**；
- **有人做过 audio prompt / acoustic prompt / projector-only speech-LLM**，但目标是 **接入音频、增强鲁棒性或做任务适配**，不是 **forgetting**。

所以最准确的文献判断不是“完全没人碰过相关点”，而是：

> **相关零件都已经有人做过，但你这个“audio classifier + corrupted audio prompts + plug-and-play ASR forgetting + utility preservation”组合点，目前看起来仍然是 research gap。**

---

## 2. 你的 idea 可以怎么 formalize

一个比较清楚的方法表述可以写成：

### 方法草图

给定一个 LLM-based ASR 系统：

- speech encoder `E`
- projector / adapter `P`
- decoder-only LLM `M`

在推理时额外加入两个轻量模块：

1. **audio embedding classifier** `C`
   - 输入：encoder 输出的 audio embeddings，或 projector 后的 speech-to-LLM embeddings
   - 输出：当前输入是否属于 **forget target**

2. **corruption module** `Delta`
   - 当 `C` 判定命中 forget target 时，
   - 对 audio embeddings / projected audio prompts 注入可学习 corruption，
   - 使 LLM 的输出更接近“该知识从未学过”的状态，或者更接近“安全替代输出 / 空白输出 / refusal-like output”。

### 两个最自然的实现位点

#### 方案 A：pre-projector corruption

- 在 speech encoder 输出的 frame-level embeddings 上动手；
- 更接近“音频端遗忘”；
- 但对时序对齐和局部定位要求更高。

#### 方案 B：post-projector corruption

- 在映射到 LLM token space 之后的 audio prompt embeddings 上动手；
- 概念上更接近 **ECO prompts**；
- 也更容易直接继承 SLAM-ASR 的“接口层可控”叙事。

如果你想让论文故事更干净，我会更推荐先从 **post-projector corruption** 起步，因为它和：

- ECO 的 embedding corruption，
- SLAM-ASR 的 projector-only interface，
- “plug-and-play, no base-model update”

三者的逻辑更一致。

---

## 3. 最相关的文献：谁已经做了什么

下面是最值得盯的几篇。

| 论文 | 直接回答了什么 | 和你 idea 的关系 | 还缺什么 |
|---|---|---|---|
| **Large Language Model Unlearning via Embedding-Corrupted Prompts** (arXiv 2406.07933, NeurIPS 2024) | 用 **prompt classifier** 识别要 forget 的文本 prompt，并在推理时对 prompt embeddings 加 corruption，实现 **plug-and-play / inference-time unlearning**，且 side effects 很小。 | 这是你 idea 最接近的 **方法论原型**。 | 它是 **text LLM**，不是 ASR；corruption 的对象是 **text prompt embeddings**，不是 **audio embeddings**。 |
| **Unlearning LLM-Based Speech Recognition Models** (Interspeech 2025) | 直接说明 **LLM-based ASR 的 unlearning** 是值得研究且可行的，并讨论 privacy-utility trade-off。 | 这是你 idea 最接近的 **任务原型**。 | 它不是 classifier-gated inference-time 方法，也不是 audio prompt corruption。 |
| **An Embarrassingly Simple Approach for LLM with Strong ASR Capacity** (arXiv 2402.08846) | 证明 **speech encoder + frozen LLM + only trainable linear projector** 就能做很强的 ASR。 | 说明 speech-to-LLM 的接口层本身就是一个非常干净的 intervention point。 | 这篇不是 unlearning。 |
| **Acoustic Prompt Tuning: Empowering Large Language Models with Audition Capabilities** (arXiv 2312.00249; TASLP 2025) | 证明可以把 **audio embeddings 当作 soft prompts** 注入 LLM。 | 说明“audio prompt / audio soft prompt”这个操作本身是成立的。 | 目标是音频能力接入，不是 forgetting。 |

---

## 4. 为什么我认为这还是一个明确 research gap

你的 idea 同时要求满足 4 个条件：

1. **对象是 LLM-based ASR**
2. **forgetting 发生在 audio embedding / audio prompt 层**
3. **只新增 classifier + corruption module，不改原模型主体**
4. **目标是 forget 成功，同时尽量不伤原模型 utility**

现在公开文献里：

- **ECO** 满足 **(2)(3)(4)**，但不满足 **(1)**；
- **ASR unlearning** 满足 **(1)(4)**，但不满足 **(2)(3)**；
- **SLAM-ASR / APT** 满足 **(1)(2)** 的结构前提，但不满足 **forgetting**。

所以 gap 很清楚：

> **还没有公开论文把“ASR-specific forgetting”与“embedding-corruption-style inference-time intervention”真正接起来。**

---

## 5. 这个 idea 的 novelty 应该怎么讲

我觉得最好的 novelty 不是：

- “我们也用了 embeddings”
- “我们也用了 prompt classifier”
- “我们也做 unlearning”

这些单点都不够新。

更强的 claim 应该是：

> **We propose classifier-gated audio embedding corruption for plug-and-play forgetting in LLM-based ASR, achieving unlearning-like behavior without updating the base speech encoder, projector, or LLM.**

进一步强化后，可以讲成：

> **Prior work studies text-side embedding-corrupted prompts for LLM unlearning and post-training updates for LLM-based ASR unlearning separately. We bridge them by introducing an audio-side, inference-time, plug-and-play forgetting mechanism for speech-LLM ASR.**

这个表述的优点是：

- 你不是在 claim “从零到一发明 embedding unlearning”；
- 而是在 claim **跨领域迁移 + 新任务形态 + 新干预位点**；
- 这个 novelty 更稳，也更容易说服 reviewer。

---

## 6. 这件事最大的优点

### 优点 1：非常符合 plug-and-play 叙事

如果方法真的只需要：

- 一个 classifier
- 一个 corruption vector / corruption subspace / corruption generator

那它的故事会非常干净：

- 不需要 full retraining；
- 不需要改 ASR backbone；
- 不需要重新训练整个 speech-LLM；
- 推理时按需触发；
- 很适合作为部署层的“forget filter”。

### 优点 2：很有希望保住原模型 utility

这也是你 idea 最吸引人的地方。

如果 corruption 只在 classifier 命中时触发，那么 retain data 理论上不会被频繁扰动，因此很可能比：

- full fine-tuning unlearning
- LoRA-based unlearning
- projector-only weight update

更容易保持整体 WER / CER / decoding fluency。

### 优点 3：天然适合多 backbone 迁移

如果 corruption 作用在一个比较标准化的 audio prompt space 上，那么它可能有机会迁移到：

- 不同 speech encoders，
- 不同 projectors，
- 不同 LLM decoders，
- 甚至不同 speech-LLM ASR backbone。

这和 ECO 的“跨 0.5B-236B LLM 可扩展”叙事是平行的。

---

## 7. 但 reviewer 最可能质疑什么

这里我觉得要提前警惕 5 个问题。

### 风险 1：这到底算不算“真 unlearning”？

这是最大问题。

因为如果你不更新 base model，只是在 inference 时屏蔽 / 扰乱一部分 audio prompts，reviewer 很可能会说：

> 这更像 **suppression / filtering / access control / output steering**，而不是严格意义上的 machine unlearning。

所以你需要提前决定你的 positioning：

#### Position A：把它写成 strict unlearning

风险较大，因为 reviewer 会追问：

- 底层知识真的被删了吗？
- 还是只是当前输入通路被阻断了？
- 若关闭 classifier / corruption，知识是不是立刻又回来了？

#### Position B：把它写成 unlearning-inspired forgetting layer

更稳一些，例如：

> plug-and-play forgetting for LLM-based ASR

或者：

> inference-time unlearning approximation for speech-LLM ASR

我个人更建议先走 **Position B**，因为可 defend 性更强。

### 风险 2：audio target 的定义必须非常清楚

你的 classifier 要判断“该忘什么”，这件事在语音里比文本更难。

你至少要先定一个 threat model：

1. **speaker-level forgetting**
   - 忘某个说话人
   - classifier 更容易做
   - 但更像 speaker deletion

2. **phrase / content-level forgetting**
   - 忘某些私密短语、实体名、电话号码、地址
   - 更贴近 privacy
   - 但 classifier 难度更高

3. **sample-level memorization forgetting**
   - 忘训练集中的具体录音样本
   - 最接近 data deletion
   - 也是最难做的

如果是第一篇 paper，我会建议优先做：

> **phrase-level 或 speaker-level**，

因为这两种设置更容易让 classifier 真正学到稳定边界。

### 风险 3：语音是连续时序信号，forget target 可能只出现在局部片段

文本 prompt 往往整段都与 forget query 对齐，但语音输入可能只有 1 秒里出现了敏感词。

所以你要回答：

- classifier 是对整段音频判别，还是 frame/chunk 判别？
- corruption 是加在整段 embeddings 上，还是只加在命中的局部片段上？

如果这点不处理好，副作用会很大。

### 风险 4：false positives 会直接伤 utility

你 claim 的核心是“**不影响原本 model utility**”。

那就意味着 classifier 的误报率必须非常低，否则 benign audio 也会被错误 corruption。

这个方向里，classifier 其实不是配角，而是决定 utility 的关键模块。

### 风险 5：可能只是切断表达，不是消除记忆

即便结果表面上成功了，也要回答：

- 若攻击者绕过 classifier，知识是否仍然存在？
- 若换一种声学变体、口音、说话速度，是否还能恢复出来？
- 若把同一语义内容换成文本 prompt + audio context，是否仍能泄漏？

这决定了你方法更像：

- **robust forgetting**
- 还是 **narrow-path blocking**

---

## 8. 我会怎么建议你定义第一版 paper

如果想把故事做得最清楚，我建议第一版不要一上来 claim 太大，而是把 paper 收窄成：

### 推荐问题定义

> **Can classifier-gated corruption of projected audio prompts provide plug-and-play forgetting behavior in LLM-based ASR while preserving retain-set utility better than weight-update baselines?**

这个问题有 3 个好处：

1. 避开“严格 machine unlearning”定义争议；
2. 突出你的核心卖点：**plug-and-play + utility preservation**；
3. baseline 非常自然。

---

## 9. 一个可执行的方法设计

### 9.1 模型结构

从一个标准 speech-LLM ASR 系统出发：

- frozen speech encoder `E`
- frozen projector `P` 或只保留预训练好的 `P`
- frozen LLM decoder `M`

新增：

- **binary / multi-label classifier** `C`
- **corruption module** `Delta`

### 9.2 classifier 输入可以怎么选

#### 选项 A：encoder outputs

- 优点：更接近 acoustic evidence；
- 缺点：维度高、帧长、边界难对齐。

#### 选项 B：projected audio prompts

- 优点：更贴近 LLM 输入空间；
- 缺点：可能已丢掉一部分细粒度声学信息。

如果你的主叙事是“corrupted prompts”，我会优先选 **projected audio prompts**。

### 9.3 corruption 可以怎么做

可以从简单到复杂做三档：

#### 最简单：additive corruption vector

对命中的 audio prompt embeddings 加一个学习得到的 `delta`。

#### 中等：subspace projection / redirection

把命中的 embeddings 从某个 forget-sensitive subspace 投影掉，或重定向到一个安全子空间。

#### 更强：conditioned corruption generator

由 classifier 的输出决定使用哪一种 corruption pattern，例如不同 forget class 对应不同 `delta_k`。

第一版论文最稳的起点通常是：

> **classifier + class-specific additive delta / low-rank delta**

因为实现简单、变量少、容易做消融。

---

## 10. 最该做的 baseline

我觉得至少要有下面几类。

### Baseline 1：No forgetting

原始 LLM-based ASR 模型。

### Baseline 2：Text-side refusal / output filtering

不动 audio embeddings，只在 decoder 输出端做规则或文本过滤。

这能证明你的改动不是“任何后处理都一样”。

### Baseline 3：Weight-update ASR unlearning

参考 **Unlearning LLM-Based Speech Recognition Models** 的 post-training 方案。

这是最重要的对照组，因为它代表当前 ASR unlearning 主流叙事。

### Baseline 4：Projector-only weight update

这是最自然的轻量基线：

- 只更新 projector，
- 不动 encoder 和 LLM。

如果你的方法比这个更保 utility，那故事就很强。

### Baseline 5：Random corruption / zero-out corruption

证明你的 corruption 不是“随便扰动一下都有效”。

---

## 11. 评价指标应该怎么设计

这里不能只看普通 WER。

### 11.1 Forgetting side

- forget-set target phrase recall
- forget-set exposure / extraction success
- targeted entity recovery rate
- ASR memorization leakage rate

如果你沿用 ASR unlearning 论文的设置，还可以考虑：

- WER-based memorization rate
- prompting-based memorization rate
- homophone-based memorization rate

### 11.2 Utility side

- retain-set WER / CER
- long-form WER
- rare-word recognition on benign words
- speaker / accent robustness on non-target audio
- latency overhead

### 11.3 Classifier side

- false positive rate
- false negative rate
- AUROC / AUPRC
- calibration under domain shift

### 11.4 Robustness side

- speech perturbation robustness
  - speed perturbation
  - noise
  - reverberation
  - accent shift
- paraphrase-like robustness
  - same entity in different acoustic realization
- bypass robustness
  - if the target is spoken indirectly or with disfluency, does forgetting still trigger?

---

## 12. 我觉得这个 idea 最值得回答的 4 个研究问题

### Q1. Audio-side corruption 是否能逼近“未学过该知识”的输出？

这是最核心的问题。

如果命中 forget target 后，模型输出与“从未见过该知识的模型”更接近，那你的故事会比“只是输出坏掉了”强很多。

### Q2. 它是否比权重更新方法更保 utility？

这是你最强的卖点。

如果：

- forget 成功率接近，
- retain WER 更稳，
- 延迟成本更低，

那么这个方法的意义就很明确。

### Q3. 忘记是否定位在局部 audio prompts 上更有效？

整段 corruption 和局部 chunk corruption 的比较会很有信息量。

### Q4. 这到底是 forgetting 还是 blocking？

你最好主动回答，而不是等 reviewer 来问。

比如可以在文中明确承认：

> 我们的方法更接近 inference-time forgetting approximation，而不是 parameter-level deletion。

这反而会让论文更诚实、更好 defend。

---

## 13. 我会怎么给这个想法下最终判断

### 文献层面的最终判断

> **我目前没有找到公开论文直接做：classifier-gated corruption of audio embeddings / projected audio prompts for plug-and-play forgetting in LLM-based ASR.**

### 创新性判断

> **这个点有创新性，但创新点不在“embedding”本身，而在“ASR + audio-side + classifier-gated + inference-time forgetting + utility-preserving”这个组合。**

### 风险判断

> **最大的风险不是方法能不能跑，而是 reviewer 会不会认为这只是 suppression / filtering，而不是真 unlearning。**

### 研究价值判断

> **即便最后不把它包装成 strict machine unlearning，这个方向作为“plug-and-play forgetting layer for speech-LLM ASR”依然很值得做。**

---

## 14. 一个更稳的标题方向

如果你后面真写 paper，我觉得下面几种标题会比直接把 “unlearning” 写太死更稳一些：

1. **Plug-and-Play Forgetting for LLM-based ASR via Classifier-Gated Audio Prompt Corruption**
2. **Inference-Time Forgetting in Speech-LLM ASR with Corrupted Audio Prompts**
3. **Classifier-Guided Audio Embedding Corruption for Utility-Preserving Forgetting in LLM-based ASR**

如果实验特别强，再往 “unlearning” 靠也可以。

---

## 15. 你现在最值得优先读的 4 篇

1. **Large Language Model Unlearning via Embedding-Corrupted Prompts**  
   <https://arxiv.org/abs/2406.07933>

2. **Unlearning LLM-Based Speech Recognition Models**  
   <https://www.isca-archive.org/interspeech_2025/liu25b_interspeech.html>

3. **An Embarrassingly Simple Approach for LLM with Strong ASR Capacity**  
   <https://arxiv.org/abs/2402.08846>

4. **Acoustic Prompt Tuning: Empowering Large Language Models with Audition Capabilities**  
   <https://arxiv.org/abs/2312.00249>

---

## 16. Bottom line

如果只留一句最实用的话，我会写成：

> **“非常接近的零件都已经存在了，但我还没看到有人公开把 ECO-style embedding-corrupted prompts 迁到 SLAM-ASR，并只靠 audio embedding classifier 做 utility-preserving forgetting。”**

所以，这个 idea 目前看起来：

- **不是完全无前置基础**，
- 但也 **不是已有成熟文献把你想做的点做完了**，
- 更像是一个 **可以被清楚表述、能找到直接 baseline、而且 reviewer 也容易理解 novelty 的新切口**。
