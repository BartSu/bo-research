# Idea: Plug-and-Play Audio Embedding Corruption for LLM-based ASR Unlearning

## 核心构想

> 在 SLAM-ASR 类架构（speech encoder + projector + frozen LLM）中，**不修改 speech encoder 和 LLM 的任何参数**，仅通过加入一个轻量的 **audio embedding classifier**，在推理时识别需要遗忘的音频输入，并对其 audio embedding 进行 corruption，使 LLM-based ASR **选择性地遗忘特定内容**，同时不影响原模型对其他输入的识别能力（model utility）。

这是一种 **plug-and-play、inference-time、training-free（对原模型而言）** 的 unlearning 方案。

---

## 0. 这个想法为什么值得做

三个独立的技术趋势正在汇聚，但尚未在 **audio/ASR 领域**交汇：

1. **LLM-based ASR 的 unlearning 需求已被确认**（Interspeech 2025, Liu et al.），但现有方法仍需修改模型参数。
2. **Embedding-corrupted prompts 已在 text LLM 上被证明可行**（ECO Prompts, NeurIPS 2024），但从未被迁移到 audio/speech 模态。
3. **Audio encoder 级别的对抗攻击已证明，仅操纵 audio embedding 就能完全控制 LLM 输出**（EMNLP 2025, ICML 2025 等），但这些工作是攻击视角，从未被用于"正向"的 unlearning。

本 idea 的核心贡献在于：**把 embedding corruption 从 text-only LLM unlearning 推广到 audio-LLM ASR 的跨模态场景，并利用 audio embedding classifier 实现完全 plug-and-play 的选择性遗忘。**

---

## 1. 最直接相关的工作：ECO Prompts（NeurIPS 2024）

### 1.1 论文信息

- **标题**: Large Language Model Unlearning via Embedding-Corrupted Prompts
- **作者**: Chris Yuhao Liu, Yaxuan Wang, Jeffrey Flanigan, Yang Liu
- **发表**: NeurIPS 2024
- **链接**: <https://arxiv.org/abs/2406.07933>
- **代码**: <https://github.com/chrisliu298/llm-unlearn-eco>

### 1.2 核心方法

ECO Prompts 是目前与本 idea **最接近的已有工作**，其技术路线为：

1. **Prompt Classifier**: 训练一个分类器，识别哪些输入 prompt 属于需要遗忘的内容。
2. **Embedding Corruption**: 对被标记的 prompt，通过 zeroth-order optimization 离线学习一组 corruption 向量，在推理时加到 prompt embedding 上。
3. **Inference-time Unlearning**: 被 corrupt 的 embedding 导致 LLM 对目标内容输出接近"从未学习过"的状态，同时对非目标 prompt 完全不影响。

### 1.3 关键结论

- 在 entity unlearning、hazardous knowledge unlearning、copyright unlearning 上均有效。
- **几乎零副作用**：general domain 和 closely related domain 的性能不受影响。
- **极度可扩展**：适用于 0.5B 到 236B 参数的 100+ LLM，参数量增大不增加额外成本。
- 这是 **text-only** 的工作，**从未在 audio/speech 模态上做过**。

### 1.4 与本 idea 的关键区别

| 维度 | ECO Prompts (NeurIPS 2024) | 本 Idea |
|---|---|---|
| 模态 | Text-only LLM | Audio → LLM (跨模态) |
| Corruption 对象 | Text prompt embedding | Audio embedding (speech encoder 输出) |
| Classifier 输入 | Text prompt | Audio embedding / 原始音频 |
| 应用场景 | 知识遗忘、版权删除 | ASR 隐私保护、说话人遗忘、特定内容遗忘 |
| 架构适配 | 通用 LLM | SLAM-ASR 类架构 (encoder + projector + LLM) |
| 跨模态瓶颈 | 无 | 需要处理 speech-to-LLM projector 这个跨模态接口 |

---

## 2. LLM-based ASR Unlearning 的现有工作

### 2.1 直接命中：Unlearning LLM-Based Speech Recognition Models（Interspeech 2025）

- **链接**: <https://www.isca-archive.org/interspeech_2025/liu25b_interspeech.html>
- **核心贡献**:
  - 首次定义和测量 LLM-based ASR 的 memorization（WER-based、homophone-based、prompting-based memorization rate）。
  - 使用 **gradient ascent** 对 forget set 做 post-training unlearning。
  - 在 LibriSpeech 上验证 privacy-utility trade-off。
- **方法特点**:
  - Audio encoder 侧：只有 adapter 可训练，其余冻结。
  - LLM 侧：dense layers 用 LoRA 微调。
  - **需要修改模型参数**（adapter + LoRA），不是 plug-and-play。
- **与本 idea 的对比**: 这是目前唯一直接做 LLM-based ASR unlearning 的论文，但它是 **training-based** 方案。本 idea 是 **inference-time、training-free（对原模型）** 方案，两者互补。

### 2.2 OrthoGrad: Per-Sample Gradient Orthogonalization（arXiv:2503.02312）

- **链接**: <https://arxiv.org/abs/2503.02312>
- **核心贡献**:
  - 针对 retain set 有限（如 Whisper 的完整训练集不可获取）的场景。
  - 将 unlearn gradient 投影到与 retain set gradient 正交的子空间。
  - 在 Whisper ASR 上验证。
- **与本 idea 的关系**: 同样关注 ASR unlearning，但仍是 **parameter-update** 方案。本 idea 完全不需要 gradient 计算。

### 2.3 Speech Unlearning（arXiv:2506.00848）

- **链接**: <https://arxiv.org/abs/2506.00848>
- **核心贡献**:
  - 首次系统定义 speech unlearning 的两个子任务：sample unlearning 和 class unlearning。
  - 发现 speech data 的 unlearning 比 image/text 更难。
  - 在 keyword spotting 和 speaker identification 上实验。
- **局限**: 面向传统 speech model（非 LLM-based ASR），方法也是 training-based。

### 2.4 UnSLU-BENCH（Interspeech 2025）

- **标题**: "Alexa, can you forget me?" Machine Unlearning Benchmark in Spoken Language Understanding
- **链接**: <https://www.isca-archive.org/interspeech_2025/koudounas25c_interspeech.html>
- **核心贡献**:
  - 首个 spoken language understanding 的 unlearning benchmark，覆盖 4 种语言。
  - 评估 8 种 unlearning 技术的 efficacy、utility、efficiency。
  - 聚焦 speaker-level unlearning。
- **局限**: 面向 SLU 而非 ASR，方法仍是 training-based。

### 2.5 Paralinguistic Speech Unlearning（Interspeech 2025）

- **标题**: Towards Machine Unlearning for Paralinguistic Speech Processing
- **链接**: <https://arxiv.org/abs/2506.02230>
- **核心贡献**:
  - 将 machine unlearning 应用于 speech emotion recognition 和 depression detection。
  - 提出 SISA++ 方法。
- **局限**: 非 ASR 场景，传统 speech model。

### 2.6 Do Not Mimic My Voice（ICML 2025）

- **标题**: Do Not Mimic My Voice: Speaker Identity Unlearning for Zero-Shot Text-to-Speech
- **链接**: <https://arxiv.org/abs/2507.20140>
- **核心贡献**:
  - 首个 zero-shot TTS 的说话人身份 unlearning 方法。
  - Teacher-Guided Unlearning (TGU) 使模型遗忘特定说话人声音。
  - 提出 spk-ZRF 评估指标。
- **与本 idea 的关系**: 同为 speech 领域的 unlearning，但面向 TTS 而非 ASR，且是 training-based。

---

## 3. Audio Embedding 级别的"操纵"已被证明可行（来自对抗攻击文献）

以下论文从攻击视角证明了一个关键技术前提：**仅操纵 audio embedding / encoder 输出，就足以完全控制 LLM 的输出行为**。

### 3.1 Breaking Audio LLMs by Attacking Only the Encoder（arXiv:2512.23881）

- **核心发现**: 学习一个 universal perturbation 加到 audio encoder 的 latent 表示上，就能在 Qwen2-Audio-7B 上诱导任意目标输出。
- **关键意义**: **只改 encoder embedding，不接触 LLM**，就能控制输出 —— 这正是本 idea 的技术前提。
- 攻击是 universal 的，可跨输入、跨说话人泛化。

### 3.2 Universal Acoustic Adversarial Attacks for Flexible Control of Speech-LLMs（EMNLP 2025 Findings）

- **链接**: <https://aclanthology.org/2025.findings-emnlp.990/>
- **核心发现**:
  - 一段固定的 universal adversarial audio segment 可以 prepend 到输入前，覆盖模型的 prompt 指令。
  - **选择性攻击**：可以根据说话人性别、语言等属性选择性激活，只影响目标输入。
  - 在 Qwen2-Audio 和 Granite-Speech 上验证。
- **关键意义**: 证明了 **attribute-selective 的 audio embedding manipulation** 是可行的 —— 这与本 idea 中"classifier 决定是否 corrupt"的思路高度一致。

### 3.3 Sirens' Whisper: Inaudible Near-Ultrasonic Jailbreaks（arXiv:2603.13847, 2026）

- 通过近超声波编码实现 covert prompt injection，在商用 speech LLM 上达到 0.94 的 non-refusal rate。
- 进一步证明 audio 层面的操纵可以绕过 LLM 的安全机制。

### 3.4 AudioJailbreak / AdvWave（2024-2025）

- 多种方法证明可以通过 audio perturbation 对 GPT-4o-Audio、Llama-Guard 等实施 jailbreak。
- 核心启示：**audio embedding 是 speech-LLM 的关键攻击面，同时也是一个有效的控制接口。**

### 3.5 TrojanWave: Prompt Poisoning for Audio-Language Models（EMNLP 2025）

- **链接**: <https://aclanthology.org/2025.emnlp-main.940/>
- 通过 learnable prompts 在冻结的 audio-language model 上注入后门。
- 证明 **在模型完全冻结的情况下，仅通过操纵 prompt/embedding 就能改变模型行为**。

### 3.6 综合判断

这些攻击论文给本 idea 提供的最重要证据是：

> **Audio embedding 是一个高效的控制接口：仅操纵 encoder 的输出（或向 embedding 添加 perturbation），就足以在不修改 LLM 参数的情况下，根本性地改变 speech-LLM 的输出行为。**

本 idea 把这个能力从"攻击"翻转为"正向 unlearning"。

---

## 4. Inference-time / Plug-and-Play Unlearning 方法（Text LLM 领域）

除 ECO Prompts 外，text LLM 领域还有几种 inference-time unlearning 方法，它们共同构成了 plug-and-play unlearning 的方法论基础。

### 4.1 GUARD: Generation-time LLM Unlearning（ICML 2025 Workshop / ICLR 2026）

- **链接**: <https://arxiv.org/abs/2505.13312>
- **核心方法**:
  - 用 **prompt classifier（MLP）** 检测需要遗忘的输入。
  - 在生成时动态惩罚和过滤候选 token（token matching + semantic matching）。
  - 完全不需要微调模型。
- **与本 idea 的关系**: GUARD 在 text 领域证明了"classifier + inference-time restriction"的框架有效。本 idea 将类似框架迁移到 audio 模态。

### 4.2 MLLMEraser: Test-Time Unlearning for Multimodal LLMs（2025）

- **链接**: <https://arxiv.org/abs/2510.04217>
- **核心方法**:
  - Training-free 的 test-time unlearning。
  - 构造 multimodal erasure direction（对比 adversarially perturbed image-text pairs）。
  - 在推理时通过 activation steering 注入 erasure direction。
  - Input-aware steering 机制自适应决定何时应用。
- **关键意义**: 这是 **multimodal（vision-language）领域第一个 test-time unlearning** 方法，与本 idea 的方向最接近，但面向 vision 而非 audio。
- 在 LLaVA-1.5 和 Qwen-2.5-VL 上验证。

### 4.3 SAUCE: Selective Concept Unlearning with Sparse Autoencoders（2025）

- **链接**: <https://arxiv.org/abs/2503.14530>
- 在 VLM 上使用 sparse autoencoders 捕获语义特征，识别与目标概念相关的特征并在推理时修改。
- 证明 **inference-time 的特征级操纵可以实现细粒度 concept unlearning**。

### 4.4 小结：Inference-time Unlearning 的技术共识

| 方法 | 模态 | Classifier | Corruption 方式 | 需要训练原模型？ |
|---|---|---|---|---|
| ECO Prompts | Text | Prompt classifier | Embedding corruption (zeroth-order opt) | 否 |
| GUARD | Text | Prompt classifier (MLP) | Token-level restriction | 否 |
| MLLMEraser | Vision-Language | Input-aware steering | Activation steering | 否 |
| SAUCE | Vision-Language | Sparse autoencoder | Feature modification | 否 |
| **本 Idea** | **Audio-Language (ASR)** | **Audio embedding classifier** | **Audio embedding corruption** | **否** |

**本 idea 填补的空白：在 Audio-Language / ASR 模态上，还没有任何 inference-time plug-and-play unlearning 方法。**

---

## 5. SLAM-ASR 架构与 Projector 的特殊地位

### 5.1 SLAM-ASR 的核心发现（arXiv:2402.08846）

- **标题**: An Embarrassingly Simple Approach for LLM with Strong ASR Capacity
- 冻结 speech encoder 和 LLM，**只训练 linear projector**，就能做出竞争力很强的 ASR。
- 这说明 projector 是 speech-to-LLM 的关键跨模态瓶颈。

### 5.2 Prompt Projector（ICASSP 2026, arXiv:2601.20898）

- **标题**: Reducing Prompt Sensitivity in LLM-based Speech Recognition Through Learnable Projection
- 以 SLAM-ASR 为 base model，引入一个 **prompt projector** 模块。
- 证明在冻结 encoder 和 LLM 的前提下，projector 是一个稳定、低成本、可插拔的更新位置。

### 5.3 Steer-MoE（arXiv:2510.13558）

- **标题**: Steer-MoE: Efficient Audio-Language Alignment with a Mixture-of-Experts Steering Module
- 冻结 audio encoder 和 LLM，仅训练 MoE steering module。
- 通过 learned steering vectors 动态变换 audio 表示到 LLM 可理解的空间。
- 进一步证明 **encoder-LLM 之间的接口模块是 audio-LLM 中最灵活的控制点**。

### 5.4 架构启示

在 SLAM-ASR 类架构中：

```
[Audio] → [Speech Encoder (frozen)] → [Audio Embedding] → [Projector] → [LLM (frozen)] → [Transcription]
```

本 idea 提出在 Audio Embedding 和 Projector 之间（或在 Speech Encoder 输出处）插入：

```
[Audio] → [Speech Encoder (frozen)] → [Audio Embedding] → [Classifier] → corrupt? → [Corrupted/Original Embedding] → [Projector] → [LLM (frozen)] → [Transcription]
```

- **Classifier** 判断当前 audio embedding 是否属于需要遗忘的内容（如特定说话人、特定词汇、特定隐私数据）。
- 如果属于，则对 audio embedding 施加 learned corruption。
- 如果不属于，则原样通过。

---

## 6. Multimodal Unlearning 中的 Projector 相关工作

### 6.1 SineProject（arXiv:2511.18444）

- **标题**: SineProject: Machine Unlearning for Stable Vision Language Alignment
- **核心发现**: 在 VLM unlearning 中，projector 的 Jacobian 会变得 severely ill-conditioned，导致优化不稳定和 cross-modal embedding 漂移。
- **方法**: 给冻结的 projector 补充 sinusoidally modulated trainable parameters，改善 Jacobian 的 spectral conditioning。
- **关键意义**: 这是 **multimodal unlearning 中第一篇正面指出 projector 在 unlearning 中的关键角色** 的论文。虽然面向 vision-language，但其 insight 直接适用于 audio-language。

### 6.2 ViKeR: Visual-Guided Key-Token Regularization（arXiv:2601.22020）

- 利用 irrelevant visual input 来预测 unlearning 后的 ideal token-level distribution。
- 证明 **在 multimodal unlearning 中，可以用"与目标无关的模态输入"来引导遗忘行为**。
- 启示：在 audio-LLM 中，可以用 random/irrelevant audio 的 embedding 作为 corruption target。

---

## 7. 与 Audio Backdoor Detection 的联系

### 7.1 STEP: Stability-based Trigger Exposure Profiling（arXiv:2603.18103, 2026）

- Black-box、retraining-free 的 audio backdoor 检测方法。
- 利用 backdoor trigger 的"dual anomaly"：语义破坏扰动下 label 异常稳定 + 语义保持扰动下 label 异常脆弱。
- 启示：类似的稳定性分析可以用来 **验证** audio embedding corruption 是否真的实现了遗忘。

### 7.2 Poisoned Acoustics（arXiv:2602.22258, 2026）

- 仅 0.5% 的训练数据 corruption 就能达到 95.7% 的攻击成功率。
- 进一步证明 audio 领域的 embedding-level 操纵效率极高。

---

## 8. 综合文献判断：这个 idea 的 novelty 定位

### 8.1 已有的

1. **Text LLM 上的 embedding corruption unlearning** → ECO Prompts（NeurIPS 2024）
2. **Text LLM 上的 classifier-gated inference-time unlearning** → GUARD（ICML 2025 WS）
3. **Vision-Language 上的 test-time unlearning** → MLLMEraser（2025）
4. **Vision-Language 上的 projector-aware unlearning** → SineProject（2025）
5. **LLM-based ASR 上的 training-based unlearning** → Liu et al.（Interspeech 2025）
6. **Audio embedding manipulation 可以控制 speech-LLM** → 多篇对抗攻击论文（2024-2026）

### 8.2 没有的（Research Gap）

> **在 Audio-Language / LLM-based ASR 领域，没有任何工作做 inference-time、plug-and-play 的 unlearning。**

具体而言：

1. **没有人把 ECO Prompts 的 embedding corruption 框架迁移到 audio 模态。**
2. **没有人训练 audio embedding classifier 来实现 ASR 的选择性遗忘。**
3. **没有人研究在 SLAM-ASR 的 projector 接口处做 plug-and-play unlearning。**
4. **没有人把对抗攻击中证明的"audio embedding 操纵能力"翻转为正向 unlearning 工具。**

### 8.3 Novelty 声明

一个清晰的 novelty claim：

> Prior work has established (1) embedding-corrupted unlearning for text LLMs (ECO Prompts, NeurIPS 2024), (2) test-time unlearning for vision-language models (MLLMEraser, 2025), and (3) training-based unlearning for LLM-based ASR (Interspeech 2025). Adversarial attack research has also demonstrated that manipulating audio embeddings alone can fully control speech-LLM outputs. However, **no existing work combines these insights to achieve plug-and-play, inference-time unlearning for LLM-based ASR**. We propose the first framework that uses an audio embedding classifier to identify forget-target inputs and applies learned embedding corruptions at the speech-to-LLM interface, achieving selective ASR unlearning without modifying any parameter of the original model.

---

## 9. 具体方法构想

### 9.1 整体框架

```
Phase 1: Offline Preparation
├── (a) 定义 forget set F（需要遗忘的音频及其转写）和 retain set R
├── (b) 用 speech encoder 提取 F 和 R 的 audio embeddings
├── (c) 训练 audio embedding classifier C：区分 F vs R
└── (d) 学习 embedding corruption δ：使得 corrupted embedding → LLM 输出满足 unlearning 目标

Phase 2: Inference-time Plug-and-Play
├── (1) 输入音频 → speech encoder → audio embedding e
├── (2) Classifier C(e) 判断是否属于 forget 类
├── (3) 如果 C(e) = forget → 施加 corruption: e' = e + δ (或其他 corruption 方式)
├── (4) 如果 C(e) = retain → 原样传递: e' = e
└── (5) e' → projector → LLM → 输出
```

### 9.2 Classifier 设计选项

| 选项 | 描述 | 优点 | 风险 |
|---|---|---|---|
| MLP on pooled embedding | 对 audio embedding 做 mean pooling 后接 MLP | 简单高效 | 可能对细粒度区分不够 |
| Attention-based classifier | 在 frame-level embedding 上用 attention pooling + MLP | 保留时序信息 | 稍复杂 |
| Prototype-based | 计算 forget set 的 embedding centroid，用距离阈值判断 | 无需训练 | 边界模糊时不鲁棒 |
| Contrastive learned | 用 contrastive learning 学习 forget/retain 的判别表示 | 对 overlap 区域更鲁棒 | 需要更多训练数据 |

### 9.3 Embedding Corruption 设计选项

| 选项 | 描述 | 来源启示 |
|---|---|---|
| Additive perturbation δ | 学习一个 universal 或 instance-specific 的加性 perturbation | ECO Prompts (NeurIPS 2024) |
| Random embedding replacement | 将 forget embedding 替换为随机噪声或 irrelevant audio 的 embedding | ViKeR (2025) 用 irrelevant visual input |
| Projection to null space | 将 forget embedding 投影到 projector 的 null space | SineProject (2025) 的 Jacobian 分析 |
| Steering vector | 加一个学习的 steering direction 使 LLM 输出 "I don't know" 类响应 | MLLMEraser (2025) 的 activation steering |
| Zeroth-order optimized corruption | 通过 ZO optimization 离线学习最优 corruption，不需要 LLM 梯度 | ECO Prompts 的核心技术 |

### 9.4 Unlearning 目标

遗忘目标可以灵活定义：
- **Speaker unlearning**: 遗忘特定说话人的所有音频 → ASR 对该说话人的语音输出乱码/空白/拒绝
- **Content unlearning**: 遗忘特定内容（如隐私信息、版权内容）→ ASR 对含该内容的音频输出掩码/替换
- **Domain unlearning**: 遗忘特定领域的识别能力 → ASR 对该领域音频表现为"未训练"状态

---

## 10. 核心假设与可验证实验

### H1: Audio embedding corruption 可以有效诱导 LLM-based ASR 的选择性遗忘
- **实验**: 在 SLAM-ASR 上，对 forget set 的 audio embedding 施加 learned corruption，测量 forget set 的 WER 上升幅度和 retain set 的 WER 变化。
- **成功标准**: forget set WER 显著上升（接近随机输出），retain set WER 几乎不变。

### H2: Audio embedding classifier 可以高精度区分 forget vs retain
- **实验**: 在 speaker unlearning 和 content unlearning 两个场景下训练 classifier，测量 precision/recall/F1。
- **关键挑战**: forget-retain overlap 区域的处理。

### H3: Plug-and-play 方案的 utility 保持优于 training-based unlearning
- **实验**: 对比本方法与 Liu et al. (Interspeech 2025) 的 gradient ascent 方法在 retain set 上的 WER。
- **假设**: 由于本方法不修改任何模型参数，retain utility 应严格优于 training-based 方法。

### H4: 不同 corruption 策略的效果差异显著
- **实验**: 对比 additive perturbation、random replacement、null-space projection、steering vector 等策略。
- **预期**: Zeroth-order optimized additive perturbation 可能是最佳选择（基于 ECO Prompts 的经验）。

### H5: 本方法对 paraphrase / relearning attack 具有一定鲁棒性
- **实验**: 测试在 forget content 被 paraphrase 后，classifier 和 corruption 是否仍然有效。
- **挑战**: 这是 inference-time 方法的固有弱点 —— 如果输入足够不同，classifier 可能 miss。

---

## 11. 预期贡献与局限性

### 11.1 预期贡献

1. **首个 audio-LLM ASR 的 inference-time unlearning 框架**。
2. **将 ECO Prompts 的 embedding corruption 范式从 text 推广到跨模态 audio-LLM 场景**。
3. **完全 plug-and-play**：不修改 speech encoder、projector、LLM 的任何参数，可即插即用。
4. **将对抗攻击的 insight（audio embedding 操纵可控制 LLM）翻转为正向 unlearning 工具**。
5. **模型无关性**：理论上适用于任何 speech encoder + projector + LLM 架构。

### 11.2 潜在局限性

1. **Classifier 精度是瓶颈**: 如果 classifier 误分类，可能导致误遗忘（false positive）或遗漏（false negative）。
2. **Embedding corruption 的"遗忘"是否深层**: 不修改模型参数意味着知识仍在 LLM 中 —— 如果绕过 classifier，知识可被恢复。这是所有 inference-time 方法的固有问题。
3. **跨模态 corruption 的设计空间更大但也更复杂**: Audio embedding 的维度、时序结构与 text embedding 不同，ECO Prompts 的 ZO optimization 是否直接适用需要验证。
4. **Forget-retain overlap 在 audio 中可能更严重**: 同一说话人的不同内容、不同说话人的相同内容 —— 这种纠缠在 audio 中比 text 更突出。

---

## 12. 与本 workspace 其他 idea 的关系

### 与 non-forgettable-safety-knowledge.md 的关系

本 idea 关注"让模型遗忘"，而 non-forgettable idea 关注"防止模型遗忘安全知识"。两者是 complementary 的两面：
- 本 idea 的 classifier 可以被反向利用 —— 检测"不应被遗忘"的安全关键内容，在训练时增强保护。
- Boundary-aware unlearning 的 insight 可以帮助设计更精确的 forget-retain 分离策略。

### 与 unlearning-emergent-capabilities.md 的关系

如果 audio embedding corruption 导致 LLM "遗忘"了某些内容，是否会在 speech-LLM 中产生 emergent capability change？这是一个有趣的衍生问题。

### 与 SURF framework (llm-unlearning-research-gap-table.md) 的关系

SURF 的 pre-unlearning data refinement 思路可以帮助设计更好的 forget set 和 retain set，从而训练出更好的 audio embedding classifier。

---

## 13. 推荐阅读优先级

### 第一优先级：直接技术基础
1. **ECO Prompts** (NeurIPS 2024) — <https://arxiv.org/abs/2406.07933>
2. **Unlearning LLM-Based ASR** (Interspeech 2025) — <https://www.isca-archive.org/interspeech_2025/liu25b_interspeech.html>
3. **Breaking Audio LLMs by Attacking Only the Encoder** — <https://arxiv.org/abs/2512.23881>

### 第二优先级：框架设计参考
4. **GUARD** (ICML 2025 WS) — <https://arxiv.org/abs/2505.13312>
5. **MLLMEraser** — <https://arxiv.org/abs/2510.04217>
6. **SLAM-ASR** — <https://arxiv.org/abs/2402.08846>

### 第三优先级：补充 insight
7. **Universal Acoustic Adversarial Attacks for Speech-LLMs** (EMNLP 2025) — <https://aclanthology.org/2025.findings-emnlp.990/>
8. **SineProject** — <https://arxiv.org/abs/2511.18444>
9. **Steer-MoE** — <https://arxiv.org/abs/2510.13558>
10. **Prompt Projector for SLAM-ASR** (ICASSP 2026) — <https://arxiv.org/abs/2601.20898>

### 第四优先级：Speech unlearning landscape
11. **Speech Unlearning** — <https://arxiv.org/abs/2506.00848>
12. **UnSLU-BENCH** (Interspeech 2025) — <https://www.isca-archive.org/interspeech_2025/koudounas25c_interspeech.html>
13. **OrthoGrad** — <https://arxiv.org/abs/2503.02312>
14. **Do Not Mimic My Voice** (ICML 2025) — <https://arxiv.org/abs/2507.20140>
15. **Paralinguistic Speech Unlearning** (Interspeech 2025) — <https://arxiv.org/abs/2506.02230>

### 第五优先级：Audio backdoor/poisoning (作为技术前提)
16. **TrojanWave** (EMNLP 2025) — <https://aclanthology.org/2025.emnlp-main.940/>
17. **STEP** — <https://arxiv.org/abs/2603.18103>
18. **Poisoned Acoustics** — <https://arxiv.org/abs/2602.22258>
19. **Sirens' Whisper** — <https://arxiv.org/abs/2603.13847>
20. **AdvWave** — <https://arxiv.org/abs/2412.08608>

---

## 14. 一句话总结

> **在 LLM-based ASR（特别是 SLAM-ASR 架构）中，通过加一个 audio embedding classifier + learned embedding corruption 实现 plug-and-play 的选择性遗忘，是一个明确的 research gap。Text LLM 的 ECO Prompts（NeurIPS 2024）、multimodal 的 MLLMEraser（2025）、以及 audio adversarial attack 文献共同提供了充分的技术前提，但这三条线还从未在 audio-LLM ASR unlearning 上汇合。**
