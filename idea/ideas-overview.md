# 研究 Idea 总览（Ideas Overview）

> 三条正在验证与推进中的研究方向，按统一结构整理：**优化标题 → 一句话问题定义 → Research Gap → Novelty 分析**。
> 详细方案与文献支撑见各自的 idea / papers 文件链接。

---

## Idea 1 — Audit Before Unlearn：遗忘数据驱动的 Pre-Unlearning 知识腐蚀审计框架

**原标题**：unlearning knowledge corruption, audit before unlearn
**优化后标题（候选，由稳到激进）**：

1. **Audit Before Unlearn: Predicting Knowledge Corruption from Forget-Set Features in LLM Unlearning**
2. **Pre-Unlearning Corruption Audit: A Data-Centric Framework for Estimating Collateral Damage in LLM Unlearning**
3. **遗忘前审计：基于遗忘数据特征预测 LLM Unlearning 的知识腐蚀**

**推荐**：#1（"Audit Before Unlearn" 作为记忆点强、expression 直接，"Predicting … from Forget-Set Features" 明确数据→结果的预测范式）

**详细方案**：[`idea/forget-data-knowledge-corruption-correlation.md`](./forget-data-knowledge-corruption-correlation.md)
**支撑文献**：[`papers/llm-unlearning-research-gap-table.md`](../papers/llm-unlearning-research-gap-table.md)、[`papers/pre-unlearning-data-augmentation-shortlist.md`](../papers/pre-unlearning-data-augmentation-shortlist.md)

### 一句话问题定义

> **能否仅凭遗忘数据（forget set）本身的可测量属性，在执行任何 unlearning 更新之前，预测出无关知识将被腐蚀（knowledge corruption）的严重程度？**

类比 ACL Findings 2024 论文 *A Curious Case of Searching for the Correlation between Training Data and Adversarial Robustness*：
- 他们研究 `训练数据 → 对抗鲁棒性`，在 fine-tune 前用 Random Forest 预测攻击成功率，节省 30×–193× 评估开销。
- 我们研究 `遗忘数据 → 无关知识腐蚀`，在 unlearn 前预测腐蚀严重程度，形成"审计 → 干预 → 执行"的闭环。

### Research Gap

| 已有工作 | 缺失的那一步 |
|---|---|
| 事后诊断腐蚀：TOFU / WMDP / MUSE / BLUR / knowledge holes / blind spots | **Pre-unlearning** 基于遗忘数据特征**预测**腐蚀 |
| 样本级遗忘难度预测：CUD (circuit-guided unlearning difficulty) | 预测的是"这一条样本有多难忘"，**不是"会波及多少无关知识"** |
| 知识纠缠分析：EGUP / CLReg / SKeB / UIPE / CIR | 在 unlearning **过程中**利用纠缠调节强度，**不是事前量化腐蚀风险** |
| Data-centric 行为预测：TuneAhead / Data2Behavior / PRISM | 应用于 **fine-tuning**，**从未用于 unlearning 腐蚀** |
| Pre-unlearning 数据精炼：SURF 等 | **启发式**操作数据，**缺证据驱动的决策依据** |
| ACL Findings 2024 训练数据 → 对抗鲁棒性预测 | **没有对应的 "遗忘数据 → 知识腐蚀" 版本** |

**核心空白**：没有任何公开工作系统地从 *遗忘数据及其与模型现有知识的关系* 提取特征，来在 unlearning 之前预测**无关知识腐蚀**的严重程度。

### Novelty 分析

- **不是**一个新的 unlearning 方法；**不是**一个新的 benchmark；**不是** unlearning 过程中做纠缠分析。
- **是**第一个 data-centric、可在 unlearning **之前**运行的**腐蚀审计层**（audit layer）：
  1. **特征体系（Feature Taxonomy）**：5 类 18+ 个特征（embedding 分布 / 知识图谱 / circuit / 模型行为 probing / token 表层），首次面向 unlearning 腐蚀系统整理。
  2. **轻量级预测器**：Random Forest / XGBoost / 小 MLP 输出连续的腐蚀预测值（保留准确率下降、知识空洞、多跳退化等）。
  3. **跨方法与跨模型迁移性**：在 GA / NPO / RMU / LUNAR / FALCON 等方法上验证单一预测器是否通用。
  4. **从相关到因果**：用 causal tracing 把"数据层特征 → circuit 共享 → 腐蚀"接起来，给经验预测提供机制解释。
  5. **审计 → 干预闭环**：预测结果驱动 pre-unlearning 数据精炼（拆分高风险子集、加 retain anchors、重写歧义样本、调整超参），把 SURF 类启发式操作升级为**证据驱动**。

**最强新颖性声明**：
> 在 LLM unlearning 场景下，首次提出 "pre-unlearning audit"——从遗忘数据的 embedding、知识图谱、circuit 与轻量 probing 特征中预测无关知识腐蚀，并据此指导数据干预。将 data-centric 预测范式（fine-tuning 前预测对抗鲁棒性）迁移到 unlearning 这一更新但尚未被覆盖的重要问题。

---

## Idea 2 — 分类器门控的音频嵌入腐蚀：面向 LLM-based ASR 说话人/短语级遗忘

**原标题**：llm-based ASR unlearning, unlearn speaker
**优化后标题（候选，由稳到激进）**：

1. **Plug-and-Play Forgetting for LLM-based ASR via Classifier-Gated Audio-Prompt Corruption**
2. **Speaker Unlearning in Speech-LLMs: Classifier-Gated Audio Embedding Corruption at the Projector Interface**
3. **Inference-Time Speaker Forgetting for SLAM-ASR without Retraining the Backbone**

**推荐**：#1（最中性、可以同时覆盖 speaker-level 与 phrase-level，"plug-and-play" 直接命中核心卖点；第一版避免过早 commit 到严格 "unlearning" 定义）

**详细方案**：[`idea/slam-asr-audio-embedding-corrupted-prompts.md`](./slam-asr-audio-embedding-corrupted-prompts.md)
**支撑文献**：[`papers/llm-asr-unlearning-landscape.md`](../papers/llm-asr-unlearning-landscape.md)、[`papers/llm-based-asr-research-map.md`](../papers/llm-based-asr-research-map.md)

### 一句话问题定义

> **在 LLM-based ASR（SLAM-ASR / SALMONN / Qwen2-Audio 等）中，能否只新增一个 audio embedding classifier + 一个 corruption 模块，在推理时对命中遗忘目标（说话人 / 短语 / 实体）的 projected audio prompts 做可控腐蚀，实现 plug-and-play forgetting，同时保住原模型 utility？**

三条已有的、但**尚未被合并**的证据链：
- **ECO (NeurIPS 2024)**：classifier-gated + prompt embedding corruption，但对象是**文本 prompt**。
- **Liu et al. (Interspeech 2025)**：LLM-based ASR 的 unlearning 是**权重更新 / PEFT**，**不是** inference-time intervention。
- **SLAM-ASR (arXiv:2402.08846) / APT (TASLP 2025)**：证明 projector / audio soft prompt 是稳定、可插拔的接口层，但目标是**能力接入**而非 forgetting。

**缺的正是把这三条线合起来的那一步。**

### Research Gap

你这个 idea 同时要求四个条件，而现有文献只各自满足其中一部分：

| 条件 | ECO (text LLM) | Liu 2025 (ASR unlearn) | SLAM-ASR / APT | **本 idea** |
|---|:---:|:---:|:---:|:---:|
| (1) 对象是 LLM-based ASR | × | ✓ | ✓ | ✓ |
| (2) forgetting 发生在 audio embedding / audio prompt 层 | × | × | n/a | ✓ |
| (3) 只新增 classifier + corruption，不改主体 | ✓ | × | × | ✓ |
| (4) 保 utility | ✓ | partial | n/a | ✓ |

最接近的新竞品 **TruS (arXiv:2601.20481, 2026-01)**：training-free TTS speaker unlearning via hidden-activation steering。
- 共同点：inference-time、无 retraining。
- 差异点：**模态**（TTS decoder vs. ASR projector）、**机制**（activation steering vs. classifier-gated corruption）、**无 gating classifier**。
- → TruS 应该在论文里作为**主要对比坐标**，但它**不占据本 idea 的方法位点**。

**其他可利用的资产**：UnSLU-BENCH (arXiv:2505.15700) 用于 speaker-level RTBF 评估协议（虽然覆盖 SLU 而非 ASR）。

### Novelty 分析

**不要**把 novelty 押在 "用了 embedding" / "用了 classifier" / "做 unlearning" 这些单点上——都已经被单独做过。

**真正的 novelty** 是四个维度的**组合点**：
1. **任务侧**：speech-LLM ASR（而非 text LLM / TTS / SLU classifier）；
2. **干预位点**：projected audio prompts 层（SLAM-ASR 唯一可训练组件，天然 plug-and-play 接口）；
3. **干预方式**：classifier-gated inference-time corruption（而非 post-training weight update / activation steering without gating）；
4. **目标形态**：speaker-level + phrase/entity-level forgetting，且显式量化 retain-set utility 保持度。

**最强新颖性声明**：
> 此前工作分别研究了：文本侧 embedding-corrupted prompts 做 LLM unlearning；LLM-based ASR 的 post-training unlearning。我们首次将二者打通——在 speech-LLM ASR 的 projector 接口上，以 classifier-gated audio-prompt corruption 实现 inference-time、plug-and-play、utility-preserving 的说话人 / 短语级遗忘，无需更新 speech encoder、projector 或 LLM。

**可同时回应的 reviewer 质疑（也就是要在论文里主动处理的点）**：
- "这是 unlearning 还是 suppression？"
  → 建议第一版 position 为 **"inference-time unlearning approximation / forgetting layer"**，而非严格 parameter-level deletion，主动承认这一定位。
- 说话人 vs. 短语 vs. 样本级遗忘三种 threat model 选哪个？
  → 第一版优先 **speaker-level + phrase-level**（classifier 更容易学到稳定边界）。
- 局部 vs. 全段 corruption？
  → 用 frame / chunk 级 classifier 支撑局部 corruption，避免整段污染伤 utility。
- False positive 会直接吃 utility。
  → classifier calibration 与 FPR/FNR 要作为一等评估指标。
- Bypass 风险（换声学变体、换口音、换说话速度）。
  → robustness-to-bypass 是额外的 section，反而强化 "audio-specific forgetting" 叙事。

---

## Idea 3 — LLM-based ASR 中的 Attention Sink：跨模型图谱 + 干预位点

**原标题**：attention-sink in llm-based ASR
**优化后标题（候选，由稳到激进）**：

1. **Attention Sinks in LLM-based ASR: A Cross-Model Atlas of SLAM-ASR, SALMONN, Qwen2-Audio, and Kimi-Audio**
2. **Where Do Speech-LLMs Sink? Cross-Model Analysis of Attention Sinks and Massive Activations in LLM-based ASR**
3. **Audio Registers for Speech-LLMs: Understanding and Intervening on Attention Sinks in LLM-based ASR**

**推荐**：
- 分析型 paper（低风险、Interspeech / ICASSP 级别）选 **#1**
- 干预型 paper（中等风险、带 WER / streaming / unlearning 应用）选 **#3**
- #2 作为两者之间的通用框架题

**详细文献与假设**：[`papers/attention-sink-llm-asr.md`](../papers/attention-sink-llm-asr.md)
**唯一直接相关工作**：Anand et al., *Mitigating Attention Sinks and Massive Activations in AVSR with LLMs* (arXiv:2510.22603, ICASSP 2026)

### 一句话问题定义

> **在 SLAM-ASR 家族的纯音频 ASR（而非 AVSR）中，attention sink 出现在哪些 token、何时形成、是否编码声学/说话人/域信息？能否借 sink 作为轻量干预位点支撑 streaming、长音频 RoPE、unlearning、说话人擦除等下游任务？**

### Research Gap

1. **纯分析论文完全缺位**：SLAM-ASR / SALMONN / Qwen2-Audio / Kimi-Audio 上系统性 sink + massive activation 图谱**没人做过**。Anand et al. 只在 **Llama-AVSR**（多模态 AVSR）上做了一瞥，且：
   - 没有 text-only 条件对照 → sink 是否因模态而出现因果不清；
   - 没有 audio token 索引级定位（首帧 vs 末帧）；
   - 没有随音频长度变化的分析；
   - 场景是 AVSR，**纯 SLAM-ASR 仍待独立验证**。

2. **流式 ASR 效率**：StreamingLLM 的 "sink + rolling window" 范式**没人真正搬到 speech-LLM 解码**。AudioKV 等通用 KV eviction 没利用 audio sink 特性。

3. **长音频 RoPE 合理性**：如果 sink 随音频长度漂移，直接影响 RoPE interpolation 是否合法；目前**完全未分析**。

4. **干预位点空白**：sink token 是 unlearning / 说话人擦除 / 幻觉抑制的天然介入点。**如果 sink 编码声学/说话人/域信息（待 probing 验证），就可以在 sink 上做 activation patching 做超轻量干预**——但目前**没人在 speech-LLM 上验证 sink 是否真的承载这些信息**。

5. **Audio register token 未被提出**：Darcet / Victor 的 visual register 思路到 speech-LLM 的对应版本不存在。Anand et al. 用 decorrelation loss 间接处理，但没提出 audio register token 这个更结构化的解法。

### Novelty 分析

**不要**把 novelty 押在 "发现 sink 存在" 上——StreamingLLM / Anand 已经给过；单独发现 sink 不足以支撑。

**真正的 novelty 坐标**：

| 维度 | 与 Anand et al. (ICASSP 2026) 的差异点 |
|---|---|
| 覆盖范围 | 他：Llama-AVSR 一个模型；我：SLAM-ASR / SALMONN / Qwen2-Audio / Kimi-Audio 跨模型图谱 |
| 模态隔离 | 他：AVSR（音 + 视）；我：纯音频 ASR，且做 text-only 条件对照 |
| 长度标度 | 他：固定长度；我：10 / 30 / 60 / 180 秒 sink 漂移分析 |
| Probing | 他：没做；我：probe sink token 是否编码 domain / SNR / speaker |
| Head 级分析 | 他：没做；我："audio-critical heads"（AudioKV 启发）与 "sink heads" 重叠度 |
| 干预 | 他：decorrelation loss；我：audio register token + sink-targeted activation patching，且对接下游 streaming / unlearning / 说话人擦除 |

**最强新颖性声明**：
> 首个针对 LLM-based ASR 的 attention sink 系统性研究：跨模型（SLAM-ASR / SALMONN / Qwen2-Audio / Kimi-Audio）图谱化 sink 位置与 massive activation，做纯音频 vs. text-only 对照、长度标度分析、sink 的 probing、以及 sink head 与 audio-critical head 的分离度分析，并据此提出 audio register token 与 sink-targeted intervention，直接对接流式解码、长音频 RoPE、说话人擦除等下游任务。

### 与 Idea 1 / Idea 2 的交叉点（为什么三条线值得一起推进）

- **与 Idea 2 的强协同**：如果 sink token 被 probe 出承载说话人 / 声学域信息，那么 Idea 2 的 classifier-gated corruption 就有了**更高效、更可解释的干预位点**——不必对所有 projected audio prompts 做 corruption，只需在 sink token 上做 activation patching。这会把 Idea 2 从 "corruption on audio prompts" 精细化为 "sink-targeted audio activation patching"，论文可读性与机制性都明显加强。
- **与 Idea 1 的弱耦合**：sink token 的 hidden state 可作为 Idea 1 特征体系中 "circuit / 机制特征" 的一个新维度（sink 重叠度、sink 份额变化），用来预测 unlearning 腐蚀。这是 Idea 1 特征集在多模态场景下的自然扩展。

---

## 三条线的推进优先级建议（非 commit，仅作参考）

| Idea | 风险 | 执行门槛 | 叙事独立性 | 预期 venue | 建议优先级 |
|---|---|---|---|---|---|
| Idea 3（分析型） | 低 | 低（画图 + probing） | 强 | Interspeech / ICASSP | **先做**（快速出一篇，且其 probing 结论能直接反哺 Idea 2） |
| Idea 2 | 中 | 中（classifier + corruption + ASR 全流程评估） | 强 | ACL / EMNLP / Interspeech | **中期主推**（可借 Idea 3 的 sink 结果加强机制性） |
| Idea 1 | 高 | 高（多模型 × 多方法 × 多遗忘集 × 全评估） | 最强 | NeurIPS / ICLR / ICML | **长线大坑**（data-centric × unlearning，novelty 最足但实验量最大） |

---

## 文件导航

- 本文档：**统一总览** — [`idea/ideas-overview.md`](./ideas-overview.md)
- Idea 1 详细方案：[`idea/forget-data-knowledge-corruption-correlation.md`](./forget-data-knowledge-corruption-correlation.md)
- Idea 2 详细方案：[`idea/slam-asr-audio-embedding-corrupted-prompts.md`](./slam-asr-audio-embedding-corrupted-prompts.md)
- Idea 3 详细文献：[`papers/attention-sink-llm-asr.md`](../papers/attention-sink-llm-asr.md)
- 其他 idea（备选，未纳入本次整理）：[`idea/non-forgettable-safety-knowledge.md`](./non-forgettable-safety-knowledge.md)、[`idea/unlearning-emergent-capabilities.md`](./unlearning-emergent-capabilities.md)
