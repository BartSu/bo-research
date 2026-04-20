# Attention Sink in LLM-based ASR / Speech-LLMs

**Cycle date:** 2026-04-19
**Window:** 基础文献不限时间；speech-LLM 分析主要 2025-H2 – 2026-Q2。

## 1. 一句话结论

> **在 SLAM-ASR 家族的纯音频 ASR 上，attention sink 的系统性分析基本是空白。** 目前只有一篇直接证据 (Anand et al., ICASSP 2026) 做了 AVSR 场景的 sink 分析，且其结论指向一些非平凡的、值得在纯 ASR 上复现和扩展的观察（intermediate sinks 出现在 `<audio>`/`</audio>` 这类特殊 token 上而非音频帧内部；sink token 的 hidden state 与 BOS 极度相似，像"克隆"）。这既是分析 gap，也是可以直接产出一篇 Interspeech / ICASSP 的机会。

---

## 2. 基础文献 (Foundations)

| 论文 | 一句话 |
|---|---|
| **StreamingLLM** (Xiao et al., arXiv:2309.17453, ICLR 2024) | 首次系统提出 "attention sink" ——自回归 LLM 把大量注意力无关语义地分配给前几个 token（尤其 BOS）；保留这几个 KV 槽让 sliding window 推到 4M tokens。 |
| **Massive Activations** (Sun et al., arXiv:2402.17762) | 少数固定维度上出现比其他 feature 大数万倍的激活 outlier，位于 sink token 上，等价于隐式注意力 bias；ViT 上同样存在。 |
| **Vision Transformers Need Registers** (Darcet et al., arXiv:2309.16588) | ViT 在低信息背景 patch 上产生 high-norm artifact token；加入可学习 "register" 吸收该角色可同时提升分割和稠密任务。为"sink 作为计算寄存器"假说提供视觉端证据。 |
| **When Attention Sink Emerges** (Gu et al., ICLR 2025 Spotlight, arXiv:2410.10781) | Sink 由 softmax 归一化 + 有效优化 + 足量数据共同触发；换成 sigmoid attention 后 sink 消失；sink 更像 "key bias"，几乎不贡献 value。 |
| **Why do LLMs attend to the first token?** (arXiv:2504.02732) | 从 over-mixing 视角给理论解释——sink 是 LLM 用于防止深层 token 表征过度混合的机制；context 长度、深度、data packing 都会调制 sink。 |
| **Sinks and Compression Valleys** (arXiv:2510.06477, ICLR 2026) | 把 sink、"compression valley"、massive activation 统一到残差流同一过程，提出 Mix-Compress-Refine 三阶段信息流理论。 |

---

## 3. LLM-based ASR 的直接证据

**直接证据极其稀少，只有一篇真正切题：**

### Anand, Cappellazzo, Petridis, Pantic — *Mitigating Attention Sinks and Massive Activations in Audio-Visual Speech Recognition with LLMs*

arXiv:2510.22603, **ICASSP 2026**；Llama-AVSR 框架（Whisper encoder + LoRA LLaMA）在 ASR / VSR / AVSR 三任务均观察到 sink。

**关键发现：**

1. **Intermediate sinks 的位置**：除 BOS sink 外，出现"中间 sink"，分布在 `<audio>` / `</audio>` 特殊 token 以及 prompt 文本中的低语义 token 上，**而不是在投影后的音频特征位置本身**。
2. **形成时点**：BOS sink 在预训练 LLM 中就已存在；**intermediate sinks 是多模态 fine-tuning 阶段新出现的**。
3. **Sink token ≈ BOS 的克隆**：massive activation 从第 2 层 MLP 开始出现；intermediate sink 的 hidden state 与 BOS 的 hidden state 具有**极高余弦相似度**。
4. **方法**：提出 decorrelation loss（降低 BOS 与其他 token 的余弦相似度），在高下采样（压缩率 32）下把 AVSR WER 从 12.92% 改善到 11.50%。

**局限（也是 gap）：**
- 未做纯 text-only 条件对照（sink 随模态出现的因果不清楚）
- 未做 audio token 索引级定位（例如首帧 vs 末帧）
- 未分析 sink 是否随音频长度漂移
- 场景是 AVSR 而非纯 SLAM-ASR；纯音频 ASR 的现象需独立验证

### 其他 speech-LLM 论文的间接线索

SLAM-ASR (arXiv:2402.08846)、SALMONN (arXiv:2310.13289)、Qwen2-Audio、LLaMA-Omni 等**原始论文未对 attention sink 做直接分析**。相关间接线索：

- **Towards Audio Token Compression in LALMs** (arXiv:2511.20973) 与 **segmentwise pruning** (arXiv:2511.14293)：Qwen2-Audio 深层 token 强冗余、可大幅池化——暗示存在 sink 式信息集中，但未命名 sink。
- **AudioKV** (arXiv:2604.06694)：通用 text-KV eviction 在音频域失败，需要 audio-critical attention head 先验；间接说明 speech-LLM 的注意力结构与 text-LLM 不同。
- **Extending Audio Context for LALMs** (arXiv:2510.15231)：讨论 RoPE 在长音频下的温度缩放，但未提 sink。
- **Intermediate Representations in Spoken Language Models** (arXiv:2510.02569)：spoken LM 中间表征分析，离 sink 最近的一篇。

---

## 4. 多模态相邻证据（可迁移）

- **See What You Are Told: Visual Attention Sink in LMMs** (Kang et al., ICLR 2025, arXiv:2503.03321)：LLaVA 系存在 "visual attention sink"——某些视觉 token 长期吸引高注意力但与文本完全无关；删除不影响性能；提出 VAR 在推理阶段重分配 sink 头的注意力。
- **To Sink or Not to Sink** (arXiv:2510.08510)：ViT encoder 输出的 high-norm token 识别为 "ViT sink"，承载高层语义，LLM 恰好依赖——**视觉 sink 并非完全无用**。这与"sink 是 bias，无语义"的 text-LLM 观点存在张力。
- **Victor: Visual Compact Token Registers** (arXiv:2410.14072)：在视觉 token 后插入少量可学习 register，让前几层 LLM 把信息挤压进去，显著降低视觉 token 数量。

**迁移判断：** 投影后的音频帧速率比 visual patch 还高（Qwen2-Audio 30s → 750 token），**很可能出现类似的 audio sink token**。Anand et al. 只给了一瞥，系统化的 speech-LLM 版本尚未出现。

---

## 5. 有趣、可验证的假设

1. **BOS vs. 音频前哨 token 的 sink 竞争**
   实验：同一条音频，比较 `BOS + audio + prompt` vs. `audio + BOS + prompt` 两种顺序，测量各层 attention 落在首位置的比例。
   假设：sink 锁定**绝对第一位**而非 BOS 语义；位置交换后 sink 落到第一个音频 token 上。

2. **音频投影位置是否发展出 massive activation**
   实验：按 Sun et al. 协议画 SLAM-ASR 每层残差 norm 的 per-token 热图，重点看音频段首帧、末帧、`<audio>`/`</audio>` 三处。
   预测：特殊 token 出现 outlier，音频帧内部不出现（与 Anand et al. AVSR 的初步观察一致）；纯音频 ASR 上需验证。

3. **长度标度效应**
   实验：LibriSpeech / TED-LIUM 上把同一 utterance 重复拼接至 10 / 30 / 60 / 180 秒，监测 sink 是否"分裂"或"漂移"。
   动机：文本端已有 context-length 调制 sink 的证据 (2504.02732)；音频端未知，对长音频 ASR 的 RoPE interpolation 合理性直接相关。

4. **Modality-specific sink heads**
   实验：在 SALMONN / Qwen2-Audio / SLAM-ASR 上做 head-level sink 统计，对比 "audio-critical heads"（AudioKV 启发）与 "sink heads" 的重合度；head-ablation 验证。
   应用：若两者错开，可把 sink head 专门用作 streaming KV 保留策略。

5. **Sink 是否编码声学域信息**
   实验：对 sink token 的 hidden state 做 probing（domain / SNR / 说话人）。
   预测：若可分类，则 sink 并非纯 bias，而是**隐式承载全局声学上下文**——与 Anand et al. "sink 低语义" 结论产生张力，且为 unlearning / 说话人擦除 / 域适应提供轻量干预点。

---

## 6. 研究空白 (Research Gaps)

1. **纯分析论文完全缺位**——系统性的 "Attention Sink in Speech-LLMs" 还没有人写。对 SLAM-ASR / SALMONN / Qwen2-Audio / Kimi-Audio 做 cross-model sink + massive-activation 图谱，就是一篇 Interspeech / ICASSP 级别的工作。风险低、执行门槛低。

2. **流式 ASR 效率**——把 StreamingLLM "sink + rolling window" 直接应用到 speech-LLM 解码；当前 AudioKV 等方法只做通用 KV eviction，没人专门利用 audio sink 特性。

3. **长音频上下文**——若 sink 位置随音频长度漂移，直接影响 RoPE interpolation 的合理性；可与 arXiv:2510.15231 的 context-extension 结合。

4. **干预与可解释性钩子**——sink token 是 unlearning / 说话人擦除 / 幻觉抑制的天然介入点。如果 sink 编码域信息，可在 sink 上做 activation patching 实现轻量化域适应。**这点和本仓库 `idea/slam-asr-audio-embedding-corrupted-prompts.md` 的 classifier-gated corruption idea 高度协同**——sink token 本身就是 corruption 的天然目标。

5. **Audio register token**——把 Darcet / Victor 思路搬过来：在 projector 后加入少量可学习 audio register，看能否吸收 intermediate sink 并降低 WER 或支撑更高音频 token 压缩率。直接对标 Anand et al. 的 decorrelation loss；几乎肯定能做出的 follow-up。

---

## 7. 和本仓库其他主题的交叉点

- **LLM-based ASR** (`papers/llm-based-asr-research-map.md`)：本文是注意力机制视角的补充。
- **SLAM-ASR forgetting idea** (`idea/slam-asr-audio-embedding-corrupted-prompts.md`)：gap #4 提示——sink token 可能是 classifier-gated corruption 的天然 / 更高效干预位点。值得在 idea 更新时加一节 "sink-targeted corruption"。
- **LLM-ASR unlearning landscape** (`papers/llm-asr-unlearning-landscape.md`)：如果 sink 承载全局声学 / 说话人信息，sink 上的激活编辑是说话人擦除的轻量方案。

---

## 8. 来源

**Foundations：**
- [StreamingLLM (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453)
- [Massive Activations (arXiv:2402.17762)](https://arxiv.org/abs/2402.17762)
- [Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588)
- [When Attention Sink Emerges (arXiv:2410.10781)](https://arxiv.org/abs/2410.10781)
- [Why do LLMs attend to the first token? (arXiv:2504.02732)](https://arxiv.org/abs/2504.02732)
- [Sinks and Compression Valleys (arXiv:2510.06477)](https://arxiv.org/abs/2510.06477)

**Speech-LLM direct / adjacent：**
- [Mitigating Attention Sinks in AVSR with LLMs (arXiv:2510.22603, ICASSP 2026)](https://arxiv.org/abs/2510.22603)
- [SLAM-ASR (arXiv:2402.08846)](https://arxiv.org/abs/2402.08846)
- [SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [AudioKV (arXiv:2604.06694)](https://arxiv.org/abs/2604.06694)
- [Towards Audio Token Compression in LALMs (arXiv:2511.20973)](https://arxiv.org/abs/2511.20973)
- [Segmentwise pruning in audio-language models (arXiv:2511.14293)](https://arxiv.org/abs/2511.14293)
- [Extending Audio Context for LALMs (arXiv:2510.15231)](https://arxiv.org/abs/2510.15231)
- [video-SALMONN S (arXiv:2510.11129)](https://arxiv.org/abs/2510.11129)
- [Intermediate Representations in Spoken LMs (arXiv:2510.02569)](https://arxiv.org/abs/2510.02569)

**Multimodal adjacent：**
- [Visual Attention Sink in LMMs (arXiv:2503.03321, ICLR 2025)](https://arxiv.org/abs/2503.03321)
- [To Sink or Not to Sink (arXiv:2510.08510)](https://arxiv.org/abs/2510.08510)
- [Victor: Visual Compact Token Registers (arXiv:2410.14072)](https://arxiv.org/abs/2410.14072)

**Code：**
- [mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm)
- [sail-sg/Attention-Sink](https://github.com/sail-sg/Attention-Sink)
