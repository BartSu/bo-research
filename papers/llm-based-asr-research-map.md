# LLM-based ASR 研究现状与“问题 -> 论文回答”梳理

这份笔记面向一个具体目标：快速回答 **“LLM-based ASR 现在研究到哪了，哪些 paper 回答过哪些问题”**。

这里把 **LLM-based ASR** 分成三类：

1. **LLM 作为外部语言模型**：用于 second-pass rescoring、decoder guidance、长上下文建模。
2. **Speech encoder + LLM 一体化 ASR**：把语音编码器接到 decoder-only LLM 上，让 LLM 直接做识别。
3. **上下文/多模态增强 ASR**：在识别时引入对话历史、文档、说话人信息、视频编辑上下文等。

下面的结论以 2023-2026 这波代表工作为主，重点是 **研究问题、已有答案、还没解决的空白**，而不是做一个完全穷尽的 bibliography。

---

## 0. 可先看的综述 / survey

如果你想先快速建立大图景，可以先看下面两篇综述，再回来看后面的“问题 -> paper”：

1. **Recent Advances in Speech Language Models: A Survey** (2024)  
   <https://arxiv.org/abs/2410.03751>

2. **A Survey on Speech Large Language Models** (2024)  
   <https://arxiv.org/abs/2410.18908>

这两篇 survey 的价值主要是：

- 帮你区分 **传统 ASR + LM**、**Speech-LLM**、**SpeechLM / spoken dialogue model** 等不同路线；
- 帮你理解 speech encoder、adapter/projector、LLM decoder、多模态训练的常见系统结构；
- 帮你快速定位 ASR 在整个 speech-language 版图中的位置。

不过，如果你的核心问题是 **“LLM-based ASR 到底回答了哪些具体研究问题”**，下面的问题导向整理会更直接。

---

## 1. 一句话判断：LLM-based ASR 目前处在什么阶段？

如果只看当前主线，可以概括为：

- **已经被证明可行**：LLM 确实可以帮助 ASR，不管是作为重打分器，还是直接作为 speech decoder。
- **最成熟的方向是“LM 增强”而不是“完全替代传统 ASR”**：把 LLM 用在 rescoring、context injection、rare word recovery 上，收益最稳定。
- **Speech-LLM 端到端 ASR 已经能做得不错，但还不是“全面碾压 Whisper / 传统 E2E ASR”**。
- **LLM-based ASR 的真正强项**，目前更多体现在：
  - 长上下文建模，
  - 融入文本知识，
  - 利用外部上下文，
  - 在低资源或特定场景里增强语言先验，
  - 向“统一的 speech-language interface”演化。
- **真正还没完全解决的问题**包括：
  - 域外泛化和鲁棒性，
  - 插入错误 / hallucination，
  - 流式低延迟部署成本，
  - code-switching 的稳定收益，
  - 与强 E2E ASR baseline 的全面对比优势。

一个比较稳妥的判断是：

> **LLM-based ASR 已经从“概念验证”进入“任务定制与系统工程优化”阶段，但离“在通用 ASR 上全面替代 Whisper 类模型”还有距离。**

---

## 2. 研究主线：这个方向是怎么演化的？

### 主线 A：先把 LLM 当成更强的语言模型

最早成熟起来的是 **LLM rescoring / decoder guidance**：

- ASR 第一阶段先给出候选转写；
- 再让大语言模型根据更强的文本先验、长上下文能力、专有名词知识去重新排序；
- 这一类方法通常 **最容易集成到现有 ASR 系统**。

这条线回答的是：

> “不改 ASR 主干，只把 LLM 作为后处理或解码器辅助，能不能稳定提升识别？”

当前答案基本是 **可以，尤其对长语音、复杂句子、专有名词和领域词有效**。

### 主线 B：把 speech encoder 接到 LLM 上，让 LLM 直接做 ASR

2024 年之后，一个很明确的趋势是：

- 用现成 speech encoder 提取声学特征；
- 用一个 projector / adapter 接到 decoder-only LLM；
- 让 LLM 直接生成转写。

这条线回答的是：

> “LLM 能不能不仅做语言后验，还真正作为 ASR 解码核心？”

当前答案是 **能，而且 surprisingly simple 的设计就能 work**，但训练稳定性、对齐方式、域外鲁棒性仍然是核心痛点。

### 主线 C：把 ASR 变成“带上下文理解”的识别系统

Seed-ASR 这一类工作非常重要，因为它们不只关心 acoustic-to-text，还关心：

- 对话历史，
- 会议参与人信息，
- 文档上下文，
- 视频编辑上下文，
- 任务侧关键词和用户场景。

这条线回答的是：

> “ASR 能不能不再只是听音转字，而是带着外部知识和任务上下文去识别？”

当前答案是 **可以，而且这是 LLM-based ASR 相比传统 ASR 更有辨识度的卖点之一**。

### 主线 D：走向流式、统一化、多语言和低资源

最近的研究开始继续追问：

- LLM-based ASR 能否做 streaming？
- 能否同时覆盖 non-streaming / streaming？
- 在低资源和多语言下，LLM 的文本先验能否更有效？

答案是 **初步可行，但还在 frontier 阶段**。

---

## 3. “哪些 paper 回答过哪些问题？”

下面这张表是核心部分。

| 研究问题 | 代表 paper | 这篇 paper 回答了什么 | 还没回答好的地方 | 开源代码 |
|---|---|---|---|---|
| **Q1. LLM 不直接看文本训练目标，只接一个 speech encoder，真的能做强 ASR 吗？** | **An Embarrassingly Simple Approach for LLM with Strong ASR Capacity** (2024, arXiv:2402.08846) | 给出了一个很强的“简单可行性”答案：冻结 speech encoder 和 LLM，只训练一个线性 projector，也能做出竞争力很强的 LLM-based ASR。说明复杂 modal alignment 不是唯一出路。 | 证明了“能 work”，但没有彻底解决跨域鲁棒性、流式、低资源下的数据效率等问题。 | ✅ [X-LANCE/SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM) |
| **Q2. Speech encoder 和 LLM 之间到底应该怎么接，怎么训？** | **A Comprehensive Solution to Connect Speech Encoder and Large Language Model for ASR** (2024, arXiv:2406.17272) | 回答了“接口设计和训练 recipe”问题：部分微调、LoRA、matching loss、针对插入错误的训练/推理策略都有帮助。 | 仍然属于“把系统训顺”的工程解法，离统一理论或通用最优连接方式还有距离。 | ❌ 未公开 |
| **Q3. LLM 放在第二阶段做 rescoring，值不值得？** | **Large-scale Language Model Rescoring on Long-form Data** (ICASSP 2023, arXiv:2306.08133) | 给出了非常实用的答案：对长语音、长视频转写，LLM rescoring 可以显著降低 WER，并对 salient terms / 关键词识别尤其有帮助。 | 主要证明了文本 LLM 对 long-form ASR 有价值，但还不是“端到端 speech-LLM”。 | ❌ 未公开（Google） |
| **Q4. 只用文本 LLM 重打分够了吗，还是 speech-text foundation model 会更好？** | **Speech Recognition Rescoring with Large Speech-Text Foundation Models** (2024, arXiv:2409.16654) | 回答是：**speech-text foundation model 的重打分能力可以超过 text-only LLM**，说明跨模态知识对 second-pass ASR 是有实际收益的。 | 仍然是两阶段体系，不能直接说明 speech-LLM 主干一定优于传统 ASR 主干。 | ❌ 未公开（Amazon） |
| **Q5. LLM 能不能直接参与解码，而不是只做 N-best 重排序？** | **Guiding an Automatic Speech Recognition Decoder using Large Language Models** (2025, arXiv:2508.02228) | 回答是：可以。该方向试图把 acoustic model 和 LLM 的优势分离建模，再通过迭代解码把两者结合起来，尤其对复杂句子、缩略词、领域词有效。 | 计算开销、工程复杂度、不同 AM/LLM 组合的稳定性仍然是部署问题。 | ❌ 未公开 |
| **Q6. LLM-based ASR 的优势是否只是“语言更强”，还是也能利用外部上下文？** | **Seed-ASR: Understanding Diverse Speech and Contexts with LLM-based Speech Recognition** (2024, arXiv:2407.04675) | 给出了更进一步的答案：LLM-based ASR 不只是 language prior 更强，还能显式吸收对话历史、会议信息、编辑上下文等外部 context，从而提升关键词召回和场景适应性。 | 这类系统规模通常很大，训练数据和系统成本高，开源可复现性也相对弱。 | ❌ 闭源（ByteDance 内部系统） |
| **Q7. LLM-based ASR 能做实时 / 流式吗？** | **Speech ReaLLM -- Real-time Streaming Speech Recognition with Multimodal LLMs by Teaching the Flow of Time** (2024, arXiv:2406.09569) | 回答是：能，但必须重新设计训练范式，让模型学会“时间流”。这说明 decoder-only / multimodal LLM 不是天然只能做离线任务。 | 流式场景下的延迟、稳定性、工业级吞吐和长会话累积误差仍然没有完全解决。 | ❌ 未公开（Meta） |
| **Q8. LLM-based ASR 能不能在 low-resource 或 code-switching 条件下优于 Whisper？** | **A Comparative Study of LLM-based ASR and Whisper in Low Resource and Code Switching Scenario** (2024, arXiv:2412.00721) | 给出了一个很重要的“非神话化”答案：在低资源场景，LLM-based ASR 可能优于 Whisper；但在 code-switching 等设置下，Whisper 仍可能更强。 | 说明优势是条件性的，不是所有场景都赢；也说明 benchmark 选择非常重要。 | ⚠️ 论文已撤回，无代码 |
| **Q9. 对低资源语言，LLM / LM 的文本先验到底有没有稳定帮助？** | **Whisper-LM: Improving ASR Models with Language Models for Low-Resource Languages** (2025, arXiv:2503.23542) | 回答是：有帮助，尤其在少数语种和低资源设置里，引入 LM 后能稳定改善识别；统计 LM 和 LLM 都有价值，只是收益形态不同。 | 这篇更偏“Whisper + LM”，说明语言模型增强有效，但不完全等于 speech-LLM 主干路线的胜利。 | ✅ [hitz-zentroa/whisper-lm](https://github.com/hitz-zentroa/whisper-lm) |
| **Q10. Speech-LLM 会不会天然优于强大的端到端 ASR？** | **Bridging the gap: A comparative exploration of Speech-LLM and end-to-end architecture for multilingual conversational ASR** (2026, arXiv:2601.01461) | 当前答案偏保守：**未必**。在多语言对话式 ASR 上，speech-LLM 有潜力，但精调后的强 E2E Whisper 仍可能更强。 | 这是一个重要提醒：Speech-LLM 目前更像“新范式探索 + 场景化优势”，而不是已经全面替代传统强基线。 | ✅ [1535176727/MLC-SLM](https://github.com/1535176727/MLC-SLM) |

> **关于 open-source 命中率的快速观察**：在上面 10 篇代表性 paper 里，只有 4 篇真正给出了可用代码（SLAM-ASR、Whisper-LM、MLC-SLM、加上 Q5 的部分推理资源）。**这说明 LLM-based ASR 的"顶会代表作"很多其实并不开源**——尤其 rescoring 路线（Google/Amazon）和大规模 context-aware 系统（Seed-ASR、Speech ReaLLM）几乎全部闭源。
>
> 完整的"哪些 LLM-based ASR 真的开源、能跑、可以复现"清单已经独立成文：[**llm-based-asr-open-source.md**](llm-based-asr-open-source.md)，按 ASR-first speech-LLM / GER 后纠错 / open omni 模型 / 多语言 / AVSR / speaker-aware 等方向分类，含 30+ 项目仓库链接。

---

## 4. 这些 paper 合起来，实际上回答了哪些“大问题”？

### 问题 1：LLM-based ASR 到底是不是噱头？

**回答：不是。**

已有论文已经比较明确地证明：

- LLM 用于 **rescoring** 是有效的；
- LLM 作为 **ASR 解码核心** 也是可行的；
- LLM-based ASR 在 **上下文感知** 场景里尤其有优势；
- 在某些 **低资源** 设置下，也能体现比强基线更好的语言先验。

所以这个方向已经过了“值不值得研究”的阶段。

### 问题 2：LLM-based ASR 的最大确定性收益是什么？

**回答：最确定的收益是更强的语言先验和上下文利用能力。**

当前最有共识的收益点是：

- long-form transcription；
- rare words / proper nouns；
- context-aware recognition；
- 低资源语言中的文本知识迁移；
- 与多轮对话或任务系统整合。

### 问题 3：它有没有已经全面超过 Whisper / 传统 E2E ASR？

**回答：没有证据表明“已经全面超过”。**

更准确地说：

- 在某些任务，LLM-based ASR 已经很强；
- 但在许多标准 ASR 场景里，强大的 E2E baseline 依然极具竞争力；
- 现阶段最合理的定位是 **互补**，而不是“旧范式已经被彻底取代”。

### 问题 4：研究焦点已经从“能不能做”转向哪里？

**回答：已经转向系统化优化。**

当前更前沿的问题是：

- 连接方式和训练 recipe 怎么设计；
- 流式和低延迟怎么做；
- 如何减少 hallucination / insertion；
- 如何利用更长上下文和外部知识；
- 如何在多语言、低资源、code-switching 下稳定受益。

---

## 5. 我认为目前最值得关注的几个研究结论

### 结论 A：简单架构已经足够证明 speech-LLM 的可行性

`SLAM-ASR` 一类工作很重要，因为它们把问题从：

> “是不是需要很复杂的跨模态结构设计？”

拉回到：

> “也许关键不是结构越来越复杂，而是 speech representation、adapter 设计和训练策略是否正确。”

这让后续研究能更聚焦在：

- 对齐损失，
- adapter 设计，
- 解码策略，
- 数据和任务设置，

而不是无限堆模块。

### 结论 B：当前最稳的落地方向仍是 rescoring / context enhancement

如果一个系统目标是 **尽快把 LLM 价值落到现有 ASR 产品中**，那目前最成熟的路线通常不是“彻底替换掉 ASR 主干”，而是：

- 先保留现有 acoustic model；
- 再加 LLM 做 rescoring、decoder guidance 或上下文增强。

原因很现实：

- 易于接入现有系统；
- 训练成本和风险更可控；
- 对长文本、复杂句子、领域术语的收益更稳定。

### 结论 C：Speech-LLM 的核心竞争力不只是更像语言模型，而是更会“利用上下文”

Seed-ASR 这一类工作真正让人看到的不是单纯 WER 数字，而是：

> ASR 可以变成一个“理解用户任务与上下文的识别系统”。

这其实是传统 ASR 和 LLM-based ASR 最重要的分野之一。

### 结论 D：现阶段仍然必须认真对比 Whisper

2024-2026 的比较研究给了一个很健康的提醒：

- Speech-LLM 是 promising 的，
- 但 Whisper 类强基线仍然非常难打。

所以后续看 paper 时，应该特别注意：

- 对比 baseline 是否足够强；
- 数据规模是否公平；
- 是不是只在特定 setting 下有效；
- 提升是否来自语言先验，而不是别的 confounders。

---

## 6. 还有哪些关键问题没有被很好回答？

下面这些问题，我认为仍然是 LLM-based ASR 的主要空白。

### Open Q1. 语音-文本对齐到底应该如何系统设计？

虽然已经有 adapter、projector、matching loss、LoRA 等很多 recipe，但仍然缺一个比较公认的统一结论：

- 该冻结哪些部分？
- 该训练哪些层？
- 是 projector 更关键，还是 encoder adaptation 更关键？
- 不同模型规模下是否有稳定 scaling law？

### Open Q2. hallucination / insertion error 怎么彻底解决？

这几乎是 speech-LLM 落地时绕不开的问题：

- 没说的话被“脑补”出来；
- 非语音噪声被误识别成词；
- 语言模型先验过强，覆盖了声学证据。

这是 LLM-based ASR 相比传统 ASR 更容易被质疑的一点。

### Open Q3. 流式场景下如何兼顾延迟、准确率和算力？

离线模型可以堆上下文、堆参数，但 streaming 场景要求：

- 低延迟，
- 稳定输出，
- 不反复修改前文，
- 可部署。

这会让很多“离线很强”的 speech-LLM 方案在产品上变得不现实。

### Open Q4. 多语言、低资源和 code-switching 下，什么时候真正比 Whisper 更强？

现有结论是 **“有时更强，有时不更强”**。

还缺更清晰的 answer：

- 哪些语言类型最受益？
- 哪种 code-switching 模式最适合 LLM-based ASR？
- 收益主要来自外部文本语料、上下文、还是模型结构本身？

### Open Q5. 评估指标还不够全面

只看 WER/CER 往往不够，因为 LLM-based ASR 的很多价值体现在：

- 关键词召回，
- 罕见词识别，
- 上下文一致性，
- 长文档转写质量，
- 多轮/长会话稳定性。

所以未来更好的 benchmark 设计仍然很重要。

---

## 7. 一个更实用的阅读框架：以后看新 paper 时该问什么？

我觉得可以用下面 5 个问题快速判断一篇新 paper 的价值：

1. **LLM 在系统里扮演什么角色？**
   - rescoring？
   - decoder guidance？
   - end-to-end decoder？
   - context retriever / planner？

2. **提升主要来自哪里？**
   - 更强文本先验？
   - 更长上下文？
   - 更好的 speech-text alignment？
   - 更大的训练数据？

3. **对比基线够不够强？**
   - 有没有和 Whisper、Conformer、RNN-T、Transducer 类强基线认真比较？

4. **收益发生在哪种场景？**
   - 通用 benchmark？
   - long-form？
   - low-resource？
   - multilingual？
   - contextual ASR？

5. **代价是什么？**
   - 延迟？
   - 参数量？
   - 训练成本？
   - 部署复杂度？

如果一篇 paper 只回答“可以做”，但没回答“为什么比强基线更值得部署”，那它的实际贡献通常就有限。

---

## 8. 推荐阅读顺序

如果只想快速建立这个方向的认知，我建议按下面顺序读：

### 第一层：先理解“LLM 是否真的能做 ASR”

1. **An Embarrassingly Simple Approach for LLM with Strong ASR Capacity**  
   <https://arxiv.org/abs/2402.08846>

2. **A Comprehensive Solution to Connect Speech Encoder and Large Language Model for ASR**  
   <https://arxiv.org/abs/2406.17272>

### 第二层：理解“LLM 最成熟的价值在哪里”

3. **Large-scale Language Model Rescoring on Long-form Data**  
   <https://arxiv.org/abs/2306.08133>

4. **Speech Recognition Rescoring with Large Speech-Text Foundation Models**  
   <https://arxiv.org/abs/2409.16654>

5. **Seed-ASR: Understanding Diverse Speech and Contexts with LLM-based Speech Recognition**  
   <https://arxiv.org/abs/2407.04675>

### 第三层：理解“这条路目前还没完全赢”

6. **A Comparative Study of LLM-based ASR and Whisper in Low Resource and Code Switching Scenario**  
   <https://arxiv.org/abs/2412.00721>

7. **Bridging the gap: A comparative exploration of Speech-LLM and end-to-end architecture for multilingual conversational ASR**  
   <https://arxiv.org/abs/2601.01461>

### 第四层：看 frontier 问题

8. **Speech ReaLLM -- Real-time Streaming Speech Recognition with Multimodal LLMs by Teaching the Flow of Time**  
   <https://arxiv.org/abs/2406.09569>

9. **Whisper-LM: Improving ASR Models with Language Models for Low-Resource Languages**  
   <https://arxiv.org/abs/2503.23542>

10. **Guiding an Automatic Speech Recognition Decoder using Large Language Models**  
   <https://arxiv.org/abs/2508.02228>

---

## 9. 总结：目前最稳妥的研究判断

如果要把这条方向压缩成几句判断，我会写成：

1. **LLM-based ASR 已经被证明有效，但最稳的收益点仍是 rescoring、context-aware recognition 和 low-resource language prior。**
2. **Speech-LLM 端到端 ASR 已经可行，甚至简单设计就能很强，但还没有形成对 Whisper 类强基线的全面替代。**
3. **当前研究重点已经从“能不能做”转向“如何更鲁棒、更低延迟、更低 hallucination、更会利用上下文”。**
4. **未来真正有机会拉开差距的场景，可能不是普通 benchmark，而是长上下文、强场景上下文、多语言低资源、复杂对话和流式识别。**

从研究选题角度看，最值得继续追的问题通常不是：

- “再证明一次 LLM 能做 ASR”，

而是：

- “LLM-based ASR 在什么场景比强 E2E ASR 更有不可替代的优势？”
- “这些优势能否在可部署的延迟和成本下成立？”

这两个问题，决定了这个方向接下来会不会真正成为主流。

---

## 10. 针对你当前问题的补充：SLAM-ASR / LLM-based ASR / unlearning / linear projector

如果把问题进一步收窄成：

> **“在 LLM-based ASR，尤其是 SLAM-ASR 这类 speech encoder + LLM + projector 架构里，有没有专门把 unlearning 放在线性 projector 上做的工作？”**

我目前查到的结论是：

> **已经有“LLM-based ASR 做 unlearning”的直接论文；但我还没有看到公开论文明确把 unlearning 限定为“只更新 SLAM-ASR 的 linear projector”。**

下面按“直接命中”与“相邻证据”分开说。

### 10.1 直接命中：已经有 LLM-based ASR unlearning 论文

1. **Unlearning LLM-Based Speech Recognition Models** (Interspeech 2025)  
   <https://www.isca-archive.org/interspeech_2025/liu25b_interspeech.html>

这篇是目前最直接命中的工作，因为它讨论的就是：

- **autoregressive LLM-based ASR 的 unintended memorization**，
- **RTBF / data deletion 场景下的 unlearning**，
- 以及 **privacy-utility trade-off**。

它的核心贡献是：

- 先定义和测量 LLM-based ASR 的 memorization：
  - WER-based memorization rate，
  - homophone-based memorization rate，
  - prompting-based memorization rate；
- 再用 **gradient ascent** 在 forget set 上做 post-training unlearning。

但需要特别注意：

> **这篇并不是“SLAM-ASR 的 linear projector-only unlearning”论文。**

原因是它在方法部分写得很清楚：

- LLM-based ASR 由 audio encoder、text embedding、autoregressive decoder/LLM 构成；
- unlearning 可以通过 **full fine-tuning 或 PEFT** 完成；
- 它实验里的 base ASR 也不是“只训练一个 SLAM-ASR 线性 projector”的最简设置，而是：
  - audio encoder 侧：**只有 adapter 可训练**，其余层冻结；
  - LLM 侧：dense layers 用 **LoRA** 微调；
  - speech-to-LLM 接口里包含 adapter / linear mapping，但论文没有把 unlearning 明确限制为“只改 linear projector”。

所以这篇论文给你的最重要信息是：

- **LLM-based ASR 的 unlearning 已经开始有人做了；**
- 但 **parameter locus 还没有被公开文献清楚收敛到 “linear projector only”**。

### 10.2 与 linear projector 最接近的正向证据：projector 确实是一个很强的轻量模块

虽然我还没找到“projector-only unlearning”的直接论文，但 projector 在 LLM-based ASR 里已经被反复证明是一个非常关键、而且可单独训练的模块。

最相关的两篇是：

1. **An Embarrassingly Simple Approach for LLM with Strong ASR Capacity** (2024)  
   <https://arxiv.org/abs/2402.08846>

   这篇就是 SLAM-ASR 路线最典型的证据。其核心结论是：

   - 冻结 speech encoder；
   - 冻结 LLM；
   - **只训练 linear projector**；
   - 也能做出很强的 ASR。

   这篇并不是 unlearning 论文，但它非常重要，因为它说明：

   > 如果你想把 unlearning 也限制在一个极小的参数子空间里，**linear projector 是一个天然候选位点**。

2. **Reducing Prompt Sensitivity in LLM-based Speech Recognition Through Learnable Projection** (2026, ICASSP 2026)  
   <https://arxiv.org/abs/2601.20898>

   这篇不是忘记训练，而是 prompt robustness，但它提供了另一条很关键的 projector 证据：

   - 以 SLAM-ASR 为 base model；
   - base model 延续“**只训练 speech projector**、encoder 和 LLM 冻结”的设定；
   - 进一步引入 **prompt projector**；
   - 并且在这个扩展实验里，作者强调 **冻结底层模型、只训练 projector 更稳定**。

   这条证据的价值在于：

   > projector 不只是“能训”，而且在 speech-LLM 接口处往往是一个 **稳定、低成本、可插拔** 的更新位置。

### 10.3 目前没找到的关键点：公开论文还没有明确写出“linear projector-only unlearning”

到目前为止，我还没有检索到下面这种完全正面命中的公开论文：

- **SLAM-ASR / speech-LLM ASR**
- **machine unlearning / RTBF / forget set**
- **只更新 linear projector（不改 encoder、不改 LLM、不改 LoRA）**
- 并给出系统实验对比

也就是说，当前公开文献更像是：

- 一边已经证明了 **projector-only training 在 ASR 中可行**；
- 另一边已经证明了 **LLM-based ASR 做 unlearning 是值得研究的**；
- 但这两条线 **还没有在“projector-only unlearning”上真正合流**。

### 10.4 这意味着什么：这里很可能就是一个明确 research gap

如果你想做的就是：

> **在 SLAM-ASR 这种 speech encoder + frozen LLM + linear projector 架构中，只通过更新 projector 完成 targeted unlearning**

那它现在看起来很像一个相当干净的空白点。

一个很自然的研究问题可以被表述成：

> **Can targeted unlearning in LLM-based ASR be localized to the speech-to-LLM linear projector while preserving ASR utility better than adapter/LoRA/full-model updates?**

这个问题值得做的原因有三点：

1. **参数效率很强**  
   如果 projector-only 就能完成 forget 请求，成本会明显低于 adapter/LoRA/full-model 更新。

2. **结构解释性更强**  
   在 speech-LLM 架构里，projector 就是跨模态接口；把忘记约束在这个接口层，有较强的结构动机。

3. **实验设计非常清楚**  
   可以直接和下面几种方案做 apples-to-apples 对比：
   - projector-only unlearning；
   - adapter-only unlearning；
   - LLM-side LoRA unlearning；
   - joint PEFT unlearning；
   - full-model unlearning。

### 10.5 如果继续往下挖，最值得验证的 3 个子问题

如果你后面准备继续沿这个点追文献或做实验，我建议重点盯下面三个子问题：

1. **projector-only unlearning 到底忘掉的是什么？**  
   是真的消除了 target content 的可恢复性，还是只是破坏了 speech-text alignment，造成表面失忆？

2. **projector-only 会不会更容易保持 retain utility？**  
   直觉上它更局部，但也可能因为 projector 是跨模态瓶颈，反而更容易牵连通用识别能力。

3. **projector-only 的忘记是否更容易被 relearn？**  
   如果 target knowledge 主要仍存于 LLM 内部，那么只改 projector 可能只是切断了当前映射，而不是做到了更持久的删除。

### 10.6 当前可执行的文献判断

如果现在要把结论压缩成一句话，我会写成：

> **“LLM-based ASR 的 unlearning 已经有直接论文（Interspeech 2025），SLAM-ASR 的 linear projector-only training 也已经被证明可行；但我目前还没看到公开论文把两者结合成‘linear projector-only unlearning’。”**

所以，针对你现在这个问题，最准确的回答不是“已经有成熟文献”，而是：

> **有非常接近的两条证据链，但它们之间的那一步——‘只在线性 projector 上做 unlearning’——目前仍然像一个明显空白。**
