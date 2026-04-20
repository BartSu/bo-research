# LLM-based ASR × Machine Unlearning — 交叉领域 landscape

**Cycle date:** 2026-04-19
**Window:** 2025-H2 – 2026-Q2（重点补充 `idea/slam-asr-audio-embedding-corrupted-prompts.md` 写于 2026-03-26 之后未覆盖的工作）

本文档是"LLM-based ASR + 机器遗忘"这一窄交叉的近期 landscape。精确落在交叉点上的论文仍然很少（仅 2 篇），但周边生态（TTS / SER / SLU / 说话人级遗忘）在 2025-H2 明显加速，可作为对比基线或方法论建材。

---

## 1. 精确交叉：LLM-based ASR × unlearning

| 论文 | 一句话 | 和 SLAM-ASR forgetting idea 的关系 |
|---|---|---|
| **Unlearning LLM-Based Speech Recognition Models** (Liu et al., Interspeech 2025) | LLM-ASR unlearning 的首篇；post-training gradient-based update，评估 WER / homophone / prefix prompt 三招记忆化测试 | 任务原型与评估协议的直接来源；必引对比；尚未检索到 v2 或期刊扩展版 |
| _(user's proposed idea)_ classifier-gated audio-prompt corruption | 见 `idea/slam-asr-audio-embedding-corrupted-prompts.md` | **仍是开放空位** |

---

## 2. 语音侧遗忘的基础性工作（近 6 个月新出）

### 2.1 Speech Unlearning (arXiv 2506.00848, 2025-06)

Liao et al. 定义语音遗忘的两类基础任务：**sample unlearning** 和 **class unlearning**（整个说话人类别），在关键词检测与说话人识别上给出基线。核心观察：语音遗忘因 **feature entanglement + temporal dependency** 比图像/文本遗忘显著更难。

- **用作：** 基础 benchmark 和定义参考；必引对比。
- **不触及 LLM-ASR；作用对象是传统分类器。**

### 2.2 UnSLU-BENCH / "Alexa, can you forget me?" (Koudounas et al., Interspeech 2025, arXiv 2505.15700)

首个 **SLU** 机器遗忘基准。覆盖 FSC / SLURP / ITALIC / SpeechMASSIVE 四数据集四语言，评测 8 种 MU 方法在**说话人级 RTBF** 上的 efficacy / utility / efficiency，提出联合度量。

- **用作：** 评测框架的直接竞品，也是可复用 benchmark。做 SLAM-ASR unlearning 基本绕不过它。
- **差距：** 任务是 SLU 分类，不是 ASR 序列转写；也不是 LLM-based。

### 2.3 SISA++ for Paralinguistic Speech (Phukan et al., Interspeech 2025, arXiv 2506.02230)

把 SISA（Sharded-Isolated-Sliced-Aggregated）扩展为 SISA++（分片训练 + 权重平均），应用于 SER 和抑郁检测。提供 "cookbook recipes"——选择 feature / 下游架构以减少遗忘后的性能塌陷。

- **用作：** "重训式基线"代表，用来对比"plug-and-play 不动参数"的优势。
- **不适用：** SISA 需要分片重训，本身违背 plug-and-play 精神。

### 2.4 Forget-Set-Only SER Unlearning (Ren et al., arXiv 2510.04251, v2 2025-12)

只用 forget set（无需 retain set）做对抗式微调来遗忘 SER 样本。

- **哲学相通：** "仅在遗忘侧施加干预"的取向与 ECO 的"仅在推理端拦截"精神一致。
- **差异点：** 他们仍然修改模型参数；你的 classifier-gated corruption 可以做到**完全不动模型参数**——这是差异化卖点。

### 2.5 QPAudioEraser (Pathak et al., arXiv 2507.22208, 2025-07)

"量子启发"的音频生物特征类级遗忘：权重初始化 + 叠加式标签变换 + 不确定性最大化损失。AudioMNIST / Speech Commands / LibriSpeech / Speech Accent Archive 四数据集；报告 0% forget accuracy，保留集仅降 0.05%。

- **用作：** 说话人类级遗忘的 SOTA 对比。
- **不适用：** 作用于分类器（ResNet18/ViT/CNN），不是 LLM-ASR。

---

## 3. TTS 侧对称工作（说话人身份遗忘）

### 3.1 TGU / "Do Not Mimic My Voice" (Kim et al., ICML 2025, arXiv 2507.20140)

首个 ZS-TTS 说话人身份遗忘框架；提出 Teacher-Guided Unlearning 和新指标 spk-ZRF；代码开源。

- **用作：** "说话人遗忘"家族的姊妹篇引用，证明方向社区关心。
- **差异：** 模态相反（合成 vs 识别）。

### 3.2 TruS / "Erasing Your Voice Before It's Heard" (arXiv 2601.20481, 2026-01-28)

**训练无关**的 ZS-TTS 说话人遗忘：提取 ID-prototype，在推理时 **steering hidden activations** 压制目标说话人；对 seen/unseen 说话人都有效。

- **最接近用户 idea 的哲学：** "inference-time control, no retraining"。
- **差异：**
  - 作用点：TTS 解码端 vs. ASR 编码/projector 端
  - 方法：直接 steer hidden activations vs. classifier-gated audio-prompt corruption
  - 没有 gating classifier；你可以强调 gating 带来的 utility 保持优势
- **做强对比：** TruS 是目前最重要的"inference-time speech unlearning"参照系。

---

## 4. LLM 通用侧的可迁移新方法

这些是纯文本 LLM unlearning 工作，但哲学或结构可跨模态迁移：

- **GUARD** (arXiv 2505.13312, 2025-05)：生成时无训练遗忘，prompt classifier + token 级动态惩罚。哲学上与 ECO 同宗；是"audio-side 生成时遗忘"的文本端最强类比。
- **GRUN** (arXiv 2502.17823)：soft-gate + ReFT 抑制模块。结构上与 "gate + 表征干预" 同构，仍在文本端。
- **Pre-Forgettable Models** (arXiv 2509.15230)：把 prompt learning 当作原生遗忘机制，免重训。
- **BLUR benchmark** (arXiv 2506.15699) 与 **Unlearning Verification Survey** (arXiv 2506.15115)：评估/验证工具箱。
- **Unlearning Isn't Invisible** (arXiv 2506.14003)：检测"遗忘痕迹"的反向研究；评估隐蔽性时可引。

---

## 5. 语音侧记忆化 / 泄漏（严重空白）

除 Liu Interspeech 2025 那三个测试（WER / homophone / prefix prompt），**针对 Whisper / SALMONN / Qwen-Audio / SLAM-ASR 的系统性训练数据提取攻击研究几乎空白**。

邻近工作：

- **Jailbreak-AudioBench** (arXiv 2501.13772)
- **AudioJailbreak** (arXiv 2505.14103)

两者都是内容安全（绕过安全对齐），不是 PII / memorization 抽取。**这是一个清晰的独立论文机会。**

---

## 6. 研究空位 (Research Gaps)

1. **音频 projector 层干预尚无人做。** ECO 在文本 embedding 加噪；TruS 在 TTS hidden states 做 steering；**没有工作在 SLAM-ASR 的 projector 输出上做 classifier-gated corruption**。SLAM-ASR 的可训练组件几乎只有 projector，使其成为"最小入侵遗忘"的理想接入点。论文可强调"只干预 projector，不触碰 LLM 或 encoder"。

2. **Phrase / entity-level 遗忘在语音侧几乎空白。** 现有语音 unlearning 几乎全是：
   - speaker-level（UnSLU-BENCH, QPAudioEraser, TGU, TruS）
   - 或 sample-level（Liu Interspeech, Ren 2510.04251）
   
   **让 ASR 忘记特定人名 / 商标 / 地址**这种 phrase-level、跨说话人的遗忘任务，目前没有标准 benchmark——可顺带定义一个。

3. **Training-free + classifier-gated 的组合点空缺。**
   - TruS: training-free 但无 classifier gate
   - GUARD: classifier-gated 但在 text token 侧
   - ECO: 二者结合但纯文本
   
   **语音侧的"audio-prompt classifier + audio-embedding corruption"尚未出现**——这是用户方法的**核心独占坐标**。

4. **评估协议碎片化。** Speech Unlearning 用 KWS/SID；UnSLU-BENCH 用 intent classification；Liu 2025 用 WER + homophone + prefix prompt；Ren 2025 用 SER 精度。LLM-ASR 场景下"遗忘成功"需要自己的指标（forget-phrase WER、homophone rate、prefix-leakage rate、retain-set WER degradation）。可作为 **"LLM-ASR Unlearning Suite"** 副产出提出。

5. **语音 LLM 的记忆化攻击实证仍缺位。** 除 Liu 的三招，**没有人在 SLAM-ASR / SALMONN / Qwen-Audio 上系统做 membership inference 或 training-data extraction**。可作为论文独立的"威胁模型"章节，强化遗忘工作的必要性。

---

## 7. 对 `idea/slam-asr-audio-embedding-corrupted-prompts.md` 的增量结论

写于 2026-03-26 的 idea 判断"classifier-gated audio-prompt corruption for SLAM-ASR forgetting 是 gap"——**到 2026-04-19 这个判断仍然成立**。TruS (2026-01) 最接近的哲学对照，但作用点（TTS 解码 vs. ASR projector）和机制（activation steering vs. classifier-gated corruption）都不同，可以做清晰的对比。

建议加到论文叙事里的新坐标：

- TruS 是 "inference-time speech unlearning" 最强参照系，要正面讨论。
- UnSLU-BENCH 是可直接复用的评测框架（尽管任务是 SLU 而非 ASR）。
- Liu Interspeech 2025 的 WER / homophone / prefix prompt 三招是最小评估套件，需要加上 phrase-level / utility-retention 指标。

---

## 8. 本轮来源 (entry points)

Direct intersection:
- Liu et al., *Unlearning LLM-Based Speech Recognition Models*, Interspeech 2025 — https://www.isca-archive.org/interspeech_2025/liu25b_interspeech.html

Speech-side unlearning:
- Liao et al., *Speech Unlearning*, arXiv:2506.00848
- Koudounas et al., *UnSLU-BENCH*, arXiv:2505.15700
- Phukan et al., *SISA++ for Paralinguistic Speech*, arXiv:2506.02230
- Ren et al., *Forget-Set-Only SER Unlearning*, arXiv:2510.04251
- Pathak et al., *QPAudioEraser*, arXiv:2507.22208

TTS speaker forgetting:
- Kim et al., *TGU / Do Not Mimic My Voice*, ICML 2025, arXiv:2507.20140
- *TruS / Erasing Your Voice Before It's Heard*, arXiv:2601.20481

LLM-side transferable methods:
- *GUARD*, arXiv:2505.13312
- *GRUN*, arXiv:2502.17823
- *Pre-Forgettable Models*, arXiv:2509.15230
- *BLUR benchmark*, arXiv:2506.15699
- *Unlearning Verification Survey*, arXiv:2506.15115
- *Unlearning Isn't Invisible*, arXiv:2506.14003

Audio security (adjacent, not memorization):
- *Jailbreak-AudioBench*, arXiv:2501.13772
- *AudioJailbreak*, arXiv:2505.14103
- *Targeted Speaker Poisoning in ZS-TTS*, arXiv:2603.07551
