# Open-source LLM-based ASR：可复现项目清单

这份文档专门解决一个问题：

> **"我现在想跑 / 复现 / 在上面做实验，哪些 LLM-based ASR 是真的开源的？"**

它不是论文综述（研究脉络见 [llm-based-asr-research-map.md](llm-based-asr-research-map.md)），而是一份按"项目 / 仓库"组织的实用清单，全部代码均已公开。

---

## 0. 快速决策

按你的目标直接挑：

| 你想做的事 | 推荐起点 |
|---|---|
| **最简 speech-LLM ASR baseline**（projector-only training） | [SLAM-LLM](#1-asr-first-speech-llm) — 事实标准底座 |
| **最强开源 ASR checkpoint**（直接跑、不训） | [Phi-4-Multimodal](#3-open-omni--dialog-speech-llmasr-上很强) / [Granite-Speech 3.3](#3-open-omni--dialog-speech-llmasr-上很强) — 都曾登顶 HuggingFace OpenASR Leaderboard |
| **多任务音频理解（不止 ASR）** | [Qwen2-Audio](#1-asr-first-speech-llm) / [SALMONN](#1-asr-first-speech-llm) / [LTU](#1-asr-first-speech-llm) / [LauraGPT](#1-asr-first-speech-llm) |
| **中文 / 多方言 LLM-based ASR** | [FireRedASR-LLM](#4-多语言--长尾语言) / [Qwen3-ASR](#1-asr-first-speech-llm) |
| **超大语种覆盖** | [Omnilingual ASR](#4-多语言--长尾语言)（1600+ 语言） |
| **Post-ASR 纠错（不重训主干）** | [HyPoradise](#2-generative-error-correction-ger) / [RobustGER](#2-generative-error-correction-ger) / [Whispering-LLaMA](#2-generative-error-correction-ger) |
| **视听 ASR (AVSR)** | [Llama-AVSR](#5-audio-visual-speech-recognition-avsr) / [MMS-LLaMA](#5-audio-visual-speech-recognition-avsr) |
| **会议 / 多说话人 ASR** | [SpeakerLM](#6-speaker-aware--multi-talker) / [LLM-Diarize-ASR-Agnostic](#6-speaker-aware--multi-talker) |
| **参数局部化研究（unlearning / efficient adaptation）** | [SLAM-LLM](#1-asr-first-speech-llm) + [idiap/llm-asr-prompt](#1-asr-first-speech-llm) |
| **数据效率：自蒸馏路线** | [DiVA](#3-open-omni--dialog-speech-llmasr-上很强) — 用 ASR 数据自蒸馏，比 Qwen-Audio 少 100× 算力 |

---

## 1. ASR-first speech-LLM

speech encoder + adapter / projector + LLM 主干路线，论文目标就是 ASR（或以 ASR 为核心任务之一）。

| 项目 | 论文 | 架构定位 | 代码仓库 |
|---|---|---|---|
| **SLAM-LLM (SLAM-ASR)** | An Embarrassingly Simple Approach for LLM with Strong ASR Capacity (arXiv:2402.08846) | speech encoder + 线性 projector + 冻结 LLM；同 repo 还有 ST、AAC 等多任务 recipe | [X-LANCE/SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM) |
| **Qwen-Audio** | Qwen-Audio: Advancing Universal Audio Understanding (arXiv:2311.07919) | Whisper encoder + Qwen LLM，多任务联合训练 | [QwenLM/Qwen-Audio](https://github.com/QwenLM/Qwen-Audio) |
| **Qwen2-Audio** | Qwen2-Audio Technical Report (arXiv:2407.10759) | Whisper-style encoder + Qwen2 LLM | [QwenLM/Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio) |
| **Qwen3-ASR / Qwen3-Omni** | Qwen3 Technical Reports (2025) | Qwen3 系 LLM + 音频编码器；52 语种，开源 SOTA | [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), [QwenLM/Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) |
| **SALMONN** | SALMONN: Towards Generic Hearing Abilities for LLMs (ICLR 2024, arXiv:2310.13289) | dual encoder (Whisper + BEATs) + Q-Former + Vicuna | [bytedance/SALMONN](https://github.com/bytedance/SALMONN) |
| **WavLLM** | WavLLM: Towards Robust and Adaptive Speech LLM (EMNLP 2024 Findings, arXiv:2404.00656) | WavLM + Whisper dual encoder + LLaMA | [aka.ms/wavllm](https://aka.ms/wavllm) |
| **LTU / LTU-AS** | Listen, Think, and Understand (ICLR 2024, arXiv:2305.10790) | audio encoder + LLaMA，QA 风格训练；含 OpenAQA / OpenASQA 数据 | [YuanGongND/ltu](https://github.com/YuanGongND/ltu) |
| **LauraGPT** | LauraGPT: Listen, Attend, Understand, and Regenerate Audio with GPT (arXiv:2310.04673) | decoder-only 统一音频-文本 LLM；ASR 是 7+ 个统一任务之一 | [lauragpt.github.io](https://lauragpt.github.io)（FunAudioLLM/FunASR 内集成） |
| **MooER** | MooER: LLM-based Speech Recognition and Translation Models from Moore Threads (arXiv:2408.05101) | Paraformer-style encoder + adapter + LLM；5k / 80k 小时模型 | [MooreThreads/MooER](https://github.com/MooreThreads/MooER) |
| **LLaMA-Omni** | LLaMA-Omni: Seamless Speech Interaction with LLMs (arXiv:2409.06666) | speech encoder + speech adaptor + LLaMA + streaming speech decoder | [ictnlp/LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni) |
| **OSUM** | OSUM: Advancing Open Speech Understanding Models with Limited Resources in Academia (arXiv:2501.13306) | Whisper encoder + Qwen2 LLM；ASR+X 多任务 | [ASLP-lab/OSUM](https://github.com/ASLP-lab/OSUM) |
| **Whisper-LM** | Whisper-LM: Improving ASR Models with Language Models for Low-Resource Languages (2025, arXiv:2503.23542) | Whisper + (统计 LM / LLM) 后融合，主打低资源语种 | [hitz-zentroa/whisper-lm](https://github.com/hitz-zentroa/whisper-lm) |
| **MLC-SLM** | Bridging the gap: Speech-LLM vs E2E for multilingual conversational ASR (2026, arXiv:2601.01461) | Speech-LLM vs Whisper-E2E 对照实验 | [1535176727/MLC-SLM](https://github.com/1535176727/MLC-SLM) |
| **Prompt-projector for LLM-ASR** | Reducing Prompt Sensitivity in LLM-based Speech Recognition Through Learnable Projection (ICASSP 2026, arXiv:2601.20898) | SLAM-ASR base + 额外 prompt projector；**冻结底层只训 projector** | [idiap/llm-asr-prompt](https://github.com/idiap/llm-asr-prompt) |

---

## 2. Generative Error Correction (GER)

把 LLM 放在 ASR **之后**做生成式纠错——可以修正 N-best 列表里根本没出现的 token。这一线和 SLAM-ASR 主干路线完全不同的"LLM 介入位点"，几乎全开源。

| 项目 | 论文 | 代码仓库 |
|---|---|---|
| **HyPoradise** | HyPoradise: An Open Baseline for Generative Speech Recognition with LLMs (NeurIPS 2023, arXiv:2309.15701) | [Hypotheses-Paradise/Hypo2Trans](https://github.com/Hypotheses-Paradise/Hypo2Trans) |
| **Whispering-LLaMA** | Whispering LLaMA: A Cross-Modal Generative Error Correction Framework for ASR (EMNLP 2023, arXiv:2310.06434) | [Srijith-rkr/Whispering-LLaMA](https://github.com/Srijith-rkr/Whispering-LLaMA) |
| **RobustGER** | LLMs are Efficient Learners of Noise-Robust Speech Recognition (ICLR 2024, arXiv:2401.10446) | [YUCHEN005/RobustGER](https://github.com/YUCHEN005/RobustGER) |
| **NeKo** | NeKo: Cross-Modality Post-Recognition Error Correction with Tasks-Guided MoE LM (arXiv:2411.05945) | 跟随论文（NVIDIA Research） |
| **GenSEC Challenge** | LLM-Based Generative Error Correction for ASR/Speaker Tagging/Emotion (arXiv:2409.09785) | 基于 HyPoradise / RobustGER 数据 |

---

## 3. Open omni / dialog speech-LLM（ASR 上很强）

不是 ASR-first 设计，但开源 ASR 成绩盖过大多数专门 speech-LLM——**做 baseline 比较时必须纳入**。

| 项目 | 论文 / 报告 | ASR 相关性 | 代码仓库 |
|---|---|---|---|
| **SpeechGPT** | arXiv:2305.11000 | 最早一批开源 speech-LLM，离散 token + LLaMA | [0nutation/SpeechGPT](https://github.com/0nutation/SpeechGPT) |
| **Phi-4-Multimodal** | Phi-4-Mini Technical Report (arXiv:2503.01743) | Microsoft 5.6B 多模态，**HuggingFace OpenASR Leaderboard 一度第一 (WER 6.14%)**，强于 Whisper-V3 | [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) |
| **Granite-Speech 3.3 / 4.0** | arXiv:2505.08699 | IBM；Granite-3.3 + modality alignment + LoRA；**也曾登顶 OpenASR Leaderboard** | [ibm-granite/granite-speech-models](https://github.com/ibm-granite/granite-speech-models) |
| **Kimi-Audio** | Kimi-Audio Technical Report (arXiv:2504.18425) | Moonshot AI 7B；13M 小时音频预训练，ASR SOTA 级开源结果 | [MoonshotAI/Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio) |
| **GLM-4-Voice** | arXiv:2412.02612 | 智谱 9B 端到端语音对话；1T speech-text token 预训练 | [zai-org/GLM-4-Voice](https://github.com/zai-org/GLM-4-Voice) |
| **Step-Audio / Step-Audio 2** | arXiv:2502.11946 / arXiv:2507.16632 | StepFun 130B 双码本 speech-LLM；ASR 是 understanding 部分主任务 | [stepfun-ai/Step-Audio](https://github.com/stepfun-ai/Step-Audio), [/Step-Audio2](https://github.com/stepfun-ai/Step-Audio2) |
| **Mini-Omni / Mini-Omni 2** | Mini-Omni 系列 (2024) | Qwen2 backbone + Whisper encoder；端到端流式 | [gpt-omni/mini-omni](https://github.com/gpt-omni/mini-omni), [/mini-omni2](https://github.com/gpt-omni/mini-omni2) |
| **Moshi** | Moshi (Kyutai, 2024) | 全双工 speech-text foundation；理论延迟 160ms | [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) |
| **Baichuan-Audio / Baichuan-Omni** | Baichuan-Audio / Baichuan-Omni Tech Reports | 开源端到端语音交互框架 | [baichuan-inc/Baichuan-Audio](https://github.com/baichuan-inc/Baichuan-Audio), [westlake-baichuan-mllm/bc-omni](https://github.com/westlake-baichuan-mllm/bc-omni) |
| **DiVA** | Distilling an End-to-End Voice Assistant Without Instruction Training Data (ACL 2025, arXiv:2410.02678) | Stanford/GT；**用 ASR 数据自蒸馏**得到 speech-LLM；72% 胜率 vs Qwen-Audio，100× 更少算力 | [diva-audio.github.io](https://diva-audio.github.io/) → [WillHeld/DiVA-llama-3-v0-8b](https://huggingface.co/WillHeld/DiVA-llama-3-v0-8b) |
| **DeSTA2 / DeSTA2.5-Audio** | ICASSP 2025 (arXiv:2409.20007) / arXiv:2507.02768 | 描述式 speech-text alignment，无需 speech instruction-tuning data | [kehanlu/DeSTA2](https://github.com/kehanlu/DeSTA2), [/DeSTA2.5-Audio](https://github.com/kehanlu/DeSTA2.5-Audio) |
| **Audio Flamingo / AF2 / AF3** | arXiv:2402.01831, 2503.03983, 2507.08128 | NVIDIA 系列 audio-language reasoning | [NVIDIA/audio-flamingo](https://github.com/NVIDIA/audio-flamingo) |

---

## 4. 多语言 / 长尾语言

| 项目 | 论文 | LLM-based 程度 | 代码仓库 |
|---|---|---|---|
| **FireRedASR-LLM** | FireRedASR (arXiv:2501.14350) | **是**：Encoder-Adapter-LLM，LLM 用 **Qwen2-7B-Instruct** 初始化；中文/方言/英文 SOTA | [FireRedTeam/FireRedASR](https://github.com/FireRedTeam/FireRedASR) |
| **Omnilingual ASR (7B-LLM-ASR)** | Omnilingual ASR (Meta FAIR, arXiv:2511.09690) | **部分**：7B wav2vec encoder + LLM 风格 transformer decoder，覆盖 1600+ 语言；支持 zero-shot in-context learning 适配新语种。decoder 不是从预训练文本 LLM 初始化，所以是"LLM 风格"而非真正 speech-LLM | [facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr) |

> **SenseVoice** (FunAudioLLM) 和 **Whisper / WhisperX** 这类常被一并提到的多语言系统：架构上更接近传统 encoder-decoder，不算 LLM-based ASR 主干路线，所以这里不收。但它们是 SLAM-LLM、Qwen-Audio 等真正 speech-LLM 的常用 encoder/数据源，做实验时绕不开。

---

## 5. Audio-Visual Speech Recognition (AVSR)

| 项目 | 论文 | 架构 | 代码仓库 |
|---|---|---|---|
| **Llama-AVSR** | Large Language Models are Strong AVSR Learners (ICASSP 2025) + Mitigating Attention Sinks ... | AV-HuBERT (visual) + Whisper (audio) + LLaMA-3.1-8B (LoRA) | [umbertocappellazzo/Llama-AVSR](https://github.com/umbertocappellazzo/Llama-AVSR) |
| **MMS-LLaMA** | MMS-LLaMA (ACL 2025 Findings, arXiv:2503.11315) | 强调 token 压缩与计算效率 | [JeongHun0716/MMS-LLaMA](https://github.com/JeongHun0716/MMS-LLaMA) |
| **Zero-AVSR** | Zero-AVSR (ICCV 2025) | LLaMA-3.2-3B 基础上做**跨语种零样本** AVSR | [JeongHun0716/zero-avsr](https://github.com/JeongHun0716/zero-avsr) |
| **VSP-LLM** | Visual Speech Processing with LLMs | 视觉语音处理（VSR + 翻译）多任务 LLM 框架 | [Sally-SH/VSP-LLM](https://github.com/Sally-SH/VSP-LLM) |
| **AV-HuBERT** (基础设施) | AV-HuBERT | 不是 LLM-based，但**几乎所有 AVSR-LLM 的 visual encoder**都用它 | [facebookresearch/av_hubert](https://github.com/facebookresearch/av_hubert) |

---

## 6. Speaker-aware / Multi-talker

| 项目 | 论文 | 这篇做了什么 | 代码仓库 |
|---|---|---|---|
| **SpeakerLM** | SpeakerLM (arXiv:2508.06372) | 端到端 E2E-SDR：用一个多模态 LLM 同时做说话人分割、识别、ASR | 跟随论文 |
| **Diarization-Aware Multi-Speaker ASR via LLMs** | arXiv:2506.05796 | 把 diarization 输入和 frame-level speaker embedding 一起喂给 LLM-based ASR | 跟随论文 |
| **LLM-Diarize-ASR-Agnostic** | LLM-based speaker diarization correction: A generalizable approach | 不重训 ASR / diarization，用 LLM 做事后修正 | [GeorgeEfstathiadis/LLM-Diarize-ASR-Agnostic](https://github.com/GeorgeEfstathiadis/LLM-Diarize-ASR-Agnostic) |

> 这条线和第 2 章 GER 思想接近：**把 LLM 放在 ASR/diarization 之后做修正**。GenSEC challenge（见第 2 章）也包含 speaker tagging 任务。

---

## 7. 领域特定（医疗 / 法律 / 教育）：现状是稀缺

公开论文里基本 **没有真正的"领域 speech-LLM 主干"**：

- 医疗 ASR 的开源工作（United-MedASR, arXiv:2412.00055；ASR-LLM Medical Benchmarking, arXiv:2502.13982）几乎都是 **Whisper / Wav2Vec2 + LLM-style 后处理（语义纠错、术语补全）**，主干仍然是传统 ASR。
- 也就是说，**第 1–6 章列出的开源 speech-LLM 在领域适配上仍是个开放空白**——目前 publicly available 的"医疗 speech-LLM"接近于 0。
- 实际路径：拿 SLAM-LLM / Qwen-Audio / Granite-Speech 做 base，自行做领域微调，**而不是寻找一个开箱即用的"医疗 speech-LLM"**——它目前不存在。

---

## 8. 关于"open-source"覆盖率的一个观察

[llm-based-asr-research-map.md](llm-based-asr-research-map.md) 里 10 篇问题导向的代表 paper，**只有 4 篇真正给出了可用代码**（SLAM-ASR、Whisper-LM、MLC-SLM、加上部分 Q5）：

- **rescoring 路线（Google/Amazon）**——全部闭源
- **大规模 context-aware 系统（Seed-ASR、Speech ReaLLM）**——全部闭源
- **真正能复现的 LLM-based ASR 生态**集中在三条线：
  1. SLAM-LLM 框架及其衍生
  2. Qwen-Audio / Granite-Speech / Phi-4-Multimodal 等大厂"顺手 ASR"模型
  3. Whisper + LM/LLM 增强 / 后纠错

这意味着研究选题时，**复现性最强的位置是第 1 章和第 2 章**——它们的开源生态最完整，也是最方便做参数局部化、unlearning、prompt 鲁棒性等扩展的实验底座。

---

## 9. Meta-resource：开源 speech/audio-LLM 索引仓库

如果想持续跟踪：
- [ga642381/speech-trident](https://github.com/ga642381/speech-trident) — speech/audio LLM、表征学习、codec 模型大全
- [AudioLLMs/Awesome-Audio-LLM](https://github.com/AudioLLMs/Awesome-Audio-LLM) — Audio LLM 论文与代码索引
- [huangcanan/Awesome-Large-Speech-Model](https://github.com/huangcanan/Awesome-Large-Speech-Model) — 大语音模型（论文/数据/工具）汇总
- [halsay/ASR-TTS-paper-daily](https://github.com/halsay/ASR-TTS-paper-daily) — 每日更新 ASR/TTS 论文
- [DongKeon/Awesome-Speaker-Diarization](https://github.com/DongKeon/Awesome-Speaker-Diarization) — speaker diarization（含 LLM 路线）综合索引
