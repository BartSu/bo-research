# 实验设计：Plug-and-Play Audio-Embedding Corrupted Prompts for LLM-based ASR Unlearning

本文档是对 [slam-asr-audio-embedding-corrupted-prompts.md](slam-asr-audio-embedding-corrupted-prompts.md) 中 idea 的 **完整实验设计**，覆盖：unlearning 效果怎么验证、数据集如何选取与预处理、forget/retain 怎么划分、评价指标怎么设、baseline 怎么对比、消融实验怎么做。

---

## 目录

1. [实验总体目标](#1-实验总体目标)
2. [数据集选择：为什么用 LibriSpeech](#2-数据集选择为什么用-librispeech)
3. [数据集结构与统计](#3-数据集结构与统计)
4. [Forget / Retain 数据划分方案](#4-forget--retain-数据划分方案)
5. [数据预处理流水线](#5-数据预处理流水线)
6. [Base ASR 系统搭建](#6-base-asr-系统搭建)
7. [Classifier 训练](#7-classifier-训练)
8. [Corruption Module 训练](#8-corruption-module-训练)
9. [Baseline 方案](#9-baseline-方案)
10. [评价指标体系](#10-评价指标体系)
11. [消融实验设计](#11-消融实验设计)
12. [鲁棒性测试](#12-鲁棒性测试)
13. [实验流程总览](#13-实验流程总览)
14. [补充数据集（可选扩展）](#14-补充数据集可选扩展)

---

## 1. 实验总体目标

核心研究问题：

> **Can classifier-gated corruption of projected audio prompts provide plug-and-play forgetting behavior in LLM-based ASR while preserving retain-set utility better than weight-update baselines?**

实验需要回答的 4 个子问题：

| 编号 | 子问题 | 成功标准 |
|------|--------|---------|
| Q1 | Audio-side corruption 能否让 forget target 的输出接近"从未学过该知识"？ | Forget-set 上 memorization rate 显著下降 |
| Q2 | 是否比权重更新方法更保 utility？ | Retain-set WER 损失 < weight-update baseline |
| Q3 | 局部 corruption 是否比全局 corruption 更有效？ | Chunk-level vs. utterance-level corruption 消融 |
| Q4 | Forgetting 是否具有对声学变体的鲁棒性？ | 在 speed/noise/accent 变体下 forget 仍然有效 |

---

## 2. 数据集选择：为什么用 LibriSpeech

### 首选：LibriSpeech

| 优势 | 说明 |
|------|------|
| 标准基准 | LLM-based ASR 领域（包括 SLAM-ASR）的标准评测集，结果直接可比 |
| Speaker ID 标注 | 天然支持 speaker-level forget 设定 |
| 文本对齐 | 有 word-level / utterance-level transcript，支持 content-level forget 设定 |
| 规模合适 | train-clean-100 已足够训练 projector；更大的 360h/500h 可做 scaling 实验 |
| 社区基础设施 | torchaudio、HuggingFace datasets 均有现成 loader |
| 已有 ASR unlearning 先例 | Interspeech 2025 的 ASR unlearning 论文就用 LibriSpeech |

### 为什么不单独用其他数据集

| 数据集 | 问题 |
|--------|------|
| Common Voice | 可用于 accent/multilingual 扩展实验，但 speaker ID 标注质量不如 LibriSpeech |
| VoxCeleb | 主要是 speaker verification，不是 ASR 任务 |
| GigaSpeech | 规模太大，初期实验成本高 |
| SPGISpeech | 金融领域，不通用 |

**建议**：主实验用 **LibriSpeech**，鲁棒性实验用 **LibriSpeech + 扰动** 或小规模 Common Voice 子集验证跨域泛化。

---

## 3. 数据集结构与统计

### LibriSpeech 各子集

| 子集 | 时长 | 样本数 | 用途 |
|------|------|--------|------|
| `train-clean-100` | 100h | ~28,539 | **主训练集**（projector 训练 + classifier 训练） |
| `train-clean-360` | 360h | ~104,014 | 可选扩展训练 |
| `train-other-500` | 500h | ~148,688 | 可选 noisy 扩展 |
| `dev-clean` | ~5h | 2,703 | 开发验证 |
| `dev-other` | ~5h | 2,864 | 开发验证（noisy） |
| `test-clean` | ~5h | 2,620 | **主测试集** |
| `test-other` | ~5h | 2,939 | 鲁棒性测试 |

### 每条样本包含

```
waveform (16 kHz, mono)
sample_rate: 16000
transcript: str
speaker_id: int
chapter_id: int
utterance_id: str
```

### 关键统计（需要在预处理阶段实际计算确认）

- `train-clean-100` 约 251 个 speaker
- 每个 speaker 平均约 100+ 条 utterance
- 平均 utterance 时长约 10-15 秒

---

## 4. Forget / Retain 数据划分方案

**核心原则**：实验需要同时设计 **speaker-level** 和 **content-level** 两种 forget 设定，以回答不同粒度的 unlearning 问题。

### 4.1 Setting A：Speaker-Level Forgetting

> 目标：忘掉特定说话人的所有语音。

#### 划分策略

```
train-clean-100 speakers
├── Forget speakers: 随机选 K 个 speaker（如 K = 10, 20, 50）
│   └── 这些 speaker 的所有 utterance → Forget Set (D_f)
├── Retain speakers: 剩余 speaker
│   └── 这些 speaker 的所有 utterance → Retain Set (D_r)
└── Classifier 训练也需要正负样本对
```

#### 推荐配置

| 实验组 | Forget speakers | 约占比 | 说明 |
|--------|----------------|--------|------|
| Small forget | 10 speakers | ~4% | 最简单，baseline sanity check |
| Medium forget | 25 speakers | ~10% | 主实验配置 |
| Large forget | 50 speakers | ~20% | 压力测试 |

#### 选 speaker 的注意事项

- 避免只选 utterance 极少的 speaker（否则 forget 太容易）
- 确保 forget speakers 和 retain speakers 的性别比例大致平衡
- 记录每个 forget speaker 的 utterance 数和总时长

### 4.2 Setting B：Content-Level Forgetting（Phrase / Entity）

> 目标：忘掉包含特定敏感实体或短语的 utterance 的转写。

#### 划分策略

1. 从 `train-clean-100` 的 transcript 中选择 **target entities**
   - 可以是人名、地名、数字序列（模拟电话号码 / 地址）
   - 或者是特定的 n-gram 短语
2. 包含 target entity 的 utterance → **Forget Set**
3. 不包含的 → **Retain Set**

#### 具体操作

```python
# 实体选择策略
target_entities = [
    # 方案 1：从 transcript 中提取 NER 实体
    # 方案 2：选高频人名/地名
    # 方案 3：人工定义一组"敏感词"
]

forget_utterances = [u for u in train_set if any(e in u.transcript for e in target_entities)]
retain_utterances = [u for u in train_set if u not in forget_utterances]
```

#### 推荐配置

| 实验组 | Target entities | Forget set 大小 | 说明 |
|--------|----------------|----------------|------|
| Narrow | 5 个高频实体 | ~500-1000 条 | 简单场景 |
| Medium | 20 个混合实体 | ~2000-4000 条 | 主实验 |
| Broad | 50+ 个实体 | ~5000+ 条 | 压力测试 |

### 4.3 Setting C：Sample-Level Forgetting（可选）

> 目标：忘掉训练集中的特定录音样本。

- 随机抽取 N 条 utterance 作为 forget set
- 最接近 RTBF（Right to Be Forgotten）的场景
- 但 classifier 更难做，因为要精确识别"这条 utterance 是否是某个特定样本"

**建议**：第一版论文以 **Setting A + Setting B** 为主，Setting C 作为附录实验。

### 4.4 测试集划分

测试阶段需要 4 个子集：

| 测试子集 | 来源 | 用途 |
|----------|------|------|
| **Forget-test** | forget speakers/entities 在 test-clean 中的样本 | 评估 forgetting 成功率 |
| **Retain-test** | retain speakers/entities 在 test-clean 中的样本 | 评估 utility 保持 |
| **Unseen-test** | test-clean 中都未在训练集出现的 speaker/entity | 评估泛化能力 |
| **Noisy-test** | test-other | 评估 noisy 条件下的鲁棒性 |

> **关键**：LibriSpeech 的 test set 和 train set 的 speaker 不重叠。所以对 speaker-level forgetting，需要在 **train set 内部** hold out 一部分作为 forget-test，或者在 train set 中用 utterance-level 的 hold-out。

#### Speaker-Level 测试集构造

```
对每个 forget speaker:
├── 80% utterances → 用于训练 classifier
└── 20% utterances → forget-test（验证 corruption 是否对未见过的该 speaker utterance 也生效）

对每个 retain speaker:
├── 全部用于训练
└── 从 dev-clean 或 test-clean 中选样本做 retain-test
```

---

## 5. 数据预处理流水线

### 5.1 原始音频预处理

```
步骤 1: 下载与组织
├── 下载 LibriSpeech（train-clean-100, dev-clean, dev-other, test-clean, test-other）
├── 统一格式确认：16 kHz, mono, FLAC → WAV (如需)
└── 建立 metadata 表：utterance_id, speaker_id, chapter_id, transcript, duration, file_path

步骤 2: 音频质量检查
├── 过滤静音 / 过短 / 过长 utterance
├── 设定时长范围：[1s, 30s]（太短无意义，太长影响 batch 效率）
└── 验证所有文件可正常加载

步骤 3: Speaker 统计
├── 统计每个 speaker 的 utterance 数、总时长、平均时长
├── 按 utterance 数排序，确保 forget set 选到的 speaker 有足够样本
└── 统计性别分布（LibriSpeech 有 speaker metadata）
```

### 5.2 Speech Encoder 特征提取

```
步骤 4: 提取 speech encoder 输出
├── 选定 encoder: HuBERT-XL / WavLM-Large / Whisper encoder
├── 对每条 utterance 提取 frame-level embeddings
│   ├── 输入：raw waveform (16 kHz)
│   ├── 输出：T × D 的特征矩阵（T = 帧数, D = embedding 维度）
│   └── 保存为 .pt / .npy 文件
├── 记录每条 utterance 的帧数 T
└── 验证特征维度一致性

注意：
- HuBERT-XL: D = 1280, 下采样率 320 (50 fps at 16kHz)
- WavLM-Large: D = 1024, 下采样率 320
- Whisper encoder: D = 1280 (large-v3), 下采样率 320
```

### 5.3 Projector 输出特征提取

```
步骤 5: 提取 projected audio prompts
├── 在训好的 base ASR 系统上
├── 将 speech encoder 输出过 linear projector
├── 得到 T' × D_llm 的 projected embeddings（D_llm = LLM embedding dim）
├── 保存用于 classifier 和 corruption module 训练
└── T' 可能因 projector 设计不同而与 T 不同（如带 downsampling）
```

### 5.4 Forget / Retain 标注

```
步骤 6: 标注 forget/retain 标签
├── Speaker-level: 根据 speaker_id 标注 is_forget = 1/0
├── Content-level: 根据 transcript 中是否包含 target entity 标注
├── 生成 JSONL 格式的标注文件:
│   {
│     "utterance_id": "1272-128104-0000",
│     "speaker_id": 1272,
│     "transcript": "...",
│     "audio_path": "...",
│     "encoder_feat_path": "...",
│     "projected_feat_path": "...",
│     "is_forget": 0,
│     "forget_reason": null,  // or "speaker" / "entity:John"
│     "split": "train" / "forget_test" / "retain_test"
│   }
└── 划分 train / val / test
```

### 5.5 NER 实体提取（Content-Level 专用）

```
步骤 7: 对 transcript 做 NER（用于 Setting B）
├── 工具: spaCy / Stanza
├── 提取: PERSON, GPE, ORG, DATE, CARDINAL 等实体
├── 统计实体频率，选取合适的 target entities
├── 注意：LibriSpeech 是有声书朗读，实体分布偏文学
└── 可能需要补充人工定义的"模拟敏感实体"列表
```

### 5.6 数据增强（用于鲁棒性测试）

```
步骤 8: 生成声学变体
├── Speed perturbation: 0.9x, 1.1x
├── Noise injection: 加 white noise / babble noise，SNR = {5, 10, 20} dB
├── Reverberation: room impulse response (RIR) 模拟
├── Pitch shift: ±2 semitones
└── 只用于测试阶段，不用于训练（目的是验证 forget 鲁棒性）

工具: torchaudio, audiomentations, SoX
```

### 5.7 预处理流水线汇总

```
Raw Audio (FLAC/WAV, 16kHz)
    │
    ▼
[质量过滤] → 过滤静音/过短/过长
    │
    ▼
[Speech Encoder] → frame-level embeddings (T × D)
    │
    ▼
[Projector] → projected audio prompts (T' × D_llm)
    │
    ▼
[Forget/Retain 标注] → 每条 utterance 标记 is_forget
    │
    ▼
[NER (可选)] → content-level target entity 提取
    │
    ▼
[声学增强 (仅测试)] → speed/noise/reverb 变体
    │
    ▼
[导出 JSONL + .pt features] → 可供 classifier/corruption module 训练
```

---

## 6. Base ASR 系统搭建

### 推荐配置

| 组件 | 推荐选择 | 替代选择 | 说明 |
|------|---------|---------|------|
| Speech Encoder | HuBERT-XL (finetuned on LS960) | WavLM-Large | SLAM-ASR 论文验证过的最强配置 |
| Projector | Linear (D_enc → D_llm) | 2-layer MLP | SLAM-ASR 的核心 minimal 设计 |
| LLM | Vicuna-7B-v1.5 | LLaMA-2-7B / Qwen2-7B | SLAM-ASR 原始配置；更新的 LLM 也可以 |
| 训练数据 | train-clean-100 | train-clean-100 + 360 | 100h 已经够用，且控制训练成本 |

### 训练协议

```
冻结: Speech Encoder (全部冻结)
冻结: LLM (全部冻结)
可训练: Linear Projector (~18-22M 参数)

训练目标: next-token prediction (CTC 或 attention-based)
优化器: AdamW
学习率: 1e-3 ~ 1e-4
Batch size: 16-32
Epochs: 10-20
```

### 预期性能

| 测试集 | 目标 WER |
|--------|---------|
| test-clean | < 3.0 |
| test-other | < 6.0 |

如果 base model 的 WER 不在这个范围内，需要先调通 base ASR 再进行 unlearning 实验。

---

## 7. Classifier 训练

### 7.1 Classifier 架构

```
输入: projected audio prompts (T' × D_llm)
    │
    ▼
[Temporal Pooling] → 方案 A: mean pooling
                     方案 B: attention pooling
                     方案 C: [CLS]-style token
    │
    ▼
[MLP head]
    │
    ▼
输出: P(forget | audio_prompts)
```

参数量目标：< 5M（保持 lightweight）

### 7.2 训练数据

| 设定 | 正样本（forget） | 负样本（retain） | 训练策略 |
|------|----------------|----------------|---------|
| Speaker-level | forget speakers 的 projected embeddings | retain speakers 的 projected embeddings | Binary CE + class balancing |
| Content-level | 含 target entity 的 projected embeddings | 不含 target entity 的 projected embeddings | Binary CE + hard negative mining |

### 7.3 训练注意事项

- **类别不平衡**：forget set 通常远小于 retain set → 使用过采样 / focal loss / class weight
- **验证集**：从 forget speakers 的 held-out utterances 和 retain speakers 的 dev-clean 构建
- **阈值校准**：训练结束后在 dev set 上做 threshold calibration（Platt scaling 或 isotonic regression），使 FPR 尽可能低

### 7.4 Classifier 性能要求

| 指标 | 目标 | 说明 |
|------|------|------|
| AUROC | > 0.95 | 区分 forget/retain 的整体能力 |
| FPR @ TPR=0.95 | < 0.05 | 在 95% 召回 forget 的前提下，误报率要低 |
| FPR @ TPR=0.90 | < 0.02 | 更保守的配置，最大限度保 utility |

如果 classifier 性能不达标，corruption 的副作用会很大——这一点必须在论文中明确报告。

---

## 8. Corruption Module 训练

### 8.1 三种 corruption 方案

#### 方案 1：Additive Corruption Vector（最简单，推荐首选）

```
corrupted_prompts = original_prompts + delta

delta: 可学习向量 (D_llm,) 或 (T', D_llm)
训练目标: 使 LLM(corrupted_prompts) 的输出 ≈ "不知道" / 乱码 / 空白
```

#### 方案 2：Subspace Projection

```
corrupted_prompts = original_prompts - proj_forget(original_prompts)

proj_forget: 投影到 forget-sensitive subspace 的投影矩阵
训练: 通过 SVD 或 PCA 从 forget set 的 projected embeddings 中提取主方向
```

#### 方案 3：Conditioned Corruption Generator

```
corrupted_prompts = original_prompts + G(original_prompts, classifier_logits)

G: 小型条件生成器 (MLP / small Transformer)
训练: 端到端优化，目标是 forget + retain trade-off
```

### 8.2 Corruption 训练目标

```
L_total = L_forget + lambda * L_retain_guard

L_forget: 使 forget-set 输入经 corruption 后，LLM 输出 ≈ target_forget_output
    选项 A: target = 空白 / 空字符串
    选项 B: target = "I don't know" / refusal
    选项 C: target = 随机 token 序列（知识已损坏的模拟）
    选项 D: target = retrained-from-scratch model 的输出（gold standard）

L_retain_guard: 使 retain-set 输入不受影响（classifier 应该过滤掉，但作为额外保护）
    = KL(LLM(original_prompts) || LLM(original_prompts))  # 恒等，即只通过 classifier gating 保护

lambda: 权衡系数
```

### 8.3 优化协议

```
优化器: Adam / SGD
冻结: Speech Encoder, Projector, LLM (全部冻结)
可训练: delta 向量 (方案 1) 或 generator 参数 (方案 3)
学习率: 1e-3 ~ 1e-2 (delta 向量) / 1e-4 (generator)
迭代: zeroth-order optimization (参考 ECO) 或 first-order (如果 LLM 可以反向传播)
```

---

## 9. Baseline 方案

### 9.1 完整 Baseline 列表

| Baseline | 方法描述 | 对应的对比维度 |
|----------|---------|---------------|
| **B0: No Forgetting** | 原始 LLM-based ASR，不做任何 forgetting | Upper bound for utility, lower bound for forgetting |
| **B1: Output Filtering** | 不动 audio embeddings，只在 decoder 输出端用规则/文本分类器过滤 | 证明 audio-side intervention 有独立价值 |
| **B2: Random Corruption** | classifier 命中后加随机噪声（非学习得到的 delta） | 证明 learned corruption 优于盲目扰动 |
| **B3: Zero-out Corruption** | classifier 命中后将 projected prompts 置零 | 最简单的 non-learned corruption |
| **B4: Projector-only Weight Update** | 在 forget set 上对 projector 做 gradient ascent（参考 Interspeech 2025） | Weight-update 的最轻量版本 |
| **B5: LoRA-based Unlearning** | 在 LLM 上加 LoRA，用 gradient ascent unlearn | 参考 Interspeech 2025 的 PEFT 方案 |
| **B6: Full Fine-tuning Unlearning** | 对 projector + LLM adapter 做 full unlearning | 最彻底但最重的 weight-update 方案 |
| **B7: Retrain-from-scratch** | 在 retain set 上从头训练 projector | Gold standard（理想 unlearning 的 upper bound） |

### 9.2 最核心的 3 组对比

如果只选 3 个 baseline，优先级是：

1. **B0 (No Forgetting)** vs. **Ours** → 证明 forgetting 确实发生了
2. **B4 (Projector Weight Update)** vs. **Ours** → 证明 plug-and-play 方法比最自然的 weight-update 替代方案更保 utility
3. **B7 (Retrain-from-scratch)** vs. **Ours** → 衡量离理想 unlearning 的距离

---

## 10. 评价指标体系

### 10.1 Forgetting 指标（越低越好 = 忘得越干净）

| 指标 | 定义 | 来源 |
|------|------|------|
| **Forget-set WER** | Forget-test 上的 WER（高 = 忘得好；但 WER 过高可能只是输出乱码） | — |
| **WER-based Memorization Rate (WMR)** | 在 forget-test 上，verbatim 复现训练转写的比例 | Interspeech 2025 |
| **Prompting-based Memorization Rate (PMR)** | 给一段 forget-test 音频的前缀，看模型能否补全剩余转写 | Interspeech 2025 |
| **Homophone-based Memorization Rate (HMR)** | 用 forget target 的同音异形词测试，看模型是否还能恢复原始词 | Interspeech 2025 |
| **Entity Recovery Rate (ERR)** | Content-level: 对 forget entities，看模型能否从含该 entity 的音频中正确识别出来 | 本文设计 |
| **KL Divergence to Retrained Model** | 本方法输出分布与 B7 (retrain-from-scratch) 的 KL 散度 | Gold standard 度量 |

### 10.2 Utility 指标（越低越好）

| 指标 | 定义 | 说明 |
|------|------|------|
| **Retain-set WER** | Retain-test 上的 WER | 最核心指标 |
| **Retain-set CER** | Retain-test 上的 CER | 更细粒度 |
| **test-clean WER** | 标准 test-clean 上的整体 WER | 通用 ASR 能力 |
| **test-other WER** | 标准 test-other 上的 WER | 噪声鲁棒性 |
| **Rare Word Accuracy** | Retain-test 中低频词的识别准确率 | 检测 collateral damage |
| **Utility Drop (ΔU)** | WER_after - WER_before 在 retain set 上 | 直接量化 side effect |

### 10.3 Classifier 指标

| 指标 | 说明 |
|------|------|
| **AUROC** | 整体区分能力 |
| **AUPRC** | 在类别不平衡下更可靠 |
| **FPR @ 固定 TPR** | 误报率（直接影响 utility） |
| **FNR @ 固定 FPR** | 漏报率（直接影响 forget 成功率） |
| **Calibration ECE** | 分类器概率是否可靠 |

### 10.4 Efficiency 指标

| 指标 | 说明 |
|------|------|
| **额外参数量** | Classifier + corruption module 的总参数 |
| **推理延迟开销** | 加入 classifier + corruption 后的 RTF (Real-Time Factor) 增加 |
| **训练成本** | GPU hours to train classifier + corruption module |
| **可逆性** | 关闭 module 后原始 ASR 能力是否完全恢复 |

### 10.5 综合评分

建议使用一个类似 MUSE 的多维雷达图：

```
维度:
1. Forget Success (↑)
2. Utility Preservation (↑)  
3. Classifier Accuracy (↑)
4. Robustness (↑)
5. Efficiency (↑)
6. Reversibility (↑ 或 ↓，取决于 positioning)
```

---

## 11. 消融实验设计

### 11.1 核心消融

| 消融变量 | 比较 | 回答的问题 |
|----------|------|-----------|
| **Corruption 位点** | Pre-projector vs. Post-projector | 在哪层做 corruption 更有效？ |
| **Corruption 粒度** | Utterance-level vs. Chunk-level | 全局 vs. 局部 corruption 的 trade-off |
| **Corruption 方法** | Additive delta vs. Subspace projection vs. Zero-out | 哪种 corruption 方式效果最好？ |
| **Classifier 输入** | Encoder outputs vs. Projected prompts | 用哪层特征做分类更准确？ |
| **Classifier 池化** | Mean pooling vs. Attention pooling | 池化策略的影响 |
| **Forget set 规模** | 10 / 25 / 50 speakers | 规模如何影响 forget/utility trade-off？ |
| **Lambda 权衡** | λ ∈ {0.01, 0.1, 1.0, 10.0} | 超参数敏感性 |

### 11.2 Corruption 强度消融

```
对 additive delta:
├── delta 的 L2 norm 约束: ||delta||_2 ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
├── 画 forget_success vs. utility_drop 的 Pareto 曲线
└── 找到最优 corruption 强度
```

### 11.3 Classifier 错误传播分析

```
模拟 classifier 在不同 FPR 水平下对最终结果的影响:
├── Oracle classifier (FPR=0, FNR=0) → upper bound
├── 真实 classifier → 实际性能
├── Noisy classifier (人工注入额外 FP/FN) → 灵敏度分析
└── 画 FPR vs. ΔUtility 曲线
```

---

## 12. 鲁棒性测试

### 12.1 声学变体鲁棒性

> 问题：如果 forget target 的声学特征变了（说快/说慢/加噪/混响），corruption 还能触发吗？

| 扰动类型 | 参数 | 期望结果 |
|----------|------|---------|
| Speed perturbation | 0.9x, 1.1x | Forget 仍然成功 |
| Additive noise | SNR = 5, 10, 20 dB | Forget 可能退化，量化退化程度 |
| Reverberation | RT60 = 0.3s, 0.6s, 1.0s | 同上 |
| Pitch shift | ±2 semitones | 同上 |

### 12.2 语义变体鲁棒性（Content-Level 专用）

> 问题：如果 forget entity 以不同方式出现（缩写、昵称、同义词），forgetting 还有效吗？

| 变体类型 | 示例 | 测试方法 |
|----------|------|---------|
| 同音异形词 | "John" vs. "Jon" | 构造包含同音词的测试音频 |
| 缩写 | "New York" vs. "NY" | 看 classifier 是否能泛化 |
| 上下文变化 | 同一实体在不同句子中 | 测试 classifier + corruption 的稳定性 |

### 12.3 Bypass 攻击鲁棒性

> 问题：如果攻击者试图绕过 classifier，能不能恢复被 forget 的知识？

| 攻击方式 | 描述 | 防御 |
|----------|------|------|
| **关闭 corruption module** | 如果 module 是可拆卸的，关了就恢复 | 这是 plug-and-play 方法的固有限制，需在论文中明确讨论 |
| **对抗音频** | 构造使 classifier 判负的对抗音频 | 测试 classifier 的对抗鲁棒性 |
| **间接提问** | 不直接说 forget target，但通过上下文暗示 | Content-level setting 的关键挑战 |

### 12.4 跨模型迁移性（可选）

> 问题：在一个 LLM 上训练的 classifier + corruption，能否迁移到另一个 LLM？

| 迁移设定 | 说明 |
|----------|------|
| 换 LLM (Vicuna-7B → LLaMA-2-7B) | 测试 corruption 的可迁移性 |
| 换 Encoder (HuBERT → WavLM) | 测试 classifier 的可迁移性 |
| 换 Projector (Linear → MLP) | 测试 corruption module 的可迁移性 |

---

## 13. 实验流程总览

```
Phase 0: 环境准备
├── 安装依赖: torch, torchaudio, transformers, datasets, fairseq
├── 下载 LibriSpeech (train-clean-100, dev-clean, dev-other, test-clean, test-other)
├── 下载预训练模型: HuBERT-XL, Vicuna-7B / LLaMA-2-7B
└── 验证 SLAM-ASR 代码可运行

Phase 1: Base ASR 训练 (约需 1-2 轮 GPU 时间)
├── 用 train-clean-100 训练 linear projector
├── 在 test-clean / test-other 上验证 WER
└── 确保 base model 质量达标 (test-clean WER < 3.0)

Phase 2: Forget/Retain 划分 + 特征提取
├── 确定 forget set (speaker-level 和 content-level)
├── 提取 speech encoder 特征
├── 提取 projected audio prompt 特征
├── 生成标注文件
└── 对 content-level: 运行 NER，确定 target entities

Phase 3: Classifier 训练
├── 在 projected embeddings 上训练 binary classifier
├── 在 dev set 上调参、校准阈值
├── 报告 AUROC / FPR / FNR
└── 如果 classifier 性能不达标，先解决此问题

Phase 4: Corruption Module 训练
├── 先做 additive delta (方案 1)
├── 用 zeroth-order 或 first-order 优化
├── 在 dev set 上调 lambda 和 corruption 强度
└── 验证 forget + utility 指标

Phase 5: 主实验
├── 在 test set 上评估所有 baselines 和 ours
├── 报告 Table 1: Forget 指标对比
├── 报告 Table 2: Utility 指标对比
├── 报告 Table 3: Efficiency 对比
└── 报告 Figure 1: Forget-Utility Pareto 曲线

Phase 6: 消融实验
├── corruption 位点消融
├── corruption 粒度消融
├── corruption 方法消融
├── forget set 规模消融
└── classifier 错误传播分析

Phase 7: 鲁棒性实验
├── 声学变体测试
├── 语义变体测试 (content-level)
├── bypass 攻击测试
└── (可选) 跨模型迁移测试

Phase 8: 分析
├── 可视化 corruption 前后 embedding space 的变化 (t-SNE / PCA)
├── 案例分析: 成功 forget 和失败 forget 的典型样本
├── 分析 classifier 误判对下游结果的影响
└── 讨论 forgetting vs. blocking 的区别
```

---

## 14. 补充数据集（可选扩展）

如果主实验效果好，可以在以下数据集上做扩展验证：

| 数据集 | 用途 | 说明 |
|--------|------|------|
| **Common Voice (English)** | 跨域泛化 | 不同录音条件、更多口音 |
| **Common Voice (Multilingual)** | 多语言 forget | 验证方法是否对非英语也有效 |
| **VoxPopuli** | 欧盟议会演讲 | 更长的 utterance，更正式的语体 |
| **AISHELL-1/2** | 中文 ASR unlearning | 跨语言验证 |
| **TED-LIUM** | TED 演讲 | Speaker-level forget 的另一个验证集 |

---

## 附录 A: 关键依赖

```
# Python 环境
torch >= 2.0
torchaudio >= 2.0
transformers >= 4.35
datasets (HuggingFace)
fairseq (HuBERT)
soundfile
librosa
spacy (NER, content-level)
scikit-learn (classifier metrics)
audiomentations (数据增强)
```

## 附录 B: 预期计算资源

| 阶段 | 预计 GPU 时间 | GPU 类型 |
|------|-------------|---------|
| Speech encoder 特征提取 (100h) | 2-4h | 1× A100 |
| Projector 训练 | 4-8h | 1× A100 |
| Classifier 训练 | 1-2h | 1× A100 |
| Corruption module 训练 | 2-8h | 1× A100 (需要 LLM forward pass) |
| 全部 baseline 实验 | 20-40h | 1× A100 |
| 消融 + 鲁棒性 | 10-20h | 1× A100 |
| **总计** | **~40-80h** | **1× A100** |

## 附录 C: 与 ECO 实验设计的关键差异

| 维度 | ECO (文本 LLM) | 本方案 (ASR) |
|------|---------------|-------------|
| 输入模态 | Text prompt | Audio waveform → audio prompt |
| Classifier 输入 | Text token embeddings | Projected audio embeddings |
| Corruption 对象 | Text prompt embedding dimensions | Audio prompt embedding dimensions |
| Forget target | Entity / hazardous knowledge / copyright | Speaker / phrase / entity in speech |
| 时序结构 | Token sequence (离散) | Frame sequence (连续、密集) |
| 局部性挑战 | 通常整个 prompt 与 forget 相关 | Forget target 可能只出现在局部音频片段 |
| Evaluation | Text generation metrics | WER / CER / memorization rate |

## 附录 D: 新发现的相关论文

在检索过程中发现了一篇值得关注的新论文：

**Speech Unlearning** (arXiv 2506.00848, Jun 2025)
- 首次系统定义了 **speech 领域的 machine unlearning**
- 定义了两种任务：**sample unlearning** 和 **class unlearning**
- 在 keyword spotting 和 speaker identification 上做实验
- 发现 speech unlearning 比 image/text unlearning 更难
- 虽然不是 LLM-based ASR，但其 setting 定义和 difficulty 分析对本 idea 有参考价值
