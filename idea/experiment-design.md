# 实验设计：Plug-and-Play Audio-Embedding Corrupted Prompts for LLM-based ASR

---

## 0. 你的问题的直接回答

> 选不同数量的 speaker_id 去 train classifier，然后比较 WER？

这个方向是对的，但 **只变 speaker 数量 + 只看 WER 还不够**。

原因很简单：你的论文需要同时讲清楚 **四件事**：

1. **Forgetting 有没有成功？** — 目标 speaker 的内容是否被有效遗忘
2. **Utility 保住了没有？** — 非目标 speaker 的识别质量有没有被伤
3. **Classifier 靠不靠谱？** — 分类器对 speaker 的识别有没有误判
4. **方法比 baseline 好在哪？** — 和权重更新方案的对比

所以实验应该围绕一个更完整的矩阵来设计。下面是具体方案。

---

## 1. Base ASR System 选择

### 推荐选择

| 组件 | 推荐 | 理由 |
|---|---|---|
| Speech Encoder | **WavLM-large** 或 **HuBERT-xtralarge** | SLAM-ASR 原文验证过的组合 |
| Projector | **Linear projector**（单层线性映射） | SLAM-ASR 核心设定，只训练这个 |
| LLM | **Vicuna-7B-v1.5** 或 **LLaMA-2-7B** | 和 SLAM-ASR 原文对齐 |
| 训练数据 | **LibriSpeech train-clean-100** 或 **train-clean-460** | 2484 speakers，speaker 信息完整 |

用 SLAM-ASR 的好处：
- 公开代码（[SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM)）；
- 只训练 projector，encoder 和 LLM 全冻结；
- 这意味着你方法的"plug-and-play"叙事在这个 base system 上最干净。

### 第一步：复现 base ASR

先确认 base system 在 LibriSpeech test-clean / test-other 上的 WER，作为所有实验的 anchor point。

预期参考值（来自 SLAM-ASR 原文）：
- test-clean WER ≈ 1.84%
- test-other WER ≈ 3.39%

---

## 2. Forget Target 定义：Speaker-Level Forgetting

### 为什么先做 speaker-level

1. **Speaker ID 是 LibriSpeech 天然提供的元数据**，不需要额外标注；
2. **Classifier 更容易做**——speaker verification / identification 是成熟任务；
3. **Forget-retain 分割天然干净**——一个 speaker 的所有 utterances 属于 forget set，其余属于 retain set；
4. **和 "Speech Unlearning" (arXiv:2506.00848) 的 class unlearning 设定对齐**；
5. **和 "Unlearning LLM-Based Speech Recognition Models" (Interspeech 2025) 的 privacy 动机对齐**。

### Forget Set 的构造

从 LibriSpeech train-clean-100 的 ~251 speakers 中选择若干 speaker 作为 forget target。

---

## 3. 核心实验变量：Forget Speaker 数量

这是你问的核心问题。实验应该设计成一个 **scaling experiment**：当要遗忘的 speaker 数量增加时，forgetting 效果和 utility 保持如何变化？

### 3.1 实验组设计

| 设定名 | Forget Speakers 数量 | 占 train speakers 比例 (≈) | 目标 |
|---|---|---|---|
| **F-1** | 1 | ~0.4% | 最简场景 |
| **F-5** | 5 | ~2% | 小规模 |
| **F-10** | 10 | ~4% | 中等规模 |
| **F-25** | 25 | ~10% | 大规模 |
| **F-50** | 50 | ~20% | 极端压力测试 |

### 3.2 Speaker 选择策略

不要随机选一次就完事。需要考虑 **选哪些 speaker** 也会影响结果。

#### 方法 A：随机采样 × 3 seeds

对每个数量级（1/5/10/25/50），随机采样 3 次不同的 speaker 集合，报告 mean ± std。

#### 方法 B：控制变量采样

另外设计几组对照，验证 speaker 属性的影响：

| 对照维度 | 设计 | 目标 |
|---|---|---|
| **数据量多 vs 少** | 选择 utterance 数最多的 N 个 speaker vs 最少的 N 个 | 数据量是否影响 forgetting 难度 |
| **声学特征典型 vs 异常** | 选择 embedding 离群程度不同的 speaker | 声学独特性是否影响 classifier 和 corruption 效果 |
| **性别均衡** | 全男 / 全女 / 混合 | 性别是否是混淆变量 |

在第一版论文里，**方法 A（随机 × 3 seeds）是必须的，方法 B 可以作为 ablation**。

---

## 4. 训练流程：三阶段

### Stage 0：训练 Base ASR（或直接用预训练好的 SLAM-ASR）

输入：全量 LibriSpeech train → 输出：一个正常工作的 speech-LLM ASR

### Stage 1：训练 Speaker Classifier `C`

**输入**：projected audio prompts（projector 输出的 embedding 序列）

**任务**：binary classification — 当前 utterance 是否来自 forget speaker set

**训练数据**：
- 正样本：forget speakers 的 utterances
- 负样本：retain speakers 的 utterances（下采样到与正样本均衡，或使用 focal loss）

**关键设计选择**：

| 选择 | 推荐 | 理由 |
|---|---|---|
| Classifier 输入 | projected audio prompts (post-projector) | 和 corruption 作用位点一致 |
| Classifier 架构 | 简单：mean-pooling + 2-layer MLP | 保持轻量，避免 classifier 本身引入太多参数 |
| 训练方式 | 独立训练，base ASR 全冻结 | 不改 base model |

**如何评估 classifier 本身**：
- 用一个 held-out validation set（从 forget/retain speakers 的未见 utterances 中取）
- 报告：Accuracy, Precision, Recall, F1, AUROC, FPR@95TPR

### Stage 2：训练 Corruption Module `Δ`

**目标**：学习一个 corruption 向量/矩阵 `δ`，使得当 classifier 命中时，corrupted audio prompts 送入 LLM 后，模型输出趋向于"未学过该知识"的行为。

**训练方式**（参考 ECO）：
- Classifier 冻结
- Base ASR 全冻结（encoder, projector, LLM 都不动）
- 只优化 `δ` 本身
- Loss 设计：让 corrupted output 的分布逼近某个 target 分布（如 uniform token distribution / refusal template / 与 retrained-without-forget-data 的 model 对齐）

**Corruption 实现选择**（消融实验用）：

| 方案 | 描述 | 预期 |
|---|---|---|
| **Additive-global** | 所有 forget speakers 共享一个 `δ` | 最简单，baseline |
| **Additive-per-speaker** | 每个 forget speaker 有自己的 `δ_k` | 更精细 |
| **Low-rank** | `δ = U V^T`，rank r 可调 | 更灵活，参数可控 |
| **Zero-out** | 直接将命中的 audio prompts 置零 | 消融：corruption 不需要学习？ |
| **Random noise** | 加高斯噪声 | 消融：学习过的 δ 比随机好多少？ |

---

## 5. 评估指标（完整矩阵）

### 5.1 Forgetting Side（目标 speakers 有没有被遗忘）

| 指标 | 含义 | 怎么测 |
|---|---|---|
| **Forget-WER** | 对 forget speakers 的 test utterances 的 WER | WER 越高 = 遗忘越成功 |
| **Memorization Rate (MR)** | 模型对 forget speakers 训练数据的记忆程度 | 参考 Interspeech 2025 的 WER-based MR |
| **Speaker Extraction Rate** | 用 prompting 方法尝试从模型中提取 forget speaker 的具体内容 | 提取成功率越低越好 |
| **Oracle Gap** | Forget-WER 与 "retrained-without-forget-data" 模型的 WER 之间的差距 | 越接近 = 遗忘越像 "从未学过" |

**Oracle Model**：为了计算 Oracle Gap，需要一个 **ground-truth retrained model**——从 train set 中移除 forget speakers 的数据后重新训练的 ASR。这很贵，但对论文说服力非常重要。至少要为 F-1 和 F-10 两个设定各做一个。

### 5.2 Utility Side（非目标 speakers 有没有被伤）

| 指标 | 含义 | 怎么测 |
|---|---|---|
| **Retain-WER** | 对 retain speakers 的 test utterances 的 WER | 越接近 base model 越好 |
| **test-clean WER** | LibriSpeech test-clean 全集 WER | 整体能力 |
| **test-other WER** | LibriSpeech test-other 全集 WER | 鲁棒性 |
| **Retain-WER Δ** | Retain-WER 与 base model Retain-WER 的绝对差 | 核心 utility 指标 |

**这是你的核心卖点**：如果 Retain-WER Δ ≈ 0 而权重更新 baseline 的 Retain-WER Δ 明显大于 0，论文就有了最强的实验证据。

### 5.3 Classifier Side

| 指标 | 含义 |
|---|---|
| **Accuracy** | 整体准确率 |
| **FPR (False Positive Rate)** | 非 forget speaker 被误判为 forget 的比率 → 直接伤 utility |
| **FNR (False Negative Rate)** | forget speaker 被漏判的比率 → 遗忘不完整 |
| **AUROC** | 分类器的整体判别力 |

### 5.4 效率 Side

| 指标 | 含义 |
|---|---|
| **Classifier 参数量** | 新增了多少参数 |
| **Corruption 参数量** | δ 有多少参数 |
| **Inference latency overhead** | 推理时增加了多少延迟（ms） |
| **Forget 请求响应时间** | 新增一个 forget speaker 需要多久（训练 classifier + δ 的时间） |

---

## 6. Baseline 设计

| Baseline | 描述 | 对比目的 |
|---|---|---|
| **No Forgetting** | 原始 base ASR，不做任何操作 | 上界（utility）和下界（forgetting） |
| **Oracle Retrain** | 移除 forget speakers 重新训练 | Gold standard，forgetting 和 utility 的理想参考 |
| **Gradient Ascent** | 在 forget data 上做 gradient ascent 更新权重 | 最经典的 unlearning baseline |
| **LoRA Unlearning** | 用 LoRA 对 LLM 做 unlearning 微调 | 代表 PEFT-based unlearning |
| **Projector-only Unlearning** | 只更新 projector 权重做 gradient ascent | 最公平的轻量对照 |
| **Random Corruption** | classifier 命中后加随机噪声而非学习的 δ | 消融：corruption 需要学习吗 |
| **Zero-out** | classifier 命中后将 audio prompts 置零 | 消融：简单屏蔽够不够 |
| **Output Filtering** | 不动 embeddings，在 decoder 输出端做文本关键词过滤 | 证明不是任何后处理都一样 |

---

## 7. 实验矩阵总览

### 主实验表格模板（Table 1: Speaker-Level Forgetting Results）

```
方法 \ 指标    | Forget-WER ↑ | Oracle Gap ↓ | Retain-WER ↓ | Retain Δ ↓ | test-clean ↓ | test-other ↓
-------------|-------------|-------------|-------------|-----------|------------|------------
No Forgetting | xx.x        | xx.x        | xx.x        | 0.0       | 1.84       | 3.39
Oracle Retrain| xx.x        | 0.0         | xx.x        | xx.x      | xx.x       | xx.x
Gradient Asc  | xx.x        | xx.x        | xx.x        | xx.x      | xx.x       | xx.x
LoRA Unlearn  | xx.x        | xx.x        | xx.x        | xx.x      | xx.x       | xx.x
Proj-only UL  | xx.x        | xx.x        | xx.x        | xx.x      | xx.x       | xx.x
Ours (F-10)   | xx.x        | xx.x        | xx.x        | xx.x      | xx.x       | xx.x
```

### Scaling 实验表格模板（Table 2: Effect of Forget Set Size）

```
Forget Speakers | Forget-WER ↑ | Oracle Gap ↓ | Retain-WER ↓ | Retain Δ ↓ | FPR ↓ | FNR ↓
----------------|-------------|-------------|-------------|-----------|-------|------
1               | xx.x        | xx.x        | xx.x        | xx.x      | xx.x  | xx.x
5               | xx.x        | xx.x        | xx.x        | xx.x      | xx.x  | xx.x
10              | xx.x        | xx.x        | xx.x        | xx.x      | xx.x  | xx.x
25              | xx.x        | xx.x        | xx.x        | xx.x      | xx.x  | xx.x
50              | xx.x        | xx.x        | xx.x        | xx.x      | xx.x  | xx.x
```

---

## 8. Ablation Studies

### Ablation 1：Corruption 位点

| 位点 | 描述 |
|---|---|
| Pre-projector | corruption 加在 speech encoder 输出上 |
| **Post-projector** | corruption 加在 projector 输出上（推荐默认） |

目标：证明 post-projector 是更有效的干预位点。

### Ablation 2：Corruption 方式

| 方式 | 描述 |
|---|---|
| Learned additive δ | 默认方案 |
| Random Gaussian noise | 不学习的随机扰动 |
| Zero-out | 直接置零 |
| Low-rank δ | δ = UV^T |

目标：证明学习过的 corruption 比随机/简单方案更好。

### Ablation 3：Classifier 输入

| 输入 | 描述 |
|---|---|
| Encoder output | 用 speech encoder 的输出训练 classifier |
| **Projected prompts** | 用 projector 输出训练 classifier（推荐默认） |
| Combined | 两者拼接 |

### Ablation 4：Corruption 粒度

| 粒度 | 描述 |
|---|---|
| **Utterance-level** | 整段 utterance 的 audio prompts 全部 corrupt（推荐默认） |
| Chunk-level | 只 corrupt audio prompts 中的部分 chunk |

对于 speaker-level forgetting，utterance-level 更自然。但如果将来要做 phrase-level forgetting，chunk-level 就很重要。

### Ablation 5：Classifier 阈值

| 阈值 | 效果 |
|---|---|
| 低阈值（0.3） | 更激进，FPR 高，utility 受损但 forgetting 更完整 |
| 中阈值（0.5） | 平衡 |
| 高阈值（0.7） | 更保守，FPR 低，utility 好但可能遗忘不完整 |

画一条 **Forgetting-Utility Trade-off 曲线**（横轴 Forget-WER，纵轴 Retain-WER Δ），通过扫阈值得到不同工作点。和权重更新 baseline（通过扫 learning rate / steps 得到的不同工作点）一起画。

如果你的曲线一直 dominate baseline 的曲线（同等 forgetting 下 utility 损伤更小），这就是论文最强的一张图。

---

## 9. 实验执行优先级

论文实验量很大，不可能一次做完。建议按以下优先级推进：

### P0（必须有，否则论文不成立）

1. ✅ Base ASR 复现 → 确认 WER baseline
2. ✅ Speaker classifier 训练与评估 → 确认 classifier 可用
3. ✅ Corruption module 训练（F-10 设定）→ 确认方法可行
4. ✅ 主实验对比表（Table 1）：Ours vs No-Forgetting vs Gradient-Ascent vs Projector-only-UL
5. ✅ Scaling 实验（Table 2）：F-1/5/10/25/50

### P1（强烈推荐，显著加强论文）

6. Oracle Retrain（至少 F-1 和 F-10）→ 提供 gold standard
7. Forgetting-Utility Trade-off 曲线 → 最有说服力的图
8. Ablation: corruption 方式（learned δ vs random vs zero-out）
9. Classifier 性能详细分析（AUROC, FPR/FNR vs forget set size）

### P2（锦上添花，如果有时间做）

10. Ablation: corruption 位点（pre vs post projector）
11. Ablation: classifier 输入（encoder output vs projected prompts）
12. Speaker 属性对照（数据量多/少, 男/女）
13. LoRA Unlearning baseline
14. Output Filtering baseline
15. Inference latency 分析

---

## 10. 预期结果与故事线

### 最理想的结果

```
                    Forget-WER ↑    Retain Δ ↓
No Forgetting       低 (≈ base)     0.0
Gradient Ascent     高              大（utility 损伤明显）
Projector-only UL   高              中等
Ours                高              ≈ 0（几乎不伤 utility）
Oracle Retrain      高              小
```

如果实验结果如上，你的故事是：

> 我们的方法在 forgetting 效果和权重更新方案相当的前提下，utility 保持明显更好——因为我们完全不改 base model 参数。

### Scaling 实验预期

| Forget Speakers ↑ | Forget-WER | Retain Δ | Classifier FPR |
|---|---|---|---|
| 1 | 高 | ≈ 0 | 很低 |
| 5 | 高 | 很小 | 低 |
| 10 | 高 | 小 | 低 |
| 25 | 高 | 开始可见 | 中等 |
| 50 | 中等 | 明显上升 | 较高 |

**预期趋势**：
- Forget-WER 应该在所有设定下都较高（forgetting 基本成功）
- Retain Δ 随 forget speakers 增加而 **缓慢上升**（因为 classifier FPR 会随 forget set 变大而增加）
- 权重更新 baseline 的 Retain Δ 随 forget speakers 增加 **快速上升**

如果权重更新方法在 F-25/F-50 时 Retain Δ 飙升，而你的方法 Retain Δ 仍然可控，那就证明了 plug-and-play 方案在大规模遗忘时的优势。

### 最不想看到的结果及应对

| 坏结果 | 原因可能 | 应对 |
|---|---|---|
| Forget-WER 不高 | Corruption 不够强 | 加大 δ norm / 换更强 corruption 方式 |
| Retain Δ 也大 | Classifier FPR 太高 | 提高阈值 / 改进 classifier |
| Classifier AUROC 低 | Projected prompts 的 speaker 信息不足 | 换用 encoder output 作为 classifier 输入 |
| 比 projector-only UL 差 | 权重更新更直接 | 强调 plug-and-play / 可逆性 / 速度优势 |

---

## 11. 数据分割详细方案

以 LibriSpeech train-clean-100（~251 speakers, ~100 hours）为例：

```
LibriSpeech train-clean-100
├── Forget Speakers (N 个，N ∈ {1, 5, 10, 25, 50})
│   ├── 他们的所有 utterances → Forget Set
│   │   ├── 80% → Classifier 训练正样本 + Corruption 训练
│   │   └── 20% → Forget Test Set（测 Forget-WER）
│   └──
├── Retain Speakers (251 - N 个)
│   ├── 他们的所有 utterances → Retain Set
│   │   ├── 80% → Base ASR 训练 / Classifier 训练负样本
│   │   └── 20% → Retain Test Set（测 Retain-WER）
│   └──
└──

外部测试集（不在 train 里的 speakers）
├── LibriSpeech test-clean → 测 test-clean WER
└── LibriSpeech test-other → 测 test-other WER
```

**重要细节**：
- Forget Test Set 和 Retain Test Set 都是 **utterance-level hold-out**，不是 speaker-level hold-out（speaker 在训练中见过，但这些具体 utterances 没见过）
- LibriSpeech test-clean / test-other 的 speakers 完全独立于 train，所以它们测的是 **真正的 generalization 能力**

---

## 12. 你问的具体问题的总结

### "选不同数量的 speaker_id 去 train classifier？"

**是的**，这是 scaling experiment 的核心变量。但要注意：

1. 不只是 classifier 需要重新训练——每个 forget set size 对应一组新的 **(classifier + corruption module)**
2. 每个设定需要 **3 个随机 seed** 来报告稳定性
3. Classifier 本身的性能（AUROC, FPR, FNR）也要随 forget set size 变化而分析

### "然后比较 WER？"

WER 是核心指标之一，但不是唯一。你至少需要：

| 你需要报告的 | 为什么 |
|---|---|
| **Forget-WER** (↑ = 好) | forget 是否成功 |
| **Retain-WER** (↓ = 好) | utility 是否保住 |
| **Retain Δ** | 和 base model 的差距 |
| **test-clean / test-other WER** | 整体泛化 |
| **Classifier FPR / FNR** | classifier 可靠性 |
| **vs baselines** | 方法优势在哪 |

如果只报告 WER 而没有 baseline 对比 + classifier 分析 + oracle gap，reviewer 会认为评估不完整。

---

## 13. 一个最小可行实验计划（如果时间/算力有限）

如果只能做最少的实验，以下是精简方案：

### 固定设定

- Base ASR: SLAM-ASR (WavLM-large + Vicuna-7B + Linear Projector)
- 数据: LibriSpeech train-clean-100
- Corruption: post-projector additive learned δ
- Classifier: mean-pooling + 2-layer MLP on projected prompts

### 最小实验集

| 实验 | 变量 |
|---|---|
| Scaling | F-1, F-5, F-10, F-25（4 个设定 × 1 seed） |
| Baselines | No Forgetting + Gradient Ascent + Projector-only UL（3 个 baseline） |
| Ablation | Learned δ vs Random noise vs Zero-out（3 种 corruption） |

### 最小指标集

- Forget-WER, Retain-WER, test-clean WER, Classifier Accuracy

这样总共大约 4 × 4 = 16 个实验 run（4 settings × 4 methods），加 2 个 corruption ablation = 18 个 run。每个 run 主要开销是训练 classifier + corruption（base ASR 只训练一次）。
