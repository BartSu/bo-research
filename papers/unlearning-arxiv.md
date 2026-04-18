# Unlearning arXiv 论文解析

## Class Unlearning via Depth-Aware Removal of Forget-Specific Directions

- 链接：https://arxiv.org/abs/2604.15166

核心发现：现有 class unlearning 方法只压低了分类头输出，深层表征里仍保留 forget class 的结构，属于"表面遗忘"。作者提出 **DAMP**——一种无需梯度优化的一次性闭式权重编辑方法：在每层用 retain 类原型作参照，把 forget 方向作为残差投影消除，并按 probe separability 做 depth-aware 缩放（浅层改动小、深层改动大），在多个数据集与 CNN/Transformer 上都比基线更接近"从零重训"的金标准。

## Modeling LLM Unlearning as an Asymmetric Two-Task Learning Problem

- 链接：https://arxiv.org/abs/2604.14808

核心发现：把 LLM unlearning 重新建模成**非对称双任务**——retention 是主目标，forgetting 是辅助任务，而不是过去那种等权重 loss 相加。作者提出 retention-prioritized 梯度合成框架，并实例化出 **SAGO** 方法处理两任务间的梯度冲突；在 WMDP Bio 上目标模型能力保留从 44.6% 提升到 96.0%，说明调整梯度几何比重新平衡 loss 权重更能解开遗忘-保留冲突。

## CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge

- 链接：https://arxiv.org/abs/2604.14644

核心发现：绕开"改权重"的路线,把 unlearning 重新定义为**输入侧行为过滤**——训练一个句向量模型(基座 multi-qa-mpnet-base-dot-v1, 109M)用对比损失(margin=0.5,cosine)学会识别"forget request",推理时用最大余弦相似度 s_max 与阈值 δ(0.8–0.9) 比较,命中就从 229 条拒答模板里随机返回,否则走正常 LLM 输出。训练数据用 NQ 6k seed 问题构造三类对:同义改写正例 + 高词法重叠但语义不同的硬负例。优势全在**连续场景**:每来一条新 forget request 只需 0.04s 入库(基线 GA/NPO/O3 等需要 178–328s 微调),权重完全不动所以"零遗忘"——TOFU stage 3 上 retain set 0.961、world facts 0.913,同时 forget set 压到 0.043;RETURN 十阶段连续 unlearning 后知识保留几乎无损。代价是**无法抵御 jailbreak/改写攻击**(作者承认),本质上是把 unlearning 降维成一个检索+拒答分类器,和参数编辑路线(DAMP、SAGO)正交——可以组合使用。

## MAGE: Memory-Graph Guided Corpus-Free Unlearning

- 链接：https://arxiv.org/abs/2604.13777

核心发现：指出现有 unlearning 流程存在一个被忽视的**二次泄露**风险——用户为了申请遗忘必须把完整 forget corpus 交给模型持有方,反而扩大了敏感数据暴露面。MAGE 把输入从"语料"降格为**轻量 anchor**(一个实体标识符),然后分三步反过来让模型自己供出记忆:(1) 以 anchor 为探针诱导 LLM 复现相关内容;(2) 把复现片段组织成**加权本地记忆图**,边权刻画内容片段与 anchor 的关联强度;(3) 基于图合成监督信号,塞给任意标准 unlearning 方法(GA / NPO / ...)做 forget 更新。在 TOFU 与 RWKU 上,这种自生成监督的 unlearning 效果与用外部参考语料生成监督"可比",同时保留通用能力。意义是把 unlearning 做成**可审计、最小暴露**的工作流——与 CURaTE(输入侧过滤)、DAMP/SAGO(权重编辑)关注的环节不同,MAGE 攻的是更上游的"监督信号从哪来"。

## CausalDetox: Causal Head Selection and Intervention for Detoxification

- 链接：https://arxiv.org/abs/2604.14602

核心发现：不是严格的 unlearning,但属于"**定向抹除模型某类行为**"的近亲——去毒。用**因果统计量 PNS**(Probability of Necessity and Sufficiency)度量每个注意力头在毒性生成中的必要性与充分性,筛出最小充分子集(比暴力扫描快 7×),再给两种互补干预:(a) 推理时的 **局部 ITI**,按输入动态生成 steering vector(而非固定方向)加到选中头上;(b) **PNS-guided fine-tuning**,对同一组头做永久性参数修正。在 ToxiGen / ImplicitHate / ParaDetox 上毒性下降比基线多 5.34%,流畅度不掉,并放出了 ParaTox(毒/非毒对齐句对)基准。相对既往 ITI/DPO 类工作的增量:**因果选头**给出了"改哪里"的原理性依据,而不是凭探针或梯度相关性拍板。与 unlearning 的连接点是——如果把"毒性"换成"某类知识",这套 PNS 选头 + 双路径干预的骨架应该是可迁移的。
