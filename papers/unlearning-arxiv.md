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
