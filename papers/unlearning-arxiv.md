# Unlearning arXiv 论文解析

## Class Unlearning via Depth-Aware Removal of Forget-Specific Directions

- 链接：https://arxiv.org/abs/2604.15166

核心发现：现有 class unlearning 方法只压低了分类头输出，深层表征里仍保留 forget class 的结构，属于"表面遗忘"。作者提出 **DAMP**——一种无需梯度优化的一次性闭式权重编辑方法：在每层用 retain 类原型作参照，把 forget 方向作为残差投影消除，并按 probe separability 做 depth-aware 缩放（浅层改动小、深层改动大），在多个数据集与 CNN/Transformer 上都比基线更接近"从零重训"的金标准。

## Modeling LLM Unlearning as an Asymmetric Two-Task Learning Problem

- 链接：https://arxiv.org/abs/2604.14808

核心发现：把 LLM unlearning 重新建模成**非对称双任务**——retention 是主目标，forgetting 是辅助任务，而不是过去那种等权重 loss 相加。作者提出 retention-prioritized 梯度合成框架，并实例化出 **SAGO** 方法处理两任务间的梯度冲突；在 WMDP Bio 上目标模型能力保留从 44.6% 提升到 96.0%，说明调整梯度几何比重新平衡 loss 权重更能解开遗忘-保留冲突。
