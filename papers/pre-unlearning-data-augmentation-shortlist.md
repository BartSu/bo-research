# LLM Unlearning Shortlist: Geometry-Aware + Data-Augmentation + One-Shot-Oriented

Target focus: papers related to **geometry-aware unlearning** and/or **data augmentation before unlearning** to improve one-shot forgetting quality (instead of iterative unlearn-patch-unlearn loops).

## 1) ReLearn: Unlearning via Learning for Large Language Models
- **Authors (Affiliation):**  
  Haoming Xu (Zhejiang University); Ningyuan Zhao (Xiamen University); Liming Yang (Tsinghua University); Sendong Zhao (Harbin Institute of Technology); Shumin Deng (National University of Singapore, NUS-NCS Joint Lab); Mengru Wang (Zhejiang University); Bryan Hooi (National University of Singapore); Nay Oo (National University of Singapore); Huajun Chen (Zhejiang University); Ningyu Zhang (Zhejiang University)
- **Venue:** ACL 2025 (reported in scholarly indexes) / arXiv:2502.11190
- **TL;DR (research gap):** Existing gradient-reversal style LLM unlearning often harms language quality and utility; this work identifies the need for a **data-centric augmentation pipeline** that boosts forgetting while preserving fluent generation.
- **Link:** https://arxiv.org/abs/2502.11190

## 2) Align-then-Unlearn: Embedding Alignment for LLM Unlearning
- **Authors (Affiliation):**  
  Philipp Spohn (Technical University of Munich); Leander Girrbach (Technical University of Munich; Munich Center for Machine Learning; MDSI; Helmholtz Munich); Jessica Bader (Technical University of Munich; Munich Center for Machine Learning; MDSI; Helmholtz Munich); Zeynep Akata (Technical University of Munich; Munich Center for Machine Learning; MDSI; Helmholtz Munich)
- **Venue:** arXiv preprint (ICML format submission), 2025
- **TL;DR (research gap):** Token-level unlearning fails under prompt rephrasing and tends to over-forget unrelated knowledge; this paper argues for a **pre-alignment stage in embedding space** to support more robust, concept-level forgetting.
- **Link:** https://arxiv.org/abs/2506.13181

## 3) Geometric-disentangelment Unlearning
- **Authors (Affiliation):**  
  Duo Zhou, Yuji Zhang, Tianxin Wei, Ruizhong Qiu, Ke Yang, Xiao Lin, Cheng Qian, Jingrui He, Hanghang Tong, Chengxiang Zhai, Heng Ji, Huan Zhang (all: University of Illinois Urbana-Champaign)
- **Venue:** arXiv preprint, 2025
- **TL;DR (research gap):** Prior unlearning methods lack a principled way to avoid retain-set side effects; this paper frames the problem geometrically and introduces update-space disentanglement to reduce forget/retain gradient conflict.
- **Link:** https://arxiv.org/abs/2511.17100

## 4) SOUL: Unlocking the Power of Second-Order Optimization for LLM Unlearning
- **Authors (Affiliation):**  
  Jinghan Jia, Yihua Zhang, Yimeng Zhang, Jiancheng Liu, Bharat Runwal (Michigan State University); James Diffenderfer, Bhavya Kailkhura (Lawrence Livermore National Laboratory); Sijia Liu (Michigan State University; MIT-IBM Watson AI Lab, IBM Research)
- **Venue:** arXiv preprint, 2024
- **TL;DR (research gap):** First-order unlearning updates are often too coarse for balancing forgetting and retention; this work highlights the need for **curvature-aware (second-order)** updates to improve one-shot optimization quality.
- **Link:** https://arxiv.org/abs/2404.18239

## 5) Gauss-Newton Unlearning for the LLM Era
- **Authors (Affiliation):**  
  Lev McKinney (University of Toronto; Vector Institute for AI); Anvith Thudi (University of Toronto; Vector Institute for AI); Juhan Bae (University of Toronto); Tara Rezaei (MIT); Nicolas Papernot (University of Toronto; Vector Institute for AI); Sheila A. McIlraith (University of Toronto; Vector Institute for AI); Roger Grosse (University of Toronto)
- **Venue:** IEEE SaTML 2026 (accepted; also arXiv:2602.10568)
- **TL;DR (research gap):** Retraining-based exact unlearning is impractical at LLM scale, and common approximations under-enforce retain constraints; this paper proposes a **Gauss-Newton constrained update** to better approximate one-shot unlearning with smaller utility drift.
- **Link:** https://arxiv.org/abs/2602.10568

## 6) Data Augmentation Improves Machine Unlearning *(non-LLM, but highly relevant design signal)*
- **Authors (Affiliation):**  
  Andreza M. C. Falcao, Filipe R. Cordeiro (Visual Computing Lab, Department of Computing, Universidade Federal Rural de Pernambuco, Brazil)
- **Venue:** SIBGRAPI 2025 / arXiv:2508.18502
- **TL;DR (research gap):** Many unlearning pipelines underuse data-centric interventions; this paper demonstrates that **augmentation before unlearning** can improve forgetting behavior and is a direct inspiration for LLM pre-unlearning augmentation frameworks.
- **Link:** https://arxiv.org/abs/2508.18502

---

## Best matches for your specific direction

If your goal is a **pre-unlearning data augmentation framework for one-shot performance boosting** (vs iterative unlearn-patch-unlearn), the most aligned starting points are:
1. **ReLearn** (explicit data augmentation for LLM unlearning),
2. **Align-then-Unlearn** (explicit pre-stage before unlearning),
3. **SOUL / Gauss-Newton Unlearning** (one-shot-oriented geometry/curvature optimization),
4. **Data Augmentation Improves Machine Unlearning** (strong evidence from non-LLM unlearning).

These papers still leave open a clear gap: a unified framework that combines **(i) pre-unlearning augmentation** and **(ii) geometry-aware constrained updates** in a single one-shot pipeline for robust forgetting under paraphrase/relearning stress tests.
