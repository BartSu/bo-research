# LLM Unlearning Shortlist: Geometry-Aware + Data-Augmentation + One-Shot-Oriented

Target focus: papers related to **geometry-aware unlearning** and/or **data augmentation before unlearning** to improve one-shot forgetting quality (instead of iterative unlearn-patch-unlearn loops).

## 1) ReLearn: Unlearning via Learning for Large Language Models
- **Authors (Affiliation):**  
  Haoming Xu (Zhejiang University); Ningyuan Zhao (Xiamen University); Liming Yang (Tsinghua University); Sendong Zhao (Harbin Institute of Technology); Shumin Deng (National University of Singapore, NUS-NCS Joint Lab); Mengru Wang (Zhejiang University); Bryan Hooi (National University of Singapore); Nay Oo (National University of Singapore); Huajun Chen (Zhejiang University); Ningyu Zhang (Zhejiang University)
- **Venue:** ACL 2025 (reported in scholarly indexes) / arXiv:2502.11190
- **TL;DR (research gap):** Existing gradient-reversal style LLM unlearning often harms language quality and utility; this work identifies the need for a **data-centric augmentation pipeline** that boosts forgetting while preserving fluent generation.
- **Link:** https://arxiv.org/abs/2502.11190
- **Code:** https://github.com/zjunlp/unlearn

## 2) Align-then-Unlearn: Embedding Alignment for LLM Unlearning
- **Authors (Affiliation):**  
  Philipp Spohn (Technical University of Munich); Leander Girrbach (Technical University of Munich; Munich Center for Machine Learning; MDSI; Helmholtz Munich); Jessica Bader (Technical University of Munich; Munich Center for Machine Learning; MDSI; Helmholtz Munich); Zeynep Akata (Technical University of Munich; Munich Center for Machine Learning; MDSI; Helmholtz Munich)
- **Venue:** arXiv preprint (ICML format submission), 2025
- **TL;DR (research gap):** Token-level unlearning fails under prompt rephrasing and tends to over-forget unrelated knowledge; this paper argues for a **pre-alignment stage in embedding space** to support more robust, concept-level forgetting.
- **Link:** https://arxiv.org/abs/2506.13181
- **Code:** https://github.com/ExplainableML/align-then-unlearn

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
- **Code:** https://github.com/OPTML-Group/SOUL

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

---

## 2024–2026 Expanded Coverage (Appended)

Coverage rule used in this append: papers published/released in **2024–2026** only, with priority on benchmark/evaluation blind spots, proactive data handling, and one-shot stability.

## 7) TOFU: A Task of Fictitious Unlearning for LLMs
- **Authors (Affiliation):**  
  Pratyush Maini; Zhili Feng; Avi Schwarzschild; Zachary C. Lipton; J. Zico Kolter (**Carnegie Mellon University**)
- **Venue:** arXiv preprint, 2024
- **TL;DR (research gap):** Establishes a foundational LLM unlearning benchmark and shows most methods fail to jointly satisfy forgetting + utility goals; motivates stronger pre-unlearning design.
- **Link:** https://arxiv.org/abs/2401.06121
- **Code:** https://github.com/locuslab/tofu

## 8) The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning
- **Authors (Affiliation):**  
  Nathaniel Li et al. (**multi-institution collaboration**, incl. Center for AI Safety, UC Berkeley, MIT, SecureBio, Scale AI, NYU, Stanford, Harvard, USC, UIUC, UCLA, CMU, Caltech, etc.)
- **Venue:** ICML 2024 (PMLR 235) / arXiv:2403.03218
- **TL;DR (research gap):** Introduces safety-centric unlearning benchmark and demonstrates that utility-preserving removal is hard in hazardous domains, reinforcing need for proactive robustness controls.
- **Link:** https://arxiv.org/abs/2403.03218
- **Code:** https://github.com/centerforaisafety/wmdp

## 9) MUSE: Machine Unlearning Six-Way Evaluation for Language Models
- **Authors (Affiliation):**  
  Weijia Shi; Jaechan Lee; Daogao Liu; Luke Zettlemoyer; Noah A. Smith (**University of Washington**); Yangsibo Huang; Sadhika Malladi (**Princeton University**); Jieyu Zhao (**University of Southern California**); Ari Holtzman (**University of Chicago**); Chiyuan Zhang (**Google Research**)
- **Venue:** arXiv preprint, 2024 (released as preprint under review)
- **TL;DR (research gap):** Expands unlearning evaluation to six dimensions (forgetting, privacy, utility, scalability, sustainability), revealing blind spots missed by single-metric evaluations.
- **Link:** https://arxiv.org/abs/2407.06460
- **Code:** https://github.com/jaechan-repo/muse_bench

## 10) BLUR: A Benchmark for LLM Unlearning Robust to Forget-Retain Overlap
- **Authors (Affiliation):**  
  Shengyuan Hu; Neil Kale; Pratiksha Thaker; Yiwei Fu; Steven Wu; Virginia Smith (**Carnegie Mellon University**)
- **Venue:** arXiv preprint, 2025
- **TL;DR (research gap):** Shows many prior benchmark setups are too “separable”; with realistic forget-retain overlap, existing methods degrade substantially and utility side effects become clearer.
- **Link:** https://arxiv.org/abs/2506.15699

## 11) Unlearning's Blind Spots: Over-Unlearning and Prototypical Relearning Attack
- **Authors (Affiliation):**  
  SeungBum Ha; Saerom Park; Sung Whan Yoon (**Ulsan National Institute of Science and Technology, UNIST**)
- **Venue:** arXiv preprint, 2025 (updated 2026)
- **TL;DR (research gap):** Directly formalizes blind-spot failure modes (over-unlearning and relearning attacks), supporting your claim that current protocols can miss latent post-unlearning damage.
- **Link:** https://arxiv.org/abs/2506.01318

## 12) Verifying Robust Unlearning: Probing Residual Knowledge in Unlearned Models
- **Authors (Affiliation):**  
  Hao Xuan; Xingyu Li (**University of Alberta**)
- **Venue:** arXiv preprint, 2025
- **TL;DR (research gap):** Argues that “unlearning happened” verification is insufficient; proposes probing residual knowledge leakage, again pointing to hidden side effects beyond standard utility checks.
- **Link:** https://arxiv.org/abs/2504.14798

## 13) Probing Knowledge Holes in Unlearned LLMs
- **Authors (Affiliation):**  
  Myeongseob Ko; Hoang Anh Just; Ming Jin; Ruoxi Jia (**Virginia Tech**); Charles Fleming (**Cisco Research**)
- **Venue:** NeurIPS 2025 / arXiv:2511.00030
- **TL;DR (research gap):** Shows unlearning can create broad “knowledge holes” that static benchmarks fail to detect; strongly aligned with your blind-spot corruption motivation.
- **Link:** https://arxiv.org/abs/2511.00030

## 14) Reveal and Release: Iterative LLM Unlearning with Self-generated Data
- **Authors (Affiliation):**  
  Linxi Xie; Xin Teng; Shichang Ke; Hongyi Wen; Shengjie Wang (**New York University Shanghai, Center for Data Science**)
- **Venue:** Findings of EMNLP 2025 / arXiv:2509.14624
- **TL;DR (research gap):** Uses model-generated forget data and iterative updates; relevant because it is explicitly data-centric, but differs from your proactive one-shot, pre-unlearning stabilization angle.
- **Link:** https://arxiv.org/abs/2509.14624

## 15) Unlearning That Lasts: Utility-Preserving, Robust, and Almost Irreversible Forgetting in LLMs
- **Authors (Affiliation):**  
  Naman Deep Singh; Maximilian Müller; Matthias Hein (**University of Tübingen & Tübingen AI Center**); Francesco Croce (**EPFL**)
- **Venue:** arXiv preprint, 2025 (OpenReview submission track in 2025–2026 cycle)
- **TL;DR (research gap):** Targets robust and persistent forgetting with strong utility preservation criteria; adjacent to your goal but optimization-centric rather than pre-unlearning data synthesis-centric.
- **Link:** https://arxiv.org/abs/2509.02820
- **Code:** https://github.com/nmndeep/JensUn-Unlearning

## 16) A Comprehensive Evaluation of LLM Unlearning Robustness under Multi-Turn Interaction
- **Authors (Affiliation):**  
  Ruihao Pan; Suhang Wang (**Pennsylvania State University**)
- **Venue:** arXiv preprint, 2026
- **TL;DR (research gap):** Extends robustness evaluation into multi-turn settings, showing one-shot pass/fail evaluations can miss failure recovery or leakage over interaction trajectories.
- **Link:** https://arxiv.org/abs/2603.00823

## 17) Toward Understanding Unlearning Difficulty: A Mechanistic Perspective and Circuit-Guided Difficulty Metric
- **Authors (Affiliation):**  
  Jiali Cheng (**University of Massachusetts Lowell**); Ziheng Chen (**Walmart Global Tech**); Chirag Agarwal (**University of Virginia**); Hadi Amiri (**University of Massachusetts Lowell**)
- **Venue:** arXiv preprint, 2026
- **TL;DR (research gap):** Introduces sample-level unlearning difficulty analysis; relevant to your “algorithm-dependent behavior” premise and can support adaptive synthesis policies.
- **Link:** https://arxiv.org/abs/2601.09624

## 18) AGT^AO: Robust and Stabilized LLM Unlearning via Adversarial Gating Training with Adaptive Orthogonality
- **Authors (Affiliation):**  
  Pengyu Li; Lingling Zhang; Zhitao Gao; Yanrui Wu; Yuxuan Dong; Huan Liu; Bifan Wei; Jun Liu (**Xi'an Jiaotong University / MOE KLINNS Lab / Shaanxi Province Key Laboratory of Big Data Knowledge Engineering**)
- **Venue:** arXiv preprint, 2026
- **TL;DR (research gap):** Shows utility-preserving stabilization can benefit from geometry-aware constraints; useful adjacent evidence for your algorithm-adaptive preparation narrative.
- **Link:** https://arxiv.org/abs/2602.01703
- **Code:** https://github.com/TiezMind/AGT-unlearning

## 2024–2026 Representation-Shift + Mechanistic-Interpretability Addendum

This addendum targets papers that are especially useful for a literature review on **representation shift**, **activation-space interventions**, and **mechanistic interpretability** in LLM unlearning.  
Note: **Item 17** above ("Toward Understanding Unlearning Difficulty: A Mechanistic Perspective and Circuit-Guided Difficulty Metric") already fits this theme; the papers below extend that line with activation steering, circuit localization, and latent-space disentanglement.

## 19) Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization
- **Authors (Affiliation):**  
  Phillip Huang Guo; Aaquib Syed; Abhay Sheshadri; Aidan Ewart; Gintare Karolina Dziugaite
- **Venue:** ICML 2025 (PMLR 267) / arXiv:2410.12949
- **TL;DR (research gap):** Prior unlearning/editing localization is often based only on output effects, which can be brittle under paraphrase or relearning; this paper argues for **mechanism-level circuit localization** so edits target the actual factual-recall pathway rather than a superficial output symptom.
- **Link:** https://arxiv.org/abs/2410.12949
- **Code:** https://github.com/magikarp01/mechanistic-unlearning

## 20) FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model
- **Authors (Affiliation):**  
  Jinwei Hu; Zhenglin Huang; Xiangyu Yin; Wenjie Ruan; Guangliang Cheng; Yi Dong; Xiaowei Huang (**University of Liverpool**)
- **Venue:** NeurIPS 2025 / arXiv:2502.01472
- **TL;DR (research gap):** Standard forget/retain objectives are too coarse when harmful and benign knowledge overlap; FALCON motivates **layer-selective activation manipulation** and orthogonal unalignment to separate target knowledge in representation space with fewer retain-set side effects.
- **Link:** https://arxiv.org/abs/2502.01472
- **Code:** https://github.com/CharlesJW222/FALCON

## 21) LLM Unlearning via Neural Activation Redirection
- **Authors (Affiliation):**  
  William F. Shen; Xinchi Qiu; Meghdad Kurmanji; Alex Iacob; Lorenzo Sani; Yihong Chen; Nicola Cancedda; Nicholas D. Lane (**Meta / Facebook Research-associated release**)
- **Venue:** NeurIPS 2025 poster / arXiv:2502.07218
- **TL;DR (research gap):** Refusal-style unlearning can alter outputs without explicitly controlling where the underlying activations move; LUNAR frames unlearning as **representation redirection** toward "cannot answer" regions, making internal shift itself the optimization target.
- **Link:** https://arxiv.org/abs/2502.07218
- **Code:** https://github.com/facebookresearch/LUNAR

## 22) Reliable Unlearning Harmful Information in LLMs with Metamorphosis Representation Projection
- **Authors (Affiliation):**  
  Chengcan Wu; Zeming Wei; Huanran Chen; Yinpeng Dong; Meng Sun
- **Venue:** arXiv preprint, 2025
- **TL;DR (research gap):** Many unlearning pipelines appear effective only because they suppress harmful responses at the surface level; this paper pushes a stronger **representation-projection view** aimed at irreversible hidden-state transformation and resistance to relearning attacks.
- **Link:** https://arxiv.org/abs/2508.15449
- **Code:** https://github.com/ChengcanWu/MRP

## 23) Collapse of Irrelevant Representations (CIR) Ensures Robust and Non-Disruptive LLM Unlearning
- **Authors (Affiliation):**  
  Filip Sondej (**Jagiellonian University**); Yushi Yang (**University of Oxford**)
- **Venue:** arXiv preprint, 2025
- **TL;DR (research gap):** Existing methods often erase broad, shared features instead of the fact-specific subspace that should be forgotten; CIR argues for first collapsing **irrelevant/common representations** so the final update hits the harmful representation more selectively.
- **Link:** https://arxiv.org/abs/2509.11816

## 24) Representation-Aware Unlearning via Activation Signatures: From Suppression to Knowledge-Signature Erasure
- **Authors (Affiliation):**  
  Syed Naveed Mahmood; Md. Rezaur Rahman Bhuiyan; Tasfia Zaman; Jareen Tasneem Khondaker; Md. Sameer Sakib; K. M. Shadman Wadith; Nazia Tasnim; Farig Sadeque
- **Venue:** arXiv preprint, 2026
- **TL;DR (research gap):** Benchmark scores alone cannot tell whether a model truly forgot or merely learned to stay silent; this paper explicitly studies the **suppression-vs-erasure distinction** through activation signatures and uses internal knowledge traces as the unlearning target.
- **Link:** https://arxiv.org/abs/2601.10566

## 25) From Logits to Latents: Contrastive Representation Shaping for LLM Unlearning
- **Authors (Affiliation):**  
  Haoran Tang; Rajiv Khanna
- **Venue:** arXiv preprint, 2026
- **TL;DR (research gap):** Logit-level forget/retain losses ignore latent entanglement between what should be forgotten and what should be preserved; this paper shifts the focus to **contrastive latent-space shaping** to reduce representation overlap before it manifests as output interference.
- **Link:** https://arxiv.org/abs/2601.22028

## 26) KUDA: Knowledge Unlearning by Deviating Representation for Large Language Models
- **Authors (Affiliation):**  
  Ce Fang; Zhikun Zhang; Min Chen; Qing Liu; Lu Zhou; Zhe Liu; Yunjun Gao
- **Venue:** arXiv preprint, 2026
- **TL;DR (research gap):** Forcing random or refusal-like outputs does not necessarily break the internal association that stores target knowledge; KUDA instead performs unlearning by **deviating the representation itself**, with null-space protection for knowledge that should remain.
- **Link:** https://arxiv.org/abs/2602.19275

---

## Code Resources (quick access)

- **Align-then-Unlearn:** https://github.com/ExplainableML/align-then-unlearn
- **ReLearn:** https://github.com/zjunlp/unlearn
- **SOUL:** https://github.com/OPTML-Group/SOUL
- **TOFU benchmark:** https://github.com/locuslab/tofu
- **WMDP benchmark:** https://github.com/centerforaisafety/wmdp
- **MUSE benchmark:** https://github.com/jaechan-repo/muse_bench
- **Unlearning That Lasts (JensUn):** https://github.com/nmndeep/JensUn-Unlearning
- **AGT^AO:** https://github.com/TiezMind/AGT-unlearning
- **Mechanistic Unlearning:** https://github.com/magikarp01/mechanistic-unlearning
- **FALCON:** https://github.com/CharlesJW222/FALCON
- **LUNAR:** https://github.com/facebookresearch/LUNAR
- **MRP:** https://github.com/ChengcanWu/MRP

---

## Gap Check Against Your Proposed Direction (2024–2026 view)

Dimensions:
- **D1:** Pre-unlearning preparation stage exists  
- **D2:** Explicit data synthesis over forget/retain (add/edit/remove)  
- **D3:** Algorithm-adaptive strategy via parameter sensitivity/behavior  
- **D4:** Explicit objective: prevent blind-spot corruption *before deployment*  
- **D5:** One-shot stability emphasis (avoid iterative patch loops)

| Paper | D1 | D2 | D3 | D4 | D5 | Overlap with your gap |
|---|---|---|---|---|---|---|
| ReLearn (2025) | Partial | Partial (augmentation) | No | No | Partial | **Partial overlap** |
| Align-then-Unlearn (2025) | Yes (alignment pre-stage) | No | No | Partial | Partial | **Partial overlap** |
| Data Augmentation Improves MU (2025, non-LLM) | Yes | Partial | No | No | Partial | **Partial overlap** |
| Reveal and Release (2025) | Partial (self-generated data) | Partial | No | No | No (iterative) | **Partial overlap** |
| SOUL / Gauss-Newton / AGT^AO | No | No | Partial (optimizer/geometry sensitivity) | No | Yes | **Adjacent** |
| Blind Spots / BLUR / Probing Knowledge Holes / Multi-turn Eval | No | No | No | Yes (diagnose blind spots) | No | **Adjacent** |

### Evidence-based conclusion
- **Already well explored:** benchmarking and post-hoc diagnosis of unlearning failures (TOFU/WMDP/MUSE/BLUR + blind-spot and probing papers).  
- **Partially explored:** data-centric interventions and pre-stages (ReLearn, Align-then-Unlearn, Reveal-and-Release), but usually not unified with algorithm-adaptive control and proactive blind-spot prevention in one-shot settings.  
- **Still open (defensible claim):** a **unified proactive pre-unlearning data synthesis framework** that jointly performs forget/retain data shaping (add/edit/remove), adapts to algorithm-specific sensitivity, and targets blind-spot prevention before deployment for stable one-shot unlearning.

### Suggested intro positioning sentence
> Prior work has separately advanced unlearning benchmarks, blind-spot diagnostics, and selective data-centric/optimization-centric methods; however, a unified pre-unlearning framework that proactively synthesizes forget/retain data in an algorithm-adaptive manner to prevent blind-spot corruption before one-shot unlearning remains under-explored.
