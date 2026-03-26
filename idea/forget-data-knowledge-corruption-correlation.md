# How Do Forget Data Correlate to Knowledge Corruption? Estimating Knowledge Corruption Before Unlearning

## Motivation and Analogy

### The Reference Paper

**A Curious Case of Searching for the Correlation between Training Data and Adversarial Robustness of Transformer Textual Models** (Findings of ACL 2024, arXiv:2402.11469) asks:

> How do training data correlate to adversarial robustness? Can we estimate the adversarial robustness before models are fine-tuned without generating adversarial examples?

The authors take a **data-first** approach: instead of modifying model architecture or loss functions, they extract 13 features from training corpora across four categories (embedding distribution, label distribution, surrogate model learnability, token-based statistics) and use a lightweight Random Forest classifier to predict attack success rates. The framework saves 30x–193x runtime over traditional adversarial evaluation and transfers across models (BERT, RoBERTa, BART, ELECTRA, GPT-2).

The key insight is that **properties of the training data themselves are predictive of a downstream model behavior (adversarial robustness)**, and this prediction can be made **before fine-tuning even occurs**.

### Our Analogous Question

We propose the parallel question for LLM unlearning:

> **How do forget data correlate to knowledge corruption? Can we estimate knowledge corruption before the model is unlearned?**

Where the reference paper studies `training data → adversarial robustness`, we study `forget data → irrelevant knowledge corruption`.

| Dimension | Reference Paper | Our Proposal |
|---|---|---|
| Input data | Training (fine-tuning) corpus | Forget data (data to be unlearned) |
| Model operation | Fine-tuning | Unlearning |
| Target outcome | Adversarial robustness (attack success rate) | Knowledge corruption (degradation of unrelated knowledge) |
| Prediction timing | Before fine-tuning | Before unlearning |
| Approach | Data-first: extract features from data, predict outcome | Data-first: extract features from forget data, predict corruption |

The broader research direction is: **LLM unlearning causes irrelevant knowledge corruption**, and we want to understand, predict, and ultimately mitigate this corruption by analyzing the forget data before unlearning happens.

---

## Part I: Literature Review

### 1. LLM Unlearning: Foundations and Benchmarks

LLM unlearning aims to remove specific knowledge from trained models while preserving general capabilities. Several benchmarks now exist:

- **TOFU** (Maini et al., 2024; arXiv:2401.06121) — A foundational LLM unlearning benchmark using fictitious author profiles; shows most methods fail to jointly satisfy forgetting and utility goals.
- **WMDP** (Li et al., ICML 2024; arXiv:2403.03218) — Safety-centric benchmark for removing hazardous knowledge (biosecurity, cybersecurity); demonstrates utility-preserving removal is hard in hazardous domains.
- **MUSE** (Shi et al., 2024; arXiv:2407.06460) — Six-way evaluation covering forgetting, privacy, utility, scalability, sustainability; reveals blind spots missed by single-metric evaluations.
- **BLUR** (Hu et al., 2025; arXiv:2506.15699) — Benchmark with realistic forget-retain overlap; existing methods degrade significantly when forget and retain sets are not cleanly separable.

**Relevance:** These benchmarks document that unlearning causes knowledge corruption but do not systematically analyze *which properties of the forget data predict the severity of corruption*.

### 2. Knowledge Corruption and Utility Degradation in LLM Unlearning

A central challenge in LLM unlearning is the **utility-forgetting trade-off**: aggressive unlearning causes collateral damage to unrelated knowledge.

- **Rethinking Machine Unlearning for LLMs** (Fan et al., Nature Machine Intelligence 2025; arXiv:2402.08787) — Comprehensive survey identifying data-model interaction dynamics and unlearning scope as often-overlooked elements. Highlights that unlearning should avoid affecting causally unrelated information.
- **Does Unlearning Truly Unlearn?** (Doshi & Stickland, 2024; arXiv:2411.12103) — Black-box evaluation showing LLMU and RMU cause notable degradation of general capabilities. Training on *unrelated* data can almost completely recover pre-unlearning performance, suggesting methods fail to truly unlearn.
- **Probing Knowledge Holes in Unlearned LLMs** (Ko et al., NeurIPS 2025; arXiv:2511.00030) — Demonstrates unlearning creates broad "knowledge holes" that static benchmarks fail to detect, strongly aligned with the irrelevant knowledge corruption concern.
- **Unlearning's Blind Spots** (Ha et al., 2025; arXiv:2506.01318) — Formalizes blind-spot failure modes including over-unlearning, where removal of target knowledge spills over into unrelated domains.
- **A Comprehensive Evaluation of LLM Unlearning Robustness under Multi-Turn Interaction** (Pan & Wang, 2026; arXiv:2603.00823) — Shows one-shot evaluations miss failure recovery or leakage over interaction trajectories.

**Key finding across these works:** Knowledge corruption is well-documented as a *consequence* of unlearning, but the literature largely diagnoses it **post-hoc**. No work systematically studies which **properties of the forget data** predict how severe this corruption will be.

### 3. Knowledge Entanglement: Why Forget Data Properties Matter

Recent work establishes that knowledge entanglement — where forget data and retain data share overlapping representations — is a primary driver of collateral damage.

- **EGUP: Entanglement-Guided Unlearning with Proxy Constraint** (2025; arXiv:2508.20443) — Uses inter-sample and intra-sample entanglement metrics to adaptively reweight unlearning strength. Forget samples semantically closer to retained knowledge receive more careful treatment. Shows consistent improvement in the unlearning-utility trade-off.
- **SKeB: Stimulus-Knowledge Entanglement-Behavior Framework** (2025; arXiv:2510.25732) — Models information entanglement via domain graphs. Demonstrates knowledge can be recalled from unlearned models through persuasive prompting, with recovery inversely correlated to model size.
- **From Logits to Latents: CLReg** (Tang & Khanna, 2026; arXiv:2601.22028) — Shows that operating at the logit level leaves forget-retain entanglement intact in the latent space. Proposes contrastive representation shaping to push forget and retain features apart. Provides theoretical insights relating representation shaping to entanglement reduction.
- **UIPE: Enhancing LLM Unlearning by Removing Knowledge Related to Forgetting Targets** (EMNLP 2025; arXiv:2503.04693) — Identifies that models can reconstruct forgotten content through logically *related* knowledge. Addresses this by removing knowledge highly correlated with forgetting targets.
- **CIR: Collapse of Irrelevant Representations** (Sondej & Yang, 2025; arXiv:2509.11816) — Argues existing methods erase broad shared features instead of fact-specific subspaces. Proposes first collapsing common representations so the update hits the harmful representation more selectively.

**Key finding:** The degree of representational entanglement between forget data and other knowledge is a critical driver of corruption. This entanglement is a *property of the forget data's relationship to the model's knowledge structure* — precisely the kind of property that could be measured before unlearning.

### 4. Predicting Unlearning Difficulty and Effects

A small but growing body of work attempts to understand *why* some data is harder to unlearn, and whether this can be predicted.

- **Circuit-Guided Unlearning Difficulty (CUD)** (Cheng et al., 2026; arXiv:2601.09624) — A **pre-unlearning metric** that assigns each sample a continuous difficulty score using circuit-level signals. Easy-to-unlearn samples are associated with shorter, shallower circuit interactions concentrated in earlier-to-intermediate layers; hard samples rely on longer, deeper pathways. CUD is stable across different unlearning methods. **This is the closest existing work to our proposal**, but it predicts *unlearning difficulty* (how hard it is to forget a sample), not *knowledge corruption* (how much unrelated knowledge is damaged).
- **When to Forget? Complexity Trade-offs in Machine Unlearning** (van Waerebeke et al., ICML 2025) — Establishes first theoretical bounds on unlearning efficiency as a function of data dimensionality, number of forget samples, and privacy constraints. Identifies three regimes with different feasibility characteristics.
- **Mechanistic Unlearning** (Guo et al., ICML 2025; arXiv:2410.12949) — Uses mechanism-level circuit localization to target the actual factual-recall pathway rather than surface symptoms. More robust to relearning than surface-level methods.
- **KUDA: Knowledge Unlearning by Deviating Representation** (Fang et al., 2026; arXiv:2602.19275) — Uses causal tracing to locate knowledge storage layers, then deviates representations from original positions. The causal tracing step is a pre-unlearning diagnostic of where knowledge lives.

**Key finding:** CUD demonstrates that pre-unlearning prediction of sample-level properties is feasible. However, CUD predicts *forget difficulty*, not *corruption extent*. No existing work extracts features from forget data to predict how much *irrelevant* knowledge will be corrupted.

### 5. Data-Centric Approaches and Pre-Training Prediction

Several works predict model behavior from data characteristics, either before training or before fine-tuning.

- **TuneAhead** (OpenReview, 2025) — Predicts LLM fine-tuning performance before training begins using static dataset descriptors + dynamic probe features from short simulated runs. Achieves 89.4% accuracy in predicting successful/failed runs, enabling 58.4% computational savings. Uses SHAP-based interpretability to identify which features drive performance.
- **Data2Behavior** (2025; arXiv:2602.04735) — Predicts unintended model behaviors before training using Manipulating Data Features (MDF). Summarizes candidate training data through mean representations and injects them into a base model's forward pass, revealing potential biases without updating parameters. Uses ~20% of GPU resources vs. fine-tuning.
- **PRISM** (GuideAI, 2025) — Traces LLM predictions to training data prototypes in a single forward pass by decomposing outputs into contributions from learned prototypes. Identifies how different training data categories contribute to predictions.
- **Knowledge Immunization Framework (KIF)** (2026; arXiv:2601.10566) — Targets internal activation signatures rather than surface outputs, achieving near-oracle erasure while preserving utility. Represents a shift toward understanding internal knowledge traces.

**Key finding:** The idea of predicting model outcomes from data features *before the model operation occurs* is well-established for fine-tuning (TuneAhead, Data2Behavior) but has **not been applied to predicting unlearning corruption**. This is a clear methodological gap.

### 6. Representation-Level Understanding of Unlearning Effects

Several works operate at the representation level to understand and control unlearning effects.

- **LUNAR: Neural Activation Redirection** (Shen et al., NeurIPS 2025; arXiv:2502.07218) — Redirects representations to activation regions expressing inability to answer. Achieves 2.9–11.7× improvement in combined efficacy and utility scores. Shows that representation-level control is more effective than output-level control.
- **FALCON** (Hu et al., NeurIPS 2025; arXiv:2502.01472) — Fine-grained activation manipulation using contrastive orthogonal unalignment. Addresses cases where harmful and benign knowledge overlap at the activation level.
- **MRP: Metamorphosis Representation Projection** (Wu et al., 2025; arXiv:2508.15449) — Aims at irreversible hidden-state transformation with resistance to relearning attacks.
- **Representation-Aware Unlearning via Activation Signatures** (Mahmood et al., 2026; arXiv:2601.10566) — Studies the suppression-vs-erasure distinction through activation signatures. Shows benchmark scores alone cannot determine whether knowledge is truly removed or merely suppressed.

**Key finding:** Representation-level features (activation patterns, hidden state geometry, circuit structure) contain rich information about how knowledge is stored and how unlearning affects it. These features could serve as the basis for a pre-unlearning corruption predictor.

### 7. Knowledge Correlation and Evaluation

- **Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness** (2025; arXiv:2506.05735) — Uses knowledge graphs with confidence scores to capture latent inferential dependencies. Shows that facts presumed forgotten can persist through *correlated* information. Current evaluation overestimates unlearning effectiveness by overlooking implicit knowledge structures.
- **Dynamic Evaluation Framework** (COLM 2025) — Stress-tests unlearning using complex structured queries including multi-hop reasoning and entity aliasing. Single-hop queries follow dominant computation pathways easily disrupted by unlearning; multi-hop queries use alternative pathways that often remain intact.

**Key finding:** Knowledge in LLMs is interconnected via correlation structures and inferential dependencies. The *position* of forget data within this knowledge graph likely determines how much collateral corruption occurs — but this connection has not been formalized as a predictive framework.

---

## Part II: Research Gap Analysis

### Gap Summary Table

| What Exists | What Is Missing |
|---|---|
| Post-hoc diagnosis of knowledge corruption (TOFU, WMDP, MUSE, BLUR, knowledge holes, blind spots) | **Pre-unlearning prediction** of corruption severity from forget data features |
| Pre-unlearning difficulty prediction per sample (CUD) | Prediction of **corruption to irrelevant knowledge** (not just forget difficulty) |
| Knowledge entanglement analysis during unlearning (EGUP, CLReg, SKeB, UIPE) | Feature extraction from forget data to **quantify entanglement-driven corruption risk before unlearning** |
| Data-feature-based prediction of fine-tuning outcomes (TuneAhead, Data2Behavior) | Analogous data-feature-based prediction for **unlearning corruption outcomes** |
| Training data → adversarial robustness correlation (reference paper) | **Forget data → knowledge corruption correlation** |
| Representation-level unlearning methods (LUNAR, FALCON, MRP, CIR) | Using representation-level features as **predictive inputs** for corruption estimation |
| Knowledge graph-based evaluation of unlearning completeness | Using knowledge graph structure of forget data as a **predictive feature** for corruption |

### The Core Research Gap

**No existing work systematically extracts features from the forget data (and its relationship to the model's existing knowledge) to predict, before unlearning occurs, how much irrelevant knowledge will be corrupted.**

This gap is the direct analogue of what the reference paper filled for adversarial robustness: they showed that training data features predict adversarial vulnerability before fine-tuning. We propose to show that forget data features predict knowledge corruption before unlearning.

### Specific Sub-Gaps

**Sub-Gap 1: No feature taxonomy for forget data properties relevant to corruption.**
The reference paper defined 13 features in four categories (embedding distribution, label distribution, learnability, token statistics). No analogous feature taxonomy exists for forget data in the unlearning context.

**Sub-Gap 2: No lightweight corruption predictor.**
The reference paper used Random Forest on data features to predict attack success rates. No analogous lightweight predictor exists for knowledge corruption in unlearning.

**Sub-Gap 3: No cross-method transferability analysis.**
The reference paper showed their framework transfers across models. No work has studied whether forget-data-based corruption predictions transfer across unlearning methods (gradient ascent, RMU, LUNAR, etc.).

**Sub-Gap 4: No connection between data-centric prediction and mitigation.**
Even works that study entanglement (EGUP, CLReg) use entanglement to *adjust the unlearning process*, not to *predict corruption upfront* and inform whether/how to proceed.

**Sub-Gap 5: No cost-benefit analysis framework.**
The reference paper quantified computational savings (30x–193x). No analogous analysis exists for how much evaluation effort can be saved by predicting corruption before unlearning rather than measuring it after.

---

## Part III: Proposed Solution Directions

### Direction 1: Forget Data Feature Taxonomy for Corruption Prediction

**Core idea:** Define a set of measurable features that can be extracted from the forget data (and its relationship to the model's knowledge base) *before* unlearning, analogous to the 13 features in the reference paper.

**Proposed feature categories:**

#### A. Embedding Distribution Features
- **Forget-retain embedding overlap:** Average cosine similarity between forget data embeddings and retain/general knowledge embeddings. Higher overlap → higher corruption risk.
- **Forget cluster dispersion:** How spread out the forget data is in embedding space. More dispersed forget data may corrupt more diverse knowledge areas.
- **Nearest-retain distance distribution:** Distribution of distances from each forget sample to its nearest retain neighbor. Tighter proximity → higher entanglement risk.
- **Forget subspace dimensionality:** Effective rank of the forget data embedding matrix. Higher dimensionality → broader representational footprint → more potential for corruption.
- **Shared subspace ratio:** Fraction of the forget data's principal components that overlap with retain data's principal components (via SVD). Higher ratio → more entangled representations.

#### B. Knowledge Graph / Relational Features
- **Knowledge connectivity:** Number of knowledge graph edges connecting forget entities to non-forget entities. Higher connectivity → more corruption pathways.
- **Multi-hop reachability:** Fraction of retain knowledge reachable from forget knowledge within k hops in a knowledge graph. Quantifies the "blast radius" of forgetting.
- **Entity co-occurrence frequency:** How often forget-set entities co-occur with non-forget entities in the training corpus (if accessible) or in model-generated text.

#### C. Circuit / Mechanistic Features
- **Circuit depth of forget knowledge:** Average depth of circuits encoding forget data (borrowing from CUD). Deeper circuits may share more components with other knowledge.
- **Circuit overlap with retain knowledge:** Fraction of circuit edges used for forget data that are also used for representative retain data.
- **Layer concentration:** Which layers primarily encode the forget knowledge. Knowledge concentrated in earlier layers (more shared) may cause broader corruption than knowledge in later, more specialized layers.

#### D. Model Behavioral Features (lightweight probing)
- **Surrogate forget success rate:** Run a small-scale proxy unlearning (e.g., 1–5 gradient ascent steps) and measure how much retain performance drops. This is a cheap probe of corruption sensitivity.
- **Forget data perplexity:** How "surprising" the forget data is to the model. Very low perplexity (deeply memorized) data may be harder to remove cleanly.
- **Gradient alignment:** Cosine similarity between gradients on forget data and gradients on retain data. High alignment → the optimization directions for forgetting conflict with retention.

#### E. Token / Surface Statistics
- **Average forget sequence length**
- **Vocabulary overlap between forget and retain sets**
- **Named entity density in forget data**
- **Topic diversity of forget data** (measured by topic model or LDA)

### Direction 2: Lightweight Corruption Predictor

**Core idea:** Train a predictor (Random Forest, gradient boosting, or small neural network) that maps forget data features → predicted corruption metrics.

**Training procedure:**
1. Curate a diverse set of forget-set / retain-set / model combinations.
2. For each combination, extract the proposed features from the forget data.
3. Run actual unlearning and measure knowledge corruption on retain and general benchmarks.
4. Train the predictor on (features → corruption metrics) pairs.
5. Evaluate on held-out combinations.

**Target metrics to predict:**
- Retain accuracy drop (overall model utility degradation)
- Domain-specific knowledge degradation (performance drop on specific knowledge domains unrelated to the forget target)
- Knowledge hole extent (following the probing methodology of Ko et al.)
- Multi-hop reasoning degradation on non-forget topics

**Computational savings:** If the predictor can reliably estimate corruption without running unlearning + full evaluation, significant compute can be saved — especially when comparing candidate forget sets or unlearning configurations.

### Direction 3: Feature-Guided Pre-Unlearning Intervention

**Core idea:** Once we can predict which forget data properties cause the most corruption, we can intervene on the data *before* unlearning to reduce predicted corruption.

This connects to the existing SURF framework (from `papers/llm-unlearning-research-gap-table.md`) but adds a specific predictive model:

1. **Extract features** from proposed forget data.
2. **Predict corruption** using the trained predictor.
3. **If predicted corruption exceeds threshold:**
   - Decompose the forget set into subsets with different corruption risk profiles.
   - Add retain anchors for high-risk subsets (samples nearest to the corruption-prone forget data).
   - Rewrite ambiguous forget samples to reduce overlap with retain knowledge.
   - Adjust unlearning hyperparameters (learning rate, number of steps) based on predicted difficulty.
4. **Run unlearning** on the refined data.

### Direction 4: Cross-Method Transferability Study

**Core idea:** Study whether the correlation between forget data features and knowledge corruption is stable across different unlearning methods.

**Hypothesis:** Some features (e.g., embedding overlap, knowledge connectivity) are "universal" predictors of corruption regardless of unlearning method, while others (e.g., gradient alignment, circuit depth) are method-specific.

**Methods to compare:**
- Gradient ascent variants (GA, NPO)
- Representation methods (RMU, LUNAR, FALCON)
- Second-order methods (SOUL, Gauss-Newton)
- Data-augmentation methods (ReLearn)

If universal features exist, a single predictor could serve as a method-agnostic corruption estimator. If features are method-specific, the predictor needs method-conditional branches.

### Direction 5: Mechanistic Analysis of the Forget-Corruption Pathway

**Core idea:** Go beyond correlation to understand the *causal mechanism* by which forget data properties lead to corruption.

**Proposed methodology:**
1. Use causal tracing to identify the circuit components involved in encoding the forget knowledge.
2. Map the overlap between forget circuits and circuits supporting unrelated knowledge.
3. Show that unlearning methods that modify shared components cause more corruption than methods targeting forget-specific components.
4. Demonstrate that the degree of circuit sharing is predictable from data-level features.

This would provide a mechanistic explanation for *why* certain forget data features predict corruption, strengthening the empirical prediction story with causal understanding.

---

## Part IV: Concrete Research Hypotheses

### H1 — Forget Data Features Predict Corruption
Features extractable from the forget data before unlearning can predict the severity of irrelevant knowledge corruption with accuracy comparable to TuneAhead's fine-tuning performance prediction (~89%).

### H2 — Embedding Overlap Is the Strongest Predictor
Among all feature categories, forget-retain embedding overlap (shared subspace ratio, nearest-retain distance) is the most informative predictor of knowledge corruption, because it directly measures representational entanglement.

### H3 — Predictions Transfer Across Unlearning Methods
A corruption predictor trained on one unlearning method (e.g., gradient ascent) retains substantial predictive power when applied to a different method (e.g., LUNAR), because the fundamental data-knowledge entanglement is method-independent.

### H4 — Corruption Prediction Enables Better Unlearning
Using predicted corruption to guide pre-unlearning data refinement (adding retain anchors, adjusting forget data) reduces actual corruption by a significant margin compared to blind unlearning.

### H5 — Lightweight Probing Is Sufficient
A small number of proxy unlearning steps (1–5 gradient updates) combined with data features provides substantially better corruption prediction than data features alone, at minimal additional cost.

### H6 — Knowledge Graph Connectivity Predicts Corruption Scope
The number of non-forget entities reachable from forget entities within k hops in a knowledge graph predicts the *breadth* (number of affected domains) of corruption, while embedding overlap predicts the *depth* (severity within each domain).

---

## Part V: Relationship to Existing Workspace Ideas

### Connection to SURF Framework (`papers/llm-unlearning-research-gap-table.md`)

SURF proposes pre-unlearning data refinement using embedding and SVD analysis. The current proposal **adds a predictive layer**: rather than just refining data heuristically, we first *predict* how much corruption a given forget set will cause, then use that prediction to guide refinement decisions. This makes SURF's data operations evidence-based rather than heuristic.

### Connection to Non-Forgettable Safety Knowledge (`idea/non-forgettable-safety-knowledge.md`)

The existing idea identifies safety-critical knowledge that must not be corrupted. The current proposal provides a mechanism to *predict* whether a given unlearning operation will corrupt that safety-critical knowledge, enabling proactive protection rather than post-hoc detection.

### Connection to Emergent Capabilities (`idea/unlearning-emergent-capabilities.md`)

The emergent capabilities idea asks whether unlearning can produce positive side effects. The current proposal studies the *negative* side (corruption), but the feature framework could be extended to predict both positive and negative downstream effects of unlearning — a unified "unlearning impact predictor."

---

## Part VI: Experimental Design Sketch

### Phase 1: Feature Engineering and Data Collection

1. Select 3–5 models (e.g., Llama-2-7B, Llama-3-8B, Mistral-7B, Phi-3).
2. Use existing benchmarks (TOFU, WMDP) and construct additional forget sets with controlled properties (varying overlap, connectivity, depth).
3. For each forget set, extract all proposed features.
4. Run 3–5 unlearning methods on each combination.
5. Measure corruption using:
   - Retain accuracy on MMLU / TriviaQA / domain-specific benchmarks
   - Knowledge hole probing (Ko et al.)
   - Multi-hop reasoning on non-forget topics
   - General language modeling perplexity

### Phase 2: Predictor Training and Evaluation

1. Split data into train/validation/test by model or by forget set.
2. Train lightweight predictors (Random Forest, XGBoost, small MLP).
3. Evaluate:
   - Prediction accuracy (R², MAE, ranking correlation)
   - Feature importance (SHAP values)
   - Cross-method transfer
   - Cross-model transfer

### Phase 3: Predictive Intervention

1. Use the trained predictor to identify high-corruption-risk forget sets.
2. Apply feature-guided data refinement.
3. Compare unlearning outcomes with and without predictive intervention.
4. Measure computational savings vs. running full unlearning + evaluation.

### Evaluation Benchmarks

| Benchmark | Purpose |
|---|---|
| TOFU | Foundational forgetting + utility evaluation |
| WMDP | Safety-sensitive knowledge removal |
| MUSE | Multi-dimensional evaluation |
| BLUR | Realistic forget-retain overlap |
| MMLU | General knowledge retention |
| TriviaQA | Factual knowledge retention |
| Custom domain probes | Targeted corruption measurement |

---

## Part VII: Positioning and Novelty Claim

### Strongest Novelty Claim

> Prior work has extensively documented that LLM unlearning causes collateral corruption of irrelevant knowledge, and recent studies have begun to characterize knowledge entanglement as a driver of this corruption. However, no existing work provides a systematic, data-centric framework for *predicting* the severity of knowledge corruption *before unlearning occurs* based on measurable properties of the forget data. We propose the first such framework — inspired by data-feature-based prediction of adversarial robustness before fine-tuning — that extracts features from the forget data's embedding distribution, knowledge graph structure, circuit properties, and surface statistics, and uses them to predict corruption metrics before any unlearning update is applied.

### What This Is NOT

- It is NOT just another unlearning method.
- It is NOT just another evaluation benchmark.
- It is NOT just an entanglement analysis during unlearning.

### What This IS

- A **prediction framework**: given forget data, estimate corruption before unlearning.
- A **feature taxonomy**: systematic categorization of forget data properties relevant to corruption.
- A **practical tool**: save compute by predicting outcomes instead of running full unlearning + evaluation.
- A **bridge between data-centric AI and LLM unlearning**: apply data-feature prediction (a well-established paradigm) to a new and important problem.

---

## References

### Core Reference
1. A Curious Case of Searching for the Correlation between Training Data and Adversarial Robustness of Transformer Textual Models. ACL Findings 2024. arXiv:2402.11469

### LLM Unlearning Benchmarks
2. TOFU: A Task of Fictitious Unlearning for LLMs. Maini et al., 2024. arXiv:2401.06121
3. The WMDP Benchmark. Li et al., ICML 2024. arXiv:2403.03218
4. MUSE: Machine Unlearning Six-Way Evaluation. Shi et al., 2024. arXiv:2407.06460
5. BLUR: A Benchmark for LLM Unlearning Robust to Forget-Retain Overlap. Hu et al., 2025. arXiv:2506.15699

### Knowledge Corruption and Utility Degradation
6. Rethinking Machine Unlearning for LLMs. Fan et al., Nature Machine Intelligence 2025. arXiv:2402.08787
7. Does Unlearning Truly Unlearn? Doshi & Stickland, 2024. arXiv:2411.12103
8. Probing Knowledge Holes in Unlearned LLMs. Ko et al., NeurIPS 2025. arXiv:2511.00030
9. Unlearning's Blind Spots. Ha et al., 2025. arXiv:2506.01318
10. Multi-Turn Robustness Evaluation. Pan & Wang, 2026. arXiv:2603.00823

### Knowledge Entanglement
11. EGUP: Entanglement-Guided Unlearning with Proxy Constraint. 2025. arXiv:2508.20443
12. SKeB: Stimulus-Knowledge Entanglement-Behavior Framework. 2025. arXiv:2510.25732
13. CLReg: From Logits to Latents. Tang & Khanna, 2026. arXiv:2601.22028
14. UIPE: Enhancing LLM Unlearning by Removing Related Knowledge. EMNLP 2025. arXiv:2503.04693
15. CIR: Collapse of Irrelevant Representations. Sondej & Yang, 2025. arXiv:2509.11816

### Predicting Unlearning Difficulty
16. CUD: Circuit-Guided Unlearning Difficulty Metric. Cheng et al., 2026. arXiv:2601.09624
17. When to Forget? Complexity Trade-offs in Machine Unlearning. van Waerebeke et al., ICML 2025
18. Mechanistic Unlearning. Guo et al., ICML 2025. arXiv:2410.12949
19. KUDA: Knowledge Unlearning by Deviating Representation. Fang et al., 2026. arXiv:2602.19275

### Data-Centric Prediction
20. TuneAhead: Predicting Fine-tuning Performance Before Training. OpenReview, 2025
21. Data2Behavior. 2025. arXiv:2602.04735
22. PRISM: Training Data Prototypes for Language Models. GuideAI, 2025
23. Do LLMs Really Forget? Knowledge Correlation and Confidence Awareness. 2025. arXiv:2506.05735

### Representation-Level Unlearning
24. LUNAR: Neural Activation Redirection. Shen et al., NeurIPS 2025. arXiv:2502.07218
25. FALCON. Hu et al., NeurIPS 2025. arXiv:2502.01472
26. MRP: Metamorphosis Representation Projection. Wu et al., 2025. arXiv:2508.15449
27. Representation-Aware Unlearning via Activation Signatures. Mahmood et al., 2026. arXiv:2601.10566
28. ReLearn: Unlearning via Learning. Xu et al., ACL 2025. arXiv:2502.11190

### Related Methods and Surveys
29. A Comprehensive Survey of Machine Unlearning Techniques for LLMs. 2025. arXiv:2503.01854
30. Unlearning in LLMs: Methods, Evaluation, and Open Challenges. 2026. arXiv:2601.13264
31. Align-then-Unlearn. Spohn et al., 2025. arXiv:2506.13181
32. SOUL: Second-Order Optimization for LLM Unlearning. Jia et al., 2024. arXiv:2404.18239
33. Gauss-Newton Unlearning. McKinney et al., SaTML 2026. arXiv:2602.10568
34. Unlearning That Lasts (JensUn). Singh et al., 2025. arXiv:2509.02820
