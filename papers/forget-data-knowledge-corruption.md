# Forget Data -> Irrelevant Knowledge Corruption in LLM Unlearning

This note is a literature review and project-positioning memo for the question:

> **How do forget data correlate with knowledge corruption?**
>  
> **Can we estimate knowledge corruption before a model is unlearned?**

The motivation is inspired by a robustness-style question:

> Can we predict a downstream failure property before expensive post-training intervention, using properties of the data and the base model, without explicitly generating failure cases?

Here the downstream failure property is not adversarial robustness, but:

> **irrelevant knowledge corruption** caused by unlearning.

In other words, the project is not only about whether the model forgets the target content. It is about whether the **forget set** causes unintended damage to adjacent, benign, or protective knowledge.

## 1. Core problem formulation

Your proposed question can be made precise as:

### Main question

How do the properties of the forget data influence the amount and type of collateral knowledge corruption after unlearning?

### Early-warning question

Can we estimate the corruption risk **before unlearning**, using only:

- the forget set,
- the retain set or protected set,
- the pretrained or pre-unlearning model,
- and cheap diagnostics that do **not** require generating adversarial examples or running many trial unlearning procedures?

### Strongest paper framing

The strongest framing is **not**:

- "unlearning hurts utility,"
- or "knowledge holes exist,"
- or "we propose another unlearning method."

The stronger framing is:

> Existing work shows that unlearning can create hidden knowledge holes and overlap-driven side effects, but the field still lacks a principled way to **predict collateral knowledge corruption before unlearning happens** from the structure of the forget data and its interaction with the model.

## 2. What exactly is "knowledge corruption"?

For this project, "knowledge corruption" should be defined more sharply than generic utility drop.

A useful taxonomy is:

### A. Adjacent benign knowledge corruption

The model forgets or degrades nearby but benign knowledge that is semantically close to the forget target.

Example:

- forgetting a harmful bio procedure also harms benign biology explanations,
- forgetting copyrighted passages harms nearby literary or factual discussion,
- forgetting one individual's information harms related but non-target facts.

### B. Protective knowledge corruption

The model loses safety-relevant boundary knowledge:

- hazard recognition,
- refusal support,
- safe redirection,
- calibrated uncertainty around dangerous topics.

This is especially important when the forget target is harmful content.

### C. Structural knowledge-hole corruption

The model no longer gives meaningful answers in regions that were not explicitly targeted, producing:

- irrelevant responses,
- nonsensical responses,
- over-refusal,
- incoherent uncertainty,
- or blank ignorance.

### D. Latent corruption without obvious benchmark failure

The model may still pass a narrow benchmark, but the internal representation or causal pathway for benign knowledge has already been destabilized.

This is important because the project should not depend only on surface utility metrics.

## 3. What the literature already establishes

The current literature already gives strong support for the broad motivation.

### 3.1 Unlearning can create hidden side effects

Several papers already support the claim that unlearning can damage non-target knowledge in ways standard metrics miss.

#### TOFU (2024)

- Establishes a foundational benchmark for LLM unlearning.
- Shows forgetting and utility are hard to balance reliably.
- Good benchmark reference, but not yet a corruption-prediction paper.

#### WMDP (2024)

- Shows hazardous-domain unlearning is difficult while preserving broader capability.
- Useful for safety-motivated forget targets.
- Still mainly an evaluation benchmark, not a pre-unlearning predictor.

#### MUSE (2024)

- Expands evaluation beyond a single metric.
- Important because your project should explicitly argue that "knowledge corruption" is multi-dimensional rather than just accuracy drop.

#### BLUR (2025)

- Demonstrates that realistic **forget-retain overlap** makes unlearning much harder.
- This is one of the strongest pieces of evidence for your project:
  - collateral damage is not random,
  - it is strongly related to overlap structure between what should be forgotten and what should be retained.

This already hints that **forget-data geometry** matters.

#### Probing Knowledge Holes in Unlearned LLMs (2025)

This is one of the most directly relevant papers.

Its core finding is:

> unlearned models may create "knowledge holes" that standard benchmarks fail to capture.

The paper reports that many generated test cases around unlearned content receive irrelevant or nonsensical responses, even though the pretrained model could answer them.

Why this matters for your idea:

- It strongly validates the existence of the failure mode you care about.
- It supports the claim that standard retain-set utility is too weak.
- It motivates studying **where corruption happens around the forget region**.

But it is still mostly a **post-unlearning probing** paper, not a **pre-unlearning prediction** paper.

#### Unlearning's Blind Spots: Over-Unlearning and Prototypical Relearning Attack (2025/2026)

This paper is especially useful conceptually even though it focuses on class-level unlearning.

Its key insight is:

> collateral damage is concentrated near the forget set, especially in regions close to the forget boundary.

Why this matters:

- It gives a strong prior that corruption should correlate with **proximity**, **overlap**, and **boundary structure**.
- It supports a boundary-aware view rather than a global utility-only view.

Again, this is evidence for the phenomenon, but not yet a full pre-unlearning corruption estimator for LLMs.

#### Verifying Robust Unlearning / residual knowledge probing (2025)

These papers argue that output-level success can overestimate true unlearning and miss hidden residual or distorted knowledge.

This helps justify a key design choice in your project:

> prediction targets should include hidden corruption, not only visible benchmark failure.

### 3.2 Some papers already move toward pre-unlearning diagnostics

This is where the literature becomes especially important for judging whether your idea is a real gap.

#### Toward Understanding Unlearning Difficulty: A Mechanistic Perspective and Circuit-Guided Difficulty Metric (2026)

This is the most important "partial overlap" paper.

Its contribution is a **pre-unlearning metric** for sample difficulty:

- some samples are intrinsically harder to unlearn,
- and this can be estimated using circuit-level signals before unlearning.

Why it matters:

- It proves that **pre-unlearning prediction is possible in principle**.
- It gives methodological support for using circuit-level or mechanistic features.

Why it does **not** close your gap:

- it predicts **difficulty of forgetting the target**,
- not **collateral corruption of irrelevant knowledge**,
- and not the amount of benign/protective knowledge damage induced by a forget set.

So this paper is not a blocker; it is actually a key supporting citation.

#### Mechanistic Unlearning (2025)

- Shows target knowledge can be localized more precisely through mechanism-level analysis.
- Supports the idea that the *way* knowledge is encoded matters for unlearning quality.

Why it matters for your project:

- if forget knowledge and benign knowledge share mechanisms,
- then corruption risk should be predictable from that shared mechanism structure.

But the paper is still mainly about better targeting and editing, not pre-unlearning collateral-damage prediction.

### 3.3 Data-centric and representation-centric papers support the intuition

Several recent directions support your intuition that the **forget data themselves** are not neutral.

#### ReLearn (2025)

- Shows data augmentation can improve LLM unlearning.
- Important because it implies the forget set can be reshaped to change outcomes.

#### Align-then-Unlearn (2025)

- Argues token-level objectives are too shallow and uses embedding alignment before unlearning.
- This supports the idea that semantic geometry before unlearning matters.

#### Representation-aware methods: FALCON, LUNAR, MRP, KUDA, CIR, activation-signature work

These papers show:

- knowledge entanglement lives in representation space,
- overlap between harmful and benign knowledge matters,
- and targeting latent structure can reduce side effects.

Why they matter for your idea:

- They support the hypothesis that corruption risk should be inferable from representation overlap, shared subspaces, and mechanism entanglement.

Why they do not close the gap:

- Most of them are intervention methods,
- not predictive frameworks that estimate corruption **before** running unlearning.

## 4. Evidence-based research-gap judgment

The cleanest answer is:

### Short answer

**Yes, this looks like a real and defensible research gap, but only under a specific framing.**

### What is already explored

The following are **not** good novelty claims by themselves:

1. **"Unlearning causes side effects."**
   - Already well supported.

2. **"Knowledge holes exist after unlearning."**
   - Already supported by recent probing work.

3. **"Forget-retain overlap matters."**
   - Already strongly suggested by BLUR and representation-aware papers.

4. **"Some pre-unlearning prediction is possible."**
   - Already partially supported by CUD-like difficulty estimation.

### What still looks under-explored

The strongest under-explored gap is:

> **Predicting collateral corruption of irrelevant or protected knowledge before unlearning, from properties of the forget data and their overlap with retained knowledge.**

More concretely, I do **not** see a mature literature that already answers:

1. Which forget sets are likely to produce large irrelevant knowledge corruption?
2. Which forget samples are high-risk for benign or protective collateral damage?
3. Can we estimate this risk before unlearning without generating adversarial examples?
4. Can those estimates guide:
   - data refinement,
   - algorithm selection,
   - or risk-aware unlearning constraints?

That is a much sharper and more novel question than generic "knowledge holes after unlearning."

## 5. A precise novelty claim you can defend

If you write this up as a paper, a good novelty claim is:

> Prior work has shown that unlearning may induce hidden knowledge holes, over-unlearning, and overlap-driven failures, while recent mechanistic work begins to estimate pre-unlearning forgetting difficulty. However, it remains under-explored whether **the forget data themselves can be used to predict collateral corruption of irrelevant or protected knowledge before unlearning**, and whether such predictions can support risk-aware data curation or unlearning.

This positioning is much safer than:

- "nobody studies knowledge corruption,"
- or "nobody studies pre-unlearning diagnostics."

## 6. Recommended research question hierarchy

To make the project publishable, I would split the problem into a hierarchy of questions.

### RQ1: Correlation

Which properties of forget data correlate with post-unlearning knowledge corruption?

Candidate properties:

- semantic overlap with retain/protect data,
- embedding-space density and cluster spread,
- paraphrase diversity of the forget concept,
- entity/relation centrality,
- hidden-state subspace overlap,
- gradient conflict with retain data,
- circuit or mediator overlap with protected knowledge.

### RQ2: Prediction

Can we estimate corruption risk before unlearning from those properties?

This is the direct analogue of the robustness motivation.

### RQ3: Transfer

Do these risk indicators transfer across:

- different unlearning methods,
- different model scales,
- different forget domains?

This matters because otherwise the predictor may just overfit one unlearning algorithm.

### RQ4: Intervention

Can predicted corruption risk be reduced by acting **before** or **during** unlearning?

For example:

- editing or splitting forget data,
- adding protect anchors,
- reweighting high-risk forget samples,
- or choosing a safer unlearning backend.

## 7. A concrete solution direction

The cleanest solution is not to start with a new unlearning optimizer.

The cleaner first contribution is:

> a **pre-unlearning corruption-risk estimation framework**

that can later be paired with data refinement or risk-aware unlearning.

### Working name

Possible names:

- **PUCR**: Pre-Unlearning Corruption Risk
- **KCR**: Knowledge Corruption Risk
- **FORGE**: Forget-set Overlap and Representation Geometry Estimator

### 7.1 Prediction target

Define one or more post-unlearning corruption metrics that the pre-unlearning predictor will estimate.

Good targets:

1. **Adjacent benign corruption**
   - performance drop on benign neighbors of forget concepts.

2. **Protective knowledge corruption**
   - drop on hazard recognition, refusal support, safe redirection, or other protected knowledge.

3. **Knowledge-hole rate**
   - rate of irrelevant, nonsensical, or over-refusal outputs in nearby but non-target regions.

4. **Mechanistic corruption**
   - drift of mediator overlap or protected-circuit integrity after unlearning.

The first paper can start with output-based corruption and later add mechanistic corruption.

### 7.2 Pre-unlearning features

The predictor can combine three families of features.

#### A. Data-only features

These are the cheapest and easiest to scale.

- lexical diversity of forget data,
- semantic cluster count,
- cluster imbalance,
- forget concept centrality in an entity/relation graph,
- nearest-neighbor overlap with retain data,
- ambiguity score for forget samples,
- paraphrase coverage or under-coverage.

#### B. Representation features

These are extracted from the pre-unlearning model.

- embedding overlap between forget and retain sets,
- hidden-state subspace angle,
- shared singular directions from SVD/PCA,
- layer-wise representation similarity,
- influence of forget samples on nearby benign samples.

This is likely one of the strongest feature families.

#### C. Mechanistic features

These are more expensive but can give the highest novelty.

- attention-head or MLP mediator overlap,
- circuit depth / pathway length,
- concentration of forget behavior in late vs early layers,
- mediator overlap between forget prompts and protected prompts,
- gradient conflict or activation conflict between forget and protect sets.

This is where the project can connect directly to the existing CMA and mechanistic-unlearning notes in this repository.

### 7.3 A simple prediction pipeline

A practical first pipeline:

1. Start with a base model and several forget sets.
2. Build:
   - forget set `F`,
   - retain set `R`,
   - protected set `P` of benign or safety-critical neighboring knowledge.
3. Compute pre-unlearning features from `F`, `R`, and the base model.
4. Run one or more standard unlearning methods.
5. Measure post-unlearning corruption on `P` and on generated neighborhood probes.
6. Train or fit a risk estimator:
   - linear model,
   - gradient-boosted trees,
   - or a simple neural regressor.
7. Evaluate:
   - within-method prediction,
   - cross-method transfer,
   - cross-domain transfer,
   - and whether high-risk examples truly correspond to more corruption.

This gives a clean paper story:

- first, show correlation;
- second, show prediction;
- third, optionally reduce risk using the prediction.

## 8. Best intervention ideas after prediction

Once a risk score exists, you have several natural second-step methods.

### Idea A: Risk-aware data refinement before unlearning

For high-risk forget samples:

- rewrite broad forget requests into more target-specific ones,
- split entangled forget samples into narrower concepts,
- add benign neighbors as retain or protect anchors,
- add contrastive retain samples near the boundary.

This is probably the cleanest extension of the prediction framework.

### Idea B: Risk-aware reweighting during unlearning

- Downweight high-corruption-risk forget samples unless supported by sufficient protect anchors.
- Upweight retain/protect samples near the forget boundary.

### Idea C: Method selection

Use the pre-unlearning risk profile to choose among:

- gradient-based unlearning,
- representation-redirection methods,
- alignment-based methods,
- or more constrained geometry-aware methods.

This would let you argue:

> different forget sets call for different unlearning strategies.

### Idea D: Boundary-aware mechanistic preservation

If mediator overlap between forget and protect knowledge is high:

- freeze or regularize protect-side mediators,
- constrain updates on shared mediators,
- only target forget-specific subcircuits when possible.

This is the closest bridge to the "non-forgettable safety-critical knowledge" direction.

## 9. Recommended experimental setup

### 9.1 Task regimes

Use at least two regimes:

1. **Harmful knowledge unlearning**
   - useful for the protective-knowledge story.

2. **Non-harmful factual/domain unlearning**
   - useful to show the idea is not limited to safety.

### 9.2 Data partition

Create:

- **forget set**: target knowledge to remove,
- **retain set**: general non-target utility data,
- **protect set**: adjacent benign or safety-critical knowledge that must remain,
- **neighbor probes**: generated or mined prompts near the forget region.

The key addition is the **protect set**. Without it, "knowledge corruption" will remain vague.

### 9.3 Evaluation metrics

Minimum recommended metrics:

- forgetting success,
- retain utility,
- adjacent benign corruption rate,
- knowledge-hole rate,
- protective-knowledge retention,
- paraphrase robustness,
- multi-turn robustness if feasible.

If you want one headline metric:

> **Corruption@k**: corruption on the top-k nearest benign/protect neighbors of the forget set.

That would make the paper more memorable.

## 10. Where the idea is strongest vs weakest

### Strongest version of the idea

The idea is strongest when framed as:

> Before unlearning, estimate how much a forget set will damage nearby benign or protective knowledge, using overlap and mechanistic signals from the base model.

Why this is strong:

- tightly motivated by recent knowledge-hole findings,
- not yet saturated,
- naturally measurable,
- compatible with both mechanistic and data-centric methods.

### Weaker version of the idea

The idea is weaker if framed as:

> Can we predict all bad outcomes of unlearning before unlearning?

That is too broad and hard to validate.

It is also weaker if framed as:

> We predict utility drop.

That sounds too generic and less novel than knowledge corruption.

## 11. Bottom-line assessment

My overall judgment is:

### Is it a research gap?

**Yes, probably yes, if you frame it narrowly and correctly.**

### Precise gap statement

What looks open is:

> a principled framework for estimating **irrelevant/protected knowledge corruption risk before unlearning**, based on the forget set's semantic, geometric, and mechanistic overlap with retained knowledge.

### Why it is not already solved

Because the current literature mostly does one of the following:

- documents corruption **after** unlearning,
- improves the unlearning update itself,
- estimates **forgetting difficulty** rather than **collateral corruption**,
- or studies overlap without converting it into a predictive risk model.

### Recommended first paper angle

If you want the cleanest first project, I would recommend:

> **Predicting Knowledge Corruption Before LLM Unlearning**

with the core claim:

> forget-set overlap and mechanistic entanglement in the pre-unlearning model predict which benign or protective knowledge will be corrupted after unlearning.

## 12. Suggested one-paragraph abstract draft

Here is a draft positioning paragraph you could reuse later:

> Recent work shows that LLM unlearning can induce hidden knowledge holes, over-unlearning, and failures on benign knowledge adjacent to the forget target. However, existing methods largely diagnose these failures only after unlearning has already been performed. We ask whether collateral knowledge corruption can be estimated *before* unlearning, directly from the forget data and their interaction with the pretrained model. We propose to model pre-unlearning corruption risk using semantic overlap, representation geometry, and mechanistic entanglement between forget and protected knowledge. This framing shifts LLM unlearning from purely reactive evaluation toward proactive risk estimation and risk-aware unlearning design.

## 13. Practical recommendation

If you continue this project, the best order is:

1. **Define corruption carefully** using a protect set and neighborhood probes.
2. **Show correlation first**, not full method novelty immediately.
3. **Build a pre-unlearning risk score** from overlap + representation features.
4. **Only then** add a risk-aware refinement or regularization method.
5. Use mechanistic analysis as a differentiator, not as the only pillar.

## 14. Short bibliography to anchor this idea

Use these as the main citation backbone:

- **TOFU** (2024) - foundational LLM unlearning benchmark.
- **WMDP** (2024) - safety-sensitive unlearning benchmark.
- **MUSE** (2024) - multi-dimensional unlearning evaluation.
- **BLUR** (2025) - forget-retain overlap matters.
- **Probing Knowledge Holes in Unlearned LLMs** (2025) - hidden collateral damage and knowledge holes.
- **Unlearning's Blind Spots: Over-Unlearning and Prototypical Relearning Attack** (2025/2026) - boundary-proximal collateral damage.
- **Toward Understanding Unlearning Difficulty: A Mechanistic Perspective and Circuit-Guided Difficulty Metric** (2026) - pre-unlearning difficulty prediction.
- **Mechanistic Unlearning** (2025) - mechanism-aware localization for targeted unlearning.
- **Align-then-Unlearn** (2025) - pre-stage representation alignment.
- **ReLearn** (2025) - data-centric unlearning augmentation.

## 15. One-sentence takeaway

The most defensible version of your idea is:

> **not** "unlearning causes corruption,"
> but **"forget-set overlap and mechanistic entanglement may let us predict irrelevant knowledge corruption before unlearning happens."**
