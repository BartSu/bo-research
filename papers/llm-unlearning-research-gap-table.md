# LLM Unlearning Research Gap Table and a Pre-Unlearning Data Refinement Framework

This note consolidates the current shortlist in `papers/pre-unlearning-data-augmentation-shortlist.md` and turns it into a more explicit research-gap map for a proactive, data-centric LLM unlearning agenda.

## 1. Prior research objective distilled from the existing notes

The existing shortlist already implies a clear objective:

1. **Act before unlearning**, not only during or after unlearning.
2. Build a **pre-unlearning data refinement / synthesis stage** over forget and retain data.
3. Make the policy **algorithm-adaptive**, since different unlearning algorithms fail in different ways.
4. Optimize for **stable one-shot unlearning**, instead of repeated unlearn-patch-unlearn loops.
5. Reduce **blind-spot corruption before deployment**, especially under:
   - paraphrase and prompt rewording,
   - forget-retain overlap,
   - relearning attacks,
   - multi-turn interaction,
   - hidden knowledge holes beyond surface-level refusal.

In short, the target is not "just another unlearning optimizer", but a **proactive pre-conditioning framework for the training data used by unlearning**.

## 2. Research gap table

The table below groups the current literature into clusters and evaluates each cluster against the above objective.

| Cluster | Representative work | What is already established | Gap vs. your objective | Opportunity level |
|---|---|---|---|---|
| Benchmarks and diagnostics | TOFU (2024), WMDP (2024), MUSE (2024), BLUR (2025), OpenUnlearning (2025), Blind Spots (2025), Verifying Robust Unlearning (2025), Probing Knowledge Holes (2025), Multi-Turn Robustness (2026) | The field now has strong evidence that forgetting, utility, privacy, overlap robustness, and multi-turn robustness must all be evaluated together. Blind spots and knowledge holes are real. | These works mostly **diagnose failure after the fact**. They do not provide a proactive policy for refining forget/retain data before the unlearning run. | **Open** |
| Data-centric augmentation before/during unlearning | ReLearn (2025), Data Augmentation Improves MU (2025, non-LLM), Reveal and Release (2025) | Augmentation and synthetic data can improve forgetting behavior. Data-centric intervention is a real lever, not just an implementation detail. | Existing methods do not yet unify **forget-side augmentation, retain-side protection, overlap-aware sample editing, and algorithm-adaptive control** in one framework. Some are iterative rather than one-shot. | **Open** |
| Embedding alignment / concept-level forgetting | Align-then-Unlearn (2025) | Embedding-space alignment helps with paraphrase robustness and concept-level forgetting. This is strong evidence that token-level supervision is too shallow. | Embeddings are used mainly to shape the unlearning objective itself, not to drive **pre-unlearning data refinement** such as add/edit/remove/reweight decisions over forget and retain sets. | **Open** |
| Representation redirection / projection / activation shaping | LUNAR (2025), MRP (2025), FALCON (2025), CIR (2025), Activation Signatures (2026), KUDA (2026), SSPU (2025) | Representation-level methods are increasingly effective at targeting latent knowledge instead of just output suppression. Orthogonality, projection, and subspace control are useful. | Most of these methods are still **post-hoc intervention methods** on weights or activations. They do not answer: *how should we prepare the forget/retain data beforehand so that a later one-shot update becomes easier and safer?* | **Open** |
| Geometry-aware and second-order optimization | SOUL (2024), Gauss-Newton Unlearning (2026), Geometric Disentanglement Unlearning (2025), AGT^AO (2026) | Better optimizers matter. Curvature, update-space disentanglement, and constrained one-shot updates improve the forget-retain trade-off. | These methods mostly assume the provided forget/retain data are already suitable. The missing piece is **data geometry shaping before optimization**. | **Open** |
| Mechanistic localization and difficulty estimation | Mechanistic Unlearning (2025), Circuit-Guided Difficulty Metric (2026) | Target knowledge can be localized more precisely, and some samples are intrinsically harder to unlearn than others. | The literature does not yet convert these insights into a **data refinement policy** that changes which data are added, removed, rewritten, or upweighted before unlearning. | **Open** |
| Robustness to relearning and persistence | Unlearning That Lasts (2025), MRP (2025), Blind Spots (2025) | Robustness and irreversibility are now recognized as first-class goals. | Most methods try to make the **update** more persistent. Fewer works ask whether **pre-unlearning data preparation** can increase persistence by covering latent variants and overlap regions in advance. | **Open** |
| Spectral / low-rank efficiency | SEMU (2025), Embarrassingly Efficient Unlearning with SVD (adjacent) | SVD and low-rank decomposition can make unlearning more efficient and parameter-sparse. | The spectral idea has mostly been used for **parameter-efficient updates**, not for **refining the data geometry before unlearning in LLMs**. This is important because your objective is proactive data refinement, not only efficient post-hoc editing. | **Open with novelty potential** |

## 3. High-confidence research gaps that still look under-explored

The most defensible gaps are the ones below.

### Gap G1: No unified pre-unlearning data refinement framework

The literature has:
- data augmentation papers,
- embedding/representation-aware unlearning papers,
- geometry-aware optimizers,
- strong diagnostic benchmarks.

What is still missing is a **single framework that runs before unlearning** and decides how to:
- add forget data,
- add retain anchors,
- edit ambiguous samples,
- remove redundant or harmful samples,
- reweight training pairs,
- adapt the above to the downstream unlearning algorithm.

### Gap G2: Weak handling of forget-retain overlap before optimization

BLUR, knowledge-hole work, and representation papers all suggest that overlap is one of the core reasons existing methods fail. However, most current methods try to fix overlap **during optimization** rather than **before optimization** by shaping the training data.

This is exactly where a pre-unlearning refinement framework can contribute:
- detect overlap early,
- preserve retain-relevant directions,
- stress-test forget coverage on paraphrases and nearby concepts.

### Gap G3: Missing bridge from latent geometry to data operations

Representation-aware papers show that latent geometry matters, but they usually use that information to guide weight or activation updates.

The missing bridge is:

> Use embedding/subspace diagnostics to decide concrete data operations on forget/retain datasets before unlearning.

This bridge is a strong novelty candidate because it connects:
- representation learning,
- data curation,
- one-shot unlearning.

### Gap G4: Algorithm-adaptive data preparation is still rare

Different unlearning families need different kinds of support:
- gradient-based methods need stronger retain anchors and overlap guards,
- embedding/projection methods need better coverage of latent variants,
- second-order methods benefit from cleaner high-leverage examples.

Most papers use a fixed recipe. An **algorithm-conditioned refinement policy** is still not well established.

### Gap G5: Existing robustness work is mostly reactive, not preventive

Blind spots, relearning attacks, and multi-turn leakage are well documented. But the field still lacks a strong answer to:

> Can we reduce those failures *before* the unlearning run by reshaping the forget/retain data geometry?

That is a strong and timely objective.

## 4. Where embedding-based ideas help

An embedding-based direction is plausible, but the positioning has to be careful.

### What is already done

- **Align-then-Unlearn** already uses embedding alignment for concept-level forgetting.
- Several representation-aware papers operate in latent space rather than on logits alone.

So a claim like "we use embeddings for unlearning" is **not novel enough** by itself.

### What is still promising

An embedding-based novelty claim becomes much stronger if the embeddings are used for **pre-unlearning data refinement**:

1. **Forget coverage estimation**
   - Cluster forget examples in embedding space.
   - Detect under-covered semantic regions.
   - Add paraphrases or counterfactual rewrites to fill those gaps.

2. **Retain-anchor mining**
   - Find retain examples nearest to the forget clusters.
   - Use them as hard retain anchors to protect adjacent benign knowledge.

3. **Ambiguity detection**
   - Flag examples whose embeddings sit in the overlap between forget and retain neighborhoods.
   - Edit or relabel them instead of blindly feeding them into unlearning.

4. **Difficulty-aware weighting**
   - Use embedding overlap or local density to upweight hard cases and downweight redundant easy cases.

### Embedding-first thesis statement

A defensible thesis is:

> Embeddings should not only define the unlearning loss; they should define how the forget/retain training data are refined before unlearning.

That is substantially more original than a pure embedding-alignment paper.

## 5. Where SVD / spectral ideas help

SVD is also plausible, but again the novelty depends on **how** it is used.

### What is already done

- **SEMU** and related spectral work already use SVD for efficient unlearning updates.
- Some representation-aware methods already reason in terms of subspaces, orthogonality, or projections.

So a claim like "we use SVD in unlearning" is also **not enough** by itself.

### What is still promising

SVD becomes interesting for your agenda if it is used to refine data *before* unlearning:

1. Build hidden-state matrices:
   - `H_f` for forget data,
   - `H_r` for retain data.

2. Compute low-rank structure:
   - `H_f = U_f Sigma_f V_f^T`
   - `H_r = U_r Sigma_r V_r^T`

3. Separate subspaces:
   - shared directions between forget and retain,
   - forget-specific directions,
   - retain-specific directions.

4. Use those subspaces to drive data operations:
   - add data covering forget-specific directions that are currently weak,
   - add retain anchors concentrated near shared directions,
   - remove redundant examples that only repeat already-dominant singular directions,
   - rewrite ambiguous examples that project too strongly onto shared subspaces.

### Why SVD is a good fit for your objective

SVD gives a clean way to transform a vague idea ("forget and retain are entangled") into measurable quantities:
- overlap ratio,
- subspace angle,
- per-sample projection score,
- coverage of rare but important directions.

This is valuable because your proposed contribution is not only algorithmic; it is also a **data-selection and data-editing policy**.

### SVD-first thesis statement

> Spectral decomposition can be used not merely to compress the unlearning update, but to expose which semantic directions should be expanded, protected, or rewritten in the pre-unlearning dataset.

That framing is much stronger than "SVD for efficient unlearning".

## 6. Recommended direction: a hybrid embedding + SVD pre-unlearning refinement framework

The most promising direction is not embedding-only or SVD-only, but a hybrid:

1. **Embeddings** for local semantic neighborhood detection,
2. **SVD / subspace analysis** for global overlap structure,
3. **Algorithm-adaptive rules** for deciding how refined data should support the chosen unlearning method.

Below is a concrete framework sketch.

## 7. Framework sketch: SURF

Working name:

**SURF: Subspace-guided Unlearning Refinement Framework**

Goal:

> Refine forget and retain data *before* unlearning so that a later one-shot unlearning method can forget more robustly while preserving adjacent benign knowledge.

### Stage 1: Representation profiling

For forget data `F` and retain data `R`:

1. Encode each example with:
   - sentence embedding,
   - prompt embedding,
   - response embedding,
   - optionally layer-wise hidden states from the base LLM.

2. Compute:
   - local nearest-neighbor overlap,
   - cluster structure of forget concepts,
   - retain examples closest to forget clusters,
   - hidden-state subspaces via SVD or PCA.

3. Produce three diagnostic scores per sample:
   - **forget-specificity**: how much the sample lies in forget-specific directions,
   - **overlap-risk**: how much it lies in shared forget-retain directions,
   - **coverage-value**: how much new semantic/subspace coverage it provides.

### Stage 2: Data refinement actions

Use the above scores to perform four types of data operations.

#### A. Add

- Add forget paraphrases for under-covered forget clusters.
- Add boundary prompts that probe nearby benign concepts.
- Add hard retain anchors nearest to forget clusters.
- Add multi-turn or adversarial variants for high-risk concepts.

#### B. Edit

- Rewrite ambiguous forget examples into more target-specific versions.
- Rewrite retain samples to become stronger contrastive anchors when benign and harmful knowledge are easily confusable.
- Convert broad prompts into minimal pairs that isolate the exact concept to forget.

#### C. Remove

- Remove duplicate or low-value forget samples that only reinforce already-dominant singular directions.
- Remove noisy retain samples that are far from the decision boundary and contribute little protection.

#### D. Reweight

- Upweight high-overlap and high-difficulty samples.
- Downweight redundant easy samples.
- Allocate more budget to forget directions that are weakly represented but likely to survive paraphrase attacks.

### Stage 3: Algorithm-adaptive policy

The same refined data should not be fed identically to all unlearning algorithms.

#### If the downstream method is gradient-based

Examples: NPO-style, RMU-style, gradient-ascent / descent variants.

Policy:
- emphasize hard retain anchors,
- emphasize overlap-risk examples,
- include contrastive retain pairs,
- penalize updates that erase shared benign directions.

#### If the downstream method is embedding / representation alignment based

Examples: Align-then-Unlearn, redirection/projection methods.

Policy:
- emphasize paraphrase diversity,
- emphasize cluster coverage in embedding space,
- include target concept variants that are semantically close but lexically different.

#### If the downstream method is second-order or geometry-aware

Examples: SOUL, Gauss-Newton, constrained low-rank methods.

Policy:
- emphasize high-leverage examples aligned with important singular directions,
- keep the refined set compact but information-dense,
- prioritize samples that most clearly separate forget-specific vs shared subspaces.

### Stage 4: One-shot unlearning

Run a single downstream unlearning step or a small fixed-budget unlearning procedure on the refined dataset.

The key research claim is:

> Better pre-unlearning data geometry should make one-shot unlearning both safer and more robust.

### Stage 5: Evaluation

A convincing evaluation should include:

- **TOFU** for foundational forgetting behavior,
- **WMDP** for safety-sensitive removal,
- **MUSE** for multi-dimensional evaluation,
- **BLUR** for realistic forget-retain overlap,
- **OpenUnlearning** for cross-method comparison,
- paraphrase tests,
- relearning stress tests,
- multi-turn leakage tests,
- knowledge-hole probing.

## 8. Concrete hypotheses

The following hypotheses are strong and testable.

### H1

Pre-unlearning data refinement improves one-shot forgetting robustness under paraphrase more than naive augmentation.

### H2

Subspace-aware retain-anchor mining reduces knowledge holes and over-unlearning under forget-retain overlap.

### H3

SVD-guided sample selection and rewriting outperform random or frequency-based data augmentation.

### H4

Algorithm-adaptive refinement policies outperform a single fixed refinement recipe across multiple unlearning backends.

## 9. What is the cleanest novelty claim?

The cleanest and safest novelty claim is:

> Prior LLM unlearning work has explored data augmentation, embedding-space objectives, subspace-aware updates, and failure diagnostics largely in isolation. We propose a **pre-unlearning data refinement framework** that uses embeddings and spectral subspace analysis to proactively reshape forget/retain data before one-shot unlearning, with algorithm-adaptive policies designed to reduce overlap-driven blind spots and preserve adjacent benign knowledge.

This is stronger than any of the following weaker claims:
- "we use embeddings for unlearning",
- "we use SVD for unlearning",
- "we do data augmentation for unlearning".

## 10. Practical recommendation

If you want the most publishable and differentiated version of this idea, the priority order should be:

1. **Pre-unlearning data refinement as the main contribution**,
2. **Embedding + SVD as the mechanism for deciding data operations**,
3. **Algorithm-adaptive support for multiple unlearning backends**,
4. **One-shot robustness under BLUR / relearning / multi-turn as the main evaluation story**.

In other words:

- Do **not** sell this as just an embedding paper.
- Do **not** sell this as just an SVD paper.
- Sell it as a **proactive data-geometry shaping framework for one-shot LLM unlearning**.

That positioning is the best match to the current shortlist and still appears under-explored.
