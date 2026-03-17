# Idea: Non-Forgettable Safety-Critical Knowledge in LLMs

This note turns the broad direction

> some knowledge in large language models must not be forgotten, because forgetting it can lead to severe downstream harm

into a set of more concrete research ideas, using insights from **causal mediation analysis (CMA)** and related mechanistic interpretability work.

The most important framing is:

- We usually study **what should be forgotten**.
- A complementary question is **what must remain causally active** even after fine-tuning, alignment, unlearning, compression, or domain adaptation.
- The risk is not only performance drop. The risk is **catastrophic safety regression**:
  - failure to recognize a dangerous request,
  - failure to maintain refusal behavior,
  - failure to preserve uncertainty or deference in high-risk settings,
  - failure to remember basic hazard concepts needed to avoid harmful answers,
  - failure to preserve guardrail-like behavior under paraphrase or multi-turn pressure.

## 1. Why CMA is a good fit

The CMA literature suggests several useful facts:

1. Important behaviors are often mediated by a **small subset of components** rather than the whole model uniformly.
2. Fine-tuning and debiasing can change **which layers, heads, or modules** causally matter.
3. Mediators can be used not only for explanation, but also for:
   - targeted fine-tuning,
   - targeted mitigation,
   - activation steering,
   - and before/after model comparison.
4. A model can appear behaviorally improved while its **internal causal pathway** has become more brittle or shortcut-based.

This makes CMA a strong tool for studying **safety-critical retention**:

> not just "does the model still answer safely?" but "which internal mechanisms are still carrying the safety-relevant behavior?"

## 2. Core research thesis

A strong thesis statement is:

> Some safety-critical knowledge should be treated as non-forgettable causal infrastructure. Fine-tuning, unlearning, and adaptation should be evaluated not only by output behavior, but also by whether they preserve the mediating circuits that support hazard recognition, safe refusal, calibrated uncertainty, and protection of adjacent benign knowledge.

This has two strong implications:

1. We need to identify **which knowledge is safety-critical**.
2. We need to measure whether that knowledge is preserved at the level of **causal mediators**, not only final answers.

## 3. What counts as "non-forgettable safety-critical knowledge"?

A useful taxonomy:

### A. Hazard recognition knowledge

The model should continue to recognize that a request touches a dangerous domain or unsafe intent.

Examples:

- identifying that a prompt is asking for harmful misuse,
- recognizing that a medical or legal request is high-stakes,
- recognizing uncertainty when evidence is insufficient,
- recognizing that the request should be refused or redirected.

### B. Protective meta-knowledge

The model should remember not just facts, but the policy-like knowledge that some actions are unsafe.

Examples:

- knowing that certain classes of assistance should not be provided,
- knowing when to defer to experts,
- knowing when low confidence should change response style,
- knowing that partial compliance can still be unsafe.

### C. Boundary knowledge around unlearning

When harmful factual/procedural content is removed, the model should still retain the **adjacent protective knowledge** needed to:

- detect that the topic is dangerous,
- explain why it cannot help,
- give safe alternatives,
- preserve nearby benign knowledge without collapsing into ignorance.

This is especially important for unlearning:

> forgetting dangerous procedural content should not also erase the model's ability to recognize danger.

## 4. Concrete research ideas

### Idea 1: SafeCore -- Identify the non-forgettable safety core

### Main question

Which internal components causally mediate safety-critical behavior, and how stable are they across fine-tuning?

### Hypothesis

Safety-critical behaviors such as hazard recognition, refusal, and calibrated uncertainty rely on a partially reusable but fragile subset of mediators. Some of these mediators drift or disappear after task tuning even when average benchmark scores remain acceptable.

### Method sketch

1. Start from a base model and several post-training variants:
   - base checkpoint,
   - instruction-tuned checkpoint,
   - domain-tuned checkpoint,
   - safety-tuned checkpoint,
   - optionally compressed or unlearned checkpoint.
2. Build a prompt suite for:
   - unsafe request recognition,
   - refusal robustness,
   - uncertainty calibration,
   - safe redirection,
   - benign neighboring tasks.
3. Define mediators:
   - attention heads,
   - MLP blocks,
   - residual directions,
   - or sparse latent features.
4. Use CMA or activation-patching-style mediation scores to measure:
   - total effect,
   - natural indirect effect,
   - direct effect,
   - layer-wise mediator concentration.
5. Compare checkpoints by:
   - mediator overlap,
   - rank shifts,
   - drift of top-k mediators,
   - safety-performance vs. mediator-stability correlation.

### Expected contribution

- A map of the model's "safety core"
- Evidence that some safety behaviors are supported by narrow causal pathways
- A benchmark for evaluating whether tuning preserves or damages that core

### Why this is interesting

It reframes safety alignment from:

> "did outputs look safe?"

to:

> "did the model preserve the internal mechanism that makes safe behavior robust?"

### Idea 2: Mediation-Preserving Fine-Tuning (MPFT)

### Main question

Can we improve a model on new tasks while explicitly preserving the mediators of safety-critical knowledge?

### Hypothesis

Many harmful regressions happen because fine-tuning optimizes task loss while unintentionally overwriting safety-mediating components. Preserving the indirect effect of those mediators should reduce catastrophic safety forgetting.

### Method sketch

1. Use Idea 1 to identify safety-critical mediators on a base or safety-tuned model.
2. During fine-tuning, add a regularizer that penalizes:
   - large change in safety mediator activations,
   - large change in mediator rankings,
   - or large drop in NIE on a small retained safety prompt set.
3. Compare:
   - standard SFT,
   - LoRA / adapter tuning,
   - mediation-preserving variants.

### Candidate regularizers

- activation anchoring on safety prompts,
- subspace preservation for safety-mediating directions,
- top-k mediator consistency penalty,
- NIE preservation penalty on a held-out calibration set.

### Expected contribution

- A training recipe that protects safety-critical knowledge during adaptation
- A new way to formalize "do not forget this" beyond standard replay
- Mechanistic evidence that preservation of mediators improves robustness

### Strong claim if it works

> Preserving the causal mediators of safety behavior is more effective than preserving only outputs or logits.

### Idea 3: Boundary-Aware Unlearning

### Main question

How do we unlearn dangerous or undesirable content without erasing the protective knowledge that tells the model the content is dangerous?

### Hypothesis

A major failure mode of unlearning is **boundary collapse**:

- the model forgets the target content,
- but also forgets adjacent hazard-recognition or refusal-support knowledge,
- leading to blind spots, weaker refusals, or incoherent safe alternatives.

### Method sketch

1. Define two nearby regions:
   - **forget region**: content that should be removed,
   - **protect region**: safety-critical knowledge that must remain.
2. Use CMA to measure mediator overlap between the two regions.
3. Categorize mediators into:
   - forget-specific,
   - shared boundary mediators,
   - protect-specific.
4. Design unlearning updates that:
   - target forget-specific mediators,
   - protect protect-specific mediators,
   - carefully constrain updates on shared boundary mediators.

### Evaluation

- forgetting success on target content,
- preservation of hazard recognition,
- preservation of refusal quality,
- preservation of safe redirection,
- robustness under paraphrase and multi-turn pressure.

### Why this is especially promising

This direction directly connects your concern to unlearning:

> some knowledge should indeed be forgotten, but the protective knowledge around it must remain causally intact.

That is a sharp and defensible research question.

### Idea 4: Mediator Drift as an Early Warning Signal

### Main question

Can internal mediator drift predict catastrophic forgetting before behavioral failure becomes obvious?

### Hypothesis

Safety regressions may begin as hidden mediator drift before they appear clearly in benchmark outputs. A model may still pass a shallow test while its safety-relevant causal pathway has already become fragile.

### Method sketch

1. Track checkpoints during fine-tuning or continual learning.
2. Measure at each checkpoint:
   - safety accuracy,
   - refusal robustness,
   - uncertainty behavior,
   - mediator overlap with the reference safety core,
   - layer-wise NIE concentration shifts.
3. Test whether mediator drift predicts later failure under:
   - harder prompts,
   - paraphrases,
   - adversarial rewording,
   - multi-turn jailbreak pressure.

### Expected contribution

- An early-warning metric for dangerous post-training regressions
- A more mechanistic safety monitoring tool than output-only validation

### Practical value

This could be very useful for model release pipelines:

> detect hidden safety degradation before deployment.

### Idea 5: Cross-Model Safety Core Transfer

### Main question

Are safety-critical mediators universal across models, or model-family-specific?

### Hypothesis

Some safety-critical mediator patterns are reusable across architectures or scales, while others are highly model-specific. Knowing which is which would help transfer safety interventions more efficiently.

### Method sketch

Compare the same task family across:

- multiple sizes within one family,
- multiple model families,
- base and post-tuned checkpoints.

Measure:

- top-k mediator overlap,
- cross-model causal-effect rank correlation,
- whether mediators found in one model transfer as useful intervention targets in another.

### Why this matters

If safety cores partly transfer, we can build:

- cheaper diagnostics,
- faster safety audits,
- more targeted adaptation recipes across models.

## 5. A concrete first paper idea

If you want one version that is especially focused and realistic, the cleanest first paper is:

### Paper candidate: Boundary-Aware Unlearning of Hazardous Knowledge

### One-sentence pitch

Unlearning should remove hazardous know-how without erasing the causal mechanisms that support hazard recognition, refusal, and safe redirection.

### Why this one is strong

- It is directly motivated by a real safety concern.
- It fits naturally with your existing unlearning-related notes.
- It uses CMA in a way that is both mechanistic and practically actionable.
- It is narrower and easier to defend than a very broad "all non-forgettable knowledge" project.

### Concrete setup

#### Model states

- base model
- safety-tuned model
- unlearned model

#### Data split

- forget set: hazardous target content to remove
- protect set: hazard recognition and refusal-support prompts
- retain set: nearby benign knowledge

#### CMA targets

- identify mediators for forget behavior,
- identify mediators for protect behavior,
- measure overlap and conflict.

#### Main method

- constrain unlearning to preserve protect-side mediators or protect-side NIE,
- compare against standard unlearning baselines.

#### Main story

Standard unlearning can create hidden safety holes because it treats all nearby knowledge as expendable. A boundary-aware, mediation-guided method can forget the target while preserving the safety-critical causal infrastructure around it.

## 6. A second strong paper idea

### Paper candidate: Mediation-Preserving Fine-Tuning for Safety-Critical Retention

### One-sentence pitch

Task adaptation should preserve the causal mediators of hazard recognition and calibrated refusal, not just their surface outputs.

### Main comparison

- standard fine-tuning
- replay-only retention
- logit-distillation retention
- mediation-preserving fine-tuning

### Main claim

Mechanism-preserving adaptation reduces catastrophic safety forgetting better than output-preserving baselines.

## 7. Suggested experiments and metrics

### A. Behavioral metrics

- hazard recognition accuracy
- refusal robustness
- safe alternative quality
- uncertainty calibration in high-risk settings
- benign utility retention

### B. Mechanistic metrics

- TE / NIE / NDE changes
- top-k mediator overlap
- mediator rank stability
- layer-wise concentration shift
- mediator drift over training

### C. Stress tests

- paraphrase
- adversarial wording
- multi-turn escalation
- neighboring benign prompts
- partial-information prompts that require uncertainty

## 8. Key novelty claim

The clean novelty claim is not:

- "we use causal mediation analysis",
- or "we study safety",
- or "we preserve knowledge".

The stronger novelty claim is:

> Prior work studies what to forget, how to fine-tune, or how to localize mechanisms largely in isolation. We study safety-critical knowledge that must remain causally active, and we use causal mediation analysis to identify, monitor, and preserve the mediators of that knowledge across fine-tuning and unlearning.

## 9. Bottom-line recommendation

If you want the most promising version of this direction, prioritize the following order:

1. **Boundary-aware unlearning of hazardous content while preserving protective knowledge**
2. **Mediation-preserving fine-tuning for safety-critical retention**
3. **Mediator drift as an early-warning signal for hidden safety regression**
4. **Cross-model transfer of the safety core**

In short:

- Do not frame the project as generic catastrophic forgetting.
- Do not frame it as generic safety alignment either.
- Frame it as:

> identifying and preserving the causal mediators of non-forgettable safety-critical knowledge in LLMs.
