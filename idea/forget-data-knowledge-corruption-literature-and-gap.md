# Forget Data, Knowledge Corruption, and Pre-Unlearning Estimation

This note connects **Cuong et al. (ACL Findings 2024)**—training-data features that predict adversarial robustness *before* expensive attack-based evaluation—to a parallel question in **LLM machine unlearning**: how properties of the **forget set** relate to **collateral / irrelevant knowledge corruption** on the **retain** distribution, and whether that corruption can be **estimated before** the unlearning update is applied.

---

## 1. Anchor paper: data-centric prediction without the expensive inner loop

**Cuong, Le, and Le (2024),** [*A Curious Case of Searching for the Correlation between Training Data and Adversarial Robustness of Transformer Textual Models*](https://aclanthology.org/2024.findings-acl.800/) (Findings of ACL 2024).

**Stated motivation (paraphrased from the abstract):** Fine-tuned textual transformers are strong but adversarially fragile. Standard robustness evaluation runs **after** fine-tuning and largely **ignores the training corpus**. The authors argue there is a **strong correlation between fine-tuning data properties and robustness**, and support this by extracting **13 corpus-level features**, then using a **lightweight predictor** (e.g., Random Forest) to estimate **attack success rate**—reporting large **runtime savings** versus full adversarial evaluation, plus transfer across several model families.

**Structural analogy for your idea:**

| Cuong et al. (robustness) | Your framing (unlearning) |
|---------------------------|---------------------------|
| Training / fine-tuning corpus | Forget set (and its relation to retain / pretraining) |
| Adversarial robustness (attack success) | Knowledge corruption on **non-forget** behavior (retain accuracy, related facts, reasoning, safety behaviors) |
| Expensive loop: generate attacks, evaluate | Expensive loop: run unlearning, then full retain / blind-spot benchmarks |

The analogy is **methodological** (predict downstream damage from **data statistics** and cheap surrogates) more than a claim that robustness and unlearning are the same phenomenon.

---

## 2. Your research questions (precise formulation)

1. **Correlation / mechanism:** *How do forget data (distribution, difficulty, overlap with retain, format, rarity, etc.) correlate with the degree and pattern of **irrelevant knowledge corruption** after unlearning?*
2. **Pre-unlearning estimation:** *Can we estimate expected corruption on retain (or on a probe suite) **before** applying the unlearning procedure—without a full unlearning run and without exhaustive post-hoc evaluation?*

**Terminology:** “Knowledge corruption” should be **operationalized** (see §5). Typical choices in the unlearning literature are **retain-set QA accuracy**, **general benchmarks**, **probes for related concepts**, and newer **blind-spot / multi-turn** checks; your “irrelevant” emphasis aligns with **utility preservation** and **side effects** beyond the forget target.

---

## 3. Literature review (selected clusters)

### 3.1 Benchmarks explicitly separate forget vs. retain

- **TOFU** ([arXiv:2401.06121](https://arxiv.org/abs/2401.06121), [project page](https://locuslab.github.io/tofu/)): synthetic author profiles with controlled forget fractions; evaluates **forgetting** and **retention** together. This gives a **standard yardstick** for “corruption on what should stay.”
- Follow-ons and variants (e.g., reasoning-oriented extensions) stress that evaluation must cover **more than a single accuracy number**—aligned with your “irrelevant corruption” theme.

**Takeaway:** The field knows **what to measure after unlearning**; it is weaker on **forecasting** those metrics from **forget-set design** alone.

### 3.2 “Closer look” and survey-style evidence on trade-offs and side effects

- Work such as **[A Closer Look at Machine Unlearning for Large Language Models](https://arxiv.org/abs/2410.08109)** synthesizes limitations of current LLM unlearning: methods can be **insufficient** or **destructive**, metrics can be **misleading**, and behavior under **relearning / probing** matters.

**Takeaway:** **Collateral damage** is recognized; your novelty is shifting attention to **predictive, forget-data-centric** models of that damage **before** optimization.

### 3.3 Gradient-based unlearning and catastrophic side effects

- **Gradient ascent** and related objectives on the forget set are standard but often **destabilize** the model; recent work asks whether GA is always necessary and proposes alternatives (e.g., memorization / extrapolation lines such as [“Is Gradient Ascent Really Necessary? Memorize to Forget…”](https://arxiv.org/abs/2602.06441)).

**Takeaway:** The **optimization path** matters, but your question can still be posed at the **data level**: which forget sets make **any** reasonable unlearning update prone to damaging retain?

### 3.4 Influence- and importance-aware unlearning

- **Influence functions** and influence-guided unlearning (e.g., IMU-style ideas: [Influence-guided Machine Unlearning](https://arxiv.org/html/2508.01620v1)) allocate unlearning intensity by **data importance**. That is close in spirit to “not all forget examples are equal,” but the typical goal is **effectiveness of forgetting**, not necessarily a **pre-run regression** from raw corpus features to **retain corruption**.

**Takeaway:** You can **build on** influence scores as **features** for your predictor, or as **theory** linking forget examples to parameter movement that harms retain.

### 3.5 Data-centric and pre-unlearning agendas (overlap with this repo)

- The internal note `papers/llm-unlearning-research-gap-table.md` already argues for **acting before unlearning**, **overlap-aware** forget/retain handling, and **algorithm-adaptive** data preparation.

**Takeaway:** Your idea **fits** that agenda. The **specific** Cuong-style question—**dataset-level features → pre-unlearning estimate of retain corruption**—is not yet fully spelled out there as a standalone research program.

### 3.6 Adjacent area: model editing and “corruption” under adaptation

- Work on **knowledge editing** studies **instability**, **consistency**, and **degradation** under further training (e.g., [“Can Fine-Tuning Erase Your Edits?”](https://arxiv.org/abs/2511.05852)). This is **not identical** to unlearning but shares the theme: **localized update → global side effects**.

**Takeaway:** Methods and metrics from editing **may transfer** if you define corruption broadly (e.g., drift on unrelated QA).

---

## 4. Is this a research gap?

**Verdict: Yes, with the right scope statement—partial overlap exists, but the core predictive question is under-explored.**

**Already crowded or partially covered:**

- **Post-hoc** evaluation of retain / utility / blind spots is **mature** and improving.
- **Influence-based** methods already use data importance **during** unlearning.
- **Pre-unlearning data refinement** is an active framing in this repository’s literature map.

**Comparatively open (defensible novelty):**

1. **Explicit predictive modeling:** Treat **forget-set + retain-set (or proxy) statistics** as inputs to a model that forecasts **retain corruption metrics** *before* any unlearning step—mirroring Cuong et al.’s **13 features → robustness** pipeline.
2. **Mechanistic / geometric hypotheses:** Link forget-retain **overlap**, **subspace alignment**, **difficulty**, and **frequency** to **which unrelated skills break first** (not only average retain loss).
3. **Algorithm-conditional predictors:** Same forget data may imply different corruption under **GA vs. preference optimization vs. representation projection**; learning **conditional predictors** is closer to an engineering product than a single-number heuristic.

**Risk to manage:** Reviewers may say “this is just better evaluation” unless you show **(a)** real forecasting accuracy, **(b)** actionable decisions (e.g., data curation, subset selection, budget allocation), or **(c)** theory tying features to update directions.

---

## 5. Operationalizing “knowledge corruption”

To make the problem falsifiable, fix one or more **corruption vectors** \(C\), for example:

- **Retain accuracy** on TOFU-style retain authors or domain benchmarks.
- **Probing suites** for **neighbor concepts** (entities co-mentioned with forget subjects, same relation type, etc.).
- **Behavioral probes:** refusals, formatting, reasoning steps, tool use—if your “irrelevant” concern includes **emergent capability** regression.
- **Stability under relearning** or **jailbreak pressure** (optional second stage).

Your estimator targets \(\mathbb{E}[C \mid \mathcal{D}_\text{forget}, \mathcal{D}_\text{retain}, \text{algo}]\) or a **distribution** over \(C\) if stochastic training is used.

---

## 6. Solution directions (from cheap to heavy)

### 6.1 Cuong-style **corpus feature regression**

- Engineer **forget-set features**: length, lexical diversity, entity density, **embedding dispersion**, **nearest-neighbor overlap** with retain in embedding space, **perplexity under the frozen base model**, **gradient norm** of forget minibatches (one forward-backward pass **without** applying the update), etc.
- Train a **small regressor** to predict measured \(C\) **after** cheap or partial runs on a **meta-dataset** of synthetic forget partitions.

**Pros:** Fast, interpretable feature ablations. **Cons:** May need many **pilot** unlearning runs to build training data for the regressor unless you rely on **surrogates** (§6.2–6.3).

### 6.2 **One-step linearization / influence surrogates**

- Approximate the effect of a proposed unlearning update on retain loss using **influence functions**, **TracIn**, or a **single Newton-style step** on a retain minibatch.
- Use **only** gradients/Hessian-vector products on **small** subsets to estimate directional harm **before** full training.

**Pros:** Connects to classical “data attribution.” **Cons:** Scalability and accuracy for **LLMs** are limiting; approximations must be validated empirically.

### 6.3 **Tiny “shadow” models**

- On a **smaller** model or **low-rank adapter**, run **short** unlearning and fit a **transfer function** to the large model’s corruption.

**Pros:** Cheaper than full LLM unlearning. **Cons:** Transfer assumptions need evidence.

### 6.4 **Bayesian / ensemble forecasting**

- Train multiple unlearning runs with **hyperparameter jitter**; predict **quantiles** of corruption for **risk-aware** data curation (when to expand forget, split concepts, add retain anchors).

---

## 7. Suggested experiments (minimal viable paper)

1. **Correlation study:** On TOFU (or similar), sweep **forget subsets** that differ in **overlap**, **size**, and **difficulty**; measure **retain corruption vector** \(C\); report **which data factors explain variance** in \(C\).
2. **Pre-unlearning estimator:** Hold out **forget-set partitions**; train **feature → \(C\)** predictors; report **\(R^2\) / ranking** metrics vs. baselines (e.g., “always predict mean,” “use overlap only”).
3. **Conditional on algorithm:** Repeat for **2–3** unlearning families; show whether **one** universal predictor suffices or **algorithm tags** are required.
4. **Actionability:** Show that **rejecting** or **rewriting** high-risk forget examples (flagged by the estimator) **reduces** corruption at **iso-forget** performance.

---

## 8. References (starting set)

1. Dang Cuong, Dung Le, and Thai Le. 2024. *A Curious Case of Searching for the Correlation between Training Data and Adversarial Robustness of Transformer Textual Models.* Findings of ACL. [ACL Anthology](https://aclanthology.org/2024.findings-acl.800/), [DOI 10.18653/v1/2024.findings-acl.800](https://doi.org/10.18653/v1/2024.findings-acl.800). Code: [CaptainCuong/RobustText_ACL2024](https://github.com/CaptainCuong/RobustText_ACL2024).
2. Pratyush Maini et al. 2024. *TOFU: A Task of Fictitious Unlearning for LLMs.* [arXiv:2401.06121](https://arxiv.org/abs/2401.06121).
3. Unlearning survey / closer-look style analyses (e.g., [arXiv:2410.08109](https://arxiv.org/abs/2410.08109)) for side effects and evaluation pitfalls.
4. Influence-guided unlearning (e.g., IMU: [arXiv:2508.01620](https://arxiv.org/abs/2508.01620)) for data-importance mechanisms.
5. Gradient-ascent alternatives (e.g., [arXiv:2602.06441](https://arxiv.org/abs/2602.06441)) for optimization context.

---

## 9. One-sentence positioning statement (for proposals)

> We study whether **forget-set geometry and statistics** predict **retain-side knowledge corruption** under LLM unlearning, and build **pre-unlearning estimators**—by analogy to **data-driven robustness prediction**—so practitioners can **curate forget data and choose algorithms** before paying the full cost of unlearning and evaluation.
