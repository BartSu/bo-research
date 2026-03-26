# Idea: Plug-and-Play Audio-Embedding Gating for Unlearning in LLM-Based ASR (SLAM-ASR Line)

**Date:** 2026-03-26  
**Scope:** Literature scan for *SLAM-ASR–style* systems (frozen speech encoder + frozen LLM + trainable speech-to-LLM connector) and whether anyone has proposed **plug-and-play corruption / gating of audio embeddings** driven by a small **classifier**, to achieve **machine unlearning** while **preserving base model utility**.

---

## 1. What “plug-and-play” means in this line of work

**SLAM-ASR** (Ma et al., 2024) argues that a strong LLM-based ASR can be built with minimal moving parts: an off-the-shelf speech encoder, an off-the-shelf LLM, and **only a linear projector** trained to map speech representations into the LLM’s input space—encoder and LLM stay frozen.

- Paper: [An Embarrassingly Simple Approach for LLM with Strong ASR Capacity](https://arxiv.org/abs/2402.08846) (arXiv:2402.08846)

Follow-up work studies **robustness and failure modes** of that recipe (cross-domain shift, rate/noise perturbations):

- [Performance evaluation of SLAM-ASR: The Good, the Bad, the Ugly, and the Way Forward](https://arxiv.org/abs/2411.03866) (arXiv:2411.03866; SALMA @ ICASSP 2025)

These papers establish **why** a tiny connector is attractive: it is the natural “interface module” you can swap or retrain without touching huge backbones.

---

## 2. Has anyone done “unlearning” for LLM-based ASR?

**Yes, but not with your proposed mechanism.**

The closest direct hit is **Unlearning LLM-Based Speech Recognition Models** (Liu, Interspeech 2025), which:

- measures unintended memorization in LLM-based ASR, and  
- proposes **efficient unlearning** under data-deletion / privacy-style motivation, reporting privacy–utility trade-offs on LibriSpeech.

- Archive page: [ISCA Archive – Unlearning LLM-Based Speech Recognition Models](https://www.isca-archive.org/interspeech_2025/liu25b_interspeech.html)  
- DOI: [10.21437/Interspeech.2025-287](https://doi.org/10.21437/Interspeech.2025-287)

**Important distinction:** this line of work is **not** (in the public abstract / listing) framed as “add a classifier that corrupts audio embeddings to block forgotten content while leaving the frozen LLM untouched.” Typical unlearning recipes in speech and NLP more often involve **gradient-based updates** on subsets of parameters (adapters, LoRA, etc.), not a dedicated **embedding-space gate** trained only to scramble *forget-set-conditioned* inputs.

The repository’s longer map already notes: **LLM-based ASR unlearning exists**, while **“SLAM-ASR linear-projector–only unlearning”** is still a **clean gap** in the public literature. See section 10 in:

- `papers/llm-based-asr-research-map.md`

---

## 3. Has anyone done “classifier corrupts / reroutes audio embeddings” for forgetting?

**Not found as a published, named recipe for ASR unlearning** in this scan.

What *does* exist nearby:

| Direction | What it gives you | Why it is not your idea |
|---|---|---|
| **SLAM-ASR** | Proves a **tiny trainable connector** can carry almost all task adaptation | About **utility**, not **forgetting** |
| **Liu (Interspeech 2025)** | ASR **unlearning** for memorization / deletion requests | Methodology is **not** “classifier-gated corrupted speech prompts” |
| **Learnable prompt projection** | Adds a **second small projector** so *text prompt embeddings* land in better regions of LLM space; stresses **not modifying** the underlying ASR+LLM | About **prompt robustness**, not unlearning or forget triggers |

Prompt-projector reference (SLAM-ASR family, small add-on module):

- [Reducing Prompt Sensitivity in LLM-based Speech Recognition Through Learnable Projection](https://arxiv.org/abs/2601.20898) (arXiv:2601.20898)

**Bottom line:** the combination

> **frozen encoder + frozen LLM + (optional frozen projector) + new plug-in module that uses a classifier to conditionally corrupt / mix speech-side embeddings for forget requests, aiming for minimal utility loss**

does **not** appear as a standard published baseline as of this note. That makes it a **plausible novel axis**: *unlearning at the speech–LLM interface via conditional representation editing*, rather than *unlearning by PEFT on the LLM or encoder adapters*.

---

## 4. Your idea (recorded spec)

**Working name:** *Audio-embedding classifier gate for SLAM-ASR unlearning* (name TBD).

**High-level mechanism**

1. **Base system:** SLAM-ASR-style stack—speech encoder → speech-to-LLM projector → frozen LLM decoder. Optionally keep the pretrained projector **fixed** after initial ASR training.
2. **Plug-in:** A small **binary or multi-head classifier** operates on **audio-side features** (encoder outputs, pooled segments, or projector inputs/outputs) to detect whether the current utterance is in a **forget set** (or matches a learned “should suppress” pattern).
3. **Conditional corruption:** When the classifier fires, apply a **controlled transformation** to the sequence of speech tokens / embeddings fed to the LLM, for example:
   - additive noise in projector space with learned scale,
   - orthogonal / null-space projection relative to a “retain” subspace,
   - learned mixing with a **neutral** or **random** direction,
   - or gating that **replaces** speech embeddings with a learned “uninformative” prompt embedding.
4. **Training objective:**  
   - **Forget:** maximize loss / minimize likelihood of the original transcription (or targeted spans) on forget audio, *only through* the plug-in parameters (classifier + corruption module), with constraints so the LLM and base projector stay unchanged.  
   - **Retain:** standard ASR loss on retain data with **identity** or **near-identity** pass-through when the classifier predicts “retain.”
5. **Claim you want to test:** **Single small plug-in** can approximate unlearning **without** LoRA on the LLM or adapter fine-tunes on the encoder, preserving **general ASR utility** better than full PEFT unlearning.

**Why this could be interesting**

- **Modularity:** deployment can ship one frozen backbone + swappable “forget policy” head.
- **Audit surface:** the forget behavior is concentrated in a **small module** (easier to log, replace, or roll back than scattered LoRA weights).
- **Alignment with SLAM-ASR philosophy:** continues the story that the **cross-modal bottleneck** is where intervention should live.

**Risks / falsification paths (what reviewers will ask)**

- **Shallow forgetting:** LLM may still decode memorized text if corruption is weak or if memory lives primarily in the LLM; strong corruption may **hurt retain WER** because the bottleneck is shared.
- **Classifier robustness:** spoofing, adversarial audio, or distribution shift may **misfire** the gate.
- **Definition of unlearning:** need clear metrics beyond WER (e.g., membership inference, canary reconstruction, probing with paraphrased audio).

---

## 5. Suggested experiments (minimal viable paper)

1. **Baselines:** Liu-style unlearning (or your reimplementation) with **LoRA / adapter** vs **projector-only** vs **proposed plug-in only** (all with matched compute budget for the trainable part).
2. **Metrics:** retain-set WER, forget-set “success” (target string not emitted / high loss), and **privacy-style** probes where applicable.
3. **Ablations:** classifier input (encoder vs projector space), corruption family (noise vs subspace vs replacement embedding), threshold calibration.
4. **Stress tests:** cross-domain audio (per SLAM-ASR robustness findings), noisy speech, and **partial forget** (single speaker vs single phrase).

---

## 6. One-sentence literature verdict

> **LLM-based ASR unlearning is emerging (e.g., Interspeech 2025), and SLAM-ASR shows a tiny connector is enough for strong ASR—but a plug-and-play “classifier-conditioned corruption of audio embeddings for forgetting while freezing the LLM” does not yet read as a standard published method; it is a structurally motivated research gap.**

---

## 7. Related reading (same repo)

- `papers/llm-based-asr-research-map.md` — broader LLM-based ASR map + **§10** on projector vs unlearning gap  
- `AGENTS.md` — keywords for extending the search (machine unlearning, knowledge deletion, etc.)

When extending this note, prefer adding **venue, year, arXiv ID, and one-line “how it differs”** per new paper rather than long abstracts.
