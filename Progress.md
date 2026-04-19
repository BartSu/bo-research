# Progress

Living log of periodic literature-review cycles, organized by topic. See `AGENTS.md` for the workflow.

Each topic section contains:
- **Scope** — what the topic covers and what's out of scope.
- **Keywords** — search terms used across arXiv / OpenReview / ACL Anthology.
- **Artifacts** — files in `papers/` and `idea/` under this topic.
- **Cycles** — reverse-chronological log of each search pass (date, date range queried, papers added, gaps found/closed).
- **Open gaps** — unanswered questions currently on the radar.

---

## LLM Unlearning

**Scope.** Machine unlearning for large language models: knowledge deletion, selective forgetting, right-to-be-forgotten, relearning resistance, evaluation of unlearning. Includes pre-unlearning data refinement and geometry-aware methods. Excludes generic continual learning and catastrophic forgetting unless framed as unlearning.

**Keywords.** `LLM unlearning`, `machine unlearning`, `knowledge deletion`, `selective forgetting`, `right to be forgotten`, `relearning attack`, `unlearning evaluation`.

**Artifacts.**
- `papers/llm-unlearning-research-gap-table.md` — gap table + pre-unlearning data refinement framework.
- `papers/pre-unlearning-data-augmentation-shortlist.md` — geometry-aware + data-augmentation shortlist.
- `papers/unlearning-arxiv.md` — raw arXiv hits pending triage.
- `idea/non-forgettable-safety-knowledge.md` — safety-critical knowledge that should resist unlearning.
- `idea/forget-data-knowledge-corruption-correlation.md` — pre-unlearning corruption prediction framework.
- `idea/unlearning-emergent-capabilities.md` — unlearning impact on emergent capabilities.

**Cycles.**

### 2026-04-19 — cross-reference: LLM-ASR intersection pass

Ran under the LLM-based ASR topic (see that section for details). No new standalone LLM-unlearning papers logged here; relevant transferable methods (GUARD, GRUN, Pre-Forgettable Models, BLUR, unlearning-verification survey, Unlearning-Isn't-Invisible) are referenced in `papers/llm-asr-unlearning-landscape.md` as potential cross-modal building blocks.

**Open gaps.**
- _To be populated on first standalone LLM-unlearning cycle — migrate gaps from `papers/llm-unlearning-research-gap-table.md`._
- Generation-time unlearning with classifier gating (GUARD-style) is getting traction in text; worth tracking for cross-modal transfer.

---

## LLM-based ASR

**Scope.** LLMs applied to automatic speech recognition: speech-to-text with language-model backbones, audio-embedding prompting, SLAM-ASR-style architectures. Includes how unlearning/forgetting concepts transfer to this setting.

**Keywords.** `LLM ASR`, `speech LLM`, `SLAM-ASR`, `audio embedding prompt`, `speech-to-text large language model`, + unlearning-intersection terms: `speech unlearning`, `ASR unlearning`, `speaker unlearning`, `audio prompt corruption`.

**Artifacts.**
- `papers/llm-based-asr-research-map.md` — research map with "question → paper answer" summaries.
- `papers/llm-asr-unlearning-landscape.md` — 2025-H2 / 2026-Q2 intersection landscape with gap analysis.
- `idea/slam-asr-audio-embedding-corrupted-prompts.md` — plug-and-play corrupted audio-embedding prompts for forgetting.

**Cycles.**

### 2026-04-19 — LLM-ASR × machine unlearning intersection

- **Window queried:** 2025-H2 – 2026-Q2 (to supplement the idea file written 2026-03-26).
- **Sources:** arXiv, ISCA Archive (Interspeech 2025), ICML 2025, Google Scholar.
- **Papers added to `papers/llm-asr-unlearning-landscape.md`:** 14 papers across 4 buckets (direct intersection, speech-side unlearning, TTS speaker forgetting, LLM-transferable methods) + 3 adjacent audio-security refs.
- **Key finding:** Precise LLM-ASR × unlearning intersection still has only **one** prior paper (Liu Interspeech 2025); speech-side unlearning accelerated in 2025-H2 but stays on SLU/SER/classifier/TTS — nobody has put ECO-style classifier-gated embedding corruption on a SLAM-ASR projector. The user's idea remains an open gap.
- **Closest new competitor:** **TruS** (arXiv 2601.20481, 2026-01) — training-free TTS speaker unlearning via hidden-activation steering. Shares the "inference-time, no retraining" philosophy; differs in modality (TTS decoder vs. ASR projector), mechanism (activation steering vs. classifier-gated corruption), and lacks a gating classifier. Must be the primary positional contrast in the paper.
- **Reusable benchmark:** **UnSLU-BENCH** (arXiv 2505.15700) for speaker-level RTBF evaluation — covers SLU not ASR, but protocol transfers.

**Open gaps.**

1. **Audio projector intervention.** No work does classifier-gated corruption on the SLAM-ASR projector output. Natural minimal-intrusion entry point (only trainable component in SLAM-ASR).
2. **Phrase / entity-level speech unlearning.** All existing speech-unlearning work is speaker-level or sample-level. "ASR forgets a specific name / phone / address across speakers" has no benchmark.
3. **Training-free + classifier-gated combo.** TruS is training-free but no gate; GUARD is gated but text-only; ECO is both but text-only. Audio-prompt classifier + audio-embedding corruption is the unclaimed coordinate.
4. **ASR-specific unlearning evaluation suite.** Existing metrics (KWS/SID accuracy, intent accuracy, WER alone) don't cover forget-phrase WER, homophone leakage, prefix-leakage, retain-set degradation jointly. Potential side contribution.
5. **Memorization / extraction attacks on speech LLMs.** Systematic membership-inference or training-data extraction on Whisper / SALMONN / Qwen-Audio / SLAM-ASR is essentially unpublished. Useful threat-model chapter or standalone paper.

---

## Causal Mediation Analysis (methodology)

**Scope.** Causal mediation / interpretability methods that can attribute fine-tuning or unlearning effects to specific components. Cross-model mediation, activation patching, circuit-level attribution. Used as a methodological toolkit for the topics above, not a standalone research area.

**Keywords.** `causal mediation analysis`, `activation patching`, `circuit analysis`, `cross-model mediation`, `fine-tuning attribution`.

**Artifacts.**
- `papers/causal-mediation-analysis.md` — cross-model CMA for fine-tuning.

**Cycles.**
_No cycles logged yet._

**Open gaps.**
- _To be populated on first cycle._

---

## Adding a new topic

When a new research direction earns its own track:

1. Add a section above following the same structure (Scope / Keywords / Artifacts / Cycles / Open gaps).
2. Link any seeding papers or ideas under **Artifacts**.
3. Run the first cycle and log it.
