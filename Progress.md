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
_No cycles logged yet under the new workflow. First pass will seed the baseline._

**Open gaps.**
- _To be populated on first cycle — migrate gaps from `papers/llm-unlearning-research-gap-table.md`._

---

## LLM-based ASR

**Scope.** LLMs applied to automatic speech recognition: speech-to-text with language-model backbones, audio-embedding prompting, SLAM-ASR-style architectures. Includes how unlearning/forgetting concepts transfer to this setting.

**Keywords.** `LLM ASR`, `speech LLM`, `SLAM-ASR`, `audio embedding prompt`, `speech-to-text large language model`.

**Artifacts.**
- `papers/llm-based-asr-research-map.md` — research map with "question → paper answer" summaries.
- `idea/slam-asr-audio-embedding-corrupted-prompts.md` — plug-and-play corrupted audio-embedding prompts for forgetting.

**Cycles.**
_No cycles logged yet._

**Open gaps.**
- _To be populated on first cycle._

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
