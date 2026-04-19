# AGENTS.md — Paper Retrieval for LLM Unlearning Research

Instructions for AI agents to retrieve papers on **LLM unlearning** (machine unlearning, knowledge deletion, model forgetting) from major academic venues.

---

## Working Principles

**Think critically about each query — do not blindly comply.**

Before executing a user's request, evaluate whether the query itself is reasonable:

- **Challenge premises.** If a query rests on a flawed assumption (wrong venue name, mismatched year, unrelated keyword, a paper that likely does not exist, a method that would not answer the research question), surface the issue before acting.
- **Check scope and feasibility.** If the query is too broad ("get all unlearning papers"), too narrow (overly restrictive filters that would return zero results), or ambiguous (unclear which venue/year/task), ask for clarification or propose a refined version.
- **Flag redundancy and conflicts.** If the requested paper is already in `papers/`, or if the query contradicts an earlier decision/note in this repo, point it out rather than silently duplicating or overwriting.
- **Question research direction when relevant.** For `idea/` or literature-review tasks, if a proposed framing seems weak (unclear novelty, already addressed by cited work, confounded causal claim), raise the concern before producing output.
- **Prefer honest pushback over agreement.** Saying "this query has a problem because X — did you mean Y?" is more valuable than executing a flawed request and producing misleading results.

Only proceed once the query is sound, or once the user has acknowledged the tradeoff after being informed.

---

## Workflow: Search → Analyze → Research Gap

This repo's purpose is **periodic literature review**: search new work, analyze it against what's already catalogued, and surface research gaps worth pursuing. Each cycle should produce a concrete update to `Progress.md`.

### Loop

1. **Search** — Pick one topic from `Progress.md`. Query the sources listed below (arXiv / OpenReview / ACL Anthology) using that topic's keywords. Restrict to the date range since the last entry for that topic in `Progress.md`.
2. **Triage** — For each hit, decide: (a) already in `papers/` or `idea/` → skip; (b) clearly off-topic → skip; (c) relevant → read abstract + intro, extract the claim, method, and what it leaves unsolved.
3. **Record** — Add relevant papers to the appropriate file in `papers/` (new file per sub-theme, or append to an existing shortlist/gap table). Keep one-paragraph summaries, not full notes.
4. **Identify gaps** — After triage, explicitly answer: *what question is still unanswered across the papers just read?* Write this as a bullet under the topic's "Open gaps" section in `Progress.md`. If a gap turns into a concrete proposal, create a new file in `idea/`.
5. **Update `Progress.md`** — Log the date, queries used, papers added, and new/closed gaps under the relevant topic.

### File conventions

- `papers/` — factual notes about external work (shortlists, gap tables, research maps). One file per sub-theme or synthesis artifact.
- `idea/` — our own research proposals. One file per idea. Each should cite the `papers/` notes that motivate it.
- `Progress.md` — living log of periodic search cycles, organized by topic. Source of truth for *what's been covered and when*.
- File naming: kebab-case, descriptive (`llm-unlearning-research-gap-table.md`, not `notes1.md`).

### When to create vs. extend

- **New topic** → new section in `Progress.md` + (eventually) new files in `papers/`.
- **New angle within an existing topic** → extend the existing `papers/` file; don't fragment.
- **Idea crystallized enough to design an experiment** → new file in `idea/`.

---

## Search Terms for LLM Unlearning

Use these keywords when querying across sources:

- `machine unlearning` / `machine learning unlearning`
- `LLM unlearning` / `large language model unlearning`
- `knowledge deletion` / `knowledge removal`
- `model forgetting` / `selective forgetting`
- `catastrophic forgetting` (in unlearning context)
- `data deletion` / `right to be forgotten`
- `parameter editing` / `model editing` (related)

---

## 1. arXiv

**API:** `https://export.arxiv.org/api/query`

**Python (recommended):**
```bash
pip install arxiv
```

```python
import arxiv

client = arxiv.Client()
search = arxiv.Search(
    query="all:unlearning OR all:\"machine unlearning\" OR all:\"knowledge deletion\"",
    max_results=100,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending
)
for result in client.results(search):
    print(result.title, result.entry_id, result.pdf_url)
```

**Direct HTTP:**
```
GET https://export.arxiv.org/api/query?search_query=all:unlearning+OR+all:"machine+unlearning"&max_results=50&sortBy=submittedDate&sortOrder=descending
```

**Categories:** `cs.LG`, `cs.CL`, `cs.AI` for ML/NLP unlearning.

---

## 2. OpenReview

Covers **ICLR**, **ICML**, **NeurIPS**, **CoLM**, and other ML venues.

**Python client:**
```bash
pip install openreview-py
```

```python
import openreview

client = openreview.api.OpenReviewClient(
    baseurl='https://api2.openreview.net',
    username='YOUR_EMAIL',  # or use token
    password='YOUR_PASSWORD'
)

# Venue IDs (examples)
venues = {
    'ICLR': 'ICLR.cc/2025/Conference',
    'ICML': 'ICML.cc/2025/Conference',
    'NeurIPS': 'NeurIPS.cc/2025/Conference',
    'CoLM': 'COLM/2025/Conference'
}

# Get submissions for a venue
submissions = client.get_all_notes(
    invitation=f'{venue_id}/-/Blind_Submission',
    content={'venueid': venue_id}
)

# Search by keyword in title/abstract (post-filter)
for note in submissions:
    if 'unlearning' in (note.content.get('title', '') + note.content.get('abstract', '')).lower():
        print(note.content['title'], note.id)
```

**Guest access:** Use `openreview.api.OpenReviewClient()` without credentials for public data.

---

## 3. ICLR (International Conference on Learning Representations)

- **OpenReview:** `ICLR.cc/YYYY/Conference` (primary source)
- **Website:** https://iclr.cc
- **Years:** 2013–present

Use OpenReview API with venue ID `ICLR.cc/YYYY/Conference`.

---

## 4. NeurIPS (Neural Information Processing Systems)

- **Proceedings:** https://papers.nips.cc
- **OpenReview:** `NeurIPS.cc/YYYY/Conference` (recent years)
- **Website:** https://neurips.cc

**Options:**
- OpenReview API for recent proceedings
- Semantic Scholar API (venue filter: NeurIPS)
- Scraping: `papers.nips.cc` by year

---

## 5. ICML (International Conference on Machine Learning)

- **OpenReview:** `ICML.cc/YYYY/Conference`
- **Proceedings:** https://proceedings.mlr.press
- **Website:** https://icml.cc

Use OpenReview for submissions; PMLR for published proceedings.

---

## 6. ACL (Association for Computational Linguistics)

**ACL Anthology:**
```bash
pip install acl-anthology
```

```python
from acl_anthology.anthology import Anthology

anth = Anthology()
# Search papers by venue/event
for paper_id, paper in anth.papers.items():
    if 'ACL' in paper.get('venue', '') and 'unlearning' in str(paper).lower():
        print(paper.get('title'), paper_id)
```

**Direct:** https://aclanthology.org — browse by venue (ACL, EMNLP, etc.) or use the Anthology API.

---

## 7. EMNLP (Empirical Methods in Natural Language Processing)

- **ACL Anthology:** Same as ACL; filter by venue `EMNLP`
- **Website:** https://2024.emnlp.org

Use `acl-anthology` with venue filter for EMNLP proceedings.

---

## 8. CoLM (Conference on Language Modeling)

- **OpenReview:** `COLM/YYYY/Conference`
- **Website:** https://colmweb.org

CoLM uses OpenReview; use the OpenReview API with the CoLM venue ID.

---

## Unified Retrieval Strategy

1. **arXiv** — Preprints and early work.
2. **OpenReview** — ICLR, ICML, NeurIPS, CoLM submissions and reviews.
3. **ACL Anthology** — ACL, EMNLP, and other NLP venues.
4. **Semantic Scholar** — Cross-venue search: `https://api.semanticscholar.org/graph/v1/paper/search?query=LLM+unlearning` (API key recommended).

---

## Output Conventions

When saving retrieved papers:

- **Directory:** `papers/` or `papers/{venue}_{year}/`
- **Metadata:** JSON with `title`, `authors`, `abstract`, `url`, `pdf_url`, `venue`, `year`
- **PDFs:** Optional; store in `papers/{venue}_{year}/pdfs/`

---

## Quick Reference: Venue → Source

| Venue   | Primary Source   | API / Method          |
|---------|------------------|------------------------|
| arXiv   | arxiv.org        | arXiv API / `arxiv`    |
| ICLR    | OpenReview       | openreview-py          |
| NeurIPS | OpenReview / PMLR| openreview-py          |
| ICML    | OpenReview / PMLR| openreview-py          |
| ACL     | ACL Anthology    | acl-anthology          |
| EMNLP   | ACL Anthology    | acl-anthology          |
| CoLM    | OpenReview       | openreview-py          |
