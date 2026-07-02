# Roadmap — next-level plan (researched July 2026)

Product direction distilled from a multi-source research sweep (competitor landscape,
AI-feature state of the art, scholarly-data ecosystem, recommender-systems literature,
UX/distribution patterns, local-first best practices) plus a full codebase audit.
Key primary sources: Scholar Inbox paper (arXiv:2504.08385, ACL 2025 — includes their
production ranker recipe and an 800k-rating public dataset), PaperQA2 / `paper-qa`
(Apache-2.0), Hugging Face Papers API, OpenAlex & Semantic Scholar API policy pages.

## Ecosystem facts driving urgency (as of July 2026)

- **OpenAlex requires API keys** (since ~Feb 2026, metered with a free daily allowance).
  Keyless `mailto`-only requests degrade/fail — enrichment silently loses citations/topics.
- **Semantic Scholar** issues new keys at ~1 req/sec and prefers batch POST endpoints
  (≤500 IDs/call); the unauthenticated shared pool is throttled.
- **arXiv** has served sustained 429s since Feb 2026 and became an independent nonprofit
  on July 1, 2026 with a cloud migration ahead — expect API/RSS churn; historical
  backfill should eventually shift to the weekly Kaggle metadata dump + GCS PDF bucket.
- **Papers with Code shut down** (July 2025). The app never depended on it (verified),
  but the *replacement* opportunity matters: **HF Papers API** for ongoing code links +
  community buzz; frozen `pwc-archive` dump for history.
- **arXiv HTML** now exists for essentially all new TeX submissions (~75% of corpus
  converts cleanly, backfill ongoing) — better than PDF for section/link/math extraction.
- Watch: arXivLabs has grant funding earmarked for first-party personalized discovery;
  `sqlite-vec` is now a credible FAISS replacement at this corpus scale (would collapse
  backup/restore to one `.db` file).

## Wave 1 — Now (implemented in this pass)

1. **Ecosystem survival hardening** — OpenAlex API-key support, Semantic Scholar key +
   1 RPS pacing, arXiv 429 `Retry-After` handling, plus a Data Sources settings block
   (OpenAlex / S2 / GitHub keys as 0600 dotfiles, same pattern as `.llm_api_key`).
2. **Hugging Face Papers enrichment** — keyless per-arXiv-ID provider: upvotes, comment
   counts, code/project links (fills `github_repo`/resources when PDF mining missed
   them); 🤗 badge on cards; `cv-arxiv-backfill huggingface` subcommand.
3. **Corpus insights UI** — the already-built-and-tested clusters + emerging-topics
   backends surfaced on the Discover page (was API-only despite README billing).
4. **Feed-sources management UI** in Settings — closes the docs-vs-reality gap (help
   pages promised it; only the API existed).

## Wave 2 — Next (highest impact, medium effort)

5. **Inline figure previews** on cards + digest. Scholar Inbox's single most-praised,
   retention-linked feature; for CV papers the figures *are* the paper. The PDF-fetch +
   thumbnail pipeline already exists — extend from page-1 image to first 3–5 figure crops
   (prefer arXiv HTML `<img>` extraction for new papers, PDF fallback for backlog).
6. **Ranker upgrade on the published Scholar Inbox recipe** — per-user logistic
   regression over PCA-compressed (~256-dim) embeddings, weighted BCE with random
   negatives, retrain on every feedback event (milliseconds on CPU), decision-boundary
   active learning, temporal decay, 2–3 clearly-labeled exploration slots. Benchmark
   offline against their public 800k-rating dataset. Pair with the single biggest
   architectural fix: **dense-retrieval candidate generation** — today candidates are
   capped by author/title/affiliation whitelists and the learned model only re-ranks
   within them (`app/services/pipeline/candidate_generation.py`); the FAISS index can
   already retrieve semantically relevant papers corpus-wide.
7. **Digest 2.0** — mobile-first hero-paper layout, inline figures, per-paper scores,
   signed one-tap 👍/👎 links that hit the local API (feedback training without opening
   the app), catch-up digest after absences, user-set weekday schedule + relevance
   threshold. Also wire the orphaned `saved_searches.notify_on_match` flag into digests.
8. **Grounded per-paper "Ask this paper" chat** — agentic RAG loop (iterative retrieval,
   per-chunk evidence scoring, cited synthesis) via `paper-qa` (Apache-2.0, supports
   Ollama/LiteLLM + sentence-transformers — our exact stack). Section/quote-level
   citations are 2026 table stakes (alphaXiv, Bytez, NotebookLM).

## Wave 3 — Later (differentiators)

9. **Multiple interest profiles** (e.g. "3D reconstruction" vs "VLM efficiency") — the
   most-requested feature in the category per Scholar Inbox's own paper — each with its
   own learned ranker, feed tab, and digest section; plus an LLM-synthesized,
   *user-editable* natural-language interest profile blended with the learned model.
10. **arXiv-HTML-first extraction** — heading-aware chunking, literal `<a href>` code
    links instead of PDF regex, MathML; PDF becomes the fallback.
11. **Citation-verification layer** on every LLM output — any mentioned arXiv ID/DOI/title
    must resolve against the local corpus (green check) or be flagged. "Zero fabricated
    citations by construction" — the #1 documented failure of cloud AI research tools.
12. **Implementation-readiness score** — has-code + star velocity + license + repo
    freshness (data already extracted) as a badge/filter/ranking boost.
13. **Local MCP server** — expose `search_papers`, `get_paper`, `top_ranked_today`,
    `list_collections`, etc., so Claude Desktop / other assistants use the personalized
    corpus as a backend. Cheaper than winning the chat-UI arms race.
14. **Distribution** — PyPI publish (`uvx cv-arxiv-scraper serve`), docker-compose
    profile bundling a local model, versioned releases, `/healthz`. In this niche,
    packaging is a growth lever (zotero-arxiv-daily: 5.6k stars largely on packaging).
    Positioning: *the maintained, local-first successor to arxiv-sanity / self-hostable
    Scholar Inbox* — credible now that free tiers are closing and hosted tools keep dying.

## Deliberately not doing

- **Social/commenting features** — alphaXiv's own data shows commenting stalled while
  the AI layer scaled; consume external buzz signals (HF upvotes, star velocity) instead.
- **Bandit/exploration machinery** — RecSys 2025 evidence says greedy matches bandits
  here; a small labeled exploration quota suffices.
- **Benchmark/SOTA tracking as a hard dependency** — post-PwC sources are fragile
  (CodeSOTA is a one-person project); revisit as a best-effort bet later.

## Known technical debt (from the audit, for Wave 2 planning)

- Recall capped by whitelists (see item 6). Ranking is a static hand-tuned linear sum;
  the save/skip labels already collected train nothing. Interest model is a single
  two-centroid cosine (can't represent multi-modal interests) updated lazily.
- No evaluation loop: `RecommendationMetric` table exists but nothing measures nDCG/
  recall against held-out feedback.
- Embeddings are abstract-only SPECTER2 with exact `IndexFlatIP` search and no reranker;
  fine today, revisit with item 6 (A/B GTE-class models against logged feedback).
- LLM prompts are hardcoded CV-centric — generalizing beyond cs.CV needs prompt +
  onboarding work, not just feed sources.
- `_paper_row.html` not yet migrated to the shared `_paper_authors`/`_paper_badges`
  partials (in-flight refactor).
