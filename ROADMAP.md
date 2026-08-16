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

## Wave 2 — Delivered (July 2026)

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

### Wave 2 leftovers (deferred, small)

- Benchmark the learned ranker offline against the Scholar Inbox public 800k-rating
  dataset (currently evaluated only on the user's own feedback via `RecommendationMetric`).
- Labeled exploration slots (2–3 per feed/digest) from the Scholar Inbox recipe.
- Dense-retrieval admission is first-K-above-threshold in scrape stream order; a true
  corpus-wide top-K would need a batch pass in `scrape_engine`.
- `backfill thumbnails --figures` re-attempts papers with zero extractable figures on
  every run (no negative-result sentinel); harmless but wasteful.
- Optional `scraper.extract_figures` opt-out flag (figure fetch is always-on best-effort
  with a 25-papers/run cap and 180s deadline).

## Wave 3 — Delivered (July 2026), except items 10 & 13

9. **Multiple interest profiles** ✅ — named profiles, each with its own learned ranker
   (artifacts keyed by slug), feed switcher, and per-profile digest section, plus an
   editable natural-language description blended into scoring. New `interest_profiles`
   table + nullable `paper_feedback.profile_id` (NULL ⇒ Default profile); a zero-data-loss
   additive migration verified by `tests/test_interest_profiles.py`.
11. **Citation-verification layer** ✅ — `app/services/citation_verifier.py` resolves every
    arXiv ID/DOI/title in chat, corpus-chat, and summary output against the local corpus
    (verified chip → links to the paper; else amber "unverified"). Always-on.
12. **Implementation-readiness score** ✅ — `app/services/implementation_readiness.py`:
    has-code + star velocity + license + repo freshness → ⚙ Runnable badge, dashboard
    "Runnable (has code)" filter, and a small additive ranking bonus (honest in explain).
14. **Distribution** ✅ — `cv-arxiv serve` (single DATA_DIR), top-level `/healthz`,
    single-source version (`app/_version.py`), docker-compose `local-ai` profile
    (Ollama sidecar), CHANGELOG, README quickstart, packaging polish. PyPI publish is the
    one remaining manual release step (README documents the `uvx --from git+…` form until then).

10. **arXiv-HTML-first extraction** ✅ — `app/services/html_extraction.py` fetches
    `arxiv.org/html/{id}`, maps headings to the same canonical section vocabulary the PDF
    extractor uses (so the section index + chat/RAG are unchanged), extracts literal
    `<a href>` code/project links, and falls back to pdfplumber on any miss. Wired into the
    scrape pipeline's section step (`source` = html/pdf/none) + a backfill subcommand.
13. **Local MCP server** ✅ — `app/services/mcp_tools.py` (testable logic: `search_papers`,
    `get_paper`, `get_summary`, `top_ranked_today`, `list_collections`, `add_to_collection`,
    `ask_paper`) + a thin `app/mcp_server.py` wrapper that imports the MCP SDK lazily.
    Shipped as an optional extra (`pip install .[mcp]`) + a `cv-arxiv-mcp` console script,
    so the core install stays dependency-light and tests run without the SDK.

**The full researched roadmap (Waves 1–3) is now implemented.**

Positioning: *the maintained, local-first successor to arxiv-sanity / self-hostable
Scholar Inbox* — credible now that free tiers are closing and hosted tools keep dying.

## Wave 4 — Delivered (August 2026)

Four workstreams (re-researched Aug 2026: sqlite-vec still pre-v1 alpha; OpenAlex
metered with new semantic-search/full-text endpoints; SPECTER2 still competitive):

15. **Prove the ranker** ✅ — the holdout eval now reports nDCG@10 / recall@20 /
    MRR alongside AUC/F1 (persisted per profile, shown in Settings);
    `scripts/benchmark_scholar_inbox.py` benchmarks the production recipe on the
    public Scholar Inbox 800k-rating dataset (`--self-test` needs no data);
    dense-retrieval admission is a true per-run top-K by interest score (was
    first-K-in-stream-order); digests carry 2 labeled exploration slots
    (`digest.exploration_slots`). Closes Wave-2 leftovers 1–3.
16. **Simplify the foundation** ✅ — `faiss-cpu` replaced by an exact NumPy
    matrix (`papers.npy`/`sections.npy`; auto-migration from legacy `*.index`,
    which stays on disk for rollback) — search was always exact `IndexFlatIP`,
    so this deletes a heavy wheel and the dual-libgomp mitigations. sqlite-vec
    deliberately skipped (still alpha). `cron.py` deleted; the built-in
    scheduler gained `send_digest` and a real Settings card.
17. **Widen the audience** ✅ — neutral LLM prompts (no CV hardcoding),
    historical-search categories derived from configured feeds, tag-triggered
    PyPI trusted publishing (`publish.yml`; register the trusted publisher on
    PyPI, then tag `v0.5.0`). Version: 0.5.0.
18. **Research workflows** ✅ — "Suggest similar" collection expansion (wires
    the previously UI-less neighbors API); follow-author now also creates a
    digest alert search and the save-search prompt exposes `notify_on_match`;
    weekly "this week in your field" synthesis brief in the digest
    (`digest.synthesis_weekday`, LLM-narrated with citation verification,
    degrades to labels+counts).

Deliberately skipped in Wave 4 (with reasons): reading queue (the Saved view is
one), conference planner (no poster-metadata source), OpenAlex full-text RAG
(metered + new parsing surface), embedding-model A/B (SPECTER2 holds; the
benchmark script is the harness when wanted), bandits (standing decision).

## Graphbib adoption (2026-08-15/16, branch `feat/graphbib-adoption`, v0.6.0)

Features ported from [Graphbib](https://github.com/Lior-Falach/Graphbib) after a
gap analysis (three capabilities were genuinely absent; the rest we had):

19. **Citation edges + graph page** ✅ — reference ids persist on each paper
    (`Paper.referenced_works`), resolve locally into `PaperRelation` "cites"
    rows (a table that existed unused since Wave 1 — zero schema DDL), and
    render at `/graph` via vendored vis-network with NumPy PageRank sizing.
    Edge sources: OpenAlex `referenced_works` + Semantic Scholar `references`.
    The S2 source is load-bearing, not optional: on the real 1460-paper corpus
    (all 2026 preprints) OpenAlex had parsed references for **zero** papers,
    while S2 covered 574 immediately (32 in-corpus edges on first sync; grows
    as the corpus ages).
20. **Collection bundles** ✅ — plain-JSON export/import of one collection
    (no PDFs, no zip surface); fill-only merge on import, edges rebuild from
    shipped reference ids.
21. **Per-collection BibTeX** ✅ — collection filter on the existing export.

Graphbib's **PDF reader + annotations** was built, then cut before release
(2026-08-16): without annotations the reader adds nothing over the browser's
native PDF viewer behind the existing PDF button, and annotation itself is
better served by the Mendeley/Zotero sync — a second, disconnected annotation
store fragments notes. If in-app annotation ever returns, it should sync
*into* the reference manager, not beside it.

Deliberate ceiling (marked `ponytail:` in code): full-corpus edge recompute
per scrape.

### Wave 4 leftovers (small)

- Figure-extraction negative sentinel exists (`{id}_nofig`); delete the file to
  force a re-attempt after arXiv backfills an HTML rendition.
- ~~Run the Scholar Inbox benchmark on the real dataset and record numbers
  here.~~ Done 2026-08-16. Setup: the public release
  (github.com/avg-dev/scholar_inbox_datasets, `rated_papers.csv`, 774k
  ratings) ships only `arxiv_id`s, so titles/abstracts were joined from the
  arXiv export API; evaluated the first 200 lexicographically-sorted users
  with ≥20 ratings (137 had evaluable two-class holdouts), seed 0.
  **Results** (mean / median): AUC **0.758 / 0.778**, nDCG@10
  **0.862 / 0.931**, recall@20 **0.937 / 1.000**, MRR **0.869 / 1.000**.
  Reading: the production recipe ranks well (a relevant paper reaches the
  top-10 for the typical user) with headroom on raw AUC vs. the paper's
  reported ~0.89 — theirs trains on each user's full history; ours is a
  cold 80/20 split per user.
- One-time manual step: register the PyPI trusted publisher, then tag v0.5.0.

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
