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
- **arXiv's export API throttles by host** (since Sept 2026): every uncached request
  gets HTTP 406 with an empty body (or 429s that outlive the retries), and waiting
  minutes doesn't clear it. Chosen fallbacks, taken for 30 minutes after a refusal:
  Semantic Scholar's batch endpoint then arXiv OAI-PMH (`GetRecord`, 1 req / 3 s) for ids,
  OAI-PMH `ListRecords` (arXivRaw, by v1 date) for date windows, and Semantic Scholar bulk
  search for topic queries (CS only, no arXiv categories). Deep
  backfill still points at the Kaggle dump above.
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

## Wave 5 — Literature review (September 2026, branch `feat/wave5-lit-review`, v0.7.0)

Method: a 28-agent read-only sweep — codebase mappers, an external scan (Elicit,
Undermind, Asta/OpenScholar, PaperQA2, ResearchRabbit, Litmaps, Inciteful,
ASReview, CoCites, S2/OpenAlex APIs), 4 ideation lenses (35 ideas → 16
shortlisted), then one adversarial skeptic per idea checked against the code and
the real DB.

Core finding: a review couldn't be started in the UI (full collection CRUD in the
backend, no template to create one) and older seed papers couldn't enter the
3-month corpus. With 0 collections, 0 feedback and 0 saved searches in the real
DB, the wave ships small and then measures. The app is the grounded source (scope
→ seed → expand → read → export); synthesis happens in Claude Code over MCP.

22. **Collection manager** ✅ — sidebar "+" (works with zero collections),
    rename/delete, remove-from-collection, bulk "Add to collection"; search
    inside a collection no longer drops members below the global top-100.
23. **Seed a collection** ✅ — paste arXiv ids, URLs or a .bib
    (`POST /api/collections/import-ids`, max 100) or run
    `cv-arxiv-sync --query ... --collection NAME`; both go through the bundle
    importer, write no feedback rows (the ranker is untouched) and embed only
    the papers they create.
24. **Review exports** ✅ — venue-aware BibTeX (`@misc` preprint,
    `@inproceedings`/`@article` once accepted; private notes/tags stay out of
    the .bib) and a per-collection screening CSV.
25. **MCP read tools** ✅ — `get_collection`, `get_paper_text` for Claude Code
    as the review synthesizer.
26. **Trust fixes** ✅ — `?ids=` paper permalinks, a working Follow-author
    link, and a daily refresh of stale S2 citation counts and references.

Deliberate ceilings (marked `ponytail:` in code): synchronous id import
(100 ids; bundles over 100 new papers skip embedding), whole-stale-corpus
citation refresh per run, refreshed counts reach `paper_score` only on the
next full rescore.

### Owner actions (zero code)

- **Register the MCP server** — `pip install '.[mcp]'` in the venv, then add
  `cv-arxiv-mcp` to Claude Code/Desktop with `CV_ARXIV_DATA_DIR` pointing at
  `instance/` (it has never run).
- **Fill in the interest-profile description** — on its own this turns on
  whitelist-free admission; the gate is inert while it is empty.
- **Fix the digest recipient** — set one or disable digests (50 of 50 runs
  failed with "No recipient configured").
- **Optional: a Semantic Scholar API key** — unkeyed calls get 429s, which
  starves the citation refresh and the Tier 2/3 S2 features below.

### Tier 2 (built ahead of the usage gate, at the owner's call)

- **Prior works** ✅ — references cited by 2+ members but missing from the
  library (matched by S2 id and arXiv id), resolved with one S2 `/paper/batch`
  call; a button in "Expand this collection". Needs item 26's refresh to keep
  `referenced_works` filled.
- **Ask this collection** ✅ — corpus chat scoped by `collection_id` with exact
  vector ranking and body-section excerpts; with nothing saved it answers from
  the whole library and labels the scope. Skipped papers never become sources.
- **Paper-chat cleanup** ✅ — skip references/acknowledgments chunks, drop
  quotes that don't appear verbatim, and strip numeric in-text citations that
  would collide with the `[n]` labels.

Deliberate ceilings: prior works makes an uncached live S2 call per click
(add `lru_cache` if it gets clicked a lot), and collection members without an
embedding trail unranked in chat.

### Later: gated on evidence that collections get used

Checkpoint after 2-4 weeks:
`sqlite3 'file:instance/arxiv_papers.db?mode=ro' "select count(*) from paper_collections"`.
If it is still 0, stop — Tier 3 would have no users.

Tier 3 delivered (its trigger fired):

- **Screening decision column** ✅ — trigger: a 238-paper collection after a
  `--query` import. `paper_collections.decision` (include/maybe/exclude, NULL =
  unscreened) with filter chips, `i`/`m`/`e` keys and bulk buttons. Excluded
  papers stay members but leave the review (.bib, graph, prior works, chat,
  Suggest similar); the CSV, bundles and MCP `get_collection` carry the
  decision. It writes no feedback, so the ranker is untouched. Ceiling: chip
  counts cover the whole collection, blind to the view's search/timeframe
  filters.

Tier 3 (each only on its trigger):

- **`near=<collection>` watch filter** ("new similar this week") — when
  collections are being revisited. Its trigger fired; Wave 6 item 30 builds it
  as the MCP tool `whats_new`.
- **Review-matrix LLM cells** — when the in-app LLM is actually on.
- **S2 "outside your library" recommendations** — measured in Wave 6: 0-20%
  on-topic for three collections, so dropped (see "Deliberately not doing").
- **Related-work `outline.md`** — when `get_collection` over MCP proves
  insufficient.

## Wave 6 — Collection watch (October 2026, branch `feat/wave6-collection-watch`)

Method: read-only. Subsystem mappers plus two audits of the real DB, vectors
and logs; ten competitor tracks and ten literature themes, each re-fetched by a
fact-checker; three experiments on the real data; four design drafts; one
adversarial skeptic per surviving item; then a five-reviewer pass over the
whole plan. Evidence: [docs/wave6-research.md](docs/wave6-research.md).

Core finding: the Wave 5 checkpoint fired (ten topic collections, 1,474
papers), but the daily inbox loop is still unused (0 feedback, 0 decisions) and
the feed does not serve the collections. It had caught 28 of the 71 recent
papers that ended up in them and ranked those at chance (6 of 28 in their day's
top 10), and 98% of collection papers are imports without full text. A scorer
built from collection membership alone puts 26 of 28 in the top 10 (holdout
macro AUC 0.93). So collections become the interest model, and the agent gets
the tools to triage what the watch finds.

Baselines (2026-10-01): replay hit@10 6/28; feed recall of new collection
members 28/71; must-read papers with full text 21/111; `cites` edges 1,901
(6,706 available from cached reference lists); CI red on `main`.

27. **Green CI and an entry point that starts** — `style.css` excluded from
    the end-of-file hook, the PDFium test failures fixed at their cause,
    `run.py` moved into the package so `cv-arxiv serve` works from a wheel, and
    a wheel smoke step in CI.
28. **Collections as the interest model** — one mean-centred centroid per
    collection; a paper's score is its z against papers in no collection,
    maximum over collections. Daily admission needs z >= 2, at most 3 per
    collection, and names the collection. While collections exist the free-text
    interest description and the learned ranker do not score; whitelists are
    untouched.
29. **Eval gate** — `scripts/eval_collection_affinity.py` (`holdout`, `replay`,
    `checkpoint`, `--self-test`), read-only on the live data. Item 28 ships only
    at macro AUC >= 0.90 and replay hit@10 >= 24/28.
30. **MCP read surface** — `whats_new` (fresh non-members, attributed to their
    nearest collection), tags and full-text flags in every result,
    `get_collection` filters (`tag`, `decision`, `added_since_days`), an index
    that reloads when a scrape adds vectors, and `cv-arxiv-mcp --read-only`.
31. **MCP triage writes, attended sessions only** — `set_decision` (reason
    required), add-only `tag_papers`, `add_to_collection` no longer creates
    collections, an append-only write log, and an "Agent" chip so an agent's
    decision never passes as the owner's until confirmed.
32. **Full text on demand** — `get_paper_text` fetches and stores sections on
    first read (never under `--read-only`).
33. **Scrape hygiene** — automatic pre-scrape snapshot (7 kept), `empty` run
    status, timestamped logs, and the never-read section embeddings removed.

Items 27-30 ship as one pull request, 31-33 as a second.

### Owner actions (zero code)

- **Snapshot, then `cv-arxiv-backfill citation-edges` without API keys** —
  cached Semantic Scholar reference lists (7-day TTL) are written through; an
  OpenAlex key saved first makes the second pass skip them.
- **API keys** — Semantic Scholar and OpenAlex under Settings → Automation →
  Data Sources; unkeyed pools return 429.
- **`cv-arxiv-backfill sections --batch-size 1 --delay 3`** in an evening —
  full text for imported papers; batch size 1 keeps the write lock short.
- **After deploying item 28** — `cv-arxiv-backfill interest`, then the eval's
  `replay`; raise the "Learned interests" weight only by measurement.
- **Weekly routine** — re-run the collection queries by collection id, then a
  headless agent run against `cv-arxiv-mcp --read-only` with no built-in tools
  writes a brief of candidates per collection.

### Next (each at most a day)

- **Honest dates** — RSS replacements and late cross-lists are stored with the
  announce date (713 of 2,970 scraped rows); clamp to the id month. After the
  weekly brief works, because clamped papers leave the daily inbox.
- **Reference lists** — `backfill citations` drops the references it fetches.
- **Sections backfill** — commit per paper; stop resetting the bulk rate limit.
- **`verify_quotes`** over MCP (found is not the same as supported).
- **Scoped search and `related_papers`** over MCP.
- **Tags as links** on collection rows; confirm before deleting a tag.
- **Weekly-brief MCP prompt** — after real runs have settled the wording.
- **Per-run scrape stats** in `scrape_runs.stats`.
- **Encoder and search hygiene** — no silent fallback encoder; MCP keyword mode
  on FTS5 BM25.

### Checkpoint (about 2026-11-08)

Run the eval's `checkpoint`: of the memberships added since Wave 6 shipped, the
share whose paper the feed had already stored (baseline 28 of 71). If no paper
admitted by the collection watch became a confirmed or query-matched member,
set `candidate_top_k: 0` and keep `whats_new` plus the weekly refresh.

### Backlog (each only on its trigger)

- **Public release** (PyPI publisher, tag, README truth pass) — a month of real
  use, or the first outside user.
- **Interest description as one more scorer row** — the owner wants it kept.
- **Per-collection thresholds or TF-IDF fusion** — broad collections still
  recall poorly at the checkpoint.
- **Centroids from owner-confirmed rows only** — the checkpoint shows drift.
- **Per-collection logistic regression** — 128 or more confirmed decisions in
  one collection.
- **Better PDF text extraction** — must-read papers still without text after
  the sections backfill.
- **Encoder A/B (SPECTER2 proximity adapter) and a LitSearch search eval** —
  holdout AUC below 0.85, or before any encoder or ranking change.
- **Title whitelist matches titles only; candidate ledger** — the owner trims
  the whitelists, or runs exceed about 45 minutes.
- **Stored collection queries with a refresh command** — refresh wanted from
  the UI or over MCP.

## Deliberately not doing

- **Social/commenting features** — alphaXiv's own data shows commenting stalled while
  the AI layer scaled; consume external buzz signals (HF upvotes, star velocity) instead.
- **Bandit/exploration machinery** — there is no feedback to learn from (0 rows). The
  RecSys 2025 paper cited here earlier shows that offline evaluation is biased against
  exploration, not that greedy is as good; a small labeled exploration quota suffices.
- **Re-embedding or swapping the encoder** (Wave 6) — CLS pooling gains +0.005 AUC once
  vectors are mean-centred, which is noise.
- **Training the learned ranker from collection membership** (Wave 6) — the centroid
  scorer matches it without negatives, and naive negatives demote adjacent topics.
- **Semantic Scholar Recommendations as a source** (Wave 6) — 0-20% on-topic for three
  collections.
- **A wide MCP write surface** (Wave 6: notes, deletes, renames, config) — each is an
  injection sink for text an agent has read; three narrow, logged tools cover triage.
- **Prompt-injection detectors for paper text** (Wave 6) — they cannot tell a quoted
  prompt from an attack; the defence is on the write side.
- **Benchmark/SOTA tracking as a hard dependency** — post-PwC sources are fragile
  (CodeSOTA is a one-person project); revisit as a best-effort bet later.
- **Auto-generated surveys** — synthesis belongs to the external agent
  over MCP; the app stays the grounded source.
- **In-app related-work writer** — mis-attribution risk from a small local model;
  Claude Code over `get_collection`/`get_paper_text` does it better.
- **Draft checker** ("consider citing" per sentence) — FTS AND-semantics return
  nothing on real sentences, and a 3-month corpus makes a library audit misleading.
- **Supporting/contrasting citation classifier** — cosine measures topic, not
  stance, and it would dilute the "unverified" citation chip.
- **PRISMA diagrams** — systematic-review reporting for a single-user exploratory
  tool; the collection CSV already holds the screening record.
- **Remote MCP endpoint** — it breaks the localhost, no-auth setup.

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
