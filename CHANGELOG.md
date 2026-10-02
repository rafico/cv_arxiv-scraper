# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project aims to
follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

Wave 6: collections become the interest model (see `ROADMAP.md` and
`docs/wave6-research.md`).

## [0.7.0] — 2026-09-28

Wave 5: collections as a literature-review workspace.

### Added
- **Collection manager**: a sidebar "+" (also with zero collections),
  rename/delete, remove-from-collection on cards, and a bulk "Add to
  collection" picker.
- **Seed a collection** from pasted arXiv ids, URLs or a .bib
  (`POST /api/collections/import-ids`, up to 100 ids; old-scheme ids such as
  `hep-th/9901001` included). Hidden papers are reported, not silently linked
  out of sight. No feedback rows are written, so the ranker is untouched.
- **`cv-arxiv-sync --query ... --collection NAME [--max-results 1000]`**
  imports an arXiv search into a collection without touching sync state.
- **Collection CSV** (`GET /api/collections/<id>/table.csv`): a screening
  spreadsheet with venue, code, citations, readiness, tags and notes.
- **MCP tools** `get_collection` and `get_paper_text` (requires `mcp>=1.3`).
- **Paper permalinks** via `/?ids=1,2`, used by the graph, suggestions and
  citation chips.
- **Stale citation refresh**: the daily scrape re-fetches Semantic Scholar
  counts and references older than 7 days.
- **Prior works** (`GET /api/collections/<id>/prior-works`): outside papers
  that 2+ members cite, resolved in one Semantic Scholar batch call, with
  one-click Add. It degrades to a 502 with `{results: [], error}`, never a 500.
  Add keeps the paper's Semantic Scholar id (`import-ids` takes an optional
  `s2_ids` map), so it joins the citation graph in the same request.
- **Ask this collection**: the collection sidebar's "Ask this collection →"
  opens Discover chat with `?collection=`, which sends an optional
  `collection_id` on `POST /api/corpus/chat` and answers from that
  collection's visible members only. Answers are labelled with their scope.
- **Screening decisions**: mark collection members Include, Maybe or Exclude
  (card buttons, `i` / `m` / `e` on the focused card, or the bulk bar) and
  filter with All / Unscreened / Include / Maybe / Exclude chips that show live
  counts. The API is `PUT /api/collections/<id>/papers/<pid>/decision` and
  bulk `PUT /api/collections/<id>/decisions` (`null` clears), stored in a
  nullable `paper_collections.decision` column added on startup. Excluded
  papers stay members but leave the review: the default view, Export .bib,
  the graph, Prior works, Ask this collection, Suggest similar seeds and
  collection counts skip them, and Suggest similar never offers them again
  (its new "Not relevant" button files a suggestion as excluded). The CSV
  keeps every member plus a `decision` column, bundles carry the decision
  (import only fills a blank one), and MCP `get_collection` returns each
  member's `decision` (`include_excluded` adds the excluded papers). Decisions
  write no feedback, so the ranker is untouched.

### Changed
- BibTeX entries are `@misc` arXiv preprints (eprint, primaryclass, arXiv
  DOI), or `@inproceedings`/`@article` once an accepted venue is detected;
  cite keys are unchanged. Private notes and tags are never written to the .bib.
- Imported papers are embedded right away (only the new ones, through the
  locked index path); bundles with more than 100 new papers leave it to
  `cv-arxiv-backfill embeddings`.
- Digests (top papers, exploration slots and saved-search alerts) skip
  imported papers published long before the digest window, so a seed or topic
  import can't take over the email.
- The `mcp` extra now needs `mcp>=1.3`, so the server can send its grounding
  instructions (quote verbatim, cite arXiv ids, verify unknown ids).
- **Breaking for API clients:** `POST /api/corpus/chat` no longer returns
  `no_saved_papers` or `message`. It adds `scope` (`collection`, `saved` or
  `corpus`) and `sources[].n` and `sources[].section`. With nothing saved it
  answers from the whole library (`scope: "corpus"`) instead of refusing.
  Skipped papers are never used as sources.
- Chat excerpts come from the best-matching body section rather than the
  abstract. Numeric in-text citations such as `[3]` are removed, so they
  can't be mistaken for source numbers.
- Paper chat skips reference and acknowledgment chunks and drops quotes that
  don't appear verbatim in the excerpt.
- The citation, OpenAlex, comments, Hugging Face, GitHub and insights
  backfills rescore the whole library once when they change anything, rather
  than rescoring each updated paper. Rescoring a paper on its own used the
  current date for recency, which pushed it below papers that were not
  rescored.

### Fixed
- **arXiv refusing the export API.** Since September 2026 arXiv answers a
  throttled host with HTTP 406 on every request (or 429s that outlive the
  retries), and waiting doesn't clear it. The daily scrape lost the arXiv
  metadata (affiliations, comments, DOI, categories) of every new paper and
  its rolling window came back empty; seed import returned 502 and profile
  bootstrap resolved none of the pasted ids; historical sync and
  `cv-arxiv-sync --query` failed. A refusal is now remembered for 30 minutes
  and every path falls back: date windows list arXiv OAI-PMH (arXivRaw, kept by
  v1 date, so revised papers keep their submission date and window), id lookups
  ask Semantic Scholar's batch endpoint and then OAI-PMH within a 30 s budget at
  arXiv's 1 request per 3 s (ids not reached are reported as `deferred`, "try
  again in a minute"), and `--query` uses Semantic Scholar search (Computer
  Science only; a non-cs category or a `cat:`-only query is refused, not
  guessed; `ANDNOT` stays a negation). Papers the OAI listing already
  described aren't looked up again one by one. A historical sync reaching too
  far back for OAI-PMH fails up front instead of leaving a silent gap.
- `/api/search` (every mode) and MCP `search_papers` in hybrid and semantic
  mode no longer return skipped papers, matching the dashboard and exports;
  MCP over-fetches so skipped top hits don't leave the page empty.
- Bundle import keeps `semantic_scholar_id` only when it is a real Semantic
  Scholar paperId, so a bundle can't take over another paper's citation edges.
- Search inside a collection no longer drops members outside the global
  top-100 hits.
- Follow author links to that author's papers instead of the plain inbox.
- The sidebar "+" works on every page (the CSRF token is now injected
  shell-wide).
- Profile bootstrap (and the new seed import) no longer stores arXiv's
  "Error" entry as a paper when an id is malformed.
- The MCP setup example points `CV_ARXIV_DATA_DIR` at a real data dir
  (a source checkout's `instance/`).
- Newly scraped papers, and papers rescored by a backfill, now include the
  implementation-readiness bonus in their score. Backfills dropped it, and the
  GitHub backfill didn't rescore at all.
- The OpenAlex and citation-edge backfills merge OpenAlex references into
  `referenced_works` instead of overwriting the stored Semantic Scholar ones
  and the citation edges they produce.
- `cv-arxiv-mcp` works with mcp 2.x, which `pip install '.[mcp]'` now
  resolves. It exited with the "extra not installed" hint because `FastMCP`
  was renamed to `MCPServer`; 1.x still works.
- The onboarding and Settings text no longer says a profile description adds
  papers beyond your whitelists while learned ranking is switched off.
- A dashboard `?page=` past the last page (such as the reload after skipping
  or screening a last page empty) shows the last page instead of an empty
  state.

## [0.6.0] — 2026-08-16

Research-library features adopted from studying
[Graphbib](https://github.com/Lior-Falach/Graphbib), adapted to this app's
global paper table and offline-first constraints.

### Added
- **Citation graph** (`/graph`): your library as a force-directed network of
  real citation edges, with node color by year, size by in-corpus PageRank
  (NumPy power iteration — no new deps), collection/year filters, and
  double-click-to-open. Edges come from OpenAlex `referenced_works` **and**
  Semantic Scholar `references` — S2 covers fresh preprints months before
  OpenAlex parses them, which is most of a daily-scraper corpus. Synced
  automatically after each scrape; `cv-arxiv-backfill citation-edges`
  backfills older papers. Vendored vis-network 9.1.9 (first vendored JS —
  the app stays CDN-free).
- **Collection bundles**: export one collection as a plain-JSON file (papers,
  notes, tags, reference ids — no PDFs, they re-fetch) and import it
  elsewhere as a new collection. Imports link papers you already track
  instead of duplicating, and never overwrite local notes/tags.
- **Per-collection BibTeX** (`Export .bib` in the collection sidebar, or
  `GET /api/export/bibtex?collection=<id>`).

### Removed
- The unused `GET /api/papers/<id>/graph` TF-IDF similarity stub (no UI ever
  called it); `GET /api/graph` supersedes it with real citation edges.

### Not adopted
- Graphbib's in-app PDF reader + annotations were built, then cut before
  release: plain reading is the browser's native PDF viewer (the existing PDF
  button), and annotation is better served by the Mendeley/Zotero sync this
  app already has — a second annotation store would only fragment notes.

## [0.5.0] — 2026-08-12

Wave 4: prove the ranker, simplify the foundation, widen the audience, deepen
the research workflows.

### Added
- **Ranking metrics you can trust**: the learned-ranker holdout eval now also
  reports nDCG@10, recall@20 and MRR (per profile, shown in Settings), and
  `scripts/benchmark_scholar_inbox.py` benchmarks the exact production recipe
  on the public Scholar Inbox 800k-rating dataset (`--self-test` runs without
  the dataset).
- **Digest exploration slots** (`digest.exploration_slots`, default 2): clearly
  labeled papers from outside your usual lane; one-tap ratings teach the model
  where your interests end. Saved-search alerts take precedence.
- **Weekly field synthesis** (`digest.synthesis_weekday`, off by default): a
  "this week in your field" brief atop the digest — LLM-narrated with citation
  verification when configured, emerging-topic labels and counts otherwise.
- **Collection expansion**: a "Suggest similar" widget in every collection view
  wires the previously UI-less `/api/corpus/neighbors` endpoint to one-click
  adds.
- **Follow → digest alerts**: following an author now also creates a
  `notify_on_match` saved search, and the save-search prompt asks about digest
  alerts (the flag existed but no UI set it).
- **PyPI publishing**: tag-triggered trusted-publishing workflow
  (`.github/workflows/publish.yml`).
- `scraper.extract_figures` opt-out flag; figure extraction memoizes conclusive
  no-figure papers (`thumbnails/{id}_nofig`) instead of refetching them forever.

### Changed
- **faiss-cpu removed.** The vector index is a plain NumPy matrix
  (`papers.npy` / `sections.npy`) — search was always exact, so behavior is
  unchanged while the heaviest wheel and the dual-libgomp SIGSEGV mitigations
  disappear. Legacy `*.index` files migrate automatically on first load (the
  old file is left for rollback); without faiss installed the app runs
  read-empty and `cv-arxiv-backfill --rebuild-index` rebuilds.
- **Dense-retrieval admission is now a true per-run top-K** by interest score;
  previously the first K above threshold in stream order won and LLM enrichment
  ran before the cap.
- **The built-in scheduler replaces crontab management** (`app/services/cron.py`
  deleted — it hardcoded a `~/venv` interpreter). New `scheduler.send_digest`
  option emails the digest after each scheduled scrape; the Settings automation
  card edits `scheduler.*` in config.yaml and shows the next run.
- **Prompts and defaults generalized beyond cs.CV**: neutral analyst prompts,
  and historical search derives default categories from your configured feeds.

### Removed
- API-dead recommendation metrics (`compute_precision_at_k`,
  `measure_recommendation_quality`, …) that scored against all feedback
  (train==test) and had no callers; the holdout eval supersedes them.

### Also in 0.5.0 — activation pass (2026-07-30)

Several Wave 1–3 features shipped but stayed inert on a real install. Nothing
here adds capability — it makes what already exists run.

### Changed
- **Full-text section extraction is on by default.** `scraper.extract_sections`
  previously defaulted to `false` and was documented in no config file, so
  per-paper chat, corpus chat, and citation verification had no `PaperSection`
  rows to read. It is now on and documented in `config.example.yaml`; set it to
  `false` to opt out (it costs roughly one extra HTTP fetch per new paper).
- **The onboarding checklist tracks the real activation threshold.** The "Save or
  skip papers" step completed after a single save, while the centroid interest
  profile, the learned ranker, and whitelist-free dense-retrieval admission all
  stay inert below `MIN_POSITIVE_FEEDBACK` (5). It now shows progress (`n/5`) and
  completes at the threshold that actually switches ranking on.

### Added
- **Feature-liveness diagnostics.** `/healthz` gained a `features` block (section
  coverage, positive-feedback progress, off-whitelist admissions, last digest
  status, enrichment coverage) and Settings → Automation gained a matching
  "Feature Status" card. These are diagnostics only: the 200/503 status the
  Docker healthcheck gates on is unchanged, since an empty corpus is a fresh
  install rather than a failure.
- **A "Set a digest recipient" onboarding step** when `email.recipient` is empty —
  the digest carries the one-tap 👍/👎 links that train the ranker.

### Fixed
- **A misconfigured digest no longer fails invisibly.** A missing
  `email.recipient` raised before any `DigestRun` row was created, so nightly
  cron failures left no trace outside `cron.log` and the Settings digest panel
  showed "No digest runs yet". The misconfiguration is now recorded as an errored
  run before the error propagates.

## [0.4.0] — 2026-07-03

Wave 3 completion: HTML-first full-text extraction and a local MCP server.

### Added
- **arXiv-HTML-first section extraction.** Full-text now prefers
  `arxiv.org/html/{id}` (cleaner structure, literal `<a href>` code/project
  links, MathML), mapping headings to the same canonical section types the PDF
  extractor uses, and falling back to pdfplumber on any miss. The scrape
  pipeline records the source (`html`/`pdf`/`none`); a backfill subcommand
  re-extracts the corpus. Chat/RAG and the section index are unchanged.
- **Local MCP server.** `cv-arxiv-mcp` exposes the personalized corpus to
  Claude Desktop and other assistants via tools: `search_papers`, `get_paper`,
  `get_summary`, `top_ranked_today`, `list_collections`, `add_to_collection`,
  `ask_paper`. Ships as an optional extra (`pip install '.[mcp]'`) with the SDK
  imported lazily, so the core install stays dependency-light.

## [0.3.0] — 2026-07-03

Wave 3: distribution & operational-trust packaging, plus multi-profile ranking.

### Added
- **One-command install & run.** New `cv-arxiv-scraper` / `cv-arxiv` console
  entry points with a `serve` subcommand: `uvx cv-arxiv-scraper serve` (or, after
  `pip install .`, `cv-arxiv serve`) launches the server against a single data
  directory. The data dir (DB, FAISS index, config, secrets) resolves from
  `--data-dir`, else `$CV_ARXIV_DATA_DIR`, else `~/.local/share/cv-arxiv`, and a
  config is seeded on first run. Loopback-only stays the default; `--expose` is
  opt-in.
- **`/healthz` endpoint.** Unauthenticated, cheap liveness/readiness JSON
  (`status`, `version`, `db_ok`, `faiss_ok`, `paper_count`); 200 when healthy,
  503 when degraded. The Docker healthcheck now probes it.
- **Versioning.** A real, single-source version (`app._version.__version__`, read
  statically by the packaging metadata and surfaced by `/healthz` and
  `cv-arxiv --version`).
- **docker-compose `local-ai` profile.** `docker compose --profile local-ai up`
  brings up an Ollama sidecar (loopback-bound, GPU-optional) plus an AI-enabled
  app instance wired to it, so summaries/chat work out of the box — without
  changing the default no-AI `docker compose up` path.
- **Packaging polish.** Filled-in project metadata (description, classifiers,
  keywords, URLs), `MANIFEST.in`, and package-data that ships templates, static
  assets, and the compiled `style.css` so a plain install renders correctly.
- **Multiple interest profiles** (Wave 3 ranking) — each with its own learned
  ranker, feed tab, and digest section (see `/api/profiles`).

## [0.2.0] — 2026-07-02

Wave 2: richer feed and AI features.

### Added
- **Inline figure previews** on cards and in the digest (arXiv-HTML-first with a
  PDF fallback).
- **Learned ranker** — a per-user logistic regression over PCA-compressed
  SPECTER2 embeddings, retrained on every save/skip, with dense-retrieval
  candidate generation admitting whitelist-free "Interest" papers.
- **Grounded per-paper "Ask this paper" chat** — iterative retrieval with
  cited, evidence-scored synthesis.
- **Digest 2.0** — mobile-first hero layout, inline figures, per-paper scores,
  and signed one-tap 👍/👎 links that train the ranker without opening the app;
  catch-up digests, user-set weekday schedule, and a relevance threshold.

## [0.1.0] — 2026-07-01

Wave 1: ecosystem-survival hardening and UI gap-closing.

### Added
- **API-key support** for OpenAlex (required since Feb 2026) and Semantic Scholar
  with 1 RPS pacing; arXiv 429 `Retry-After` handling; a Data Sources settings
  block storing keys as `0600` dotfiles.
- **Hugging Face Papers enrichment** — keyless upvotes, comment counts, and
  code/project links, plus a `cv-arxiv-backfill huggingface` subcommand.
- **Corpus insights UI** (clusters + emerging topics) surfaced on Discover.
- **Feed-sources management UI** in Settings.

### Base
- Daily/on-demand arXiv scraping with interest matching, hybrid keyword +
  semantic (SPECTER2 + FAISS) search, multi-factor ranking with per-paper
  explanations, extractive TL;DR (no API required), enrichment (citations,
  topics, GitHub), collections/tags/notes, BibTeX/Zotero/Mendeley export, and a
  Gmail digest — all localhost, single-user, no auth.

[0.3.0]: https://github.com/rafico/cv_arxiv-scraper/releases/tag/v0.3.0
[0.2.0]: https://github.com/rafico/cv_arxiv-scraper/releases/tag/v0.2.0
[0.1.0]: https://github.com/rafico/cv_arxiv-scraper/releases/tag/v0.1.0
