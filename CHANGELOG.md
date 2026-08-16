# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project aims to
follow [Semantic Versioning](https://semver.org/).

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
- **In-app PDF reader** (`Read` on any paper): vendored pdf.js 4.10.38,
  PDFs downloaded on first open and cached in `instance/pdfs/`. Drag to
  **highlight**, click to pin **comments**; annotations are stored as
  page-fraction coordinates so they survive any window size.
- **Collection bundles**: export one collection as a plain-JSON file (papers,
  notes, tags, annotations, reference ids — no PDFs, they re-fetch) and
  import it elsewhere as a new collection. Imports link papers you already
  track instead of duplicating, and never overwrite local notes/tags.
- **Per-collection BibTeX** (`Export .bib` in the collection sidebar, or
  `GET /api/export/bibtex?collection=<id>`).

### Removed
- The unused `GET /api/papers/<id>/graph` TF-IDF similarity stub (no UI ever
  called it); `GET /api/graph` supersedes it with real citation edges.

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
