# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project aims to
follow [Semantic Versioning](https://semver.org/).

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
