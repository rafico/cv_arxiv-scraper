# app/services — AGENTS.md

This is the business-logic layer. Routes ([app/routes](../routes)) and CLIs
([app/cli](../cli)) are thin and delegate here. The `app/ingest`, `app/rank`,
`app/search_`, `app/enrich`, `app/web` packages are façades that **re-export**
from here — add logic here, not there.

## Map by responsibility

**Scrape orchestration**
- `scrape_engine.py` — top-level pipeline: `execute_scrape` (daily) and
  `execute_historical_scrape` (backfill). Wires ingest → enrich → rank → save →
  thumbnails → embeddings → sections. The result dict carries `pdf_content` bytes
  that the *final* steps consume — don't pop it early.
- `jobs.py` — `SCRAPE_JOB_MANAGER`: runs scrapes in a background thread and
  streams Server-Sent Events. **State is in-process** (the reason for the
  1-worker default). Exceptions from `execute_scrape` are caught here and
  published as a `scrape_error` event.
- `scheduler.py`, `cron.py` — scheduled/cron-driven runs.

**Ingestion** — see [ingest/AGENTS.md](ingest/AGENTS.md)
- `ingest/` — orchestrator + RSS and arXiv-API backends, resumable pagination.
- `arxiv_adapter.py`, `enrichment.py`, `http_client.py` (sessions, backoff,
  `request_with_backoff`), `rate_limiter.py`.

**Enrichment** (external metadata)
- `enrichment_providers/` — `OpenAlexProvider`, `SemanticScholarProvider`,
  `GitHubProvider` (repo stars/license; per-run fetch cap for the
  unauthenticated rate limit).
- `openalex.py`, `citations.py`.

**Ranking / matching** — see [pipeline/](pipeline)
- `pipeline/` — `FeatureExtractor`, candidate generation, `ranker`.
- `matching.py` (author/whitelist matching), `ranking.py`, `venues.py`
  (`parse_venue` detects conference acceptance from arXiv comments),
  `interest_model.py` (learned interest centroids from feedback + the FAISS
  index; inert below 5 saved papers), `feedback.py`
  (save/skip/priority/shared actions — **toggling**: re-applying an action clears
  it), `onboarding.py` (cold-start: ingests pasted arXiv IDs as implicit saves to
  seed the profile; active-learning `select_uncertain_papers` surfaces boundary
  papers), `metrics.py`, `recommendations.py`,
  `preferences.py` (reads/writes `config.yaml` via `save_config` — atomic with an
  in-place fallback for bind-mounted destinations).

**Search / embeddings / corpus**
- `embeddings.py` (`EmbeddingService`, FAISS index; singleton via
  `get_embedding_service`), `embed_backfill.py`, `search.py` (BM25 + semantic +
  hybrid/RRF), `rag.py` (chat-with-saved-papers: retrieves over the **saved**
  corpus via hybrid search, optionally synthesizes via the LLM client; degrades to
  retrieval-only when `llm.enabled` is false), `related.py`, `corpus_analysis.py`,
  `saved_search.py`, `pdf_extraction.py` (`extract_and_store_sections`),
  `summary.py`, `text.py`.

**Outputs / integrations**
- `email_digest.py` (Gmail OAuth + digest send; `DEFAULT_CREDENTIALS_PATH`),
  `export.py` (HTML report), `bibtex.py`, `zotero.py`, `mendeley.py`,
  `thumbnail_generator.py`, `backup.py` (one-click backup/restore: consistent
  SQLite snapshot + FAISS index + config tarball. Restore is **staged-then-
  committed** — every component is copied onto its target filesystem first so the
  commit is a same-fs rename: cross-device-safe, all-or-nothing with rollback, and
  size-bounded against decompression bombs).

**Persistence helpers**
- `_save_results` in `scrape_engine.py` maps explicit fields onto `Paper` (it
  does NOT dump whole result dicts, so transient keys like `pdf_content` are not
  persisted). `related.find_duplicates` handles near-duplicate titles.

## Conventions / gotchas

- FAISS/sentence-transformers and other heavy deps are imported **inside
  functions** to keep startup cheap and avoid cycles — match the local style.
- Network calls go through `http_client.request_with_backoff` with a
  `rate_limit_profile` (e.g. `"bulk"`); don't hand-roll `requests` calls.
- Errors from ingest backends **propagate** (callers decide how to handle):
  `jobs.py` converts them to a job error; the historical route returns 502.
  Don't reintroduce silent catch-all swallowing.
- `now_utc()` / `utc_today()` live in `text.py`; use them for timestamps.
