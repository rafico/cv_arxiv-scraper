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
- `pipeline/` — feature extraction, candidate generation, `ranker`.
- `matching.py` (author/whitelist matching), `ranking.py`, `venues.py`
  (`parse_venue` detects conference acceptance from arXiv comments),
  `interest_model.py` (the interest profile. Collections first: one mean-centred
  centroid per collection, a paper's affinity is its best z against them —
  pure NumPy `fit_collection_profile` / `affinity_scores`, which also drive the
  interest gate in `pipeline/candidate_generation.py`. Else centroids from
  feedback + the vector index; inert below 5 saved papers), `feedback.py`
  (save/skip/priority/shared actions — **toggling**: re-applying an action clears
  it), `onboarding.py` (cold-start: ingests pasted arXiv IDs as implicit saves to
  seed the profile; active-learning `select_uncertain_papers` surfaces boundary
  papers), `metrics.py`,
  `preferences.py` (reads/writes `config.yaml` via `save_config` — atomic with an
  in-place fallback for bind-mounted destinations).

**Search / embeddings / corpus**
- `embeddings.py` (`EmbeddingService`, exact NumPy vector index; singleton via
  `get_embedding_service`, which re-reads the paper index when another process —
  the daily scrape — has saved it: `reload_if_changed`, one stat per call, and
  rows not yet saved are kept; `save()` builds on the index on disk the same
  way. A pair on disk that cannot be used is logged once and not read again
  until a file changes; a reload of a whole pair also turns saving back on for
  a service that started degraded. Don't `reset_embedding_service()` just to
  see new vectors: that also drops the loaded model),
  `embed_backfill.py`, `search.py` (BM25 + semantic +
  hybrid/RRF), `rag.py` (corpus chat: scope is a collection's `paper_ids`, else the **saved**
  papers — ranked exactly by per-id vectors — else the whole corpus via hybrid
  search, labelled `scope`; optionally synthesizes via the LLM client with `[n]`
  grounding; degrades to retrieval-only when `llm.enabled` is false), `related.py`, `corpus_analysis.py`,
  `saved_search.py`, `pdf_extraction.py`,
  `summary.py`, `text.py`.

**Outputs / integrations**
- `email_digest.py` (Gmail OAuth + digest send; `DEFAULT_CREDENTIALS_PATH`),
  `export.py` (HTML report), `bibtex.py`, `zotero.py`, `mendeley.py`,
  `thumbnail_generator.py`, `backup.py` (one-click backup/restore: consistent
  SQLite snapshot + vector index + config tarball. Restore is **staged-then-
  committed** — every component is copied onto its target filesystem first so the
  commit is a same-fs rename: cross-device-safe, all-or-nothing with rollback, and
  size-bounded against decompression bombs).
- `mcp_tools.py` — the logic behind the MCP tools: plain functions returning
  compact dicts, no `mcp` SDK import (the thin wrappers are in
  `app/mcp_server.py`; `cv-arxiv-mcp --read-only` registers the read tools only
  and stops the built-in scheduler that `create_app()` may have started).
  Unknown ids and an unknown `decision` come back as `{"error": ...}`, not
  exceptions (a string of digits is an id only when `_is_row_id` says so, and is
  never a collection name); limits and day windows are clamped, and the window
  applied is said back.
  `whats_new` (new papers close to a collection) reads stored vectors only and
  must stay that way: `get_paper_vectors` / `index_size` never load the
  embedding model, `encode` does.
  The three write tools (`set_decision`, `tag_papers`, `add_to_collection`) are
  for attended sessions. Keep them narrow: validate what the agent sends (a
  reason is one printable line, a tag a pattern, a collection is never created
  by a mistyped name), never overwrite a decision the owner made, and write
  through `_commit_logged`: flush, append one line to `mcp_writes.jsonl` next to
  the database file (never `app.instance_path`), then commit; a failure rolls
  back and returns `{"error": "write_failed"}`. No read tool, and no bundle,
  returns `decision_note`.
- `screening.py` — `apply_decision`, the one writer of a membership's
  `decision` and `decision_note`. An agent write passes its reason as `note`;
  an owner write (the REST routes) leaves it unset, which clears the note.
  Non-NULL `decision_note` means "an agent's, not yet confirmed by the owner".

**Persistence helpers**
- `_save_results` in `scrape_engine.py` maps explicit fields onto `Paper` (it
  does NOT dump whole result dicts, so transient keys like `pdf_content` are not
  persisted). `related.find_duplicates` handles near-duplicate titles.

## Conventions / gotchas

- sentence-transformers and other heavy deps are imported **inside
  functions** to keep startup cheap and avoid cycles — match the local style.
- Network calls go through `http_client.request_with_backoff` with a
  `rate_limit_profile` (e.g. `"bulk"`); don't hand-roll `requests` calls.
- Errors from ingest backends **propagate** (callers decide how to handle):
  `jobs.py` converts them to a job error; the historical route returns 502.
  Don't reintroduce silent catch-all swallowing.
- `now_utc()` / `utc_today()` live in `text.py`; use them for timestamps.
- `get_embedding_service()` with no app and no app context (a worker thread, a
  unit test) resolves `FAISS_INDEX_DIR`, then `CV_ARXIV_INSTANCE_PATH/faiss_index`,
  and only then `./instance/faiss_index`. Pass the app when you have one.
- The collection scorer (`interest_model.py`: centring, z, `AFFINITY_Z_*`,
  `MIN_BACKGROUND`) was settled by measurement. Before changing it, run
  `scripts/eval_collection_affinity.py --self-test`, then its `holdout` and
  `replay` views on a real instance (read-only) and compare the numbers.
- Scrape worker threads have no app context: the interest gate reads the
  profile from `get_cached_interest_profile()`, never from the DB. In a web
  process that cache starts empty, so code that labels the interest signal
  (`resolve_interest_source`) needs `build_interest_profile` called first.
