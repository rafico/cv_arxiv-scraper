# Architecture

Deeper companion to [AGENTS.md](AGENTS.md). Covers the layering, the scrape data
flow, and step-by-step recipes for common extensions.

## Layering

```
            entry points
  run.py / wsgi.py / app/cli/*  (CLIs: cv-arxiv-scrape, -sync, -backfill, -digest)
                │
                ▼
        app/__init__.py  ── create_app(): config resolution, DB init,
                              blueprint registration, security headers, scheduler
                │
        ┌───────┴────────┐
        ▼                ▼
   app/routes/*      app/cli/*        ← thin: parse/validate, then delegate
   (HTTP Blueprints) (console)
        │                │
        └───────┬────────┘
                ▼
          app/services/*               ← business logic ("the heart")
   ┌──────────┬──────────┬───────────┬───────────┬──────────────┐
   ingest     enrichment  pipeline    search/      jobs/          outputs
   (backends) (OpenAlex,  (features,  embeddings   scheduler      (digest,
              S2,         ranker)     (FAISS,BM25) (in-process)    export,
              citations)                                          zotero…)
                │
                ▼
          app/models.py  ── SQLAlchemy (SQLite at instance/arxiv_papers.db)
```

**Façade packages** `app/{ingest,rank,search_,enrich,web}` re-export from
`app/services/*` for readable imports and hold no logic. Top-level `*_cli.py`
files are backward-compat shims via `app/_module_alias.py`. Put new logic in
`app/services/`, then re-export.

## Scrape data flow

`execute_scrape` (daily) / `execute_historical_scrape` (backfill) in
[app/services/scrape_engine.py](app/services/scrape_engine.py):

```
ingest.orchestrator.fetch(mode)            # RSS + arXiv-API backends, resumable
   → enrich_entries_with_api_metadata      # OpenAlex / Semantic Scholar / citations
   → _process_entries_with_pipeline        # feature extraction + ranking + LLM summary
   → _save_results                         # explicit field mapping onto Paper rows
   → _generate_thumbnails                   # reads result["pdf_content"]
   → _generate_embeddings                   # FAISS index update
   → _extract_sections                      # reads result["pdf_content"]  ← last consumer
```

Each result dict carries `pdf_content` (PDF bytes) fetched once and reused by the
last two steps; it is **not** persisted (`_save_results` maps explicit columns).
Don't pop it before section extraction.

Errors from ingest backends **propagate** by design (no catch-all swallow). The
background job manager converts them to a `scrape_error` SSE event; the
`/api/search/historical` route maps them to HTTP 502; the daily-watch rolling
window degrades to `[]`.

## Key invariants

- **No auth, localhost only** — `run.py` refuses non-loopback binds without
  `--expose`. Don't assume multi-user isolation.
- **Single worker** — scrape/job/streaming state is in-process
  ([app/services/jobs.py](app/services/jobs.py)). `/api/scrape/stream` breaks
  across workers; `run.py` defaults to 1 and warns otherwise.
- **Runtime config writes** — `save_config` is atomic (temp+rename) with an
  in-place fallback for non-renameable destinations (Docker single-file mount).

## Extension recipes

**Add an ingest backend**
1. Implement a backend in `app/services/ingest/` returning `list[PaperCandidate]`
   (model it on `arxiv_api_backend.py` / `rss_backend.py`; reuse `base.py` parsers).
2. Register it in `ingest/orchestrator.py` (backend registry + the relevant mode).
3. Fetch network via `http_client.request_with_backoff` (set a `rate_limit_profile`).
4. Let transient failures raise — don't add a catch-all.
5. Add a test mirroring `tests/test_ingest_backends.py` (patch `request_with_backoff`).

**Add an HTTP API endpoint**
1. Add the view to the right Blueprint in `app/routes/` (the matching
   `api/<feature>.py` module for `/api/*`).
2. Guard mutations with `validate_csrf_token()`; keep the view thin (call a service).
3. Map upstream/service failures to honest status codes (e.g. 502), not unhandled 500s.
4. Escape user data into `data-*` attributes, not inline JS, in templates.
5. Add a test in the matching `tests/test_*.py` using `FlaskDBTestCase` + the
   `_csrf_token()` helper.

**Add an enrichment provider**
1. Implement the `EnrichmentProvider` interface in
   `app/services/enrichment_providers/`.
2. Wire it into the enrichment flow and re-export via `app/enrich`.
3. Mock the HTTP calls in tests; never hit the network in unit tests.

**Add a settings-backed config option**
1. Extend the schema/validation in `app/schema.py` and `config.example.yaml`.
2. Read/update it through `app/services/preferences.py` (use `save_config`).
3. Surface it in `app/routes/settings.py` + the settings template.
