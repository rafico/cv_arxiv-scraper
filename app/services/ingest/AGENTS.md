# app/services/ingest — AGENTS.md

The ingestion subsystem: fetch paper candidates from arXiv via multiple backends,
with resumable pagination. This is the most state-heavy part of the pipeline.

## Pieces

- `base.py` — `PaperCandidate` dataclass + parsing helpers (`clean_abstract`,
  `extract_arxiv_id`, `parse_publication_dt`). Shared by all backends.
- `orchestrator.py` — `IngestOrchestrator` chooses backends and modes, aggregates
  candidates, and owns the per-backend error policy.
- `arxiv_api_backend.py` — `ArxivApiBackend.fetch(...)`: queries the arXiv Atom
  API, paginates by `page_size`, supports `resume_after_arxiv_id`.
- `rss_backend.py` — RSS/Atom feed backend for the daily-watch path.

## Modes (`IngestMode`)

- `DAILY_WATCH` — recent papers; RSS + (optionally) arXiv API rolling window.
  Wrapped in `_fetch_recent_candidates` which **catches and degrades** to `[]`
  unless `strict`.
- `BACKFILL` — historical range; calls the arXiv API backend directly
  (`_fetch_arxiv_api`, **not** error-wrapped — failures propagate to the caller).
- `CATCH_UP` — fills the gap since the last sync cursor per category.

## Error handling contract

Backends **raise** on transient/HTTP/XML failure (they do not swallow). Each
caller decides:
- daily-watch → degrades to empty results,
- the background job manager → publishes a `scrape_error` event,
- the `/api/search/historical` route → returns HTTP 502.

If you touch a backend, preserve this: don't add a catch-all that hides errors,
and don't let an unwrapped caller turn a transient error into an unhandled 500.

## Resume / pagination notes

`fetch` tracks `start`, `page_size`, and `resume_after_arxiv_id`. It computes the
current page per entry and skips everything up to (and including) the resume
anchor before appending. When editing the loop, keep the page math and the
`resume_consumed` flag consistent, and remember `max_results` bounds the total.
