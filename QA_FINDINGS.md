# Pre-Release QA Findings — cv_arxiv-scraper

Round: `qa/pre-release-sweep`. Baseline before this round: **767 passed**,
8 e2e deselected, **82% coverage**, `mypy`/`ruff`/`ruff format` all clean.

Severity: **S1** crash/data-loss/security · **S2** wrong user-visible behavior ·
**S3** cosmetic/edge. All 14 findings below are **fixed with regression tests**
that were each verified to fail on the pre-fix code and pass on the fix.

## High-impact (S1 / S2)

| # | Sev | Area | Defect | File |
|---|-----|------|--------|------|
| E1 | S1 | scheduler | Malformed-but-parseable `daily_at` (e.g. `"25:00"`) crashed app startup — the `.replace(hour=…)` sat outside the parse try/except | app/services/scheduler.py |
| E2 | S2(sec) | secrets | `write_api_key` / `_ensure_secret_key` did `write_text` then `chmod 0600` → secret world-readable in the creation window (TOCTOU) | app/services/llm_client.py, app/__init__.py |
| E3 | S2 | settings | Empty/truncated `config.yaml` → `yaml.safe_load`→`None` → every mutating settings route 500s | app/routes/settings.py |
| C2 | S2 | saved search | Unbounded `date_window_days` → `timedelta` `OverflowError` → `/run` 500 | app/services/saved_search.py |
| C3 | S2 | saved search | Unbounded `min_citations` → SQLite INTEGER overflow → create/update 500 | app/services/saved_search.py |
| C4 | S2 | preferences | Follow-author corrupted the whitelist when `whitelists[key]` was a YAML scalar string (`list("Jane")`→chars) | app/services/preferences.py |
| D1 | S2 | digest | `DigestRun` stuck `running` forever when Gmail credential load failed (load was outside the error-recording try) | app/services/email_digest.py |
| D2 | S2 | zotero | Reported success when Zotero returned HTTP 200 with a non-empty `failed` map (silent data loss) | app/services/zotero.py |
| B1 | S2 | enrichment | OpenAlex citation misattribution via substring DOI match (`2301.0001` matched `…arxiv.2301.00012`) | app/services/enrichment_providers/openalex_provider.py |
| A1 | S2 | ingest | CATCH_UP resume dropped a whole page when the saved cursor was no longer on it (shifted off by between-run submissions/withdrawals) | app/services/ingest/arxiv_api_backend.py |
| B2 | S2 | venues | A "workshop" token anywhere demoted a genuine main-conference oral | app/services/venues.py |

## Lower / edge (S3)

| # | Sev | Area | Defect | File |
|---|-----|------|--------|------|
| D3 | S3 | mendeley | Mononym / single-token authors sent as `first_name == last_name` | app/services/mendeley.py |
| B3 | S3 | venues | A stray "accepted" elsewhere defeated the "submitted" guard → false venue bonus | app/services/venues.py |
| B4 | S3 | ranking | Interest profile cached a stale `None` when <5 of 5 saved papers were indexed at build time, then the rest indexed later without new feedback | app/services/interest_model.py |
| C1 | S3 | feedback | Un-prioritizing left the save that priority had implied (asymmetric toggle) | app/services/feedback.py |

## Fix notes (judgment-call items)

- **A1** — minimal, targeted: parse the batch once and, if the saved cursor id is
  absent from the page we resumed from, treat it as "already past" and resume from
  the page's first entry. The normal cursor-present resume path is untouched
  (existing `test_fetch_resumes_from_offset_and_skips_processed_cursor` still
  passes); re-processing a few already-seen papers is harmless (de-duped on
  unique arxiv_id).
- **B2 / B3** — replaced first-match-wins status detection with proximity
  weighting: submit/accept cues are compared by distance to the venue mention,
  and "workshop" only classifies when adjacent to the venue. All 12 pre-existing
  venue tests still pass.
- **B4** — folded the FAISS index size into the interest-profile cache key (added
  `EmbeddingService.index_size()`), so a newly-embedded backlog invalidates a
  stale "disabled" profile. Callers are non-hot (scrape stages + explicit
  rebuild), so the extra cheap accessor call is fine.
- **C1** — made the toggle symmetric *without* data loss: the auto-added save is
  marked (`reason="implied_by_priority"`) so un-prioritizing removes only that
  implied save and never a save the user added explicitly.

---

# Pre-Release QA Findings — Round 3

Round: `qa/pre-release-sweep-3`. Baseline before this round: **828 passed**,
8 e2e deselected, `mypy`/`ruff`/`ruff format` all clean. After this round:
**853 passed** (25 regression tests added). 17 defects fixed; each fix has at
least one regression test verified to fail on the pre-fix code, except **F15**
(non-deterministic by nature — its test locks the post-fix invariant).

Six finder agents swept distinct subsystems; every candidate was traced against
the real code path before fixing (false positives discarded). Findings are
distinct from the 35 fixed in rounds 1–2.

## High-impact (S1 / S2)

| # | Sev | Area | Defect | File |
|---|-----|------|--------|------|
| F1 | S1 | startup/config | A null or non-dict `scheduler:` section (a natural hand-edit) crashed app startup — `.get("scheduler", {})` returns `None`/scalar when the key exists, then `.get("enabled")` raised `AttributeError` | app/__init__.py |
| F2 | S2 | venues | The oral/spotlight/highlight qualifier was matched *globally*, so a stray qualifier about a different venue ("Accepted to CVPR 2024. … our ICCV oral paper.") flipped the matched venue's status to oral → 1.5× ranking bonus. Rounds 1–2 fixed the submit/accept/workshop cues but not the qualifier | app/services/venues.py |
| F3 | S2 | related papers | Embedding neighbours (global FAISS ids) shadowed the in-pool TF-IDF fallback; the dashboard then dropped out-of-pool ids, so cards showed *no* related papers once the index was populated | app/services/related.py |
| F4 | S2 | api/feedback | Non-string `reason`/`note` reached the `String(64)`/`Text` columns → `ProgrammingError` (not caught by the api error handlers) → 500 | app/routes/api/papers.py |
| F5 | S2 | api/feedback | A non-scalar item in `bulk-feedback`'s `paper_ids` reached `session.get` as a malformed PK → `InvalidRequestError` → 500 | app/routes/api/papers.py |
| F6 | S2 | api/search | A negative `limit` defeated the result caps — SQLite treats `LIMIT -1` as unlimited — in saved-search `/run` and `/api/search` | app/routes/api/saved_searches.py, search.py |
| F7 | S2 | mendeley | `add_document` raised (uncaught) when a mid-batch 401 hit an unrefreshable token (`RuntimeError`), since `except` only caught `RequestException` → route 500 **and** an in-progress sync aborted before `commit`, losing the `mendeley_doc_id` of already-synced papers (duplicate re-uploads next run) | app/services/mendeley.py |
| F8 | S2 | settings/config | Lost-update race: the config read-modify-write happened outside the lock (only the *write* was locked), so two concurrent edits to different sections each saved over a stale snapshot, clobbering one another | app/routes/_config.py, settings.py, api/papers.py |
| F9 | S2 | feedback | Clicking "save" on a paper whose save was *implied by priority* deleted that save, leaving a still-prioritized paper with no save row (broke the priority-implies-save invariant). Round 1 fixed the un-prioritize direction only | app/services/feedback.py |
| F10 | S2 | scheduler/jobs | The scheduled scrape called `execute_scrape` directly, bypassing the job manager's single-flight gate → a scheduled + manual scrape could overlap, and each rewrites the FAISS index via atomic rename, so the later writer silently dropped the other run's vectors/sections | app/services/scheduler.py |

## Lower / edge (S3)

| # | Sev | Area | Defect | File |
|---|-----|------|--------|------|
| F11 | S3 | embeddings | `add_sections` had no dedup (unlike `add_papers`), so re-embedding an already-indexed paper appended duplicate section vectors → bloat + skewed `search_sections` | app/services/embeddings.py |
| F12 | S3 | http | `request_with_backoff(attempts<=0)` skipped the loop and did `raise last_exc` with `last_exc` still `None` → `TypeError`, no request made (reachable via `pdf_attempts: 0`) | app/services/http_client.py |
| F13 | S3 | bibtex/export | `$` was not escaped, breaking LaTeX math in titles/abstracts while `^`/`_` *were* escaped (inconsistent → invalid output) | app/services/bibtex.py |
| F14 | S3 | ranking | A DB `RankingConfig` weight of `0` was silently dropped (`0 < value`), so disabling a signal (e.g. `ai_weight: 0`) worked via preferences but not via a DB config | app/services/ranking.py |
| F15 | S3 | search | Hybrid RRF tie-break was non-deterministic (`all_pids` is a set, sort keyed on score only) → unstable top-k boundary across runs | app/services/search.py |
| F18 | S3 | feedback | A client-supplied `reason` could impersonate the internal `implied_by_priority` sentinel → an explicit save silently cascade-deleted on un-prioritize | app/services/feedback.py, api/papers.py |
| F19 | S3 | startup/config | A string `scheduler.enabled: "false"` read truthy, starting the scheduler against the user's intent | app/__init__.py |

## Fix notes (judgment-call items)

- **F2** — added `_qualifier_near_venue`, mirroring the existing `_signal_near_venue`
  proximity gating (window=30): a qualifier only sets the status when it sits near
  the matched venue. All pre-existing venue tests still pass.
- **F9** — clicking "save" on an implied save now *promotes* it (clears the marker,
  keeps the row, returns `active=True`) instead of deleting — preserving the
  priority-implies-save invariant rather than re-introducing the asymmetry C1 fixed.
- **F10** — routed the scheduler through `SCRAPE_JOB_MANAGER.start_or_get_active`
  (the single source of truth for "a scrape is running"). The scheduled run now also
  surfaces in the UI status; behaviour is otherwise unchanged (`force=False`).
- **F8** — added a re-entrant `config_write_lock()` context manager and wrapped the
  six mutating settings/API paths' load→mutate→persist cycle. Validation semantics
  per path are unchanged; only the read-modify-write is serialized.
- **F14** — relaxed the bound to `0 <= value <= 1000`; `>1000`/`NaN`/`inf` stay
  rejected (they fail the comparison) and `recency_multiplier` already clamps the
  half-life to `>= 0.5`, so a `0` freshness weight cannot divide by zero.
- **F6** — clamped with `max(1, min(..., cap))`; the other search parse paths already
  used `_parse_int_query_arg(minimum=1, ...)`.

## Investigated, not fixed (transparency)

- **(S3) thumbnail timeout** (`scrape_engine._generate_thumbnails`) — `executor.map(timeout=120)`
  doesn't bound wall-clock (shutdown waits for already-submitted tasks). It runs on the
  background scrape thread, so it never blocks request threads — only delays scrape
  completion. Cosmetic/latency with a misleading comment; left as-is.
- **(S3) thumbnail storage-key divergence** — the scrape worker and the serving route
  derive the storage key slightly differently for *non-arXiv* feeds (arXiv keys agree on
  both sides). Causes a wasted render / perpetually-"missing" thumbnail, no crash.
  Deferred.
- **(S3) no negative caching** — OpenAlex / Semantic-Scholar "not found" results aren't
  cached, so unmatched ids re-hit the API every run (the GitHub provider already
  negative-caches its 404s). Efficiency, not a correctness defect. Deferred.

---

# Pre-Release QA Findings — Round 4

Round: `qa/pre-release-sweep-3` (fourth sweep). Baseline before this round: **853 passed**,
8 e2e deselected; `mypy`/`ruff`/`ruff format` all clean. After this round: **906 passed**
(53 regression tests added across 16 new `test_qa_round4_*.py` modules). 20 defects fixed;
every fix has a regression test verified to fail on the pre-fix code and pass on the fix.

Method: 14 finder agents swept disjoint subsystems, then every candidate was put through two
independent adversarial verifiers — a correctness/reachability lens and a duplicate/already-
handled lens. 18 candidates survived (16 confirmed, 2 contested), each traced against the real
code path before fixing. Two further defects (G19, G20) were surfaced by a live run getting
429-throttled by arXiv. All findings are distinct from the 52 fixed in rounds 1–3.

## High-impact (S1 / S2)

| # | Sev | Area | Defect | File |
|---|-----|------|--------|------|
| G1 | S2 | cli/cron | `scrape_cli.py`/`sync_cli.py`/`backfill_cli.py` lost their `__main__` guard in the package reorg → pure alias shims. `python scrape_cli.py` (written by cron mode="scrape" and documented in the README) aliases the module and exits 0 **without calling `main()`** — a silent no-op, so a scheduled "Scrape only" cron never scrapes (exit 0 → cron records success) | scrape_cli.py, sync_cli.py, backfill_cli.py |
| G2 | S2 | enrichment | Semantic Scholar `fetch_batch` sent every id in one `/paper/batch` POST with no chunking; >500 ids (CLI `--batch-size`, broad-whitelist scrape) exceed the API cap and the broad `except` drops **all** citation data for the run | enrichment_providers/semantic_scholar.py |
| G3 | S2 | mendeley | `check_connection()` dereferenced `token['access_token']` under a `try` that caught only `RequestException`; a stored token missing that key → uncaught `KeyError` → **the whole Settings page 500s**, locking the user out of the re-auth UI | mendeley.py |
| G4 | S2 | api/collections | Renaming a collection to an existing name → uncaught `IntegrityError` (UNIQUE) → 500 + failed session. Create already guarded duplicates; update didn't | api/collections.py |
| G5 | S2 | dashboard | `GET /?view=saved&collection=<id>` 500s: the `view="saved"` PaperFeedback join is mutually exclusive with the collection join, but the default `sort="saved"` still issues `ORDER BY paper_feedback.created_at` → `OperationalError` | dashboard.py |
| G6 | S2 | preferences | `get_preferences()` read the muted lists with bare `list()`: a scalar `muted: {authors: "Jane Doe"}` shatters into single chars (silent filter corruption that passes validation); a non-iterable `muted: {topics: 5}` raises `TypeError` inside `_validate_config` → **app startup crash** | preferences.py |
| G7 | S2 | digest/settings | A malformed `email:` section (null or scalar) → `AttributeError` in both `_get_email_config` and `view_settings` → Settings page, digest preview, send-test, and the digest CLI all 500 | email_digest.py, settings.py |
| G9 | S2 | scrape/faiss | `execute_historical_scrape` (POST /search/historical) runs the pipeline in the request thread, outside the job-manager gate; its FAISS read-append-rename can overlap a daily/scheduled scrape's and the later `os.replace` silently drops the other run's vectors/sections — the F10 race via a still-ungated entry point | scrape_engine.py |
| G13 | S2 | embeddings | `add_papers` deduped only against the persisted reverse map (updated *after* the loop), so the same id twice in one batch (paper cross-listed across two RSS feeds — no cross-feed dedup) added the vector twice → orphaned FAISS row, inflated count, duplicate search/related cards | embeddings.py |
| G20 | S2 | enrichment/http | On an arXiv **429**, `_fetch_api_metadata_batch` recursively halves the batch ("retry in smaller chunks") — issuing *more* requests during a rate-limit window (observed live: 20→10→5), amplifying the throttling instead of backing off | enrichment.py |

## Lower / edge (S3)

| # | Sev | Area | Defect | File |
|---|-----|------|--------|------|
| G8 | S3 | api | Oversized integer query/path params (`collection_id`, bulk-bibtex `ids`, any `<int:>` PK) → uncaught `OverflowError` (not in the api error-handler set) → 500 instead of 400 | api/__init__.py |
| G10 | S3 | api/ingest | `POST /search/historical` `categories` unvalidated: a JSON string iterates char-by-char into a wrong arXiv query (silent empty result); a non-iterable → `TypeError` masked as a misleading **502 "arXiv unavailable"** | api/scrape.py |
| G11 | S3 | ranking/ui | Dashboard "Why this ranked here" called `explain_score` without `citation_count`, so the Citations chip was always 0.0 and the breakdown never summed to the shown total | dashboard.py |
| G12 | S3 | venues | `_nearest_distance` measured distance to the venue's *start* only, so for long-form aliases (IJCV, TPAMI, RSS, ACM-MM) a trailing acceptance cue lost to a preceding "submitted" word → real acceptance demoted to "mentioned" (lost bonus). Made edge-aware | venues.py |
| G14 | S3 | rate-limit | `_positive_int` caught only `(TypeError, ValueError)`; `burst: .inf` (YAML infinity) → `int(inf)` raises uncaught `OverflowError`, breaking all HTTP fetching | rate_limiter.py |
| G15 | S3 | rate-limit | `resolve_rate_limit_settings` guarded that `ingest` is a Mapping but not the nested `rate_limit`; `ingest.rate_limit: fast` (scalar/None/list) → `AttributeError`, breaking all fetching. Mirrors the existing `resolve_user_agent` guard (contested) | rate_limiter.py |
| G16 | S3 | thumbnails | Legacy slash-form arXiv ids (`cs/9901001`, allowed by the id/storage-key regexes) write to `thumbnails/cs/9901001.png` but only the top dir is created → swallowed `FileNotFoundError` → thumbnails **never** appear and every warm/backfill re-downloads + re-fails | thumbnail_generator.py |
| G17 | S3 | thumbnails | PNGs were written in place; a timeout/native crash mid-`im.save` leaves a truncated PNG that satisfies `.exists()` and is served forever. Now temp-write + `os.replace` | thumbnail_generator.py |
| G18 | S3 | embeddings | `get_embedding_service(app=None)` derived the index dir from CWD, which can diverge from `app.config['FAISS_INDEX_DIR']` under a non-default CWD/instance; now prefers `current_app` config inside an app context (defensive; contested) | embeddings.py |
| G19 | S3 | http | `request_with_backoff` retried 429s on a fixed `1.25·2ⁿ` backoff, ignoring the server's `Retry-After` → retries fire before arXiv is ready and keep getting throttled. Now honors `Retry-After` (seconds or HTTP-date), clamped | http_client.py |

## Fix notes (judgment-call items)

- **G9** — deliberately did NOT route the historical scrape through `SCRAPE_JOB_MANAGER` (it is
  fire-and-forget and would break the endpoint's synchronous summary-return contract). Instead a
  module-level `threading.Lock` serializes the FAISS read-append-rename inside `_generate_embeddings`
  / `_extract_sections`; every scrape path (daily, scheduled, historical) funnels through those two
  functions, so guarding inside them covers all paths with no duplication and no deadlock. A cross-
  *process* CLI-sync-vs-server write is a separate, lower-priority concern.
- **G13 over-reach, caught by the suite** — the finding was `add_papers` only, but the fix agent
  also added a paper-id `seen` set to `add_sections`. That is wrong (a paper contributes many
  section rows) and broke the round-3 F11 invariant (`add_sections([(1,intro),(1,method)])` must
  return 2). The full-suite gate caught it; `add_sections` was reverted to its per-paper-across-
  calls form with a regression guard added. The realistic intra-batch section-dup path isn't
  reachable — `_extract_sections` builds entries from distinct `PaperSection` rows.
- **G8** — fixed at the `api_bp` error handler (`OverflowError → 400`) rather than per-call bounds,
  so every `<int:>` PK lookup and `IN`-list degrades cleanly, not just the two named endpoints.
- **G10** — a bare-string `categories` is *rejected* (400), not silently coerced, matching the
  input-type-validation posture established in round 2.
- **G19 / G20** — surfaced by a live run getting 429-throttled by arXiv. Together they make the
  client back off politely (honor `Retry-After`) and stop *amplifying* a rate-limit storm (no
  batch-splitting on 429). The arXiv paths already use the `bulk` profile (1 req / 3 s); these
  address the *response* to throttling, not the request rate.
- **G15 / G18** — "contested" in verification (one skeptic deemed each already-handled or not
  reachable in shipped configs). Kept as low-risk one-line defensive hardening, consistent with
  sibling code (`resolve_user_agent`; the app-passed index-dir branch).

## Concurrent-edit note (not part of this round)

While this sweep ran, the working tree separately gained an **unrelated scrape progress-bar
feature** authored outside this QA session: `app/templates/partials/_shell_scripts.html`,
`app/templates/partials/_scrape_progress.html`, the `downloading`/`progress_total` streaming
in `scrape_engine.py`, and added cases in `tests/test_scrape_engine.py`. No QA agent touched
those and they are left intact. Of the `scrape_engine.py` changes, only `_INDEX_WRITE_LOCK`
(G9) belongs to this round.
