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
