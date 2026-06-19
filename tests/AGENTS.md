# tests — AGENTS.md

`unittest`-style tests run under pytest. ~60 modules, loosely mirroring `app/`.

## Running

```bash
python -m pytest -q                 # everything
python -m pytest -m unit            # fast units only
python -m pytest -m "not e2e"       # skip browser tests
python -m pytest tests/test_run.py::RunEntryPointTests::test_x -q
```

Markers (declared in `pyproject.toml`, `--strict-markers` is on): `unit`,
`integration`, `e2e`, `slow`. The full suite is slow (~minutes) because importing
`sentence-transformers`/`faiss` is heavy; prefer running the targeted module(s)
you touched, then the full suite once before committing.

## Helpers ([helpers.py](helpers.py))

- `FlaskDBTestCase` — builds an app via `create_app({...})` with `TESTING=True`, a
  temp SQLite DB, and a temp `config.yaml` seeded from `TEST_SCRAPER_CONFIG`.
  Pushes an app context in `setUp`, drops/creates tables, tears everything down.
  Use this for anything touching the DB, routes, or services.
- `DefaultConfigFlaskDBTestCase` — variant that `chdir`s into a temp root with a
  `config.example.yaml` (for testing the "no config.yaml yet" default path).

## Patterns

- **CSRF:** state-changing API/settings routes require a token. Tests fetch it via
  a `_csrf_token()` helper (GET the page, read the session token) and pass it as
  `X-CSRF-Token` header or `csrf_token` form field. Copy the pattern from
  `test_qa_scraping_ingestion.py` / `test_settings.py`.
- **External calls are patched**, not hit. `request_with_backoff`, embedding
  services, LLM clients, OAuth paths, and `execute_*_scrape` are mocked at their
  import site (e.g. `@patch("app.services.scrape_engine.execute_historical_scrape")`).
- **Filesystem side effects** should use real temp dirs and assert on outcomes
  (e.g. file contents + `stat.S_IMODE(...) == 0o600`) rather than asserting mock
  call args — it catches more (see `test_settings.py::test_upload_saves_valid_json`).
- **e2e/** holds Playwright browser tests (`pytest-playwright`); they need browsers
  installed (`playwright install`) and are excluded with `-m "not e2e"`.
- **Assert on semantic hooks, not styling.** UI tests target stable ids/classes
  and `data-*` attributes (`#paper-list`, `.feedback-btn[data-action]` +
  `data-active`, `#theme-toggle`, settings tabs' `data-active`), never Tailwind
  utility classes — restyling must not break the suite. See the hook list in
  [app/routes/AGENTS.md](../app/routes/AGENTS.md).

When adding a test, place it next to its peers (a `tests/test_<area>.py` already
exists for most areas) and prefer the existing `FlaskDBTestCase` base.
