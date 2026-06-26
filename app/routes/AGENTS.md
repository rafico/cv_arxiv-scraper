# app/routes — AGENTS.md

The HTTP layer: Flask Blueprints registered by `create_app`
([app/__init__.py](../__init__.py) → `_register_blueprints`). Routes are **thin** —
parse/validate the request, call a service, shape the response. Business logic
belongs in [app/services](../services).

## Blueprints

- `dashboard.py` — main paper feed UI (`/`), feedback actions, user tags. The
  inbox filters come in as **URL query params** (the JS `applyFilters` helper
  rewrites the querystring), not a form POST. `density` is `list` (default) or
  `visual`.
- `_shell.py` — a context processor injecting `shell` (collections, saved
  searches, inbox/saved counts) into **every** template so the sidebar renders
  app-wide. Queries are guarded against a missing DB.
- `discover.py` — discovery/recommendations/corpus views, incl. the **chat with
  your saved papers** panel (posts to `/api/corpus/chat`).
- `settings.py` — settings UI + credential uploads, Gmail OAuth callback, config
  writes (via `save_config`). Note: credential files are written with `0600`.
- `help.py` — static help/onboarding pages.
- `api/` — JSON API under `/api`, one shared `api_bp` blueprint assembled from
  feature modules (`scrape`, `search`, `papers`, `export`, `collections`,
  `saved_searches`, `feed_sources`, `chat` (corpus RAG), `onboarding`
  (cold-start bootstrap + active-learning), `backup` (export/import snapshot)).
  Add new endpoints to the matching module; routes register on import via
  `api/__init__.py`.

## Conventions

- **CSRF:** state-changing endpoints call `validate_csrf_token()`
  ([app/csrf.py](../csrf.py)). Keep new POST/mutation routes guarded.
- **Security headers** are applied app-wide in the factory; don't re-implement.
- **Escape user data in templates.** Jinja autoescaping handles HTML, but values
  interpolated into inline JS or HTML attributes need care — render them into
  `data-*` attributes and read via `dataset` rather than into `onclick="...('{{ x }}')"`
  (see the user-tag handling in `templates/partials/_paper_details.html`).
- **Stable hooks for JS and tests.** Front-end behavior and the e2e/QA tests
  target semantic hooks, not utility classes: `.paper-card`, `.paper-link`,
  `.paper-pdf-link`, `.feedback-btn[data-action]` + `data-active`,
  `.reading-status-select`, `.card-toggle`, `[data-card-details]`,
  `#paper-list` / `#paper-grid`, `#theme-toggle`, `#scrape-btn`, `#app-sidebar`,
  settings tabs' `data-active`. Don't rename these when restyling; assert on
  them rather than on Tailwind classes.
- **Map upstream failures to honest status codes.** Services raise on failure;
  surface that — e.g. `search_historical` returns **502** when the arXiv fetch
  fails rather than letting it become an unhandled 500.
- **SSE / streaming** (`/api/scrape/stream`) relies on in-process job state from
  `SCRAPE_JOB_MANAGER`; this is why the app runs a single worker.
- Heavy service imports are done **inside the view function** to keep startup
  light and avoid cycles — follow the local pattern.
