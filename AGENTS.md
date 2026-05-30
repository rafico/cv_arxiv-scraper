# AGENTS.md

Orientation for AI agents working in this repo. Keep it accurate — update it when
the architecture changes.

## What this is

A single-user, localhost web app that scrapes arXiv (computer-vision focus),
enriches papers with external metadata (OpenAlex, Semantic Scholar, citations),
ranks them against the user's interests, and serves a Flask dashboard. It also
sends email digests and exports to reference managers (Zotero, Mendeley).

Stack: **Flask 3 + Flask-SQLAlchemy (SQLite) + vanilla JS/Jinja templates +
Tailwind**. Embeddings via `sentence-transformers` + `faiss-cpu`. Python 3.10+.

## Critical constraints (read before changing behavior)

- **No authentication.** The app is designed for localhost single-user use. It
  refuses to bind to a non-loopback host unless `--expose` is passed. Don't add
  features that assume multi-user isolation or auth.
- **Single worker by default.** Scrape progress/streaming state lives in
  *in-process memory* (see [app/services/jobs.py](app/services/jobs.py)). Running
  multiple gunicorn workers breaks `/api/scrape/stream`. `run.py` defaults to 1
  worker and warns otherwise. Don't introduce cross-worker assumptions without
  moving job state to a shared store.
- **Config is read/write at runtime.** The settings UI writes `config.yaml` via
  `save_config` ([app/services/preferences.py](app/services/preferences.py)),
  which does an atomic temp-write + rename and falls back to an in-place write
  when the destination can't be renamed (e.g. a Docker single-file bind mount).
- **Secrets live as dotfiles** at the repo/instance root (`.llm_api_key`,
  `.flask_secret`, `mendeley_credentials.json`, `.mendeley_token`, Gmail
  `credentials.json`/token). They are chmod `0600` and blocked from commits by a
  pre-commit hook ([scripts/block_credential_files.py](scripts/block_credential_files.py)).
  Never commit them.

## Setup, run, test

Canonical commands live in the [Makefile](Makefile) (`make help`) and mirror CI —
prefer them so "green locally" means "green in CI".

```bash
# Setup — use a virtualenv (the README uses ./.venv)
python3 -m venv .venv && source .venv/bin/activate
make setup                                     # pip install -e ".[dev]"
cp config.example.yaml config.yaml             # optional; defaults work without it

make run            # dev server (Flask debug, opens browser)
make run-prod       # production-style, in-process gunicorn, 1 worker

make test           # full suite minus slow e2e browser tests
make test-fast      # fast inner loop (skips slow + e2e)
make check          # the pre-push gate: lint + types + test

# Or directly:
python -m pytest -m "not slow and not e2e" -q  # fast subset while iterating
python -m pytest tests/test_run.py -q          # a single module
```

The full suite is multi-minute (importing `faiss`/`sentence-transformers` is
heavy, and `e2e` needs Playwright browsers). Iterate with `make test-fast`; run
`make test` before pushing.

Docker: `docker compose up` publishes `127.0.0.1:5000:5000` and runs
`python run.py ... --no-browser --expose` (see [Dockerfile](Dockerfile)). The
`--expose` is safe there only because the published port is loopback-bound.

## Architecture map

For the full layering diagram, scrape data flow, and step-by-step extension
recipes (add an ingest backend / API endpoint / enrichment provider), see
[ARCHITECTURE.md](ARCHITECTURE.md).

The Flask app factory is [app/__init__.py](app/__init__.py) → `create_app()`:
resolves config (`CV_ARXIV_CONFIG` env > instance > project root), inits the DB,
registers blueprints, sets security headers, and optionally starts the scheduler.

```
run.py / wsgi.py ── create_app() ── blueprints (app/routes/*) ── services (app/services/*) ── models (app/models.py, SQLite)
```

- **app/routes/** — HTTP layer (Blueprints): `dashboard`, `discover`, `settings`,
  `help`, `api`. Thin; delegates to services. See
  [app/routes/AGENTS.md](app/routes/AGENTS.md).
- **app/services/** — the business logic ("the heart"). Ingestion, enrichment,
  ranking, search/embeddings, digests, jobs, persistence. See
  [app/services/AGENTS.md](app/services/AGENTS.md).
- **app/models.py** — SQLAlchemy models (`Paper`, `Collection`, feedback, etc.).
- **app/cli/** — installable console scripts (`cv-arxiv-scrape`, `-sync`,
  `-backfill`, `-digest`); see `[project.scripts]` in `pyproject.toml`.
- **app/templates/**, **app/static/** — Jinja templates + assets (Tailwind CLI
  binary is `./tailwindcss`).

### Semantic re-export packages

`app/ingest`, `app/rank`, `app/search_`, `app/enrich`, `app/web` are **façade
packages that re-export from `app/services/*`** for readable imports — they hold
no logic. The top-level `*_cli.py` files (`scrape_cli.py`, etc.) are backward-compat
shims via [app/_module_alias.py](app/_module_alias.py). When you add real logic,
put it in `app/services/` and re-export, don't fork it into the façade.

### The scrape pipeline (high level)

`execute_scrape` / `execute_historical_scrape`
([app/services/scrape_engine.py](app/services/scrape_engine.py)) orchestrate:
**ingest** (orchestrator + RSS/arXiv-API backends, resumable) → **enrich** with
API metadata → **pipeline/rank** (feature extraction + scoring) → `_save_results`
→ `_generate_thumbnails` → `_generate_embeddings` → `_extract_sections`. Note the
ordering: `pdf_content` bytes are carried in each result dict and consumed by the
*last* steps (thumbnails and section extraction), so it must not be popped early.

## Conventions

- **Ruff** governs style (line length 120; rule sets `E,F,W,I,UP,B,C4,SIM,S`).
  `S` is bandit-style security linting — expect to justify `# noqa: S...` for
  intentional cases (e.g. filename constants).
- **Imports inside functions** are common here to avoid import cycles / heavy
  startup (e.g. embeddings). Follow the local pattern in the file you're editing.
- **Tests** mirror `app/` loosely under `tests/`; helpers in
  [tests/helpers.py](tests/helpers.py). See [tests/AGENTS.md](tests/AGENTS.md).
- Commit messages follow Conventional Commits (`feat(...)`, `fix(...)`,
  `refactor(...)`).
