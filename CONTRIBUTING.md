# Contributing

Thanks for taking a look! This is a single-user, localhost Flask app — small
enough to get into quickly. This guide is the human-facing companion to the
agent-facing [AGENTS.md](AGENTS.md) and the deeper [ARCHITECTURE.md](ARCHITECTURE.md).

## Get set up

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
make setup                         # pip install -e ".[dev]"
pre-commit install                 # lint/format + credential-leak hooks on commit
cp config.example.yaml config.yaml # optional; defaults work without it
make run                           # dev server at http://127.0.0.1:5000
```

`make help` lists every command. They mirror CI, so "green locally" means "green
in CI".

## Run the tests

```bash
make test-fast   # quick inner loop (skips slow + e2e browser tests)
make test        # full suite minus e2e
make check       # the pre-push gate: lint + types + test
```

The full suite is multi-minute (importing `faiss`/`sentence-transformers` is
heavy). Iterate with `make test-fast` and run `make check` before opening a PR.

## How the code is laid out

Requests flow **routes → services → models**:

- [app/routes/](app/routes/) — thin HTTP/Blueprint layer; delegates to services.
- [app/services/](app/services/) — the business logic: ingest, enrich, rank,
  search/embeddings, digests, jobs, persistence.
- [app/models.py](app/models.py) — SQLAlchemy models on SQLite.
- [app/cli/](app/cli/) — the installable `cv-arxiv-*` console scripts.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full diagram, the scrape data
flow, and step-by-step recipes (add an ingest backend / API endpoint / enrichment
provider). Per-directory notes live in the `AGENTS.md` files under each subsystem.

A couple of constraints worth knowing up front (full list in
[AGENTS.md](AGENTS.md)): there's **no authentication** (localhost single-user by
design), the server runs a **single worker** (scrape/stream state is in-process),
and **never commit secrets** — credential dotfiles are gitignored and a pre-commit
hook blocks them.

## Conventions

- **Style:** Ruff governs formatting and linting (`make format` to auto-fix).
- **Commits:** Conventional Commits — `feat(...)`, `fix(...)`, `docs(...)`,
  `refactor(...)`, `chore(...)`.
- **UI changes:** styling is token-based; after editing template classes, rebuild
  CSS with `make tailwind`. Keep new UI work in the modern templates, not the
  frozen `app/templates/classic/` escape hatch.
