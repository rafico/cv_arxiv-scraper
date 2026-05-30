# CLAUDE.md

This project's agent guidance lives in **[AGENTS.md](AGENTS.md)** (the cross-tool
standard). Read it first — it covers setup, run/test commands, the architecture
map, and the critical constraints (no auth, single worker, runtime config writes).

Subsystem guides:
- [app/routes/AGENTS.md](app/routes/AGENTS.md) — HTTP/Blueprint layer
- [app/services/AGENTS.md](app/services/AGENTS.md) — business logic
- [app/services/ingest/AGENTS.md](app/services/ingest/AGENTS.md) — ingestion backends
- [tests/AGENTS.md](tests/AGENTS.md) — test conventions

Deeper architecture & extension recipes: [ARCHITECTURE.md](ARCHITECTURE.md).

Canonical commands are in the [Makefile](Makefile) (`make help`); they mirror CI.
