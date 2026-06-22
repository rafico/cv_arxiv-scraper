# ArXiv CV Scraper

**Your personal "daily papers" feed for computer vision research.**

Stop drowning in the arXiv firehose. This tool matches papers against authors, labs, and topics you care about, ranks them, and shows you what's worth reading — in a clean local web dashboard.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Flask](https://img.shields.io/badge/flask-3.1-green)
![License](https://img.shields.io/badge/license-MIT-gray)

![Papers dashboard](app/static/help/papers_dashboard.png)

---

## Quick Start

```bash
git clone https://github.com/rafico/cv_arxiv-scraper.git
cd cv_arxiv-scraper
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
cp config.example.yaml config.yaml
python run.py --debug
```

Open **http://127.0.0.1:5000** and click **Run Scrape** (top bar). The first run
takes ~30–60s (it fetches and ranks today's arXiv feed); an empty Inbox *before*
you scrape is normal. Matched papers then appear in the Inbox, ranked by score.

The interface is a left **sidebar** (Inbox and Saved with counts, your collections, saved searches, and filter groups) plus a **top bar** (search, sort, density toggle, and Run Scrape). The inbox is a dense, keyboard-driven triage list by default — save or skip with `s`/`x`, expand a row with `d` — and you can switch to a **Visual grid** that browses papers by their first-page teaser figure.

Prefer the older look? A **Classic UI** link at the bottom of the sidebar switches the whole app back to the pre-redesign interface; a **New UI** link in the classic top bar switches back. The choice is remembered per browser.

If you skip the copy step, the app will run from `config.example.yaml` defaults and create `instance/config.yaml` only after your first saved change.

> **Note:** this app has **no authentication** and is designed for single-user localhost use. It refuses to bind to a non-loopback address unless you pass `--expose`, which should only be used behind a reverse-proxy that adds its own auth.

Docker Compose follows the same local-only default by publishing the container as
`127.0.0.1:5000:5000`. Run `cp config.example.yaml config.yaml` first — Compose
bind-mounts that file, so `docker compose up` fails if it doesn't exist. If you
intentionally publish it on a network interface, put it behind an authenticated
reverse proxy first.

---

## Tell it what you care about

Copy `config.example.yaml` to `config.yaml`, then go to **Settings > Research Setup** in the web UI or edit the file directly:

```yaml
whitelists:
  titles:
    - "Few Shot"
    - "Remote Sensing"
  affiliations:
    - "Stanford"
    - "DeepMind"
  authors:
    - "Fei-Fei"
    - "Yann LeCun"
```

---

## Features

**Finding papers**
- Scrapes arXiv daily (or on demand) and matches against your interests
- Hybrid search — keyword, semantic (SPECTER2), or combined
- Historical sync — backfill papers from any date range
- Multiple feed sources — monitor categories beyond cs.CV

**Smart ranking**
- Personalized scoring based on authors, labs, topics, recency, citations, and your feedback
- Ranking explanations show *why* each paper was surfaced
- AI relevance scoring (optional, requires LLM setup)

**TL;DR summaries**
- Expand any paper (More details, or press `d`) to read a short summary
- Without LLM: extractive summary from the abstract (no API needed)
- With LLM enabled: AI-generated plain-language TL;DR describing what the paper does and why it matters
- Configurable number of visible lines via Settings (default: 3)

**Organization**
- Save, skip, prioritize, or share papers to train future rankings
- Collections, custom tags, notes, and reading status tracking
- Saved searches with custom filter criteria

**Export & sync**
- BibTeX export (single paper or bulk)
- Mendeley and Zotero sync
- HTML report export
- Daily email digest via Gmail

**Enrichment**
- Citation counts from Semantic Scholar and OpenAlex
- Topic classifications and open-access status from OpenAlex
- PDF thumbnails and related-paper recommendations
- Corpus analytics — topic clusters and emerging trends

---

## CLI Commands

After `pip install -e .`:

| Command | What it does |
|---|---|
| `cv-arxiv-scrape` | One-shot scrape, prints matches to terminal |
| `cv-arxiv-digest` | Send email digest (`--dry-run`, `--send-only`) |
| `cv-arxiv-sync` | Historical sync (`--from`, `--to`, `--category`) |
| `cv-arxiv-backfill` | Enrichment backfills (`embeddings`, `citations`, `openalex`, `thumbnails`, `all`) |

Standalone scripts (`python scrape_cli.py`, `python export_cli.py`, etc.) also work without installing once the environment is active.

---

## Email Digest Setup (optional)

The app works fully without email — this is only for a daily digest to your inbox.

1. In the [Google Cloud Console](https://console.cloud.google.com/), create an
   OAuth client (type **Desktop app**) with the **Gmail API** enabled, and
   download its `credentials.json`.
2. Upload that file in **Settings**, or save it at the repo root as `credentials.json`.
3. Authorize: run `python gmail_auth_setup.py` (or click through the flow in Settings).
4. Set your recipient in `config.yaml` under `email.recipient`.
5. Test with `cv-arxiv-digest --dry-run`, then send the real thing with `cv-arxiv-digest`.

Only the `gmail.send` scope is requested — the app cannot read your emails.

---

## AI summaries & relevance (optional)

LLM features are **off by default** (`llm.enabled: false` in `config.yaml`). With
them off you still get an extractive TL;DR pulled from each abstract — no API or
model needed.

To enable AI-generated summaries and relevance scoring, set `llm.enabled: true` and
pick a provider in `config.yaml`:

- **Local Ollama** (default settings, no key) — `provider: ollama`,
  `base_url: http://localhost:11434/v1`. Install [Ollama](https://ollama.com/) and
  pull the model named in `llm.model`.
- **OpenRouter** (hosted) — `provider: openrouter` plus an `OPENROUTER_API_KEY`
  (see [`.env.example`](.env.example); get a key at https://openrouter.ai/keys).

## Optional enrichment

Both are optional and the app degrades gracefully without them:

- **GitHub** repo stars/license — set a `GITHUB_TOKEN` (or `github.token` in
  `config.yaml`) to raise the rate limit from 60 to 5000 req/hr. Without it, repo
  enrichment is just capped per run.
- **OpenAlex** citations/topics — set `openalex.email` to a contact address (their
  polite-pool courtesy); it works without one.

---

## Troubleshooting

- **"Address already in use" / port 5000 busy** — another app holds the port. Run
  on another with `PORT=5001 python run.py --debug`.
- **First scrape feels slow / hangs for ~30s** — importing
  `faiss`/`sentence-transformers` is heavy on first load, and the first scrape
  fetches PDFs. This is normal; it's faster afterward.
- **A few papers log PDF-extraction warnings** — non-fatal. The paper is still
  ingested; only its thumbnail/section extraction is skipped.
- **No papers after a scrape** — your whitelists may not match today's feed. Widen
  them in **Settings > Research Setup** (or `config.yaml`) and scrape again.

---

## Development

Want to extend or contribute? See **[CONTRIBUTING.md](CONTRIBUTING.md)** for setup,
the test commands, and how the code is laid out. Run `make help` for every command.

---

## API

Full REST API at `/api/`. Key endpoints:

| Area | Endpoints |
|---|---|
| Scraping | `POST /api/scrape`, `GET /api/scrape/stream` |
| Search | `GET /api/search?q=...&mode=hybrid` |
| Papers | `/api/papers/<id>/feedback`, `explain`, `notes`, `tags`, `bibtex` |
| Collections | `GET/POST /api/collections`, manage papers in collections |
| Saved searches | `GET/POST /api/saved-searches`, `POST .../run` |
| Corpus | `/api/corpus/clusters`, `emerging`, `neighbors` |
| Export | `GET /api/export`, `GET /api/export/bibtex` |
| Feed sources | `GET/POST /api/feed-sources` |

See the in-app help at `/help` for full documentation.

---

## Testing

```bash
python -m pip install -e ".[dev]"
pre-commit install          # enable lint/format/credential hooks on commit
python -m pytest tests/ -v
```

---

## License

[MIT](LICENSE)
