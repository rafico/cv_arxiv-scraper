# ArXiv CV Scraper

**Your personal "daily papers" feed for computer vision research.**

Stop drowning in the arXiv firehose. This tool scrapes the [arXiv cs.CV](https://arxiv.org/list/cs.CV/recent) RSS feed, matches papers against authors, affiliations, and topics you care about, then ranks and surfaces the ones worth reading — all in a clean web dashboard you run locally.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Flask](https://img.shields.io/badge/flask-3.1-green)
![License](https://img.shields.io/badge/license-MIT-gray)

---

## Why use this?

- **Personalized ranking** — Papers are scored by how well they match your interests (authors you follow, labs you track, topics you study), not just recency.
- **Ranking explanations** — Each paper shows *why* it was recommended: matched authors, citation counts, similarity to saved papers, and more.
- **Hybrid search** — Find papers using keyword (BM25), semantic (SPECTER2 embeddings), or combined search modes.
- **Semantic paper similarity** — Related papers are found using SPECTER2 scientific document embeddings via a FAISS vector index — much higher quality than keyword matching.
- **OpenAlex enrichment** — Papers are enriched with topic classifications, open-access status, and additional citation data from OpenAlex.
- **Corpus analytics** — Discover topic clusters, emerging trends, and neighbor papers across your library.
- **Zero signup, local-first** — Core scraping and ranking run locally. Gmail digest and optional LLM features require extra setup only if you want them.
- **Rich triage workflow** — Save, skip, mark priority, or share papers. Your feedback tunes future rankings so the best papers rise to the top.
- **Collections & organization** — Group papers into color-coded collections, add custom tags, set reading status, and attach notes.
- **Saved searches** — Bookmark filter combinations with custom criteria (categories, keywords, authors, date windows, citation thresholds) and re-run them instantly.
- **Auto-generated summaries & tags** — Each matched paper gets a short plain-language summary and topic tags so you can scan faster.
- **Flexible time windows** — Browse today's papers, this week's, this month's, or everything.
- **Daily email digest** — Get matched papers delivered to your inbox via Gmail (OAuth2, send-only scope).
- **Reference manager sync** — Export BibTeX, or sync directly with Mendeley and Zotero.
- **Multiple feed sources** — Monitor additional arXiv categories or custom RSS feeds beyond cs.CV.
- **Historical sync** — Backfill papers from any arXiv date range, not just the daily feed.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/<your-username>/cv_arxiv-scraper.git
cd cv_arxiv-scraper
pip install -e .

# 2. Launch the web dashboard
python run.py
```

Open **http://127.0.0.1:5000** in your browser and click **Run Scrape** to pull the latest papers.

### Installable CLI commands

After `pip install -e .`, the following commands are available globally:

| Command | Description |
|---|---|
| `cv-arxiv-scrape` | One-shot scrape — prints matched papers to your terminal |
| `cv-arxiv-digest` | Build and send the daily email digest |
| `cv-arxiv-sync` | Sync historical papers over a date range |
| `cv-arxiv-backfill` | Run selective enrichment backfills |

### Standalone CLI scripts

These scripts work without installing the package:

```bash
python run.py                    # Launch web dashboard
python scrape_cli.py             # One-shot scrape (same as cv-arxiv-scrape)
python digest_cli.py             # Email digest
python sync_cli.py               # Historical sync
python backfill_cli.py           # Enrichment backfills
python export_cli.py             # Export papers to HTML report
python gmail_auth_setup.py       # One-time Gmail OAuth setup
```

---

## Tell it what you care about

Edit `config.yaml` to define the authors, labs, and topics that matter to you:

```yaml
whitelists:
  # Papers with these keywords in the title are boosted
  titles:
    - "Few Shot"
    - "Remote Sensing"
    - "NeRF"

  # Papers from these institutions are boosted
  affiliations:
    - "Stanford"
    - "DeepMind"
    - "INRIA"

  # Papers by these researchers are boosted
  authors:
    - "Fei-Fei"
    - "Yann LeCun"
```

You can also edit these lists from the web UI at **http://127.0.0.1:5000/settings** — no need to touch YAML if you don't want to.

### Scraper settings

```yaml
scraper:
  feed_url: "https://rss.arxiv.org/rss/cs.CV"   # arXiv RSS feed to monitor
  max_workers: 8          # parallel PDF download threads
  pdf_smart_header: true  # extract affiliations from PDF headers
```

### OpenAlex enrichment

```yaml
openalex:
  enabled: true
  email: ""              # Set your email for the polite pool (higher rate limits)
```

---

## Semantic Search & Embeddings

Papers are embedded using [SPECTER2](https://huggingface.co/allenai/specter2), a scientific document embedding model trained on citation graphs. These embeddings power:

- **Related papers** — High-quality similarity computed from paper meaning, not just shared keywords
- **Semantic search** — Find papers by concept even when the exact terms don't match
- **Hybrid search** — Combines keyword (BM25 via SQLite FTS5) and semantic search using Reciprocal Rank Fusion
- **Corpus analytics** — Topic clustering and emerging-trend detection

Embeddings are stored in a FAISS sidecar index alongside the SQLite database (`instance/faiss_index/`).

The first scrape will download the SPECTER2 model (~420MB) to `~/.cache/huggingface/`. Subsequent runs load it from cache.

To backfill embeddings for existing papers:
```bash
cv-arxiv-backfill embeddings
```

To fully rebuild the semantic index from scratch:
```bash
cv-arxiv-backfill index-rebuild
```

---

## Enrichment Backfills

Use the backfill CLI to enrich papers that were added before a feature was available:

```bash
cv-arxiv-backfill embeddings       # Backfill missing SPECTER2 embeddings
cv-arxiv-backfill index-rebuild    # Rebuild the full FAISS index from the database
cv-arxiv-backfill citations        # Fetch citation counts from Semantic Scholar
cv-arxiv-backfill openalex         # Fetch OpenAlex metadata (topics, OA status)
cv-arxiv-backfill thumbnails       # Generate PDF first-page thumbnails
cv-arxiv-backfill all              # Run all of the above
```

Options: `--batch-size N`, `--delay SECONDS` (rate-limit delay between batches).

---

## Historical Sync

Backfill papers from any arXiv date range, not just the daily feed:

```bash
cv-arxiv-sync --category cs.CV --from 2024-01-01 --to 2024-03-31
```

The sync processes dates in weekly chunks with progress tracking. Options: `--chunk-days N`.

You can also use the **Discover** page in the web UI to search arXiv by date range and category interactively.

---

## Daily Email Digest

Get a daily email with your matched papers — no need to open the dashboard.

### Setup

1. **Create a Google Cloud OAuth client** — Go to the [Google Cloud Console](https://console.cloud.google.com/), create a project, enable the **Gmail API**, and create an **OAuth 2.0 Client ID** (type: Web application). Download the JSON file and either upload it in **Settings** or save it as `credentials.json` in the project root.

2. **Run the one-time auth flow:**
   ```bash
   python gmail_auth_setup.py
   ```
   Most users can complete this from **Settings** in the web UI. The CLI flow above is the fallback for headless environments. The resulting `token.json` is saved locally with restricted file permissions (`600`). The app requests only the `gmail.send` scope — it **cannot** read, list, or delete your emails.

3. **Set your recipient in `config.yaml`:**
   ```yaml
   email:
     recipient: "you@example.com"
     subject_prefix: "ArXiv Digest"   # optional
   ```

4. **Test it:**
   ```bash
   cv-arxiv-digest --dry-run   # preview without sending
   cv-arxiv-digest              # scrape + send
   cv-arxiv-digest --send-only  # send from existing papers (no scrape)
   ```

5. **Schedule with cron** (e.g. every day at 8 AM):
   ```
   0 8 * * * cd /path/to/cv_arxiv-scraper && ~/venv/bin/python -m app.cli.digest
   ```

### Security notes

- Uses **OAuth2** (not app passwords) — Google's recommended approach.
- Token scope is limited to `gmail.send` only.
- `credentials.json` and `token.json` are gitignored and stored with `chmod 600`.
- All paper content is HTML-escaped before rendering in email bodies.

---

## Reference Manager Sync

Export your papers to external reference managers:

- **BibTeX** — Click the BibTeX button on any paper card for a single entry, or use the toolbar button to export all visible papers. Also available via API: `GET /api/export/bibtex`.
- **Mendeley** — Connect via OAuth in **Settings > Automation** and use **Sync Saved Papers**.
- **Zotero** — Enter your Zotero API key and user ID in **Settings > Automation** to sync to a specific collection.

---

## How ranking works

Each paper receives a composite score based on:

| Signal | Weight | Example |
|---|---|---|
| **Author match** | High | A paper by someone in your `authors` list |
| **Affiliation match** | Medium | Paper from a lab in your `affiliations` list |
| **Title keyword match** | Lower | Title contains a term from your `titles` list |
| **Recency** | Decay | Newer papers get a gentle boost (14-day half-life) |
| **Your feedback** | Adaptive | Saved/priority papers are boosted; skipped papers sink |
| **Citations** | Log-scaled | Citation counts from Semantic Scholar + OpenAlex |
| **AI relevance** | Optional | LLM-rated relevance score when enabled |

Multiple matches stack — a paper by a tracked author at a tracked lab on a tracked topic ranks highest.

Each paper also displays **ranking explanations** — human-readable reasons like "Matched author: Kaiming He", "Highly cited (342 citations)", or "Similar to saved: NeRF in the Wild".

---

## Dashboard features

- **Hybrid search** — Search by keyword, semantic meaning, or both (toggle between modes)
- **Filter by match type** — Show only Author / Affiliation / Title matches, or all
- **Advanced filters** — Filter by arXiv category, resource availability, reading status, or show hidden papers
- **Sort** — By score (trending), date (newest), recommendations, or citations
- **Time windows** — Daily, weekly, monthly, or all-time views
- **Feedback buttons** — Save, Priority, Skip, or Share papers to train rankings
- **Ranking explanations** — See why each paper was recommended
- **OpenAlex enrichment** — Topic badges and open-access status indicators
- **Collections** — Organize papers into color-coded reading lists
- **Notes & tags** — Annotate papers with personal notes and custom tags
- **Reading status** — Track papers as To Read, Reading, or Read
- **Saved searches** — Bookmark filter combinations and re-run them as quick-access pills
- **Bulk operations** — Select multiple papers and save or skip them all at once
- **BibTeX export** — Export individual papers or entire views as BibTeX
- **Follow / Mute** — Follow an author or mute a topic directly from a paper card
- **Thumbnails** — PDF first-page previews for visual scanning
- **Metadata chips** — See categories, arXiv comments, DOI links, and resource badges at a glance
- **Pagination** — Browse large result sets without slowdowns
- **Keyboard shortcuts** — Navigate and triage papers without touching the mouse

---

## API Reference

### Core

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard with search, filters, and pagination |
| `/discover` | GET | Historical arXiv search by date range |
| `/settings` | GET/POST | Whitelist and config editor |
| `/help` | GET | In-app documentation |

### Scraping

| Endpoint | Method | Description |
|---|---|---|
| `/api/scrape` | POST | Start (or join) a background scrape job |
| `/api/scrape/status` | GET | Current scrape job status |
| `/api/scrape/stream?job_id=...` | GET | SSE event stream from the active scrape |
| `/api/search/historical` | POST | Trigger a historical date-range scrape |

### Search

| Endpoint | Method | Description |
|---|---|---|
| `/api/search?q=...&mode=hybrid` | GET | Hybrid search (keyword / semantic / hybrid) |
| `/api/authors?q=...` | GET | Author name autocomplete |

### Paper actions

| Endpoint | Method | Description |
|---|---|---|
| `/api/papers/<id>/feedback` | POST | Toggle feedback action (save/skip/priority/shared) |
| `/api/papers/<id>/explain` | GET | Get ranking explanations for a paper |
| `/api/papers/<id>/reading-status` | POST | Set reading status (to_read/reading/read) |
| `/api/papers/<id>/notes` | PUT | Update paper notes |
| `/api/papers/<id>/tags` | POST/DELETE | Add or remove user tags |
| `/api/papers/<id>/follow` | POST | Follow the paper's first author |
| `/api/papers/<id>/mute` | POST | Mute the paper's primary topic |
| `/api/papers/<id>/bibtex` | GET | Get BibTeX entry for a single paper |
| `/api/papers/<id>/graph` | GET | Citation/similarity graph data |
| `/api/papers/bulk-feedback` | POST | Batch save/skip multiple papers |
| `/api/papers/bulk-bibtex?ids=...` | GET | BibTeX export for selected papers |

### Collections

| Endpoint | Method | Description |
|---|---|---|
| `/api/collections` | GET | List all collections with paper counts |
| `/api/collections` | POST | Create a new collection |
| `/api/collections/<id>` | PUT | Update collection name/description/color |
| `/api/collections/<id>` | DELETE | Delete a collection |
| `/api/collections/<id>/papers` | POST | Add paper(s) to a collection |
| `/api/collections/<id>/papers/<pid>` | DELETE | Remove a paper from a collection |

### Saved Searches

| Endpoint | Method | Description |
|---|---|---|
| `/api/saved-searches` | GET/POST | List or create saved searches |
| `/api/saved-searches/<id>` | GET/PUT/DELETE | Retrieve, update, or delete a saved search |
| `/api/saved-searches/<id>/run` | POST | Execute a saved search and return results |

### Corpus Analytics

| Endpoint | Method | Description |
|---|---|---|
| `/api/corpus/clusters` | GET | Topic clusters for a time window |
| `/api/corpus/emerging` | GET | Detect emerging topics vs. baseline |
| `/api/corpus/neighbors` | GET | Find neighbor papers by seed papers or collection |

### Export & Feed Sources

| Endpoint | Method | Description |
|---|---|---|
| `/api/export` | GET | Export papers as standalone HTML report |
| `/api/export/bibtex` | GET | Export papers as BibTeX file |
| `/api/feed-sources` | GET/POST | List or add RSS feed sources |
| `/api/feed-sources/<id>` | DELETE | Remove a feed source |
| `/api/feed-sources/<id>/toggle` | POST | Enable/disable a feed source |

**SSE event types:** `status`, `feed`, `progress`, `match`, `done`, `scrape_error`

---

## Project Structure

```
.
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── models.py                # SQLAlchemy models
│   ├── schema.py                # DB migrations / indexes / FTS5
│   ├── enums.py                 # Enums (FeedbackAction, SortOption, etc.)
│   ├── constants.py             # Shared constants
│   ├── csrf.py                  # CSRF token management
│   ├── scraper.py               # Backward-compatible scraping facade
│   ├── cli/                     # Installable CLI entry points
│   │   ├── scrape.py            # cv-arxiv-scrape
│   │   ├── digest.py            # cv-arxiv-digest
│   │   ├── sync.py              # cv-arxiv-sync (historical)
│   │   └── backfill.py          # cv-arxiv-backfill (enrichment)
│   ├── ingest/                  # Paper ingestion pipeline
│   │   ├── scrape_engine.py     # Core scrape + dedup logic
│   │   ├── orchestrator.py      # Multi-source orchestration
│   │   └── http_client.py       # HTTP session factory
│   ├── enrich/                  # Enrichment providers
│   │   ├── citations.py         # Semantic Scholar citations
│   │   └── openalex.py          # OpenAlex metadata
│   ├── search_/                 # Search & embedding layer
│   │   ├── embeddings.py        # SPECTER2 + FAISS index
│   │   ├── embed_backfill.py    # Backfill embeddings
│   │   ├── text.py              # Text utilities
│   │   └── thumbnail_generator.py
│   ├── rank/                    # Ranking subsystem
│   │   └── preferences.py       # User preference weights
│   ├── web/                     # Web-layer helpers
│   │   ├── email_digest.py      # Gmail digest build + send
│   │   └── scheduler.py         # Scheduled scrape support
│   ├── services/                # Business logic services
│   │   ├── scrape_engine.py     # Legacy scrape engine
│   │   ├── embeddings.py        # Embedding service
│   │   ├── search.py            # Hybrid search (BM25 + semantic)
│   │   ├── matching.py          # Whitelist matching engine
│   │   ├── ranking.py           # Scoring, explanations, feedback
│   │   ├── feedback.py          # User feedback handling
│   │   ├── related.py           # Related-paper recommendations
│   │   ├── corpus_analysis.py   # Topic clusters & emerging trends
│   │   ├── saved_search.py      # Saved search execution
│   │   ├── recommendations.py   # Follow/mute recommendations
│   │   ├── export.py            # HTML report generation
│   │   ├── bibtex.py            # BibTeX generation
│   │   ├── summary.py           # Auto-generated summaries & tags
│   │   ├── enrichment.py        # Metadata enrichment
│   │   ├── citations.py         # Citation fetching
│   │   ├── openalex.py          # OpenAlex enrichment
│   │   ├── llm_client.py        # LLM integration
│   │   ├── mendeley.py          # Mendeley sync
│   │   ├── zotero.py            # Zotero sync
│   │   ├── jobs.py              # Background job manager (SSE)
│   │   ├── scheduler.py         # Cron-like scheduled scrapes
│   │   ├── email_digest.py      # Legacy digest alias
│   │   └── ...
│   ├── routes/                  # Flask blueprints
│   │   ├── dashboard.py         # Main papers view
│   │   ├── api.py               # REST API endpoints
│   │   ├── discover.py          # Historical search UI
│   │   ├── settings.py          # Config editor
│   │   └── help.py              # In-app documentation
│   └── templates/               # Jinja2 HTML templates
│       └── help/                # Help documentation pages
├── run.py                       # Web server entry point
├── scrape_cli.py                # Standalone scrape CLI
├── digest_cli.py                # Standalone digest CLI
├── sync_cli.py                  # Standalone sync CLI
├── backfill_cli.py              # Standalone backfill CLI
├── export_cli.py                # Standalone export CLI
├── gmail_auth_setup.py          # One-time Gmail OAuth setup
├── config.yaml                  # Your interests & scraper settings
├── pyproject.toml               # Package config & CLI entry points
├── requirements.txt             # Runtime dependencies
└── tests/
```

---

## Testing

```bash
python -m pytest tests/ -v
```

---

## License

MIT
