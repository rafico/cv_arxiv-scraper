# ArXiv CV Scraper

**Your personal "daily papers" feed for computer vision research.**

Stop drowning in the arXiv firehose. This tool scrapes the [arXiv cs.CV](https://arxiv.org/list/cs.CV/recent) RSS feed, matches papers against authors, affiliations, and topics you care about, then ranks and surfaces the ones worth reading — all in a clean web dashboard you run locally.

![Python](https://img.shields.io/badge/python-3.9+-blue)
![Flask](https://img.shields.io/badge/flask-3.1-green)
![License](https://img.shields.io/badge/license-MIT-gray)

---

## Why use this?

- **Personalized ranking** — Papers are scored by how well they match your interests (authors you follow, labs you track, topics you study), not just recency.
- **Ranking explanations** — Each paper shows *why* it was recommended: matched authors, citation counts, similarity to saved papers, and more.
- **Hybrid search** — Find papers using keyword (BM25), semantic (SPECTER2 embeddings), or combined search modes.
- **Semantic paper similarity** — Related papers are found using SPECTER2 scientific document embeddings via a FAISS vector index — much higher quality than keyword matching.
- **OpenAlex enrichment** — Papers are enriched with topic classifications, open-access status, and additional citation data from OpenAlex.
- **Zero signup, local-first** — Core scraping and ranking run locally. Gmail digest and optional LLM features require extra setup only if you want them.
- **Rich triage workflow** — Save, skip, mark priority, or share papers. Your feedback tunes future rankings so the best papers rise to the top.
- **Auto-generated summaries & tags** — Each matched paper gets a short plain-language summary and topic tags so you can scan faster.
- **Flexible time windows** — Browse today's papers, this week's, this month's, or everything.
- **Daily email digest** — Get matched papers delivered to your inbox via Gmail (OAuth2, send-only scope).

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/<your-username>/cv_arxiv-scraper.git
cd cv_arxiv-scraper
pip install -r requirements.txt

# 2. Launch the web dashboard
python run.py
```

Open **http://127.0.0.1:5000** in your browser and click **Run Scrape** to pull the latest papers.

### Prefer the command line?

```bash
python arxiv.py
```

Runs a one-shot scrape and prints matched papers with scores and summaries to your terminal.

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

Embeddings are stored in a FAISS sidecar index alongside the SQLite database (`instance/faiss_index/`).

The first scrape will download the SPECTER2 model (~420MB) to `~/.cache/huggingface/`. Subsequent runs load it from cache.

To backfill embeddings for existing papers:
```bash
python -c "from app import create_app; from app.services.embed_backfill import backfill_embeddings; backfill_embeddings(create_app())"
```

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
   python digest_cli.py --dry-run   # preview without sending
   python digest_cli.py             # send now
   ```

5. **Schedule with cron** (e.g. every day at 8 AM):
   ```
   0 8 * * * cd /path/to/cv_arxiv-scraper && ~/venv/bin/python digest_cli.py
   ```

### Security notes

- Uses **OAuth2** (not app passwords) — Google's recommended approach.
- Token scope is limited to `gmail.send` only.
- `credentials.json` and `token.json` are gitignored and stored with `chmod 600`.
- All paper content is HTML-escaped before rendering in email bodies.

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
- **Sort** — By score (trending), date (newest), recommendations, or citations
- **Time windows** — Daily, weekly, monthly, or all-time views
- **Feedback buttons** — Save, Priority, Skip, or Share papers to train rankings
- **Ranking explanations** — See why each paper was recommended
- **OpenAlex enrichment** — Topic badges and open-access status indicators
- **Collections** — Organize papers into color-coded reading lists
- **Notes & tags** — Annotate papers with personal notes and custom tags
- **Metadata chips** — See categories, arXiv comments, DOI links, and resource badges at a glance
- **Pagination** — Browse large result sets without slowdowns

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard with search, filters, and pagination |
| `/settings` | GET/POST | Whitelist editor |
| `/api/scrape` | POST | Start (or join) a background scrape job |
| `/api/scrape/stream` | GET | SSE event stream from the active scrape |
| `/api/search?q=...&mode=hybrid` | GET | Hybrid search (keyword / semantic / hybrid) |
| `/api/papers/<id>/feedback` | POST | Toggle feedback action (save/skip/priority/shared) |
| `/api/papers/<id>/explain` | GET | Get ranking explanations for a paper |

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
│   ├── scraper.py               # Backward-compatible scraping facade
│   ├── services/
│   │   ├── scrape_engine.py     # Core scrape + dedup logic
│   │   ├── embeddings.py        # SPECTER2 embeddings + FAISS index
│   │   ├── embed_backfill.py    # Backfill embeddings for existing papers
│   │   ├── search.py            # Hybrid search (BM25 + semantic)
│   │   ├── openalex.py          # OpenAlex enrichment
│   │   ├── jobs.py              # Background job manager (SSE)
│   │   ├── matching.py          # Whitelist matching engine
│   │   ├── ranking.py           # Scoring, explanations, feedback weights
│   │   ├── feedback.py          # User feedback handling
│   │   ├── enrichment.py        # Metadata enrichment (DOI, categories)
│   │   ├── citations.py         # Semantic Scholar citations
│   │   ├── summary.py           # Auto-generated summaries & tags
│   │   ├── related.py           # Related-paper recommendations
│   │   └── email_digest.py      # Gmail digest (build + send)
│   ├── routes/                   # Flask blueprints
│   └── templates/                # Jinja2 HTML templates
├── arxiv.py                      # CLI entry point
├── digest_cli.py                 # Email digest CLI (for cron)
├── gmail_auth_setup.py           # One-time Gmail OAuth setup
├── config.yaml                   # Your interests & scraper settings
├── run.py                        # Web server entry point
└── tests/
```

---

## Testing

```bash
python -m pytest tests/ -v
# or with unittest:
python -m unittest discover -s tests -v
```

---

## License

MIT
