# ArXiv CV Scraper

**Your personal "daily papers" feed for computer vision research.**

Stop drowning in the arXiv firehose. This tool matches papers against authors, labs, and topics you care about, ranks them, and shows you what's worth reading — in a clean local web dashboard.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Flask](https://img.shields.io/badge/flask-3.1-green)
![License](https://img.shields.io/badge/license-MIT-gray)

---

## Quick Start

```bash
git clone https://github.com/<your-username>/cv_arxiv-scraper.git
cd cv_arxiv-scraper
pip install -e .
python run.py
```

Open **http://127.0.0.1:5000**, click **Run Scrape**, and you're done.

---

## Tell it what you care about

Go to **Settings > Research Setup** in the web UI, or edit `config.yaml`:

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

Standalone scripts (`python scrape_cli.py`, `python export_cli.py`, etc.) also work without installing.

---

## Email Digest Setup

1. Create a Google Cloud OAuth client with the Gmail API enabled
2. Upload credentials in **Settings** (or save as `credentials.json`)
3. Run `python gmail_auth_setup.py` (or complete setup in Settings)
4. Set your recipient in `config.yaml` under `email.recipient`
5. Test with `cv-arxiv-digest --dry-run`

Only the `gmail.send` scope is requested — the app cannot read your emails.

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
python -m pytest tests/ -v
```

---

## License

MIT
