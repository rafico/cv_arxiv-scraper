# ArXiv CV Scraper

A Flask app that monitors the [arXiv](https://arxiv.org) `cs.CV` RSS feed and surfaces papers matching your interests by author, affiliation, and title/abstract keywords.

![Python](https://img.shields.io/badge/python-3.9+-blue)
![Flask](https://img.shields.io/badge/flask-3.1-green)
![License](https://img.shields.io/badge/license-MIT-gray)

## Features

- **Three-tier matching** — author names, affiliations (extracted from PDFs), and title/abstract keywords
- **Match priority** — Author > Affiliation > Title, with compound matches (e.g. Author + Affiliation)
- **Parallel PDF processing** with configurable worker count
- **Live progress** in the web UI via Server-Sent Events
- **Persistent SQLite storage** with duplicate detection
- **YAML config** + editable settings page in the browser
- **CLI mode** for one-shot cron-friendly runs

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

Open `http://127.0.0.1:5000` and click **Run Scrape**.

### CLI usage

```bash
python arxiv.py
```

Runs a single scrape, stores matches in SQLite, and prints results to the terminal.

## Configuration

Edit `config.yaml`:

```yaml
scraper:
  feed_url: "https://rss.arxiv.org/rss/cs.CV"
  max_workers: 8
  pdf_lines_start: 2
  pdf_lines_end: 30

whitelists:
  titles:
    - "Few Shot"
    - "Remote Sensing"
  affiliations:
    - "Stanford"
    - "DeepMind"
  authors:
    - "Fei-Fei"
    - "Kaiming"
```

- **titles** match against paper title and abstract
- **authors** match against parsed individual author names (accent-normalized)
- **affiliations** match against first-page PDF text (lines `pdf_lines_start` to `pdf_lines_end`)

Settings can also be edited live at `/settings` in the web UI.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard with filters and search |
| `/settings` | GET/POST | View and edit whitelists |
| `/api/scrape` | POST | Trigger scrape (JSON response) |
| `/api/scrape/stream` | GET | Trigger scrape (SSE stream) |

SSE events: `status`, `feed`, `progress`, `match`, `done`

## Project Structure

```
.
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── models.py            # Paper model (SQLAlchemy)
│   ├── scraper.py           # Feed parsing, PDF extraction, matching
│   ├── routes/
│   │   ├── api.py           # REST + SSE endpoints
│   │   ├── dashboard.py     # Main UI
│   │   └── settings.py      # Config editor
│   └── templates/
├── config.yaml
├── arxiv.py                 # CLI entry point
├── run.py                   # Web server entry point
└── requirements.txt
```

## Dependencies

Flask, Flask-SQLAlchemy, feedparser, PyPDF2, requests, PyYAML, tqdm
