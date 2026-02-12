# ArXiv CV Scraper

A Flask app that monitors the [arXiv](https://arxiv.org) `cs.CV` RSS feed and surfaces papers that match your interests by author, affiliation, and title/abstract keywords.

![Python](https://img.shields.io/badge/python-3.9+-blue)
![Flask](https://img.shields.io/badge/flask-3.1-green)
![License](https://img.shields.io/badge/license-MIT-gray)

## Features

- Author, affiliation, and title/abstract matching with normalized regex logic
- Match priority ordering: `Author > Affiliation > Title`
- Compound matches (for example: `Author + Affiliation`)
- Parallel PDF processing with configurable workers
- Live scrape progress in the UI via Server-Sent Events (SSE)
- Persistent SQLite storage with duplicate detection by arXiv link
- YAML-based configuration + editable settings page
- CLI mode for one-shot runs (cron-friendly)

## Quick Start

From an existing checkout:

```bash
cd cv_arxiv-scraper
~/venv/bin/pip install -r requirements.txt
~/venv/bin/python run.py
```

Open `http://127.0.0.1:5000` and click **Run Scrape**.

If `~/venv` does not exist yet:

```bash
python3 -m venv ~/venv
~/venv/bin/pip install -r requirements.txt
```

## CLI Usage

```bash
~/venv/bin/python arxiv.py
```

This runs a single scrape, stores new matches in SQLite, and prints matched papers to the terminal.

## Configuration

Configuration is read from `config.yaml`.

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

Notes:

- `titles` are matched against both paper title and abstract.
- `authors` are matched against parsed individual author names.
- `affiliations` are matched against extracted first-page PDF text lines (`pdf_lines_start` to `pdf_lines_end`).

## Web and API

- Dashboard: `GET /`
- Settings page: `GET /settings`, `POST /settings`
- Trigger scrape (JSON): `POST /api/scrape`
- Trigger scrape (streaming SSE): `GET /api/scrape/stream`

SSE events emitted by `/api/scrape/stream`:

- `status`
- `feed`
- `progress`
- `match`
- `done`

## Project Structure

```text
.
├── app/
│   ├── __init__.py
│   ├── models.py
│   ├── scraper.py
│   ├── routes/
│   │   ├── api.py
│   │   ├── dashboard.py
│   │   └── settings.py
│   └── templates/
├── config.yaml
├── arxiv.py
├── run.py
├── requirements.txt
└── instance/                 # auto-created on first run
    └── arxiv_papers.db       # SQLite database
```

## Validation Commands

```bash
~/venv/bin/python -m compileall app arxiv.py run.py
```

## Dependencies

- Flask
- Flask-SQLAlchemy
- feedparser
- PyPDF2
- requests
- PyYAML
- tqdm
