# ArXiv CV Scraper

A Flask app that tracks [arXiv](https://arxiv.org) `cs.CV` papers and ranks them like a local "daily papers" feed with triage controls.

![Python](https://img.shields.io/badge/python-3.9+-blue)
![Flask](https://img.shields.io/badge/flask-3.1-green)
![License](https://img.shields.io/badge/license-MIT-gray)

## What Changed

- Modular service architecture (`app/services/*`) for matching, enrichment, ranking, feedback, summaries, related papers, and job orchestration
- Real `paper_score` ranking instead of fixed match-priority sorting
- Normalized date columns (`publication_dt`, `scraped_at`) and new indexes for faster filtering/sorting
- Time windows: `daily`, `weekly`, `monthly`, `all`
- Trending/newest sort modes
- Feedback loop: `upvote`, `save`, `skip` actions that re-rank papers
- Metadata enrichment: categories, arXiv comments/DOI-derived links, resource chips
- Auto-generated short summaries + topic tags
- Related-paper recommendations using lightweight embedding similarity
- Background scrape jobs with overlap protection and SSE progress replay
- Pagination and compound match badge fixes in UI

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

Runs one scrape and prints matched papers with score/summary details.

## Configuration

Edit `config.yaml`:

```yaml
scraper:
  feed_url: "https://rss.arxiv.org/rss/cs.CV"
  max_workers: 8
  pdf_attempts: 2
  pdf_lines_start: 2
  pdf_max_header_lines: 50
  pdf_smart_header: true

whitelists:
  titles:
    - "Few Shot"
  affiliations:
    - "Stanford"
  authors:
    - "Fei-Fei"
```

Settings can also be edited in the web UI at `/settings`.

## Agent Collaboration

If you are alternating between Codex and the Claude VS Code extension, use
`AGENT_HANDOFF.md` as the source of truth for in-progress work.

- Read the latest entry before starting.
- Append a new top entry after each work chunk.
- Record files touched, commands/tests run, decisions, and the next step.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard with search/filters/pagination |
| `/settings` | GET/POST | Whitelist editor |
| `/api/scrape` | POST | Start (or join) background scrape job |
| `/api/scrape/stream` | GET | SSE events from active scrape job |
| `/api/papers/<id>/feedback` | POST | Toggle `upvote`/`save`/`skip` |

SSE events: `status`, `feed`, `progress`, `match`, `done`, `scrape_error`

## Project Structure

```text
.
├── app/
│   ├── __init__.py
│   ├── models.py
│   ├── schema.py
│   ├── scraper.py                 # backward-compatible facade
│   ├── services/
│   │   ├── scrape_engine.py
│   │   ├── jobs.py
│   │   ├── feedback.py
│   │   ├── enrichment.py
│   │   ├── matching.py
│   │   ├── ranking.py
│   │   ├── summary.py
│   │   └── related.py
│   ├── routes/
│   └── templates/
├── arxiv.py
├── config.yaml
├── run.py
└── tests/
```

## Testing

Use your local virtualenv:

```bash
source ~/venv/bin/activate
python -m unittest discover -s tests -v
```
