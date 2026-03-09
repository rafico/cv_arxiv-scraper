# ArXiv CV Scraper

**Your personal "daily papers" feed for computer vision research.**

Stop drowning in the arXiv firehose. This tool scrapes the [arXiv cs.CV](https://arxiv.org/list/cs.CV/recent) RSS feed, matches papers against authors, affiliations, and topics you care about, then ranks and surfaces the ones worth reading — all in a clean web dashboard you run locally.

![Python](https://img.shields.io/badge/python-3.9+-blue)
![Flask](https://img.shields.io/badge/flask-3.1-green)
![License](https://img.shields.io/badge/license-MIT-gray)

---

## Why use this?

- **Personalized ranking** — Papers are scored by how well they match your interests (authors you follow, labs you track, topics you study), not just recency.
- **Zero signup, fully local** — No accounts, no cloud, no API keys. Install, configure your interests, and go.
- **Triage workflow** — Upvote, save, or skip papers. Your feedback tunes future rankings so the best papers rise to the top.
- **Auto-generated summaries & tags** — Each matched paper gets a short plain-language summary and topic tags so you can scan faster.
- **Related paper recommendations** — Lightweight embedding similarity finds related work you might have missed.
- **Flexible time windows** — Browse today's papers, this week's, this month's, or everything.

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

---

## How ranking works

Each paper receives a composite score based on:

| Signal | Weight | Example |
|---|---|---|
| **Author match** | High | A paper by someone in your `authors` list |
| **Affiliation match** | Medium | Paper from a lab in your `affiliations` list |
| **Title keyword match** | Lower | Title contains a term from your `titles` list |
| **Recency** | Decay | Newer papers get a gentle boost (14-day half-life) |
| **Your feedback** | Adaptive | Upvoted/saved papers are boosted; skipped papers sink |

Multiple matches stack — a paper by a tracked author at a tracked lab on a tracked topic ranks highest.

---

## Dashboard features

- **Filter by match type** — Show only Author / Affiliation / Title matches, or all
- **Sort** — By score (trending) or by date (newest first)
- **Time windows** — Daily, weekly, monthly, or all-time views
- **Search** — Free-text search across titles and authors
- **Feedback buttons** — Upvote, save, or skip each paper to train your rankings
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
| `/api/papers/<id>/feedback` | POST | Toggle `upvote` / `save` / `skip` on a paper |

**SSE event types:** `status`, `feed`, `progress`, `match`, `done`, `scrape_error`

---

## Project Structure

```
.
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── models.py                # SQLAlchemy models
│   ├── schema.py                # DB migrations / indexes
│   ├── scraper.py               # Backward-compatible scraping facade
│   ├── services/
│   │   ├── scrape_engine.py     # Core scrape + dedup logic
│   │   ├── jobs.py              # Background job manager (SSE)
│   │   ├── matching.py          # Whitelist matching engine
│   │   ├── ranking.py           # Composite scoring algorithm
│   │   ├── feedback.py          # User feedback handling
│   │   ├── enrichment.py        # Metadata enrichment (DOI, categories)
│   │   ├── summary.py           # Auto-generated summaries & tags
│   │   └── related.py           # Related-paper recommendations
│   ├── routes/                  # Flask blueprints
│   └── templates/               # Jinja2 HTML templates
├── arxiv.py                     # CLI entry point
├── config.yaml                  # Your interests & scraper settings
├── run.py                       # Web server entry point
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
