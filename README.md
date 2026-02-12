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

## UI Snippets

These are representative snippets from the current UI (`app/templates/*.html`) you can reuse or tweak.

### Match badges + paper card

```html
<div class="paper-card bg-white rounded-xl shadow-sm border border-gray-200 p-5">
  <div class="flex items-center gap-1.5 flex-wrap mb-3">
    {% for mtype in paper.match_type.split(' + ') %}
      {% if mtype == 'Author' %}
      <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-100 text-blue-800">
        Author
      </span>
      {% elif mtype == 'Affiliation' %}
      <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800">
        Affiliation
      </span>
      {% else %}
      <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-purple-100 text-purple-800">
        Title
      </span>
      {% endif %}
    {% endfor %}
  </div>
  <h3 class="text-sm font-semibold text-gray-900 mb-2">{{ paper.title }}</h3>
  <p class="text-xs text-gray-500">{{ paper.authors }}</p>
</div>
```

### Live scrape progress (SSE)

```javascript
const source = new EventSource('/api/scrape/stream');

source.addEventListener('progress', (e) => {
  const d = JSON.parse(e.data);
  const pct = Math.round((d.processed / d.total) * 100);
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-pct').textContent = pct + '%';
});

source.addEventListener('match', (e) => {
  const d = JSON.parse(e.data);
  document.getElementById('progress-matches').textContent = `${d.matched} matches`;
});
```

### Filter/search row

```html
<form method="GET" action="/" class="mb-8">
  <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-4">
    <div class="flex flex-col sm:flex-row gap-3 items-end">
      <input type="text" name="q" placeholder="Search titles, authors, terms..."
             class="w-full rounded-lg border-gray-300 border px-3 py-2 text-sm">
      <select name="date"
              class="w-full sm:w-44 rounded-lg border-gray-300 border px-3 py-2 text-sm">
        <option value="">All Dates</option>
      </select>
    </div>
  </div>
</form>
```

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
