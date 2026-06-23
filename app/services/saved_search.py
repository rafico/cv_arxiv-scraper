"""SavedSearch execution engine — translates structured filters into queries."""

from __future__ import annotations

import logging
from datetime import timedelta

from app.models import Paper, SavedSearch, db, inbox_freshness_clause
from app.services.text import now_utc

LOGGER = logging.getLogger(__name__)

DEFAULT_LIMIT = 100

# Upper bounds on the numeric filters. Without these, a huge ``date_window_days``
# overflows ``timedelta`` (OverflowError) when ``/run`` builds the cutoff, and a
# huge ``min_citations`` overflows the SQLite INTEGER column on commit — both
# surfacing as a 500. 100 years of window and ~2^31 citations are far beyond any
# real value while staying inside both ranges.
MAX_DATE_WINDOW_DAYS = 36_500
MAX_MIN_CITATIONS = 2_147_483_647


def _escape_like(value: str) -> str:
    """Escape SQL LIKE wildcard characters."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def validate_saved_search(data: dict) -> list[str]:
    """Validate saved search filter fields. Returns list of error messages."""
    errors = []
    if "date_window_days" in data and data["date_window_days"] is not None:
        try:
            val = int(data["date_window_days"])
            if val < 0:
                errors.append("date_window_days must be non-negative")
            elif val > MAX_DATE_WINDOW_DAYS:
                errors.append(f"date_window_days must be at most {MAX_DATE_WINDOW_DAYS}")
        except (ValueError, TypeError):
            errors.append("date_window_days must be an integer")

    if "min_citations" in data and data["min_citations"] is not None:
        try:
            val = int(data["min_citations"])
            if val < 0:
                errors.append("min_citations must be non-negative")
            elif val > MAX_MIN_CITATIONS:
                errors.append(f"min_citations must be at most {MAX_MIN_CITATIONS}")
        except (ValueError, TypeError):
            errors.append("min_citations must be an integer")

    for field in ("include_keywords", "exclude_keywords", "author_filters", "categories", "methods_mentions"):
        if field in data and data[field] is not None:
            if not isinstance(data[field], list):
                errors.append(f"{field} must be a list")
            elif not all(isinstance(item, str) for item in data[field]):
                errors.append(f"{field} must contain only strings")

    # The free-form ``filters`` dict is splatted into ``url_for`` in the sidebar.
    # A werkzeug-reserved key (``_method``/``_external``/…) raises BuildError and
    # would brick every page, so reject ``_``-prefixed keys at the boundary.
    filters = data.get("filters")
    if isinstance(filters, dict):
        reserved = sorted(k for k in filters if isinstance(k, str) and k.startswith("_"))
        if reserved:
            errors.append(f"filters may not contain reserved keys: {', '.join(reserved)}")
        if "q" in filters and filters["q"] is not None and not isinstance(filters["q"], str):
            # A non-string q would crash _escape_like at /run time.
            errors.append("filters.q must be a string")

    return errors


def execute_saved_search(
    search: SavedSearch,
    *,
    limit: int = DEFAULT_LIMIT,
) -> list[Paper]:
    """Execute a saved search and return matching papers.

    Combines multiple filter dimensions with AND logic.
    """
    query = Paper.query.filter(Paper.is_hidden.is_(False))

    # Category filter — match the quoted JSON element to avoid substring false positives.
    if search.categories:
        category_filters = []
        for cat in search.categories:
            escaped = f'%"{_escape_like(cat)}"%'
            category_filters.append(db.cast(Paper.categories, db.Text).ilike(escaped, escape="\\"))
        query = query.filter(db.or_(*category_filters))

    # Include keywords (title or abstract must contain at least one).
    if search.include_keywords:
        keyword_filters = []
        for kw in search.include_keywords:
            pattern = f"%{_escape_like(kw)}%"
            keyword_filters.append(
                db.or_(
                    Paper.title.ilike(pattern, escape="\\"),
                    Paper.abstract_text.ilike(pattern, escape="\\"),
                )
            )
        query = query.filter(db.or_(*keyword_filters))

    # Exclude keywords (none should match).
    if search.exclude_keywords:
        for kw in search.exclude_keywords:
            pattern = f"%{_escape_like(kw)}%"
            query = query.filter(
                ~Paper.title.ilike(pattern, escape="\\"),
                ~Paper.abstract_text.ilike(pattern, escape="\\"),
            )

    # Author filters (at least one author must match).
    if search.author_filters:
        author_filters = []
        for author in search.author_filters:
            author_filters.append(Paper.authors.ilike(f"%{_escape_like(author)}%", escape="\\"))
        query = query.filter(db.or_(*author_filters))

    # Date window filter. Anchored on arrival (scraped_at) with a bounded
    # publication floor, consistent with the inbox/export timeframe windows, so a
    # saved search doesn't silently drop freshly scraped, announcement-lagged papers.
    if search.date_window_days is not None:
        cutoff = now_utc() - timedelta(days=search.date_window_days)
        query = query.filter(inbox_freshness_clause(cutoff))

    # Minimum citations filter.
    if search.min_citations is not None:
        query = query.filter(
            Paper.citation_count.isnot(None),
            Paper.citation_count >= search.min_citations,
        )

    # Methods mentions (search in abstract/title).
    if search.methods_mentions:
        method_filters = []
        for method in search.methods_mentions:
            pattern = f"%{_escape_like(method)}%"
            method_filters.append(
                db.or_(
                    Paper.title.ilike(pattern, escape="\\"),
                    Paper.abstract_text.ilike(pattern, escape="\\"),
                )
            )
        query = query.filter(db.or_(*method_filters))

    # Also apply legacy free-form filters dict for backward compatibility.
    filters = search.filters or {}
    q = filters.get("q")
    if isinstance(q, str) and q:
        escaped_q = _escape_like(q)
        query = query.filter(
            db.or_(
                Paper.title.ilike(f"%{escaped_q}%", escape="\\"),
                Paper.abstract_text.ilike(f"%{escaped_q}%", escape="\\"),
                Paper.authors.ilike(f"%{escaped_q}%", escape="\\"),
            )
        )

    # Order by score descending.
    query = query.order_by(Paper.paper_score.desc())

    return query.limit(limit).all()
