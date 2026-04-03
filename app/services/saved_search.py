"""SavedSearch execution engine — translates structured filters into queries."""

from __future__ import annotations

import logging
from datetime import timedelta

from app.models import Paper, SavedSearch, db
from app.services.text import now_utc

LOGGER = logging.getLogger(__name__)

DEFAULT_LIMIT = 100


def validate_saved_search(data: dict) -> list[str]:
    """Validate saved search filter fields. Returns list of error messages."""
    errors = []
    if "date_window_days" in data and data["date_window_days"] is not None:
        try:
            val = int(data["date_window_days"])
            if val < 0:
                errors.append("date_window_days must be non-negative")
        except (ValueError, TypeError):
            errors.append("date_window_days must be an integer")

    if "min_citations" in data and data["min_citations"] is not None:
        try:
            val = int(data["min_citations"])
            if val < 0:
                errors.append("min_citations must be non-negative")
        except (ValueError, TypeError):
            errors.append("min_citations must be an integer")

    for field in ("include_keywords", "exclude_keywords", "author_filters", "categories", "methods_mentions"):
        if field in data and data[field] is not None:
            if not isinstance(data[field], list):
                errors.append(f"{field} must be a list")
            elif not all(isinstance(item, str) for item in data[field]):
                errors.append(f"{field} must contain only strings")

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

    # Category filter.
    if search.categories:
        category_filters = []
        for cat in search.categories:
            category_filters.append(Paper.categories.contains(cat))
        query = query.filter(db.or_(*category_filters))

    # Include keywords (title or abstract must contain at least one).
    if search.include_keywords:
        keyword_filters = []
        for kw in search.include_keywords:
            pattern = f"%{kw}%"
            keyword_filters.append(
                db.or_(
                    Paper.title.ilike(pattern),
                    Paper.abstract_text.ilike(pattern),
                )
            )
        query = query.filter(db.or_(*keyword_filters))

    # Exclude keywords (none should match).
    if search.exclude_keywords:
        for kw in search.exclude_keywords:
            pattern = f"%{kw}%"
            query = query.filter(
                ~Paper.title.ilike(pattern),
                ~Paper.abstract_text.ilike(pattern),
            )

    # Author filters (at least one author must match).
    if search.author_filters:
        author_filters = []
        for author in search.author_filters:
            author_filters.append(Paper.authors.ilike(f"%{author}%"))
        query = query.filter(db.or_(*author_filters))

    # Date window filter.
    if search.date_window_days is not None:
        cutoff = now_utc() - timedelta(days=search.date_window_days)
        cutoff_date = cutoff.date()
        query = query.filter(
            db.or_(
                Paper.publication_dt >= cutoff_date,
                db.and_(Paper.publication_dt.is_(None), Paper.scraped_at >= cutoff),
            )
        )

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
            pattern = f"%{method}%"
            method_filters.append(
                db.or_(
                    Paper.title.ilike(pattern),
                    Paper.abstract_text.ilike(pattern),
                )
            )
        query = query.filter(db.or_(*method_filters))

    # Also apply legacy free-form filters dict for backward compatibility.
    filters = search.filters or {}
    if filters.get("q"):
        q = filters["q"]
        query = query.filter(
            db.or_(
                Paper.title.ilike(f"%{q}%"),
                Paper.abstract_text.ilike(f"%{q}%"),
                Paper.authors.ilike(f"%{q}%"),
            )
        )

    # Order by score descending.
    query = query.order_by(Paper.paper_score.desc())

    # Update last_used_at.
    search.last_used_at = now_utc()
    db.session.commit()

    return query.limit(limit).all()
