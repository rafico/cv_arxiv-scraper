"""Shell context processor.

Injects the navigation chrome (collections, saved searches, inbox/saved
counts) into every rendered template so the sidebar shell renders on all
pages, not just the dashboard. Queries are cheap COUNTs guarded against a
missing/empty database so non-DB contexts (e.g. error pages before schema
creation) never raise.
"""

from __future__ import annotations

from datetime import timedelta

from flask import Flask
from sqlalchemy.exc import SQLAlchemyError

from app.enums import FeedbackAction
from app.models import Collection, Paper, PaperFeedback, SavedSearch, db
from app.services.text import now_utc


def _inbox_count() -> int:
    """Papers visible in the default inbox view (last day, not hidden)."""
    cutoff_dt = now_utc() - timedelta(days=1)
    cutoff_date = cutoff_dt.date()
    return (
        Paper.query.filter(Paper.is_hidden.is_(False))
        .filter(
            db.or_(
                Paper.publication_dt >= cutoff_date,
                db.and_(Paper.publication_dt.is_(None), Paper.scraped_at >= cutoff_dt),
            )
        )
        .count()
    )


def _build_shell() -> dict:
    try:
        return {
            "collections": Collection.query.order_by(Collection.name).all(),
            "saved_searches": SavedSearch.query.order_by(SavedSearch.created_at.desc()).all(),
            "saved_count": PaperFeedback.query.filter_by(action=FeedbackAction.SAVE.value).count(),
            "inbox_count": _inbox_count(),
        }
    except SQLAlchemyError:
        db.session.rollback()
        return {"collections": [], "saved_searches": [], "saved_count": 0, "inbox_count": 0}


def register_shell_context(app: Flask) -> None:
    @app.context_processor
    def inject_shell() -> dict:
        return {"shell": _build_shell()}
