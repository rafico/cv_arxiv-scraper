"""Runtime schema upgrades for existing SQLite databases."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import inspect, text

from app.models import db

LOGGER = logging.getLogger(__name__)

PAPER_COLUMN_DEFS = {
    "arxiv_id": "TEXT",
    "abstract_text": "TEXT NOT NULL DEFAULT ''",
    "summary_text": "TEXT NOT NULL DEFAULT ''",
    "topic_tags": "TEXT NOT NULL DEFAULT '[]'",
    "categories": "TEXT NOT NULL DEFAULT '[]'",
    "resource_links": "TEXT NOT NULL DEFAULT '[]'",
    "paper_score": "REAL NOT NULL DEFAULT 0",
    "llm_relevance_score": "REAL",
    "feedback_score": "INTEGER NOT NULL DEFAULT 0",
    "is_hidden": "INTEGER NOT NULL DEFAULT 0",
    "publication_dt": "DATE",
    "scraped_at": "DATETIME",
    "reading_status": "TEXT",
    "user_notes": "TEXT DEFAULT ''",
    "user_tags": "TEXT NOT NULL DEFAULT '[]'",
    "duplicate_of_id": "INTEGER REFERENCES papers(id)",
    "source_feed_id": "INTEGER REFERENCES feed_sources(id)",
    "recommendation_score": "REAL",
}

INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_papers_scraped_at ON papers (scraped_at)",
    "CREATE INDEX IF NOT EXISTS idx_papers_publication_dt ON papers (publication_dt)",
    "CREATE INDEX IF NOT EXISTS idx_papers_rank ON papers (paper_score, feedback_score)",
    "CREATE INDEX IF NOT EXISTS idx_papers_hidden ON papers (is_hidden)",
    "CREATE INDEX IF NOT EXISTS idx_feedback_paper_action ON paper_feedback (paper_id, action)",
]
UNIQUE_ARXIV_INDEX_STATEMENT = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_papers_arxiv_id "
    "ON papers (arxiv_id) WHERE arxiv_id IS NOT NULL"
)


def _try_parse_date(value: str | None):
    if not value or value == "Date Unknown":
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _try_parse_datetime(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        pass

    try:
        parsed_date = datetime.strptime(value, "%Y-%m-%d")
        return parsed_date
    except ValueError:
        return None


def ensure_schema() -> None:
    """Apply additive schema upgrades and backfill normalized date columns."""
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    if "papers" not in tables:
        return

    existing_columns = {column["name"] for column in inspector.get_columns("papers")}

    for column_name, column_type in PAPER_COLUMN_DEFS.items():
        if column_name in existing_columns:
            continue
        db.session.execute(
            text(f"ALTER TABLE papers ADD COLUMN {column_name} {column_type}")  # noqa: S608
        )

    db.session.commit()

    # Ensure all tables exist even on older DBs.
    from app.models import (  # local import to avoid circular dependency
        Collection,
        DigestRun,
        FeedSource,
        PaperCollection,
        PaperFeedback,
        PaperRelation,
        SavedSearch,
        ScrapeRun,
    )

    PaperFeedback.__table__.create(bind=db.engine, checkfirst=True)
    ScrapeRun.__table__.create(bind=db.engine, checkfirst=True)
    DigestRun.__table__.create(bind=db.engine, checkfirst=True)
    FeedSource.__table__.create(bind=db.engine, checkfirst=True)
    Collection.__table__.create(bind=db.engine, checkfirst=True)
    PaperCollection.__table__.create(bind=db.engine, checkfirst=True)
    PaperRelation.__table__.create(bind=db.engine, checkfirst=True)
    SavedSearch.__table__.create(bind=db.engine, checkfirst=True)

    for statement in INDEX_STATEMENTS:
        db.session.execute(text(statement))
    db.session.commit()
    try:
        db.session.execute(text(UNIQUE_ARXIV_INDEX_STATEMENT))
        db.session.commit()
    except Exception as exc:  # pragma: no cover - depends on legacy DB contents
        LOGGER.warning("Could not create unique arXiv id index: %s", exc)
        db.session.rollback()

    rows = db.session.execute(
        text(
            """
            SELECT id, publication_date, scraped_date, scraped_at, created_at
            FROM papers
            WHERE publication_dt IS NULL OR scraped_at IS NULL
            """
        )
    ).mappings()

    updates = []
    for row in rows:
        publication_dt = _try_parse_date(row["publication_date"])
        scraped_at = _try_parse_datetime(row["scraped_at"])
        if scraped_at is None:
            scraped_at = _try_parse_datetime(row["scraped_date"])
        if scraped_at is None:
            created_at = row["created_at"]
            if isinstance(created_at, datetime):
                scraped_at = created_at
        if scraped_at is None:
            scraped_at = datetime.now(timezone.utc).replace(tzinfo=None)

        updates.append(
            {
                "id": row["id"],
                "publication_dt": publication_dt,
                "scraped_at": scraped_at,
            }
        )

    if updates:
        db.session.execute(
            text(
                """
                UPDATE papers
                SET publication_dt = :publication_dt,
                    scraped_at = :scraped_at
                WHERE id = :id
                """
            ),
            updates,
        )
        db.session.commit()
