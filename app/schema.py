"""Runtime schema upgrades for existing SQLite databases."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from sqlalchemy import inspect, text

from app.models import db

LOGGER = logging.getLogger(__name__)

_SAFE_COLUMN_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _validate_column_name(name: str) -> None:
    """Assert column names are safe for SQL interpolation."""
    if not _SAFE_COLUMN_NAME_RE.match(name):
        raise ValueError(f"Unsafe column name for schema migration: {name!r}")

FEEDBACK_COLUMN_DEFS = {
    "reason": "TEXT",
    "note": "TEXT",
}

SAVED_SEARCH_COLUMN_DEFS = {
    "categories": "TEXT NOT NULL DEFAULT '[]'",
    "include_keywords": "TEXT NOT NULL DEFAULT '[]'",
    "exclude_keywords": "TEXT NOT NULL DEFAULT '[]'",
    "author_filters": "TEXT NOT NULL DEFAULT '[]'",
    "date_window_days": "INTEGER",
    "min_citations": "INTEGER",
    "methods_mentions": "TEXT NOT NULL DEFAULT '[]'",
    "is_active": "INTEGER NOT NULL DEFAULT 1",
    "notify_on_match": "INTEGER NOT NULL DEFAULT 0",
}

SYNC_STATE_COLUMN_DEFS = {
    "last_cursor_page": "INTEGER",
    "last_cursor_arxiv_id": "TEXT",
}

FTS5_CREATE = """
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    title, abstract_text, authors, topic_tags,
    content='papers', content_rowid='id',
    tokenize='porter unicode61'
);
"""

FTS5_TRIGGERS = [
    """CREATE TRIGGER IF NOT EXISTS papers_fts_insert AFTER INSERT ON papers BEGIN
        INSERT INTO papers_fts(rowid, title, abstract_text, authors, topic_tags)
        VALUES (new.id, new.title, new.abstract_text, new.authors, COALESCE(new.topic_tags, ''));
    END;""",
    """CREATE TRIGGER IF NOT EXISTS papers_fts_update AFTER UPDATE ON papers BEGIN
        INSERT INTO papers_fts(papers_fts, rowid, title, abstract_text, authors, topic_tags)
        VALUES ('delete', old.id, old.title, old.abstract_text, old.authors, COALESCE(old.topic_tags, ''));
        INSERT INTO papers_fts(rowid, title, abstract_text, authors, topic_tags)
        VALUES (new.id, new.title, new.abstract_text, new.authors, COALESCE(new.topic_tags, ''));
    END;""",
    """CREATE TRIGGER IF NOT EXISTS papers_fts_delete AFTER DELETE ON papers BEGIN
        INSERT INTO papers_fts(papers_fts, rowid, title, abstract_text, authors, topic_tags)
        VALUES ('delete', old.id, old.title, old.abstract_text, old.authors, COALESCE(old.topic_tags, ''));
    END;""",
]

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
    "citation_count": "INTEGER",
    "influential_citation_count": "INTEGER",
    "semantic_scholar_id": "TEXT",
    "citation_source": "TEXT",
    "citation_provenance": "TEXT NOT NULL DEFAULT '{}'",
    "citation_updated_at": "DATETIME",
    "openalex_id": "TEXT",
    "openalex_topics": "TEXT NOT NULL DEFAULT '[]'",
    "oa_status": "TEXT",
    "referenced_works_count": "INTEGER",
    "openalex_cited_by_count": "INTEGER",
}

INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_papers_scraped_at ON papers (scraped_at)",
    "CREATE INDEX IF NOT EXISTS idx_papers_publication_dt ON papers (publication_dt)",
    "CREATE INDEX IF NOT EXISTS idx_papers_rank ON papers (paper_score, feedback_score)",
    "CREATE INDEX IF NOT EXISTS idx_papers_hidden ON papers (is_hidden)",
    "CREATE INDEX IF NOT EXISTS idx_feedback_paper_action ON paper_feedback (paper_id, action)",
]
UNIQUE_ARXIV_INDEX_STATEMENT = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_papers_arxiv_id ON papers (arxiv_id) WHERE arxiv_id IS NOT NULL"
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
        _validate_column_name(column_name)
        db.session.execute(
            text(f"ALTER TABLE papers ADD COLUMN {column_name} {column_type}")  # noqa: S608
        )

    db.session.commit()

    # Ensure all tables exist even on older DBs.
    from app.models import (  # local import to avoid circular dependency
        Collection,
        DigestRun,
        EnrichmentCache,
        FeedSource,
        PaperCollection,
        PaperFeedback,
        PaperRelation,
        PaperSection,
        RankingConfig,
        RecommendationMetric,
        SavedSearch,
        ScrapeRun,
        SyncState,
    )

    PaperFeedback.__table__.create(bind=db.engine, checkfirst=True)
    ScrapeRun.__table__.create(bind=db.engine, checkfirst=True)
    DigestRun.__table__.create(bind=db.engine, checkfirst=True)
    FeedSource.__table__.create(bind=db.engine, checkfirst=True)
    EnrichmentCache.__table__.create(bind=db.engine, checkfirst=True)
    Collection.__table__.create(bind=db.engine, checkfirst=True)
    PaperCollection.__table__.create(bind=db.engine, checkfirst=True)
    PaperRelation.__table__.create(bind=db.engine, checkfirst=True)
    SavedSearch.__table__.create(bind=db.engine, checkfirst=True)
    PaperSection.__table__.create(bind=db.engine, checkfirst=True)
    RankingConfig.__table__.create(bind=db.engine, checkfirst=True)
    RecommendationMetric.__table__.create(bind=db.engine, checkfirst=True)
    SyncState.__table__.create(bind=db.engine, checkfirst=True)

    if "sync_state" in inspect(db.engine).get_table_names():
        sync_state_columns = {col["name"] for col in inspect(db.engine).get_columns("sync_state")}
        for col_name, col_type in SYNC_STATE_COLUMN_DEFS.items():
            if col_name not in sync_state_columns:
                _validate_column_name(col_name)
                db.session.execute(
                    text(f"ALTER TABLE sync_state ADD COLUMN {col_name} {col_type}")  # noqa: S608
                )
        db.session.commit()

    # Migrate paper_feedback columns for richer triage events.
    if "paper_feedback" in tables:
        feedback_columns = {col["name"] for col in inspector.get_columns("paper_feedback")}
        for col_name, col_type in FEEDBACK_COLUMN_DEFS.items():
            if col_name not in feedback_columns:
                _validate_column_name(col_name)
                db.session.execute(
                    text(f"ALTER TABLE paper_feedback ADD COLUMN {col_name} {col_type}")  # noqa: S608
                )
        db.session.commit()

    # Migrate saved_searches columns for structured query fields.
    if "saved_searches" in inspector.get_table_names():
        ss_columns = {col["name"] for col in inspector.get_columns("saved_searches")}
        for col_name, col_type in SAVED_SEARCH_COLUMN_DEFS.items():
            if col_name not in ss_columns:
                _validate_column_name(col_name)
                db.session.execute(
                    text(f"ALTER TABLE saved_searches ADD COLUMN {col_name} {col_type}")  # noqa: S608
                )
        db.session.commit()

    # Set up FTS5 full-text search index.
    try:
        db.session.execute(text(FTS5_CREATE))
        for trigger_sql in FTS5_TRIGGERS:
            db.session.execute(text(trigger_sql))
        db.session.commit()

        # Rebuild FTS index if it's empty but papers exist.
        fts_count = db.session.execute(text("SELECT COUNT(*) FROM papers_fts")).scalar()
        paper_count = db.session.execute(text("SELECT COUNT(*) FROM papers")).scalar()
        if fts_count == 0 and paper_count > 0:
            LOGGER.info("Rebuilding FTS5 index for %d papers...", paper_count)
            db.session.execute(text("INSERT INTO papers_fts(papers_fts) VALUES('rebuild');"))
            db.session.commit()
    except Exception as exc:
        LOGGER.warning("FTS5 setup failed (search will use ILIKE fallback): %s", exc)
        db.session.rollback()

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
        try:
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
        except Exception:
            # FTS5 triggers can fail if the index is out of sync.
            # Rebuild FTS and retry.
            db.session.rollback()
            try:
                db.session.execute(text("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')"))
                db.session.commit()
            except Exception:
                db.session.rollback()
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

    _backfill_arxiv_ids()
    _fix_pdf_links()


_ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/(.+?)(?:v\d+)?$")


def _backfill_arxiv_ids() -> None:
    """Fill in NULL arxiv_id values by extracting from the paper link."""
    rows = db.session.execute(text("SELECT id, link FROM papers WHERE arxiv_id IS NULL AND link IS NOT NULL")).all()
    if not rows:
        return

    updates = []
    for paper_id, link in rows:
        match = _ARXIV_ID_RE.search(link or "")
        if match:
            updates.append({"id": paper_id, "arxiv_id": match.group(1)})

    if updates:
        LOGGER.info("Backfilling arxiv_id for %d papers...", len(updates))
        db.session.execute(
            text("UPDATE papers SET arxiv_id = :arxiv_id WHERE id = :id"),
            updates,
        )
        db.session.commit()


def _fix_pdf_links() -> None:
    """Remove erroneous .pdf extension from arxiv PDF links."""
    result = db.session.execute(
        text("UPDATE papers SET pdf_link = SUBSTR(pdf_link, 1, LENGTH(pdf_link) - 4) WHERE pdf_link LIKE '%/pdf/%.pdf'")
    )
    if result.rowcount:
        LOGGER.info("Fixed .pdf extension on %d pdf_link values", result.rowcount)
        db.session.commit()
