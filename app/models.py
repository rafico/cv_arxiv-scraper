"""Database models."""

from __future__ import annotations

import json

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text as sql_text
from sqlalchemy.types import TypeDecorator, TEXT

db = SQLAlchemy()

class JSONList(TypeDecorator):
    """Custom JSON list type since SQLite driver chokes on python lists via db.JSON directly."""
    impl = TEXT
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return "[]"
        if not isinstance(value, list):
            raise ValueError(f"JSONList expected a list, got {type(value).__name__}: {value!r}")
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if not value:
            return []
        try:
            val = json.loads(value)
            if isinstance(val, list):
                return val
            # Legacy: bare JSON string — split on commas for old comma-separated rows.
            if isinstance(val, str):
                return [item.strip() for item in val.split(",") if item.strip()]
            return []
        except (json.JSONDecodeError, TypeError):
            # Legacy: raw comma-separated text that isn't valid JSON at all.
            return [item.strip() for item in str(value).split(",") if item.strip()]

class Paper(db.Model):
    __tablename__ = "papers"
    __table_args__ = (
        db.Index(
            "uq_papers_arxiv_id",
            "arxiv_id",
            unique=True,
            sqlite_where=sql_text("arxiv_id IS NOT NULL"),
        ),
        db.Index("idx_papers_scraped_at", "scraped_at"),
        db.Index("idx_papers_publication_dt", "publication_dt"),
        db.Index("idx_papers_rank", "paper_score", "feedback_score"),
        db.Index("idx_papers_hidden", "is_hidden"),
    )

    id = db.Column(db.Integer, primary_key=True)
    arxiv_id = db.Column(db.String(40))
    title = db.Column(db.Text, nullable=False)
    authors = db.Column(db.Text, nullable=False)
    link = db.Column(db.Text, nullable=False, unique=True)
    pdf_link = db.Column(db.Text, nullable=False)

    abstract_text = db.Column(db.Text, nullable=False, default="")
    summary_text = db.Column(db.Text, nullable=False, default="")
    topic_tags = db.Column(JSONList, nullable=False, default=list)
    categories = db.Column(JSONList, nullable=False, default=list)
    resource_links = db.Column(JSONList, nullable=False, default=list)

    match_type = db.Column(db.Text, nullable=False)
    matched_terms = db.Column(JSONList, nullable=False, default=list)
    paper_score = db.Column(db.Float, nullable=False, default=0.0)
    llm_relevance_score = db.Column(db.Float, nullable=True)
    feedback_score = db.Column(db.Integer, nullable=False, default=0)
    is_hidden = db.Column(db.Boolean, nullable=False, default=False)

    # Legacy string dates are preserved for compatibility with older rows.
    publication_date = db.Column(db.Text)
    scraped_date = db.Column(db.Text, nullable=False)

    publication_dt = db.Column(db.Date, index=True)
    scraped_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False, index=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    feedback = db.relationship("PaperFeedback", back_populates="paper", cascade="all, delete-orphan")

    @property
    def matched_terms_list(self) -> list[str]:
        return self.matched_terms or []

    @property
    def topic_tags_list(self) -> list[str]:
        return self.topic_tags or []

    @property
    def categories_list(self) -> list[str]:
        return self.categories or []

    @property
    def resource_links_list(self) -> list[dict]:
        return self.resource_links or []

    @property
    def rank_score(self) -> float:
        from app.services.ranking import combined_rank_score
        return combined_rank_score(float(self.paper_score or 0.0), int(self.feedback_score or 0))


class PaperFeedback(db.Model):
    __tablename__ = "paper_feedback"
    __table_args__ = (
        db.UniqueConstraint("paper_id", "action", name="uq_paper_feedback_action"),
        db.Index("idx_feedback_paper_action", "paper_id", "action"),
    )

    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.Integer, db.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)
    action = db.Column(db.String(16), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)

    paper = db.relationship("Paper", back_populates="feedback")


class ScrapeRun(db.Model):
    __tablename__ = "scrape_runs"
    __table_args__ = (
        db.Index("idx_scrape_runs_started_at", "started_at"),
        db.Index("idx_scrape_runs_status_started_at", "status", "started_at"),
    )

    id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.String(16), nullable=False, index=True)
    forced = db.Column(db.Boolean, nullable=False, default=False)
    started_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False, index=True)
    finished_at = db.Column(db.DateTime, nullable=True, index=True)
