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

class JSONDict(TypeDecorator):
    """Custom JSON dict type for SQLite."""
    impl = TEXT
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return "{}"
        if not isinstance(value, dict):
            raise ValueError(f"JSONDict expected a dict, got {type(value).__name__}: {value!r}")
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if not value:
            return {}
        try:
            val = json.loads(value)
            return val if isinstance(val, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}


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

    reading_status = db.Column(db.String(16), nullable=True)
    user_notes = db.Column(db.Text, nullable=True, default="")
    user_tags = db.Column(JSONList, nullable=False, default=list)
    duplicate_of_id = db.Column(db.Integer, db.ForeignKey("papers.id"), nullable=True)
    source_feed_id = db.Column(db.Integer, db.ForeignKey("feed_sources.id"), nullable=True)
    recommendation_score = db.Column(db.Float, nullable=True)

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
    def user_tags_list(self) -> list[str]:
        return self.user_tags or []

    @property
    def resource_links_list(self) -> list[dict]:
        return self.resource_links or []

    @property
    def rank_score(self) -> float:
        from app.services.ranking import combined_rank_score
        return combined_rank_score(float(self.paper_score or 0.0), int(self.feedback_score or 0))


class Collection(db.Model):
    __tablename__ = "collections"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False, unique=True)
    description = db.Column(db.Text, nullable=True, default="")
    color = db.Column(db.String(7), nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    updated_at = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())

    papers = db.relationship("PaperCollection", back_populates="collection", cascade="all, delete-orphan")


class PaperCollection(db.Model):
    __tablename__ = "paper_collections"
    __table_args__ = (
        db.UniqueConstraint("paper_id", "collection_id", name="uq_paper_collection"),
    )

    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.Integer, db.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)
    collection_id = db.Column(db.Integer, db.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True)
    added_at = db.Column(db.DateTime, server_default=db.func.now())

    paper = db.relationship("Paper")
    collection = db.relationship("Collection", back_populates="papers")


class PaperRelation(db.Model):
    __tablename__ = "paper_relations"
    __table_args__ = (
        db.UniqueConstraint("paper_id", "related_paper_id", "relation_type", name="uq_paper_relation"),
    )

    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.Integer, db.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)
    related_paper_id = db.Column(db.Integer, db.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)
    relation_type = db.Column(db.String(32), nullable=False, default="similar")
    similarity_score = db.Column(db.Float, nullable=True)


class SavedSearch(db.Model):
    __tablename__ = "saved_searches"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    filters = db.Column(JSONDict, nullable=False, default=dict)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    last_used_at = db.Column(db.DateTime, nullable=True)


class FeedSource(db.Model):
    __tablename__ = "feed_sources"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    url = db.Column(db.Text, nullable=False, unique=True)
    feed_type = db.Column(db.String(32), nullable=False, default="arxiv_rss")
    enabled = db.Column(db.Boolean, nullable=False, default=True)
    last_fetched_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


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


class DigestRun(db.Model):
    __tablename__ = "digest_runs"
    __table_args__ = (
        db.Index("idx_digest_runs_started_at", "started_at"),
        db.Index("idx_digest_runs_status_started_at", "status", "started_at"),
    )

    id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.String(16), nullable=False, index=True)
    recipient = db.Column(db.Text, nullable=False, default="")
    subject = db.Column(db.Text, nullable=False, default="")
    papers_count = db.Column(db.Integer, nullable=False, default=0)
    preview_only = db.Column(db.Boolean, nullable=False, default=False)
    error_message = db.Column(db.Text, nullable=True)
    started_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False, index=True)
    finished_at = db.Column(db.DateTime, nullable=True, index=True)
