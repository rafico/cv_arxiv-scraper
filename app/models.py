"""Database models."""

from __future__ import annotations

import json

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Paper(db.Model):
    __tablename__ = "papers"
    __table_args__ = (
        db.Index("idx_papers_scraped_at", "scraped_at"),
        db.Index("idx_papers_publication_dt", "publication_dt"),
        db.Index("idx_papers_rank", "paper_score", "feedback_score"),
        db.Index("idx_papers_hidden", "is_hidden"),
    )

    id = db.Column(db.Integer, primary_key=True)
    arxiv_id = db.Column(db.String(40), index=True)
    title = db.Column(db.Text, nullable=False)
    authors = db.Column(db.Text, nullable=False)
    link = db.Column(db.Text, nullable=False, unique=True)
    pdf_link = db.Column(db.Text, nullable=False)

    abstract_text = db.Column(db.Text, nullable=False, default="")
    summary_text = db.Column(db.Text, nullable=False, default="")
    topic_tags = db.Column(db.Text, nullable=False, default="")
    categories = db.Column(db.Text, nullable=False, default="")
    resource_links = db.Column(db.Text, nullable=False, default="[]")

    match_type = db.Column(db.Text, nullable=False)
    matched_terms = db.Column(db.Text, nullable=False)
    paper_score = db.Column(db.Float, nullable=False, default=0.0)
    feedback_score = db.Column(db.Integer, nullable=False, default=0)
    is_hidden = db.Column(db.Boolean, nullable=False, default=False)

    # Legacy string dates are preserved for compatibility with older rows.
    publication_date = db.Column(db.Text)
    scraped_date = db.Column(db.Text, nullable=False)

    publication_dt = db.Column(db.Date, index=True)
    scraped_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False, index=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

    feedback = db.relationship("PaperFeedback", back_populates="paper", cascade="all, delete-orphan")

    @staticmethod
    def _parse_comma_list(value: str | None) -> list[str]:
        return [item.strip() for item in (value or "").split(",") if item.strip()]

    @property
    def matched_terms_list(self) -> list[str]:
        return self._parse_comma_list(self.matched_terms)

    @property
    def topic_tags_list(self) -> list[str]:
        return self._parse_comma_list(self.topic_tags)

    @property
    def categories_list(self) -> list[str]:
        return self._parse_comma_list(self.categories)

    @property
    def resource_links_list(self) -> list[dict]:
        try:
            data = json.loads(self.resource_links or "[]")
            if isinstance(data, list):
                return data
            return []
        except json.JSONDecodeError:
            return []

    @property
    def rank_score(self) -> float:
        from app.services.ranking import FEEDBACK_BOOST
        return round(float(self.paper_score or 0.0) + (int(self.feedback_score or 0) * FEEDBACK_BOOST), 3)


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
