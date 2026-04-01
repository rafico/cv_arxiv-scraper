"""Static HTML export helpers."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from flask import render_template

from app.models import Paper, db
from app.routes.dashboard import TIMEFRAME_DAYS
from app.services.ranking import FEEDBACK_BOOST
from app.services.text import now_utc


def generate_html_report(app, timeframe: str = "daily", output_path: str | Path | None = None) -> str:
    if timeframe not in TIMEFRAME_DAYS:
        timeframe = "daily"

    generated_at = now_utc()
    days = TIMEFRAME_DAYS.get(timeframe)

    with app.app_context():
        query = Paper.query.filter(Paper.is_hidden.is_(False))
        if days is not None:
            cutoff_dt = generated_at - timedelta(days=days)
            cutoff_date = cutoff_dt.date()
            query = query.filter(
                db.or_(
                    Paper.publication_dt >= cutoff_date,
                    db.and_(Paper.publication_dt.is_(None), Paper.scraped_at >= cutoff_dt),
                )
            )

        papers = query.order_by(
            (
                db.func.coalesce(Paper.paper_score, 0.0) + db.func.coalesce(Paper.feedback_score, 0) * FEEDBACK_BOOST
            ).desc(),
            Paper.publication_dt.desc(),
            Paper.scraped_at.desc(),
        ).all()

        html = render_template(
            "export.html",
            papers=papers,
            timeframe=timeframe,
            generated_at=generated_at,
            total=len(papers),
        )

    if output_path:
        output = Path(output_path)
        output.write_text(html, encoding="utf-8")

    return html
