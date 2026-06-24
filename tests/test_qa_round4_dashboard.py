"""QA round 4 regression tests for app/routes/dashboard.py.

Covers:
- G5: GET /?view=saved&collection=<id> must return 200 (was 500 because the
  PaperFeedback join is skipped when a collection is selected, yet the default
  'saved' sort ordered by PaperFeedback.created_at -> OperationalError).
- G11: the per-paper "Why this ranked here" breakdown must include a non-zero
  citation_bonus for a paper with citation_count > 0 (explain_score was called
  without citation_count, defaulting it to None).
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from app.enums import FeedbackAction
from app.models import Collection, Paper, PaperCollection, PaperFeedback, db
from tests.helpers import FlaskDBTestCase


def _make_paper(idx: int = 0, **overrides) -> Paper:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today = date.today()
    defaults = dict(
        arxiv_id=f"2607.{3000 + idx:04d}",
        title=f"Round4 Test Paper {idx}",
        authors="Author A",
        link=f"https://arxiv.org/abs/2607.{3000 + idx:04d}",
        pdf_link=f"https://arxiv.org/pdf/2607.{3000 + idx:04d}",
        abstract_text="abstract",
        summary_text="summary",
        match_type="Title",
        matched_terms=["Vision"],
        paper_score=10.0,
        feedback_score=0,
        is_hidden=False,
        publication_date=today.isoformat(),
        publication_dt=today,
        scraped_date=today.isoformat(),
        scraped_at=now,
    )
    defaults.update(overrides)
    return Paper(**defaults)


class DashboardSavedCollectionTests(FlaskDBTestCase):
    """G5: saved view + collection filter must not 500."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_g5_saved_view_with_collection_returns_200(self):
        collection = Collection(name="C1")
        paper = _make_paper(0)
        db.session.add_all([collection, paper])
        db.session.commit()
        db.session.add(PaperCollection(paper_id=paper.id, collection_id=collection.id))
        db.session.add(PaperFeedback(paper_id=paper.id, action=FeedbackAction.SAVE.value))
        db.session.commit()

        response = self.client.get(f"/?view=saved&collection={collection.id}")
        self.assertEqual(response.status_code, 200)

    def test_g5_saved_collection_newest_sort_returns_200(self):
        collection = Collection(name="C2")
        paper = _make_paper(1)
        db.session.add_all([collection, paper])
        db.session.commit()
        db.session.add(PaperCollection(paper_id=paper.id, collection_id=collection.id))
        db.session.commit()

        response = self.client.get(f"/?view=saved&collection={collection.id}&sort=newest")
        self.assertEqual(response.status_code, 200)

    def test_g5_inbox_collection_saved_sort_returns_200(self):
        collection = Collection(name="C3")
        paper = _make_paper(2)
        db.session.add_all([collection, paper])
        db.session.commit()
        db.session.add(PaperCollection(paper_id=paper.id, collection_id=collection.id))
        db.session.commit()

        response = self.client.get(f"/?view=inbox&collection={collection.id}&sort=saved")
        self.assertEqual(response.status_code, 200)


class DashboardCitationBreakdownTests(FlaskDBTestCase):
    """G11: explain breakdown includes non-zero citation_bonus."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_g11_score_breakdown_includes_citation_bonus(self):
        from app.routes.dashboard import _enrich_cards_with_feedback_and_related

        paper = _make_paper(0, citation_count=500)
        db.session.add(paper)
        db.session.commit()

        config = self.app.config["SCRAPER_CONFIG"]
        _enrich_cards_with_feedback_and_related([paper], [paper], config)

        self.assertGreater(paper.score_breakdown["citation_bonus"], 0.0)
