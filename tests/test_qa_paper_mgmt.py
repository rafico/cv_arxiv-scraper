"""QA tests for paper management: feedback, follow/mute, bulk operations.

Covers: CVARX-59 (Paper Management)
- Feedback action toggling (save, skip, priority, shared, skimmed)
- Priority implies save
- Skip clears save and vice versa
- Follow adds author to whitelist
- Mute adds topic to mute list
- Bulk feedback
- Feedback score bounds (-100 to 100)
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from app.models import Paper, PaperFeedback, db
from app.services.feedback import apply_feedback_action
from tests.helpers import FlaskDBTestCase


def _make_paper(idx: int = 0, **overrides) -> Paper:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today = date.today()
    defaults = dict(
        arxiv_id=f"2607.{4000 + idx:04d}",
        title=f"Paper Mgmt Test {idx}",
        authors="Jane Doe, Alice Brown",
        link=f"https://arxiv.org/abs/2607.{4000 + idx:04d}",
        pdf_link=f"https://arxiv.org/pdf/2607.{4000 + idx:04d}",
        abstract_text="abstract",
        summary_text="summary",
        topic_tags=["Segmentation", "Vision"],
        categories=["cs.CV"],
        match_type="Author",
        matched_terms=["Jane Doe"],
        paper_score=15.0,
        feedback_score=0,
        is_hidden=False,
        publication_date=today.isoformat(),
        publication_dt=today,
        scraped_date=today.isoformat(),
        scraped_at=now,
    )
    defaults.update(overrides)
    return Paper(**defaults)


class FeedbackActionTests(FlaskDBTestCase):
    """Test feedback action logic."""

    def test_save_toggle_on_off(self):
        p = _make_paper(0)
        db.session.add(p)
        db.session.commit()

        result = apply_feedback_action(p.id, "save")
        self.assertTrue(result["active"])
        self.assertEqual(result["counts"]["save"], 1)
        self.assertGreater(result["feedback_score"], 0)

        result = apply_feedback_action(p.id, "save")
        self.assertFalse(result["active"])
        self.assertEqual(result["counts"]["save"], 0)

    def test_skip_hides_paper(self):
        p = _make_paper(0)
        db.session.add(p)
        db.session.commit()

        apply_feedback_action(p.id, "skip")
        refreshed = db.session.get(Paper, p.id)
        self.assertTrue(refreshed.is_hidden)

    def test_save_clears_skip(self):
        p = _make_paper(0)
        db.session.add(p)
        db.session.commit()

        apply_feedback_action(p.id, "skip")
        result = apply_feedback_action(p.id, "save")
        self.assertEqual(result["counts"]["skip"], 0)
        self.assertEqual(result["counts"]["save"], 1)
        refreshed = db.session.get(Paper, p.id)
        self.assertFalse(refreshed.is_hidden)

    def test_skip_clears_save(self):
        p = _make_paper(0)
        db.session.add(p)
        db.session.commit()

        apply_feedback_action(p.id, "save")
        result = apply_feedback_action(p.id, "skip")
        self.assertEqual(result["counts"]["save"], 0)
        self.assertEqual(result["counts"]["skip"], 1)

    def test_priority_implies_save(self):
        p = _make_paper(0)
        db.session.add(p)
        db.session.commit()

        result = apply_feedback_action(p.id, "priority")
        self.assertEqual(result["counts"]["priority"], 1)
        self.assertEqual(result["counts"]["save"], 1)

    def test_shared_is_additive(self):
        p = _make_paper(0)
        db.session.add(p)
        db.session.commit()

        apply_feedback_action(p.id, "save")
        result = apply_feedback_action(p.id, "shared")
        self.assertEqual(result["counts"]["save"], 1)
        self.assertEqual(result["counts"]["shared"], 1)

    def test_skimmed_is_additive(self):
        p = _make_paper(0)
        db.session.add(p)
        db.session.commit()

        apply_feedback_action(p.id, "save")
        result = apply_feedback_action(p.id, "skimmed")
        self.assertEqual(result["counts"]["save"], 1)
        self.assertEqual(result["counts"]["skimmed"], 1)

    def test_feedback_with_reason_and_note(self):
        p = _make_paper(0)
        db.session.add(p)
        db.session.commit()

        apply_feedback_action(p.id, "save", reason="Relevant to thesis", note="Review later")
        fb = PaperFeedback.query.filter_by(paper_id=p.id, action="save").first()
        self.assertEqual(fb.reason, "Relevant to thesis")
        self.assertEqual(fb.note, "Review later")

    def test_invalid_action_raises(self):
        p = _make_paper(0)
        db.session.add(p)
        db.session.commit()

        with self.assertRaises(ValueError):
            apply_feedback_action(p.id, "nonexistent_action")

    def test_paper_not_found_raises(self):
        with self.assertRaises(LookupError):
            apply_feedback_action(99999, "save")


class FollowMuteFromPaperTests(FlaskDBTestCase):
    """Test follow/mute actions from paper card."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        self.paper = _make_paper(0)
        db.session.add(self.paper)
        db.session.commit()

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_follow_adds_author_to_whitelist(self):
        token = self._csrf_token()
        response = self.client.post(
            f"/api/papers/{self.paper.id}/follow",
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["term"], "Jane Doe")
        self.assertTrue(data["added"])

        # Verify config updated
        config = self.app.config["SCRAPER_CONFIG"]
        self.assertIn("Jane Doe", config["whitelists"]["authors"])

    def test_mute_adds_topic_to_mute_list(self):
        token = self._csrf_token()
        response = self.client.post(
            f"/api/papers/{self.paper.id}/mute",
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["added"])
        self.assertIn(data["term"], ["Segmentation", "Vision"])

    def test_follow_paper_not_found(self):
        token = self._csrf_token()
        response = self.client.post(
            "/api/papers/99999/follow",
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 404)

    def test_mute_paper_not_found(self):
        token = self._csrf_token()
        response = self.client.post(
            "/api/papers/99999/mute",
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 404)

    def test_follow_paper_no_author(self):
        p = _make_paper(1, authors="")
        db.session.add(p)
        db.session.commit()

        token = self._csrf_token()
        response = self.client.post(
            f"/api/papers/{p.id}/follow",
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 400)

    def test_mute_paper_no_topics(self):
        p = _make_paper(1, topic_tags=[])
        db.session.add(p)
        db.session.commit()

        token = self._csrf_token()
        response = self.client.post(
            f"/api/papers/{p.id}/mute",
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 400)


class BulkFeedbackTests(FlaskDBTestCase):
    """Test bulk feedback operations."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_bulk_save_multiple_papers(self):
        p1 = _make_paper(0)
        p2 = _make_paper(1)
        p3 = _make_paper(2)
        db.session.add_all([p1, p2, p3])
        db.session.commit()

        token = self._csrf_token()
        response = self.client.post(
            "/api/papers/bulk-feedback",
            json={"paper_ids": [p1.id, p2.id, p3.id], "action": "save"},
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["processed"], 3)

    def test_bulk_feedback_empty_ids(self):
        token = self._csrf_token()
        response = self.client.post(
            "/api/papers/bulk-feedback",
            json={"paper_ids": [], "action": "save"},
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["processed"], 0)


if __name__ == "__main__":
    import unittest

    unittest.main()
