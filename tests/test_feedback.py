from datetime import date, datetime

from app.models import Paper, db
from app.services.feedback import apply_feedback_action
from tests.helpers import FlaskDBTestCase


class FeedbackTests(FlaskDBTestCase):
    def _create_paper(self) -> Paper:
        paper = Paper(
            arxiv_id="1234.5678",
            title="Test Paper",
            authors="Jane Doe",
            link="https://arxiv.org/abs/1234.5678",
            pdf_link="https://arxiv.org/pdf/1234.5678",
            abstract_text="Test abstract",
            summary_text="Test summary",
            topic_tags=["Vision"],
            categories=["cs.CV"],
            resource_links=[],
            match_type="Author",
            matched_terms=["Jane Doe"],
            paper_score=12.5,
            feedback_score=0,
            is_hidden=False,
            publication_date="2026-02-10",
            publication_dt=date(2026, 2, 10),
            scraped_date="2026-02-13",
            scraped_at=datetime(2026, 2, 13, 12, 0, 0),
        )
        db.session.add(paper)
        db.session.commit()
        return paper

    def test_save_toggle(self):
        paper = self._create_paper()

        on_result = apply_feedback_action(paper.id, "save")
        self.assertTrue(on_result["active"])
        self.assertEqual(on_result["counts"]["save"], 1)

        off_result = apply_feedback_action(paper.id, "save")
        self.assertFalse(off_result["active"])
        self.assertEqual(off_result["counts"]["save"], 0)

    def test_skip_hides_paper(self):
        paper = self._create_paper()

        result = apply_feedback_action(paper.id, "skip")
        self.assertTrue(result["active"])
        self.assertEqual(result["counts"]["skip"], 1)

        updated = db.session.get(Paper, paper.id)
        self.assertTrue(updated.is_hidden)

    def test_save_clears_skip_and_unhides_paper(self):
        paper = self._create_paper()

        apply_feedback_action(paper.id, "skip")
        result = apply_feedback_action(paper.id, "save")

        updated = db.session.get(Paper, paper.id)
        self.assertTrue(result["active"])
        self.assertEqual(result["counts"]["save"], 1)
        self.assertEqual(result["counts"]["skip"], 0)
        self.assertFalse(updated.is_hidden)

    def test_skip_clears_save(self):
        paper = self._create_paper()

        apply_feedback_action(paper.id, "save")
        result = apply_feedback_action(paper.id, "skip")

        updated = db.session.get(Paper, paper.id)
        self.assertTrue(result["active"])
        self.assertEqual(result["counts"]["save"], 0)
        self.assertEqual(result["counts"]["skip"], 1)
        self.assertTrue(updated.is_hidden)


if __name__ == "__main__":
    import unittest

    unittest.main()
