"""Regression tests for the activation defaults.

Waves 1-3 shipped features that then sat inert on a real install: full-text
sections were off by default (so per-paper chat, corpus chat and citation
verification had nothing to read), the onboarding checklist called the ranking
step done after one save while the ranker needs MIN_POSITIVE_FEEDBACK, and a
missing digest recipient raised before any DigestRun row was written, so nightly
failures were invisible outside cron.log. These tests pin all three open.
"""

from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from unittest.mock import patch

from app.models import DigestRun, Paper, PaperFeedback, PaperSection, db
from app.routes.dashboard import _build_onboarding_steps
from app.services.interest_model import MIN_POSITIVE_FEEDBACK
from app.services.scrape_engine import _extract_sections
from tests.helpers import FlaskDBTestCase


def _save_step(steps: list[dict]) -> dict:
    return next(step for step in steps if step["label"] == "Save or skip papers")


class OnboardingActivationThresholdTests(FlaskDBTestCase):
    """The checklist must track the threshold that actually switches ranking on."""

    def _steps(self, positive_count: int) -> list[dict]:
        return _build_onboarding_steps(
            {"whitelists": {"authors": ["Jane Doe"]}},
            positive_count=positive_count,
            has_successful_scrape=True,
        )

    def test_incomplete_one_short_of_threshold(self):
        step = _save_step(self._steps(MIN_POSITIVE_FEEDBACK - 1))
        self.assertFalse(step["complete"])
        # The remaining count must be visible, otherwise the user cannot tell how
        # much further they have to go before anything turns on.
        self.assertIn("1 more", step["description"])

    def test_complete_at_threshold(self):
        step = _save_step(self._steps(MIN_POSITIVE_FEEDBACK))
        self.assertTrue(step["complete"])

    def test_single_save_is_not_enough(self):
        # The pre-fix behavior: one save marked the step done and the panel vanished.
        self.assertFalse(_save_step(self._steps(1))["complete"])

    def test_digest_step_appears_only_without_a_recipient(self):
        without = _build_onboarding_steps(
            {"whitelists": {}}, positive_count=0, has_successful_scrape=True
        )
        self.assertIn("Set a digest recipient", [step["label"] for step in without])

        with_recipient = _build_onboarding_steps(
            {"whitelists": {}, "email": {"recipient": "a@b.com"}},
            positive_count=0,
            has_successful_scrape=True,
        )
        self.assertNotIn("Set a digest recipient", [step["label"] for step in with_recipient])


def _make_paper(link: str) -> Paper:
    today = date.today()
    return Paper(
        arxiv_id="2607.5001",
        title="Sections Default Paper",
        authors="Author A",
        link=link,
        pdf_link=link.replace("abs", "pdf"),
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
        scraped_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )


def _fake_run_isolated(func, *args, **kwargs):
    return [[("introduction", "extracted body text", 0)] for _pdf in args[0]]


class SectionExtractionDefaultTests(FlaskDBTestCase):
    @patch("app.services.embeddings.get_embedding_service", side_effect=RuntimeError("skip embeddings"))
    @patch("app.services.subprocess_runner.run_isolated", side_effect=_fake_run_isolated)
    def test_sections_extracted_with_no_config_key_present(self, _run, _emb):
        # TEST_SCRAPER_CONFIG carries no extract_sections key, exactly like a
        # config.yaml written before the flag existed. Sections must still land.
        self.assertNotIn("extract_sections", self.app.config["SCRAPER_CONFIG"]["scraper"])
        link = "https://arxiv.org/abs/2607.5001"
        paper = _make_paper(link)
        db.session.add(paper)
        db.session.commit()

        # No arxiv_id in the result dict keeps this off the network: the HTML path is
        # skipped and the mocked PDF fallback runs.
        _extract_sections(self.app, [{"link": link, "pdf_content": b"PDF"}])

        self.assertEqual(PaperSection.query.filter_by(paper_id=paper.id).count(), 1)

    @patch("app.services.embeddings.get_embedding_service", side_effect=RuntimeError("skip embeddings"))
    @patch("app.services.subprocess_runner.run_isolated", side_effect=_fake_run_isolated)
    def test_explicit_false_still_opts_out(self, _run, _emb):
        self.app.config["SCRAPER_CONFIG"]["scraper"]["extract_sections"] = False
        link = "https://arxiv.org/abs/2607.5001"
        db.session.add(_make_paper(link))
        db.session.commit()

        _extract_sections(self.app, [{"link": link, "pdf_content": b"PDF"}])

        self.assertEqual(PaperSection.query.count(), 0)


class DigestMisconfigurationIsRecordedTests(FlaskDBTestCase):
    def test_missing_recipient_records_an_errored_run(self):
        from app.services.email_digest import send_digest

        self.app.config["SCRAPER_CONFIG"]["email"] = {"recipient": ""}

        with self.assertRaises(ValueError):
            send_digest(self.app)

        run = DigestRun.query.one()
        self.assertEqual(run.status, "error")
        self.assertIn("No recipient configured", run.error_message)
        self.assertIsNotNone(run.finished_at)


class FeatureLivenessTests(FlaskDBTestCase):
    def test_reports_feedback_progress_toward_the_threshold(self):
        from app.services.metrics import feature_liveness

        paper = _make_paper("https://arxiv.org/abs/2607.5001")
        db.session.add(paper)
        db.session.commit()
        db.session.add(PaperFeedback(paper_id=paper.id, action="save"))
        db.session.commit()

        status = feature_liveness()
        self.assertEqual(status["positive_feedback"], 1)
        self.assertEqual(status["positive_feedback_needed"], MIN_POSITIVE_FEEDBACK - 1)
        self.assertIs(status["interest_signal_ready"], False)
        self.assertEqual(status["sections_papers"], 0)
        # A whitelist match must not be counted as dense retrieval.
        self.assertEqual(status["dense_retrieval_papers"], 0)


if __name__ == "__main__":
    unittest.main()
