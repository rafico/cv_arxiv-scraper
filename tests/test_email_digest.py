"""Tests for the email digest service."""

from __future__ import annotations

import base64
import unittest
from datetime import date, datetime, timedelta, timezone
from email import message_from_bytes
from email.header import decode_header, make_header
from unittest.mock import MagicMock, patch

from app.models import Paper, db
from app.services.email_digest import (
    _build_email_body,
    _get_email_config,
    _query_todays_papers,
    _render_paper_html,
)
from tests.helpers import FlaskDBTestCase


def _make_paper(**overrides) -> Paper:
    defaults = dict(
        title="Test Paper <script>alert(1)</script>",
        authors="Alice, Bob",
        link="https://arxiv.org/abs/0000.00000",
        pdf_link="https://arxiv.org/pdf/0000.00000",
        abstract_text="An abstract.",
        summary_text="A summary.",
        topic_tags=["vision", "detection"],
        categories=["cs.CV"],
        resource_links=[],
        match_type="Author + Title",
        matched_terms=["Alice"],
        paper_score=42.5,
        feedback_score=0,
        is_hidden=False,
        publication_date="2026-03-13",
        scraped_date="2026-03-13",
        publication_dt=date(2026, 3, 13),
        scraped_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    defaults.update(overrides)
    return Paper(**defaults)


class RenderPaperHtmlTests(unittest.TestCase):
    """Verify HTML escaping prevents injection."""

    def test_title_is_escaped(self):
        paper = _make_paper()
        html = _render_paper_html(paper)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_link_is_escaped(self):
        paper = _make_paper(link='https://evil.com/"><img src=x>')
        html = _render_paper_html(paper)
        self.assertNotIn('"><img', html)

    def test_score_uses_feedback_adjusted_rank(self):
        paper = _make_paper(paper_score=10.0, feedback_score=4)
        html = _render_paper_html(paper)
        self.assertIn("Score: 15.0", html)


class BuildEmailBodyTests(unittest.TestCase):
    def test_empty_papers_shows_message(self):
        html = _build_email_body([], date(2026, 3, 13))
        self.assertIn("No new matching papers", html)

    def test_nonempty_papers_shows_count(self):
        papers = [_make_paper(), _make_paper(title="Second Paper")]
        html = _build_email_body(papers, date(2026, 3, 13))
        self.assertIn("2 papers", html)


class QueryTodaysPapersTests(FlaskDBTestCase):
    def test_returns_recent_non_hidden_papers(self):
        recent = _make_paper(title="Recent", scraped_at=datetime.now(timezone.utc).replace(tzinfo=None))
        old = _make_paper(
            title="Old",
            link="https://arxiv.org/abs/0000.11111",
            scraped_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=48),
        )
        hidden = _make_paper(
            title="Hidden",
            link="https://arxiv.org/abs/0000.22222",
            is_hidden=True,
        )
        db.session.add_all([recent, old, hidden])
        db.session.commit()

        papers = _query_todays_papers(self.app)
        titles = [p.title for p in papers]
        self.assertIn("Recent", titles)
        self.assertNotIn("Old", titles)
        self.assertNotIn("Hidden", titles)

    def test_orders_by_feedback_adjusted_rank(self):
        lower_raw_higher_rank = _make_paper(
            title="Boosted",
            link="https://arxiv.org/abs/0000.33333",
            paper_score=50.0,
            feedback_score=7,
        )
        higher_raw_lower_rank = _make_paper(
            title="Unboosted",
            link="https://arxiv.org/abs/0000.44444",
            paper_score=55.0,
            feedback_score=0,
        )
        db.session.add_all([lower_raw_higher_rank, higher_raw_lower_rank])
        db.session.commit()

        papers = _query_todays_papers(self.app)
        titles = [paper.title for paper in papers]
        self.assertLess(titles.index("Boosted"), titles.index("Unboosted"))


class GetEmailConfigTests(FlaskDBTestCase):
    def test_defaults_when_no_email_section(self):
        cfg = _get_email_config(self.app)
        self.assertEqual(cfg["recipient"], "")
        self.assertEqual(cfg["subject_prefix"], "ArXiv Digest")

    def test_reads_from_config(self):
        self.app.config["SCRAPER_CONFIG"]["email"] = {
            "recipient": "test@example.com",
            "subject_prefix": "My Digest",
        }
        cfg = _get_email_config(self.app)
        self.assertEqual(cfg["recipient"], "test@example.com")
        self.assertEqual(cfg["subject_prefix"], "My Digest")


class SendDigestTests(FlaskDBTestCase):
    @patch("app.services.email_digest._build_gmail_service")
    @patch("app.services.email_digest._load_gmail_credentials")
    def test_dry_run_does_not_send(self, mock_creds, mock_service):
        mock_creds.return_value = MagicMock()
        self.app.config["SCRAPER_CONFIG"]["email"] = {"recipient": "a@b.com"}

        from app.services.email_digest import send_digest

        result = send_digest(self.app, dry_run=True)
        self.assertFalse(result["sent"])
        mock_service.assert_not_called()

    def test_missing_token_raises(self):
        from pathlib import Path

        from app.services.email_digest import _load_gmail_credentials

        with self.assertRaises(FileNotFoundError):
            _load_gmail_credentials(token_path=Path("/nonexistent/token.json"))

    def test_missing_recipient_raises(self):
        from app.services.email_digest import send_digest

        self.app.config["SCRAPER_CONFIG"]["email"] = {"recipient": ""}

        with patch("app.services.email_digest._load_gmail_credentials") as mock_creds:
            mock_creds.return_value = MagicMock()
            with self.assertRaises(ValueError, msg="recipient"):
                send_digest(self.app)

    @patch("app.services.email_digest.utc_today", return_value=date(2026, 3, 20))
    @patch("app.services.email_digest._build_gmail_service")
    @patch("app.services.email_digest._load_gmail_credentials")
    def test_send_digest_uses_utc_today_for_subject(
        self,
        mock_creds,
        mock_service_builder,
        _mock_today,
    ):
        from app.services.email_digest import send_digest

        mock_creds.return_value = MagicMock()
        service = MagicMock()
        mock_service_builder.return_value = service
        self.app.config["SCRAPER_CONFIG"]["email"] = {"recipient": "a@b.com"}
        db.session.add(_make_paper())
        db.session.commit()

        send_digest(self.app)

        raw = service.users.return_value.messages.return_value.send.call_args.kwargs["body"]["raw"]
        message = message_from_bytes(base64.urlsafe_b64decode(raw))
        subject = str(make_header(decode_header(message["Subject"])))
        self.assertEqual(subject, "ArXiv Digest — Mar 20, 2026 (1 papers)")


if __name__ == "__main__":
    unittest.main()
