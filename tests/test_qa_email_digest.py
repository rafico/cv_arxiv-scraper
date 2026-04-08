from __future__ import annotations

import base64
import io
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from email import message_from_bytes
from email.header import decode_header, make_header
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.models import DigestRun, Paper, db
from app.services.email_digest import (
    _get_email_config,
    _load_gmail_credentials,
    build_digest_preview,
    check_gmail_auth_status,
    finish_oauth_flow,
    send_digest,
)
from tests.helpers import FlaskDBTestCase


def _make_paper(**overrides) -> Paper:
    defaults = dict(
        title="Digest QA Paper",
        authors="Alice Example, Bob Example",
        link="https://arxiv.org/abs/2604.00001",
        pdf_link="https://arxiv.org/pdf/2604.00001",
        abstract_text="A digest QA abstract.",
        summary_text="A digest QA summary.",
        topic_tags=["vision", "detection"],
        categories=["cs.CV"],
        resource_links=[{"type": "pdf", "url": "https://arxiv.org/pdf/2604.00001"}],
        match_type="Author + Title",
        matched_terms=["Alice Example"],
        paper_score=42.5,
        feedback_score=2,
        is_hidden=False,
        publication_date="2026-04-07",
        scraped_date="2026-04-07",
        publication_dt=date(2026, 4, 7),
        scraped_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    defaults.update(overrides)
    return Paper(**defaults)


class GmailOAuthQaTests(unittest.TestCase):
    def test_check_gmail_auth_status_reports_invalid_token_clearly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            credentials_path = root / "credentials.json"
            token_path = root / "token.json"
            credentials_path.write_text("{}", encoding="utf-8")
            token_path.write_text("{}", encoding="utf-8")

            with patch(
                "google.oauth2.credentials.Credentials.from_authorized_user_file", side_effect=ValueError("bad")
            ):
                status = check_gmail_auth_status(credentials_path=credentials_path, token_path=token_path)

        self.assertEqual(status["status"], "invalid")
        self.assertIn("corrupted", status["message"])
        self.assertEqual(status["action"], "reauthorize")

    def test_finish_oauth_flow_saves_token_json_with_restricted_permissions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            credentials_path = root / "credentials.json"
            token_path = root / "token.json"
            credentials_path.write_text('{"web":{"client_id":"id","client_secret":"secret"}}', encoding="utf-8")

            fake_creds = MagicMock()
            fake_creds.to_json.return_value = '{"refresh_token":"abc"}'
            fake_flow = MagicMock()
            fake_flow.credentials = fake_creds

            with (
                patch("google_auth_oauthlib.flow.Flow.from_client_secrets_file", return_value=fake_flow),
                patch("app.services.email_digest.os.chmod") as mock_chmod,
            ):
                result = finish_oauth_flow(
                    authorization_response_url="https://example.com/callback?code=abc&state=xyz",
                    redirect_uri="https://example.com/callback",
                    credentials_path=credentials_path,
                    token_path=token_path,
                )

            written_token = token_path.read_text(encoding="utf-8")

        self.assertTrue(result["success"])
        fake_flow.fetch_token.assert_called_once()
        self.assertEqual(written_token, '{"refresh_token":"abc"}')
        mock_chmod.assert_called_once_with(token_path, 0o600)

    def test_load_gmail_credentials_refreshes_and_persists_token(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "token.json"
            token_path.write_text("{}", encoding="utf-8")

            fake_creds = MagicMock()
            fake_creds.expired = True
            fake_creds.refresh_token = "refresh-token"
            fake_creds.valid = True
            fake_creds.to_json.return_value = '{"access_token":"fresh"}'

            with (
                patch("google.oauth2.credentials.Credentials.from_authorized_user_file", return_value=fake_creds),
                patch("google.auth.transport.requests.Request", return_value=object()) as mock_request,
                patch("app.services.email_digest.os.chmod") as mock_chmod,
            ):
                loaded = _load_gmail_credentials(token_path=token_path)
                written_token = token_path.read_text(encoding="utf-8")

        self.assertIs(loaded, fake_creds)
        fake_creds.refresh.assert_called_once_with(mock_request.return_value)
        self.assertEqual(written_token, '{"access_token":"fresh"}')
        mock_chmod.assert_called_once_with(token_path, 0o600)


class DigestServiceQaTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_send_digest_uses_configured_recipient_subject_and_logs_success(self):
        self.app.config["SCRAPER_CONFIG"]["email"] = {
            "recipient": "digest@example.com",
            "subject_prefix": "Custom Digest",
        }
        db.session.add(_make_paper())
        db.session.commit()

        fake_service = MagicMock()
        fake_service.users.return_value.messages.return_value.send.return_value.execute.return_value = {"id": "msg-1"}

        with (
            patch("app.services.email_digest._load_gmail_credentials", return_value=MagicMock()),
            patch("app.services.email_digest._build_gmail_service", return_value=fake_service),
        ):
            result = send_digest(self.app)

        self.assertTrue(result["sent"])
        self.assertEqual(result["recipient"], "digest@example.com")

        raw_message = fake_service.users.return_value.messages.return_value.send.call_args.kwargs["body"]["raw"]
        message = message_from_bytes(base64.urlsafe_b64decode(raw_message))
        html_part = message.get_payload()[0].get_payload(decode=True).decode("utf-8")
        subject = str(make_header(decode_header(message["Subject"])))

        self.assertIn("Custom Digest", subject)
        self.assertEqual(message["To"], "digest@example.com")
        self.assertIn("Digest QA Paper", html_part)
        self.assertIn("Alice Example, Bob Example", html_part)
        self.assertIn("https://arxiv.org/abs/2604.00001", html_part)
        self.assertIn("Author", html_part)
        self.assertIn("Score:", html_part)

        run = DigestRun.query.one()
        self.assertEqual(run.status, "success")
        self.assertEqual(run.recipient, "digest@example.com")
        self.assertEqual(run.papers_count, 1)
        self.assertFalse(run.preview_only)
        self.assertTrue(run.subject.startswith("Custom Digest"))

    def test_send_digest_dry_run_logs_preview_status(self):
        self.app.config["SCRAPER_CONFIG"]["email"] = {"recipient": "digest@example.com"}
        db.session.add(_make_paper(title="Dry Run Paper", link="https://arxiv.org/abs/2604.00002"))
        db.session.commit()

        result = send_digest(self.app, dry_run=True)

        self.assertFalse(result["sent"])
        run = DigestRun.query.one()
        self.assertEqual(run.status, "preview")
        self.assertTrue(run.preview_only)
        self.assertEqual(run.papers_count, 1)

    def test_send_digest_logs_error_status_when_send_fails(self):
        self.app.config["SCRAPER_CONFIG"]["email"] = {"recipient": "digest@example.com"}
        db.session.add(_make_paper(title="Broken Send", link="https://arxiv.org/abs/2604.00003"))
        db.session.commit()

        with (
            patch("app.services.email_digest._load_gmail_credentials", return_value=MagicMock()),
            patch("app.services.email_digest._build_gmail_service", side_effect=RuntimeError("gmail boom")),
        ):
            with self.assertRaises(RuntimeError):
                send_digest(self.app)

        run = DigestRun.query.one()
        self.assertEqual(run.status, "error")
        self.assertIn("gmail boom", run.error_message)

    def test_digest_preview_route_matches_service_preview(self):
        self.app.config["SCRAPER_CONFIG"]["email"] = {
            "recipient": "preview@example.com",
            "subject_prefix": "Preview Digest",
        }
        db.session.add(_make_paper(title="Preview Paper", link="https://arxiv.org/abs/2604.00004"))
        db.session.commit()

        preview = build_digest_preview(self.app)
        response = self.client.get("/settings/digest-preview")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Digest-Subject"], preview["subject"])
        self.assertEqual(response.get_data(as_text=True), preview["html"])

    def test_get_email_config_defaults_and_custom_values(self):
        self.assertEqual(_get_email_config(self.app)["subject_prefix"], "ArXiv Digest")
        self.app.config["SCRAPER_CONFIG"]["email"] = {
            "recipient": "configured@example.com",
            "subject_prefix": "Configured Digest",
        }
        self.assertEqual(
            _get_email_config(self.app),
            {"recipient": "configured@example.com", "subject_prefix": "Configured Digest"},
        )


class DigestCliQaTests(unittest.TestCase):
    @patch("app.cli.digest.send_digest")
    @patch("app.cli.digest.run_scrape")
    @patch("app.cli.digest.create_app")
    def test_cli_default_runs_scrape_then_send(self, mock_create_app, mock_run_scrape, mock_send_digest):
        fake_app = MagicMock()
        mock_create_app.return_value = fake_app
        mock_run_scrape.return_value = {
            "new_papers": 2,
            "duplicates_skipped": 1,
            "total_matched": 3,
            "total_in_feed": 10,
        }
        mock_send_digest.return_value = {"papers_count": 3, "sent": True, "recipient": "digest@example.com"}

        from app.cli.digest import main

        with (
            patch.object(sys, "argv", ["cv-arxiv-digest"]),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            main()

        mock_run_scrape.assert_called_once_with(fake_app)
        mock_send_digest.assert_called_once_with(fake_app, dry_run=False)
        self.assertIn("Running scrape...", stdout.getvalue())
        self.assertIn("Digest sent - 3 papers -> digest@example.com", stdout.getvalue())

    @patch("app.cli.digest.send_digest")
    @patch("app.cli.digest.run_scrape")
    @patch("app.cli.digest.create_app")
    def test_cli_send_only_dry_run_skips_scrape_and_prepares_digest(
        self, mock_create_app, mock_run_scrape, mock_send_digest
    ):
        fake_app = MagicMock()
        mock_create_app.return_value = fake_app
        mock_send_digest.return_value = {"papers_count": 1, "sent": False, "recipient": "digest@example.com"}

        from app.cli.digest import main

        with (
            patch.object(sys, "argv", ["cv-arxiv-digest", "--send-only", "--dry-run"]),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            main()

        mock_run_scrape.assert_not_called()
        mock_send_digest.assert_called_once_with(fake_app, dry_run=True)
        self.assertIn("Digest prepared (dry run) - 1 papers -> digest@example.com", stdout.getvalue())

    @patch("app.cli.digest.send_digest", side_effect=ValueError("missing recipient"))
    @patch("app.cli.digest.create_app", return_value=MagicMock())
    def test_cli_exits_nonzero_on_digest_failure(self, _mock_create_app, _mock_send_digest):
        from app.cli.digest import main

        with (
            patch.object(sys, "argv", ["cv-arxiv-digest", "--send-only"]),
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            with self.assertRaises(SystemExit) as ctx:
                main()

        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("ERROR: missing recipient", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
