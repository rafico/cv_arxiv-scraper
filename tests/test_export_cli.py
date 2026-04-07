from __future__ import annotations

import io
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

from app.models import Paper, db
from tests.helpers import FlaskDBTestCase


class ExportCliTests(FlaskDBTestCase):
    @patch("export_cli.create_app")
    def test_cli_writes_requested_html_report(self, mock_create_app):
        mock_create_app.return_value = self.app
        paper = Paper(
            arxiv_id="2604.6101",
            title="Weekly Export Paper",
            authors="Alice Smith",
            link="https://arxiv.org/abs/2604.6101",
            pdf_link="https://arxiv.org/pdf/2604.6101",
            summary_text="Export summary",
            match_type="Title",
            matched_terms=["Vision"],
            paper_score=8.0,
            publication_date=date.today().isoformat(),
            publication_dt=date.today(),
            scraped_date=date.today().isoformat(),
            scraped_at=datetime.now(timezone.utc).replace(tzinfo=None),
            is_hidden=False,
        )
        db.session.add(paper)
        db.session.commit()

        from export_cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"

            with (
                patch.object(sys, "argv", ["export_cli.py", "--timeframe", "weekly", "--output", str(output_path)]),
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
            ):
                main()

            self.assertTrue(output_path.exists())
            self.assertIn("Weekly Export Paper", output_path.read_text(encoding="utf-8"))
            self.assertIn(str(output_path), stdout.getvalue())
