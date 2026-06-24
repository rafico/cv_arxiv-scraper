"""QA round 5 regression test — R5-con1 (S3): _extract_sections must dedup result
targets by link so a cross-listed paper (same link twice in one batch) doesn't have
the second copy's bulk delete wipe the PaperSection rows the first just added.
"""

from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from unittest.mock import patch

from app.models import Paper, PaperSection, db
from app.services.scrape_engine import _extract_sections
from tests.helpers import FlaskDBTestCase


def _make_paper(link: str) -> Paper:
    today = date.today()
    return Paper(
        arxiv_id="2607.4001",
        title="Cross-listed Paper",
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
    # Emulate extract_sections_batch: one section per pdf, text derived from the pdf
    # bytes so we can tell which duplicate's sections survived.
    pdfs = args[0]
    return [[("introduction", f"sec-{pdf.decode()}", 0)] for pdf in pdfs]


class DuplicateLinkSectionTests(FlaskDBTestCase):
    @patch("app.services.embeddings.get_embedding_service", side_effect=RuntimeError("skip embeddings"))
    @patch("app.services.subprocess_runner.run_isolated", side_effect=_fake_run_isolated)
    def test_duplicate_link_keeps_first_occurrence_sections(self, _run, _emb):
        self.app.config["SCRAPER_CONFIG"]["scraper"]["extract_sections"] = True
        link = "https://arxiv.org/abs/2607.4001"
        paper = _make_paper(link)
        db.session.add(paper)
        db.session.commit()
        paper_id = paper.id

        results = [
            {"link": link, "pdf_content": b"PDF1"},
            {"link": link, "pdf_content": b"PDF2"},
        ]
        _extract_sections(self.app, results)

        sections = PaperSection.query.filter_by(paper_id=paper_id).all()
        self.assertEqual(len(sections), 1)
        # The first occurrence's sections must survive (not be wiped by the second).
        self.assertEqual(sections[0].text, "sec-PDF1")


if __name__ == "__main__":
    unittest.main()
