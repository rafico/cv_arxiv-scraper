"""Tests for scrape_engine save/dedupe and pre-filter count logic."""

from __future__ import annotations

import unittest

from app.models import Paper, db
from tests.helpers import FlaskDBTestCase


def _make_result(link: str, title: str = "Test Paper") -> dict:
    return {
        "arxiv_id": link.split("/")[-1],
        "title": title,
        "authors": "Author A",
        "link": link,
        "pdf_link": f"{link}.pdf",
        "abstract_text": "",
        "summary_text": "",
        "topic_tags": [],
        "categories": ["cs.CV"],
        "resource_links": [],
        "match_type": "title",
        "match_types": ["title"],
        "matches": ["Vision"],
        "paper_score": 1.0,
        "publication_date": "2026-01-01",
        "publication_dt": None,
    }


class SaveResultsDedupeTests(FlaskDBTestCase):
    def test_duplicate_links_in_batch_are_deduped(self):
        """Two results with the same link should not crash; second is skipped."""
        from app.services.scrape_engine import _save_results

        results = [
            _make_result("https://arxiv.org/abs/0001"),
            _make_result("https://arxiv.org/abs/0001", title="Duplicate"),
        ]
        new_count, skipped = _save_results(self.app, results)

        self.assertEqual(new_count, 1)
        self.assertEqual(skipped, 1)
        self.assertEqual(db.session.query(Paper).count(), 1)

    def test_existing_link_in_db_is_skipped(self):
        """Results whose link is already in the DB are skipped."""
        from app.services.scrape_engine import _save_results

        db.session.add(
            Paper(
                title="Existing",
                authors="A",
                link="https://arxiv.org/abs/0001",
                pdf_link="https://arxiv.org/pdf/0001.pdf",
                match_type="title",
                matched_terms=["Vision"],
                paper_score=1.0,
                publication_date="2026-01-01",
                scraped_date="2026-01-01",
            )
        )
        db.session.commit()

        results = [
            _make_result("https://arxiv.org/abs/0001"),
            _make_result("https://arxiv.org/abs/0002"),
        ]
        new_count, skipped = _save_results(self.app, results)

        self.assertEqual(new_count, 1)
        self.assertEqual(skipped, 1)
        self.assertEqual(db.session.query(Paper).count(), 2)

    def test_all_unique_links_are_saved(self):
        """All results with unique links should be saved."""
        from app.services.scrape_engine import _save_results

        results = [
            _make_result("https://arxiv.org/abs/0001"),
            _make_result("https://arxiv.org/abs/0002"),
            _make_result("https://arxiv.org/abs/0003"),
        ]
        new_count, skipped = _save_results(self.app, results)

        self.assertEqual(new_count, 3)
        self.assertEqual(skipped, 0)


class PreFilterCountTests(FlaskDBTestCase):
    def test_pre_filtered_papers_included_in_duplicates_skipped(self):
        """Pre-filtered (already in DB) papers should be counted in duplicates_skipped."""
        from unittest.mock import patch

        # Insert an existing paper so the pre-filter catches it.
        db.session.add(
            Paper(
                title="Existing",
                authors="A",
                link="https://arxiv.org/abs/existing",
                pdf_link="https://arxiv.org/pdf/existing.pdf",
                match_type="title",
                matched_terms=["Vision"],
                paper_score=1.0,
                publication_date="2026-01-01",
                scraped_date="2026-01-01",
            )
        )
        db.session.commit()

        fake_entries = [
            {"link": "https://arxiv.org/abs/existing", "title": "Existing", "id": "1"},
            {"link": "https://arxiv.org/abs/new1", "title": "New One", "id": "2"},
        ]

        def fake_parse_feed(url):
            return fake_entries

        def fake_enrich(entries):
            pass

        def fake_process(entries, whitelists, config):
            for i, entry in enumerate(entries, 1):
                result = _make_result(entry["link"], entry["title"])
                yield i, 1, result

        captured = {}

        def capture_callback(event, data):
            if event == "done":
                captured.update(data)

        from app.services import scrape_engine

        with (
            patch.object(scrape_engine, "parse_feed_entries", side_effect=fake_parse_feed),
            patch.object(scrape_engine, "enrich_entries_with_api_metadata", side_effect=fake_enrich),
            patch.object(scrape_engine, "_process_entries_parallel", side_effect=fake_process),
        ):
            scrape_engine.execute_scrape(self.app, event_callback=capture_callback)

        # The pre-filtered existing paper should be in duplicates_skipped.
        self.assertGreaterEqual(captured.get("duplicates_skipped", 0), 1)


if __name__ == "__main__":
    unittest.main()
