"""Tests for scrape_engine save/dedupe and pre-filter count logic."""

from __future__ import annotations

import unittest
from datetime import timedelta
from unittest.mock import patch

from app.models import Paper, db
from app.services.text import now_utc
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


def _make_entry(link: str, title: str = "Paper", arxiv_id: str | None = None) -> dict:
    return {
        "arxiv_id": arxiv_id or link.split("/")[-1],
        "link": link,
        "title": title,
        "author": "Author A",
        "authors_list": ["Author A"],
        "abstract": "",
        "publication_dt": None,
        "publication_date": "2026-01-01",
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

    def test_same_arxiv_id_with_different_link_is_skipped(self):
        from app.services.scrape_engine import _save_results

        db.session.add(
            Paper(
                arxiv_id="0001",
                title="Existing",
                authors="A",
                link="https://arxiv.org/abs/0001v1",
                pdf_link="https://arxiv.org/pdf/0001v1.pdf",
                match_type="title",
                matched_terms=["Vision"],
                paper_score=1.0,
                publication_date="2026-01-01",
                scraped_date="2026-01-01",
            )
        )
        db.session.commit()

        results = [_make_result("https://arxiv.org/abs/0001v2")]
        results[0]["arxiv_id"] = "0001"

        new_count, skipped = _save_results(self.app, results)

        self.assertEqual(new_count, 0)
        self.assertEqual(skipped, 1)


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
                scraped_at=now_utc() - timedelta(days=2),
            )
        )
        db.session.commit()

        fake_entries = [
            _make_entry("https://arxiv.org/abs/existing", "Existing"),
            _make_entry("https://arxiv.org/abs/new1", "New One"),
        ]

        def fake_parse_feed(url):
            return fake_entries

        def fake_enrich(entries):
            pass

        def fake_process(entries, whitelists, config, llm_client=None, interests_text=""):
            for i, entry in enumerate(entries, 1):
                result = _make_result(entry["link"], entry["title"])
                result["arxiv_id"] = entry.get("arxiv_id")
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


class GuardTests(FlaskDBTestCase):
    def test_execute_scrape_skips_when_today_already_scraped(self):
        from app.services.scrape_engine import execute_scrape

        now = now_utc()
        db.session.add(
            Paper(
                arxiv_id="0001",
                title="Existing",
                authors="A",
                link="https://arxiv.org/abs/0001",
                pdf_link="https://arxiv.org/pdf/0001.pdf",
                match_type="Title",
                matched_terms=["Vision"],
                paper_score=1.0,
                publication_date="2026-01-01",
                scraped_date=now.date().isoformat(),
                scraped_at=now,
            )
        )
        db.session.commit()

        captured: list[tuple[str, dict]] = []

        with patch("app.services.scrape_engine.parse_feed_entries") as mock_parse:
            result = execute_scrape(
                self.app,
                event_callback=lambda event, data: captured.append((event, data)),
            )

        self.assertTrue(result["skipped"])
        self.assertEqual(captured[0][0], "skipped")
        mock_parse.assert_not_called()

    def test_force_true_bypasses_guard(self):
        from app.services.scrape_engine import execute_scrape

        now = now_utc()
        db.session.add(
            Paper(
                arxiv_id="0001",
                title="Existing",
                authors="A",
                link="https://arxiv.org/abs/0001",
                pdf_link="https://arxiv.org/pdf/0001.pdf",
                match_type="Title",
                matched_terms=["Vision"],
                paper_score=1.0,
                publication_date="2026-01-01",
                scraped_date=now.date().isoformat(),
                scraped_at=now - timedelta(hours=1),
            )
        )
        db.session.commit()

        with (
            patch("app.services.scrape_engine.parse_feed_entries", return_value=[]),
            patch("app.services.scrape_engine.enrich_entries_with_api_metadata"),
            patch("app.services.scrape_engine._process_entries_parallel", return_value=iter([])),
        ):
            result = execute_scrape(self.app, force=True)

        self.assertIn("new_papers", result)


class RollingWindowTests(FlaskDBTestCase):
    def test_rolling_window_merges_by_arxiv_id(self):
        from app.services import scrape_engine

        self.app.config["SCRAPER_CONFIG"]["scraper"]["rolling_window_days"] = 2
        rss_entries = [
            _make_entry("https://arxiv.org/abs/0001v1", "RSS One", arxiv_id="0001"),
        ]
        api_entries = [
            _make_entry("https://arxiv.org/abs/0001v2", "API Duplicate", arxiv_id="0001"),
            _make_entry("https://arxiv.org/abs/0002v1", "API Two", arxiv_id="0002"),
        ]
        seen_titles: list[str] = []

        def fake_process(entries, whitelists, config, llm_client=None, interests_text=""):
            seen_titles.extend(entry["title"] for entry in entries)
            for i, entry in enumerate(entries, 1):
                result = _make_result(entry["link"], entry["title"])
                result["arxiv_id"] = entry.get("arxiv_id")
                yield i, i, result

        with (
            patch.object(scrape_engine, "parse_feed_entries", return_value=rss_entries),
            patch.object(scrape_engine, "fetch_recent_papers", return_value=api_entries),
            patch.object(scrape_engine, "enrich_entries_with_api_metadata"),
            patch.object(scrape_engine, "_process_entries_parallel", side_effect=fake_process),
        ):
            scrape_engine.execute_scrape(self.app)

        self.assertEqual(seen_titles, ["RSS One", "API Two"])

    def test_rolling_window_zero_skips_fetch(self):
        from app.services import scrape_engine

        self.app.config["SCRAPER_CONFIG"]["scraper"]["rolling_window_days"] = 0

        with (
            patch.object(scrape_engine, "parse_feed_entries", return_value=[]),
            patch.object(scrape_engine, "fetch_recent_papers") as mock_recent,
            patch.object(scrape_engine, "enrich_entries_with_api_metadata"),
            patch.object(scrape_engine, "_process_entries_parallel", return_value=iter([])),
        ):
            scrape_engine.execute_scrape(self.app, force=True)

        mock_recent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
