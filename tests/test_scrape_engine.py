"""Tests for scrape_engine save/dedupe and pre-filter count logic."""

from __future__ import annotations

import unittest
from datetime import date, timedelta
from unittest.mock import patch

from app.models import Paper, ScrapeRun, db
from app.services.ranking import compute_paper_score
from app.services.summary import generate_summary
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


class BuildResultTests(unittest.TestCase):
    def test_build_result_uses_extractive_summary_when_llm_disabled(self):
        from app.services.scrape_engine import _build_result

        abstract = (
            "Brief intro. "
            "We introduce a transformer for dense prediction tasks and show strong "
            "segmentation performance across benchmarks. "
            "The approach also improves detection quality."
        )
        entry = {
            "title": "A Vision Model",
            "abstract": abstract,
            "author": "Author A",
            "link": "https://arxiv.org/abs/0001",
            "authors_list": ["Author A"],
        }

        result = _build_result(
            entry,
            {"Title": ["Vision"], "Author": [], "Affiliation": []},
        )

        self.assertEqual(result["summary_text"], generate_summary(entry["title"], abstract))
        self.assertNotEqual(result["summary_text"], abstract)


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

        def fake_parse_feed(url, session=None):
            return fake_entries

        def fake_enrich(entries, session=None):
            pass

        def fake_process(
            entries, whitelists, config, session=None, llm_client=None, interests_text="", product_config=None
        ):
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
            patch.object(scrape_engine, "_process_entries_with_pipeline", side_effect=fake_process),
        ):
            scrape_engine.execute_scrape(self.app, event_callback=capture_callback)

        # The pre-filtered existing paper should be in duplicates_skipped.
        self.assertGreaterEqual(captured.get("duplicates_skipped", 0), 1)


class GuardTests(FlaskDBTestCase):
    def test_execute_scrape_skips_when_successful_run_exists_today(self):
        from app.services.scrape_engine import execute_scrape

        now = now_utc()
        db.session.add(
            ScrapeRun(
                status="success",
                started_at=now,
                finished_at=now,
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

    def test_execute_scrape_does_not_skip_when_today_has_papers_but_no_successful_run(self):
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

        with (
            patch("app.services.scrape_engine.parse_feed_entries", return_value=[]),
            patch("app.services.scrape_engine.enrich_entries_with_api_metadata"),
            patch("app.services.scrape_engine._process_entries_with_pipeline", return_value=iter([])),
        ):
            result = execute_scrape(self.app)

        self.assertIn("new_papers", result)
        self.assertFalse(result.get("skipped", False))

    def test_force_true_bypasses_guard(self):
        from app.services.scrape_engine import execute_scrape

        now = now_utc()
        db.session.add(
            ScrapeRun(
                status="success",
                started_at=now - timedelta(hours=1),
                finished_at=now - timedelta(hours=1),
                forced=False,
            )
        )
        db.session.commit()

        with (
            patch("app.services.scrape_engine.parse_feed_entries", return_value=[]),
            patch("app.services.scrape_engine.enrich_entries_with_api_metadata"),
            patch("app.services.scrape_engine._process_entries_with_pipeline", return_value=iter([])),
        ):
            result = execute_scrape(self.app, force=True)

        self.assertIn("new_papers", result)

    def test_execute_scrape_records_error_run_on_failure(self):
        from app.services.scrape_engine import execute_scrape

        with patch("app.services.scrape_engine.parse_feed_entries", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                execute_scrape(self.app, force=True)

        scrape_run = db.session.query(ScrapeRun).order_by(ScrapeRun.id.desc()).first()
        self.assertIsNotNone(scrape_run)
        self.assertEqual(scrape_run.status, "error")


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

        def fake_process(
            entries, whitelists, config, session=None, llm_client=None, interests_text="", product_config=None
        ):
            seen_titles.extend(entry["title"] for entry in entries)
            for i, entry in enumerate(entries, 1):
                result = _make_result(entry["link"], entry["title"])
                result["arxiv_id"] = entry.get("arxiv_id")
                yield i, i, result

        with (
            patch.object(scrape_engine, "parse_feed_entries", return_value=rss_entries),
            patch.object(scrape_engine, "fetch_recent_papers", return_value=api_entries),
            patch.object(scrape_engine, "enrich_entries_with_api_metadata"),
            patch.object(scrape_engine, "_process_entries_with_pipeline", side_effect=fake_process),
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
            patch.object(scrape_engine, "_process_entries_with_pipeline", return_value=iter([])),
        ):
            scrape_engine.execute_scrape(self.app, force=True)

        mock_recent.assert_not_called()

    def test_ingest_backends_config_can_disable_recent_fetch(self):
        from app.services import scrape_engine

        self.app.config["SCRAPER_CONFIG"]["scraper"]["rolling_window_days"] = 2
        self.app.config["SCRAPER_CONFIG"]["ingest"] = {"backends": ["rss"]}

        with (
            patch.object(scrape_engine, "parse_feed_entries", return_value=[]),
            patch.object(scrape_engine, "fetch_recent_papers") as mock_recent,
            patch.object(scrape_engine, "enrich_entries_with_api_metadata"),
            patch.object(scrape_engine, "_process_entries_with_pipeline", return_value=iter([])),
        ):
            scrape_engine.execute_scrape(self.app, force=True)

        mock_recent.assert_not_called()


class OpenAlexEnrichmentTests(FlaskDBTestCase):
    @patch("app.services.openalex.fetch_openalex_batch")
    def test_openalex_fallback_populates_citation_count_and_rescores(self, mock_fetch):
        from app.services.scrape_engine import _enrich_results_with_openalex

        result = _make_result("https://arxiv.org/abs/0007")
        result["arxiv_id"] = "0007"
        result["match_type"] = "Title"
        result["match_types"] = ["Title"]
        result["paper_score"] = 1.0
        result["publication_dt"] = date(2026, 1, 1)
        mock_fetch.return_value = {
            "0007": {
                "openalex_id": "W7",
                "openalex_topics": [{"name": "Vision", "score": 0.9}],
                "oa_status": "green",
                "openalex_cited_by_count": 7,
                "referenced_works_count": 2,
            }
        }

        _enrich_results_with_openalex([result], session=None, config=self.app.config["SCRAPER_CONFIG"])

        self.assertEqual(result["citation_count"], 7)
        self.assertEqual(result["citation_source"], "openalex")
        self.assertEqual(result["openalex_cited_by_count"], 7)
        self.assertIsNotNone(result["citation_updated_at"])
        self.assertEqual(
            result["paper_score"],
            compute_paper_score(
                match_types=result["match_types"],
                matched_terms_count=len(result["matches"]),
                publication_dt=result["publication_dt"],
                resource_count=len(result["resource_links"]),
                llm_relevance_score=result.get("llm_relevance_score"),
                citation_count=result["citation_count"],
                config=self.app.config["SCRAPER_CONFIG"],
            ),
        )


class HistoricalScrapeTests(FlaskDBTestCase):
    def test_execute_historical_scrape_uses_orchestrator_bridge(self):
        from app.services import scrape_engine
        from app.services.ingest import PaperCandidate

        self.app.config["SCRAPER_CONFIG"]["ingest"] = {"backends": ["arxiv_api"]}

        class FakeOrchestrator:
            def fetch(self, **kwargs):
                self.kwargs = kwargs
                return [
                    PaperCandidate(
                        arxiv_id="0007",
                        link="https://arxiv.org/abs/0007",
                        title="Historical Paper",
                        author="Author A",
                        authors_list=["Author A"],
                        publication_date="2026-01-01",
                    )
                ]

        orchestrator = FakeOrchestrator()

        def fake_process(
            entries, whitelists, config, session=None, llm_client=None, interests_text="", product_config=None
        ):
            for i, entry in enumerate(entries, 1):
                result = _make_result(entry["link"], entry["title"])
                result["arxiv_id"] = entry.get("arxiv_id")
                yield i, i, result

        with (
            patch.object(scrape_engine, "_build_ingest_orchestrator", return_value=orchestrator),
            patch.object(scrape_engine, "enrich_entries_with_api_metadata"),
            patch.object(scrape_engine, "_process_entries_with_pipeline", side_effect=fake_process),
        ):
            summary = scrape_engine.execute_historical_scrape(
                self.app,
                ["cs.CV"],
                date(2026, 1, 1),
                date(2026, 1, 2),
            )

        self.assertEqual(orchestrator.kwargs["mode"].value, "backfill")
        self.assertEqual(orchestrator.kwargs["categories"], ["cs.CV"])
        self.assertEqual(orchestrator.kwargs["backend_names"], ["arxiv_api"])
        self.assertIn("new_papers", summary)


if __name__ == "__main__":
    unittest.main()
