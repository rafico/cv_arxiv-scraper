"""Tests for the Hugging Face Papers enrichment provider, pipeline hook, and backfill."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, Mock, patch

import requests

from app.models import EnrichmentCache, Paper, db
from app.services.enrichment_providers import (
    HuggingFaceProvider,
    huggingface_resource_links,
    parse_hf_paper,
)
from backfill_cli import backfill_huggingface, main
from tests.helpers import FlaskDBTestCase


def _paper(arxiv_id: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        authors="Author A",
        link=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_link=f"https://arxiv.org/pdf/{arxiv_id}",
        match_type="Title",
        matched_terms=["Vision"],
        paper_score=1.0,
        publication_date="2026-01-01",
        scraped_date="2026-01-01",
    )


def _hf_response(upvotes: int = 87, comments: int = 7, github: str | None = None, project: str | None = None):
    response = MagicMock()
    response.json.return_value = {
        "id": "2411.10442",
        "upvotes": upvotes,
        "comments": [{"id": str(i)} for i in range(comments)],
        "githubRepo": github,
        "projectPage": project,
        "ai_keywords": ["MLLMs"],
    }
    return response


def _http_error(status_code: int) -> requests.HTTPError:
    return requests.HTTPError(response=Mock(status_code=status_code))


class ParseHfPaperTests(unittest.TestCase):
    def test_parses_upvotes_comments_and_links(self):
        payload = parse_hf_paper(
            {
                "upvotes": 87,
                "comments": [{"id": "1"}, {"id": "2"}],
                "githubRepo": "https://github.com/snumprlab/dart",
                "projectPage": "https://twkang43.github.io/projects/dart/",
            }
        )
        self.assertEqual(payload["hf_upvotes"], 87)
        self.assertEqual(payload["hf_comments_count"], 2)
        self.assertEqual(payload["github_repo_url"], "https://github.com/snumprlab/dart")
        self.assertEqual(payload["project_page_url"], "https://twkang43.github.io/projects/dart/")

    def test_accepts_daily_papers_num_comments_shape(self):
        payload = parse_hf_paper({"upvotes": 3, "numComments": 5})
        self.assertEqual(payload["hf_comments_count"], 5)

    def test_sparse_object_yields_nones(self):
        payload = parse_hf_paper({"id": "2411.10442"})
        self.assertIsNone(payload["hf_upvotes"])
        self.assertIsNone(payload["hf_comments_count"])
        self.assertIsNone(payload["github_repo_url"])
        self.assertIsNone(payload["project_page_url"])

    def test_resource_links_shape(self):
        links = huggingface_resource_links(
            {
                "github_repo_url": "https://github.com/lab/model",
                "project_page_url": "https://lab.github.io/model/",
            }
        )
        self.assertEqual(
            links,
            [
                {"type": "code", "label": "Code", "url": "https://github.com/lab/model"},
                {"type": "project", "label": "Project", "url": "https://lab.github.io/model/"},
            ],
        )
        self.assertEqual(huggingface_resource_links({}), [])
        self.assertEqual(huggingface_resource_links(None), [])


class HuggingFaceProviderTests(FlaskDBTestCase):
    def test_fetch_caches_after_first_fetch(self):
        paper = _paper("2606.00001")
        db.session.add(paper)
        db.session.commit()

        request_fn = MagicMock(return_value=_hf_response(upvotes=87, comments=7))
        provider = HuggingFaceProvider(request_fn=request_fn)

        first = provider.fetch_batch(["2606.00001"])
        second = provider.fetch_batch(["2606.00001"])

        self.assertEqual(first["2606.00001"]["hf_upvotes"], 87)
        self.assertEqual(second["2606.00001"]["hf_comments_count"], 7)
        self.assertEqual(request_fn.call_count, 1)

        cache_row = EnrichmentCache.query.filter_by(paper_id=paper.id, source="huggingface").one()
        self.assertEqual(cache_row.data["hf_upvotes"], 87)

    def test_not_found_is_cached_as_empty_miss(self):
        paper = _paper("2606.00002")
        db.session.add(paper)
        db.session.commit()

        request_fn = MagicMock(side_effect=_http_error(404))
        provider = HuggingFaceProvider(request_fn=request_fn)

        first = provider.fetch_batch(["2606.00002"])
        second = provider.fetch_batch(["2606.00002"])

        # A 404 (paper never submitted to HF) is a durable, cacheable miss — an
        # empty payload, not an error, and not refetched within the TTL.
        self.assertEqual(first["2606.00002"], {})
        self.assertEqual(second["2606.00002"], {})
        self.assertEqual(request_fn.call_count, 1)

        cache_row = EnrichmentCache.query.filter_by(paper_id=paper.id, source="huggingface").one()
        self.assertEqual(cache_row.data, {})

    def test_rate_limit_aborts_remaining_fetches(self):
        db.session.add_all([_paper("2606.00003"), _paper("2606.00004")])
        db.session.commit()

        request_fn = MagicMock(side_effect=_http_error(429))
        provider = HuggingFaceProvider(request_fn=request_fn)

        payloads = provider.fetch_batch(["2606.00003", "2606.00004"])

        self.assertEqual(request_fn.call_count, 1)
        self.assertEqual(payloads, {})
        self.assertTrue(provider.rate_limited)

    def test_per_run_fetch_cap_respected(self):
        db.session.add_all([_paper("2606.00005"), _paper("2606.00006")])
        db.session.commit()

        request_fn = MagicMock(return_value=_hf_response())
        provider = HuggingFaceProvider(request_fn=request_fn, max_fetches=1)

        payloads = provider.fetch_batch(["2606.00005", "2606.00006"])

        self.assertEqual(request_fn.call_count, 1)
        self.assertEqual(len(payloads), 1)

    def test_non_json_response_skips_paper_without_aborting(self):
        db.session.add_all([_paper("2606.00007"), _paper("2606.00008")])
        db.session.commit()

        bad = MagicMock()
        bad.json.side_effect = ValueError("not json")
        request_fn = MagicMock(side_effect=[bad, _hf_response(upvotes=5, comments=0)])
        provider = HuggingFaceProvider(request_fn=request_fn)

        payloads = provider.fetch_batch(["2606.00007", "2606.00008"])

        self.assertNotIn("2606.00007", payloads)
        self.assertEqual(payloads["2606.00008"]["hf_upvotes"], 5)
        self.assertFalse(provider.rate_limited)

    def test_transient_error_is_not_cached(self):
        db.session.add(_paper("2606.00009"))
        db.session.commit()

        request_fn = MagicMock(side_effect=_http_error(500))
        provider = HuggingFaceProvider(request_fn=request_fn)

        payloads = provider.fetch_batch(["2606.00009"])

        self.assertEqual(payloads, {})
        self.assertEqual(EnrichmentCache.query.filter_by(source="huggingface").count(), 0)


class ScrapeEnrichmentTests(FlaskDBTestCase):
    @patch("app.services.enrichment_providers.HuggingFaceProvider")
    def test_fills_buzz_and_links_without_overwriting(self, mock_provider_cls):
        from app.services.scrape_engine import _enrich_results_with_huggingface

        existing = _paper("2607.00666")
        existing.github_repo = "orig/repo"
        existing.resource_links = [{"type": "code", "label": "Code", "url": "https://github.com/orig/repo"}]
        fresh = _paper("2607.00777")
        db.session.add_all([existing, fresh])
        db.session.commit()

        mock_provider_cls.return_value.fetch_batch.return_value = {
            "2607.00666": {
                "hf_upvotes": 87,
                "hf_comments_count": 7,
                "github_repo_url": "https://github.com/hf/other",
                "project_page_url": None,
            },
            "2607.00777": {
                "hf_upvotes": 3,
                "hf_comments_count": 0,
                "github_repo_url": "https://github.com/snumprlab/dart",
                "project_page_url": "https://twkang43.github.io/projects/dart/",
            },
        }

        results = [
            {
                "arxiv_id": "2607.00666",
                "resource_links": [{"type": "code", "label": "Code", "url": "https://github.com/orig/repo"}],
            },
            {"arxiv_id": "2607.00777", "resource_links": []},
        ]
        _enrich_results_with_huggingface(self.app, results, None, self.app.config["SCRAPER_CONFIG"])

        stored_existing = Paper.query.filter_by(arxiv_id="2607.00666").one()
        # Fill-only: the pre-existing repo is never overwritten, and the original
        # code link stays first (merge dedups with existing links winning).
        self.assertEqual(stored_existing.github_repo, "orig/repo")
        self.assertEqual(stored_existing.resource_links_list[0]["url"], "https://github.com/orig/repo")
        self.assertEqual(stored_existing.hf_upvotes, 87)
        self.assertEqual(stored_existing.hf_comments_count, 7)

        stored_fresh = Paper.query.filter_by(arxiv_id="2607.00777").one()
        self.assertEqual(stored_fresh.github_repo, "snumprlab/dart")
        self.assertEqual(stored_fresh.hf_upvotes, 3)
        urls = [link["url"] for link in stored_fresh.resource_links_list]
        self.assertIn("https://github.com/snumprlab/dart", urls)
        self.assertIn("https://twkang43.github.io/projects/dart/", urls)

        # The in-flight result dicts pick up the HF links so the downstream GitHub
        # stars/license pass can discover the repo in the same run.
        result_urls = [link["url"] for link in results[1]["resource_links"]]
        self.assertIn("https://github.com/snumprlab/dart", result_urls)

    @patch("app.services.enrichment_providers.HuggingFaceProvider")
    def test_config_can_disable_enrichment(self, mock_provider_cls):
        from app.services.scrape_engine import _enrich_results_with_huggingface

        db.session.add(_paper("2607.00888"))
        db.session.commit()

        config = dict(self.app.config["SCRAPER_CONFIG"])
        config["huggingface"] = {"enabled": False}
        _enrich_results_with_huggingface(self.app, [{"arxiv_id": "2607.00888"}], None, config)

        mock_provider_cls.assert_not_called()

    @patch("app.services.enrichment_providers.HuggingFaceProvider")
    def test_provider_failure_is_non_fatal(self, mock_provider_cls):
        from app.services.scrape_engine import _enrich_results_with_huggingface

        db.session.add(_paper("2607.00999"))
        db.session.commit()
        mock_provider_cls.return_value.fetch_batch.side_effect = RuntimeError("boom")

        # Must not raise: enrichment is best-effort and never crashes the scrape.
        _enrich_results_with_huggingface(
            self.app, [{"arxiv_id": "2607.00999"}], None, self.app.config["SCRAPER_CONFIG"]
        )

        self.assertIsNone(Paper.query.filter_by(arxiv_id="2607.00999").one().hf_upvotes)


class BackfillHuggingFaceTests(FlaskDBTestCase):
    @patch("app.enrich.HuggingFaceProvider")
    def test_backfill_updates_missing_papers(self, mock_provider_cls):
        paper = _paper("2601.00001")
        db.session.add(paper)
        db.session.commit()

        mock_provider_cls.return_value.fetch_batch.return_value = {
            "2601.00001": {
                "hf_upvotes": 12,
                "hf_comments_count": 2,
                "github_repo_url": "https://github.com/lab/model",
                "project_page_url": None,
            }
        }
        mock_provider_cls.return_value.rate_limited = False
        messages: list[str] = []

        updated = backfill_huggingface(self.app, batch_size=10, delay_seconds=0, emit=messages.append)

        stored = Paper.query.filter_by(arxiv_id="2601.00001").one()
        self.assertEqual(updated, 1)
        self.assertEqual(stored.hf_upvotes, 12)
        self.assertEqual(stored.hf_comments_count, 2)
        self.assertEqual(stored.github_repo, "lab/model")
        self.assertEqual(stored.resource_links_list[0]["url"], "https://github.com/lab/model")
        self.assertTrue(messages[-1].startswith("Hugging Face batch"))

    @patch("app.enrich.HuggingFaceProvider")
    def test_backfill_does_not_overwrite_existing_repo(self, mock_provider_cls):
        paper = _paper("2601.00002")
        paper.github_repo = "orig/repo"
        paper.resource_links = [{"type": "code", "label": "Code", "url": "https://github.com/orig/repo"}]
        db.session.add(paper)
        db.session.commit()

        mock_provider_cls.return_value.fetch_batch.return_value = {
            "2601.00002": {
                "hf_upvotes": 4,
                "hf_comments_count": 0,
                "github_repo_url": "https://github.com/hf/other",
                "project_page_url": None,
            }
        }
        mock_provider_cls.return_value.rate_limited = False

        updated = backfill_huggingface(self.app, batch_size=10, delay_seconds=0, emit=lambda _: None)

        stored = Paper.query.filter_by(arxiv_id="2601.00002").one()
        self.assertEqual(updated, 1)
        self.assertEqual(stored.github_repo, "orig/repo")
        self.assertEqual(stored.resource_links_list[0]["url"], "https://github.com/orig/repo")
        self.assertEqual(stored.hf_upvotes, 4)

    @patch("app.enrich.HuggingFaceProvider")
    def test_backfill_skips_cached_misses(self, mock_provider_cls):
        db.session.add(_paper("2601.00003"))
        db.session.commit()

        mock_provider_cls.return_value.fetch_batch.return_value = {"2601.00003": {}}
        mock_provider_cls.return_value.rate_limited = False

        updated = backfill_huggingface(self.app, batch_size=10, delay_seconds=0, emit=lambda _: None)

        self.assertEqual(updated, 0)
        self.assertIsNone(Paper.query.filter_by(arxiv_id="2601.00003").one().hf_upvotes)

    @patch("app.enrich.HuggingFaceProvider")
    def test_backfill_stops_on_rate_limit_without_advancing(self, mock_provider_cls):
        for idx in (4, 5):
            db.session.add(_paper(f"2601.0000{idx}"))
        db.session.commit()

        mock_provider_cls.return_value.fetch_batch.return_value = {}
        mock_provider_cls.return_value.rate_limited = True

        updated = backfill_huggingface(self.app, batch_size=1, delay_seconds=0, emit=lambda _: None)

        self.assertEqual(updated, 0)
        self.assertEqual(mock_provider_cls.return_value.fetch_batch.call_count, 1)

    @patch("backfill_cli.backfill_huggingface", return_value=0)
    @patch("backfill_cli.create_app")
    def test_main_routes_huggingface_command(self, mock_create_app, mock_backfill):
        mock_create_app.return_value = self.app

        exit_code = main(["huggingface", "--batch-size", "5", "--delay", "0"])

        self.assertEqual(exit_code, 0)
        mock_backfill.assert_called_once_with(self.app, batch_size=5, delay_seconds=0.0)


class SchemaAndBadgeTests(FlaskDBTestCase):
    def test_ensure_schema_adds_hf_columns_to_legacy_db(self):
        from sqlalchemy import inspect, text

        from app.schema import ensure_schema

        # Simulate a pre-feature DB (SQLite >= 3.35 supports DROP COLUMN), then
        # run the canonical additive migration — PAPER_COLUMN_DEFS must re-add
        # the columns, and must be idempotent on a second pass.
        db.session.execute(text("ALTER TABLE papers DROP COLUMN hf_upvotes"))
        db.session.execute(text("ALTER TABLE papers DROP COLUMN hf_comments_count"))
        db.session.commit()

        ensure_schema()
        columns = {col["name"] for col in inspect(db.engine).get_columns("papers")}
        self.assertIn("hf_upvotes", columns)
        self.assertIn("hf_comments_count", columns)

        ensure_schema()  # idempotent no-op

        db.session.add(_paper("2601.00010"))
        db.session.commit()
        self.assertIsNone(Paper.query.filter_by(arxiv_id="2601.00010").one().hf_upvotes)

    def test_badge_renders_only_for_positive_upvotes(self):
        from flask import render_template

        paper = _paper("2601.00011")
        paper.hf_upvotes = 87
        paper.hf_comments_count = 7
        db.session.add(paper)
        db.session.commit()

        html = render_template("partials/_paper_badges.html", paper=paper, BADGE=set())
        self.assertIn("🤗 87", html)
        self.assertIn("Hugging Face Papers upvotes", html)

        paper.hf_upvotes = 0
        html = render_template("partials/_paper_badges.html", paper=paper, BADGE=set())
        self.assertNotIn("🤗", html)

        paper.hf_upvotes = None
        html = render_template("partials/_paper_badges.html", paper=paper, BADGE=set())
        self.assertNotIn("🤗", html)


if __name__ == "__main__":
    unittest.main()
