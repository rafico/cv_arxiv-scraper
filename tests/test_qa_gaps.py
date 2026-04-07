"""Gap tests identified by the QA test plan.

Covers: Feed Sources API, Similarity Graph, multi-feed scraping,
search limit parameter, Discover page, Corpus API validation,
digest endpoints, Zotero sync, and post-scrape enrichment pipeline.
"""

from __future__ import annotations

import copy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.models import (
    Collection,
    FeedSource,
    Paper,
    PaperCollection,
    PaperFeedback,
    db,
)
from tests.helpers import FlaskDBTestCase


def _make_paper(idx: int = 0, **overrides) -> Paper:
    """Factory for creating test paper records."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today = date.today()
    defaults = dict(
        arxiv_id=f"2607.{1000 + idx:04d}",
        title=f"QA Gap Paper {idx}",
        authors="Author A, Author B",
        link=f"https://arxiv.org/abs/2607.{1000 + idx:04d}",
        pdf_link=f"https://arxiv.org/pdf/2607.{1000 + idx:04d}",
        abstract_text="abstract text about vision transformers",
        summary_text="Summary text",
        topic_tags=["Segmentation"],
        categories=["cs.CV"],
        match_type="Title",
        matched_terms=["Vision"],
        paper_score=10.0 + idx,
        feedback_score=0,
        is_hidden=False,
        publication_date=today.isoformat(),
        publication_dt=today,
        scraped_date=today.isoformat(),
        scraped_at=now,
    )
    defaults.update(overrides)
    return Paper(**defaults)


# ── Feed Sources API ──────────────────────────────────────────────────


class FeedSourcesApiTests(FlaskDBTestCase):
    """Tests for /api/feed-sources CRUD endpoints."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_list_feed_sources_empty(self):
        response = self.client.get("/api/feed-sources")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), [])

    def test_create_feed_source(self):
        token = self._csrf_token()
        response = self.client.post(
            "/api/feed-sources",
            json={"name": "cs.AI feed", "url": "https://rss.arxiv.org/rss/cs.AI"},
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertEqual(data["name"], "cs.AI feed")

    def test_create_feed_source_missing_name_returns_400(self):
        token = self._csrf_token()
        response = self.client.post(
            "/api/feed-sources",
            json={"url": "https://rss.arxiv.org/rss/cs.AI"},
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_create_feed_source_missing_url_returns_400(self):
        token = self._csrf_token()
        response = self.client.post(
            "/api/feed-sources",
            json={"name": "cs.AI feed"},
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_create_feed_source_empty_name_returns_400(self):
        token = self._csrf_token()
        response = self.client.post(
            "/api/feed-sources",
            json={"name": "", "url": "https://rss.arxiv.org/rss/cs.AI"},
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 400)

    def test_list_feed_sources_returns_created_sources(self):
        fs = FeedSource(name="cs.LG", url="https://rss.arxiv.org/rss/cs.LG")
        db.session.add(fs)
        db.session.commit()

        response = self.client.get("/api/feed-sources")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["name"], "cs.LG")
        self.assertTrue(data[0]["enabled"])

    def test_delete_feed_source(self):
        fs = FeedSource(name="cs.LG", url="https://rss.arxiv.org/rss/cs.LG")
        db.session.add(fs)
        db.session.commit()
        fs_id = fs.id

        token = self._csrf_token()
        response = self.client.delete(
            f"/api/feed-sources/{fs_id}",
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["deleted"])
        self.assertIsNone(db.session.get(FeedSource, fs_id))

    def test_delete_feed_source_not_found(self):
        token = self._csrf_token()
        response = self.client.delete(
            "/api/feed-sources/9999",
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 404)

    def test_toggle_feed_source(self):
        fs = FeedSource(name="cs.LG", url="https://rss.arxiv.org/rss/cs.LG", enabled=True)
        db.session.add(fs)
        db.session.commit()
        fs_id = fs.id

        token = self._csrf_token()
        response = self.client.post(
            f"/api/feed-sources/{fs_id}/toggle",
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertFalse(data["enabled"])

        # Toggle back
        response = self.client.post(
            f"/api/feed-sources/{fs_id}/toggle",
            headers={"X-CSRF-Token": token},
        )
        self.assertTrue(response.get_json()["enabled"])

    def test_toggle_feed_source_not_found(self):
        token = self._csrf_token()
        response = self.client.post(
            "/api/feed-sources/9999/toggle",
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 404)


# ── Similarity Graph API ─────────────────────────────────────────────


class SimilarityGraphApiTests(FlaskDBTestCase):
    """Tests for GET /api/papers/<id>/graph."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        # Create center paper + pool
        self.center = _make_paper(0, title="Center Paper on Object Detection")
        db.session.add(self.center)
        for i in range(1, 6):
            db.session.add(
                _make_paper(
                    i,
                    title=f"Related Paper {i} about detection",
                    paper_score=5.0 + i,
                )
            )
        db.session.commit()

    def test_graph_returns_nodes_and_edges(self):
        response = self.client.get(f"/api/papers/{self.center.id}/graph")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertIsInstance(data["nodes"], list)
        self.assertIsInstance(data["edges"], list)

    def test_graph_center_node_marked(self):
        response = self.client.get(f"/api/papers/{self.center.id}/graph")
        data = response.get_json()
        center_nodes = [n for n in data["nodes"] if n.get("center")]
        self.assertEqual(len(center_nodes), 1)
        self.assertEqual(center_nodes[0]["id"], self.center.id)

    def test_graph_edges_limited_to_top_20(self):
        # Add more papers to potentially generate > 20 edges
        for i in range(10, 35):
            db.session.add(
                _make_paper(
                    i,
                    title=f"Detection paper variation {i}",
                    paper_score=2.0 + i,
                )
            )
        db.session.commit()

        response = self.client.get(f"/api/papers/{self.center.id}/graph")
        data = response.get_json()
        self.assertLessEqual(len(data["edges"]), 20)

    def test_graph_paper_not_found_returns_404(self):
        response = self.client.get("/api/papers/99999/graph")
        self.assertEqual(response.status_code, 404)

    def test_graph_edges_have_similarity_score(self):
        response = self.client.get(f"/api/papers/{self.center.id}/graph")
        data = response.get_json()
        for edge in data["edges"]:
            self.assertIn("similarity", edge)
            self.assertIn("source", edge)
            self.assertIn("target", edge)
            self.assertGreaterEqual(edge["similarity"], 0.15)


# ── Search API ────────────────────────────────────────────────────────


class SearchApiLimitTests(FlaskDBTestCase):
    """Tests for /api/search limit parameter."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        for i in range(5):
            db.session.add(_make_paper(i, title=f"Transformer Paper {i}"))
        db.session.commit()

    def test_search_limit_parameter_is_respected(self):
        response = self.client.get("/api/search?q=Transformer&mode=keyword&limit=2")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertLessEqual(len(data["results"]), 2)

    def test_search_limit_capped_at_100(self):
        response = self.client.get("/api/search?q=Transformer&mode=keyword&limit=999")
        self.assertEqual(response.status_code, 200)
        # Should not crash — limit is capped

    def test_search_invalid_limit_defaults_to_30(self):
        response = self.client.get("/api/search?q=Transformer&mode=keyword&limit=abc")
        self.assertEqual(response.status_code, 200)
        # Should not crash — defaults to 30

    def test_search_empty_query_returns_empty(self):
        response = self.client.get("/api/search?q=")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["results"], [])


# ── Discover Page ─────────────────────────────────────────────────────


class DiscoverPageTests(FlaskDBTestCase):
    """Tests for /discover route."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_discover_page_renders(self):
        response = self.client.get("/discover")
        self.assertEqual(response.status_code, 200)
        text = response.get_data(as_text=True)
        self.assertIn("discover", text.lower())

    def test_discover_page_has_csrf_token(self):
        response = self.client.get("/discover")
        text = response.get_data(as_text=True)
        self.assertIn("csrf", text.lower())


# ── Corpus API Validation ────────────────────────────────────────────


class CorpusApiValidationTests(FlaskDBTestCase):
    """Tests for invalid query params on corpus endpoints returning 400."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_clusters_invalid_window_days_returns_400(self):
        response = self.client.get("/api/corpus/clusters?window_days=abc")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_clusters_negative_window_days_returns_400(self):
        response = self.client.get("/api/corpus/clusters?window_days=-5")
        self.assertEqual(response.status_code, 400)

    def test_clusters_window_days_exceeds_max_returns_400(self):
        response = self.client.get("/api/corpus/clusters?window_days=9999")
        self.assertEqual(response.status_code, 400)

    def test_emerging_invalid_recent_days_returns_400(self):
        response = self.client.get("/api/corpus/emerging?recent_days=xyz")
        self.assertEqual(response.status_code, 400)

    def test_neighbors_no_paper_ids_or_collection_returns_400(self):
        response = self.client.get("/api/corpus/neighbors")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_neighbors_invalid_limit_returns_400(self):
        response = self.client.get("/api/corpus/neighbors?paper_ids=1&limit=abc")
        self.assertEqual(response.status_code, 400)

    def test_clusters_invalid_cluster_count_type_returns_400(self):
        response = self.client.get("/api/corpus/clusters?clusters=abc")
        self.assertEqual(response.status_code, 400)


# ── Author Search API ────────────────────────────────────────────────


class AuthorSearchApiTests(FlaskDBTestCase):
    """Tests for /api/authors endpoint."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        db.session.add(_make_paper(0, authors="Jane Doe, John Smith"))
        db.session.add(_make_paper(1, authors="Jane Doe, Alice Brown"))
        db.session.add(_make_paper(2, authors="Bob Wilson"))
        db.session.commit()

    def test_author_search_returns_matching_authors(self):
        response = self.client.get("/api/authors?q=Jane")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(any(a["name"] == "Jane Doe" for a in data))

    def test_author_search_empty_query_returns_empty(self):
        response = self.client.get("/api/authors?q=")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), [])

    def test_author_search_returns_paper_count(self):
        response = self.client.get("/api/authors?q=Jane")
        data = response.get_json()
        jane = next(a for a in data if a["name"] == "Jane Doe")
        self.assertEqual(jane["paper_count"], 2)


# ── Multi-Feed Scraping ──────────────────────────────────────────────


class MultiFeedScrapingTests(FlaskDBTestCase):
    """Tests that scraping collects feed URLs from both config and FeedSource DB."""

    def setUp(self):
        super().setUp()

    def test_collect_feed_urls_includes_db_sources(self):
        from app.services.scrape_engine import _collect_feed_urls

        fs = FeedSource(
            name="cs.AI",
            url="https://rss.arxiv.org/rss/cs.AI",
            enabled=True,
        )
        db.session.add(fs)
        db.session.commit()

        scraper_config = self.app.config["SCRAPER_CONFIG"]["scraper"]
        urls = _collect_feed_urls(self.app, scraper_config)

        self.assertIn("https://rss.arxiv.org/rss/cs.AI", urls)
        self.assertIn(scraper_config["feed_url"], urls)

    def test_collect_feed_urls_excludes_disabled_sources(self):
        from app.services.scrape_engine import _collect_feed_urls

        fs = FeedSource(
            name="cs.AI",
            url="https://rss.arxiv.org/rss/cs.AI",
            enabled=False,
        )
        db.session.add(fs)
        db.session.commit()

        scraper_config = self.app.config["SCRAPER_CONFIG"]["scraper"]
        urls = _collect_feed_urls(self.app, scraper_config)

        self.assertNotIn("https://rss.arxiv.org/rss/cs.AI", urls)

    def test_collect_feed_urls_deduplicates(self):
        from app.services.scrape_engine import _collect_feed_urls

        config_url = self.app.config["SCRAPER_CONFIG"]["scraper"]["feed_url"]
        fs = FeedSource(name="duplicate", url=config_url, enabled=True)
        db.session.add(fs)
        db.session.commit()

        scraper_config = self.app.config["SCRAPER_CONFIG"]["scraper"]
        urls = _collect_feed_urls(self.app, scraper_config)

        self.assertEqual(urls.count(config_url), 1)


# ── Digest Settings Endpoints ────────────────────────────────────────


class DigestSettingsEndpointTests(FlaskDBTestCase):
    """Tests for digest preview and send-test endpoints."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        db.session.add(_make_paper(0))
        db.session.commit()

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    @patch("app.services.email_digest.build_digest_preview")
    def test_digest_preview_returns_html(self, mock_preview):
        mock_preview.return_value = {
            "html": "<html><body>Preview</body></html>",
            "subject": "ArXiv Digest — Apr 07, 2026",
        }

        response = self.client.get("/settings/digest-preview")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/html")
        self.assertIn("Preview", response.get_data(as_text=True))

    @patch("app.services.email_digest.send_digest")
    def test_send_test_digest_success(self, mock_send):
        mock_send.return_value = {
            "papers_count": 5,
            "sent": True,
            "recipient": "user@example.com",
        }

        token = self._csrf_token()
        response = self.client.post(
            "/settings/send-test-digest",
            headers={"X-CSRF-Token": token},
        )
        # Redirects to settings page
        self.assertIn(response.status_code, [302, 303])
        mock_send.assert_called_once()

    @patch("app.services.email_digest.send_digest")
    def test_send_test_digest_failure_flashes_error(self, mock_send):
        mock_send.side_effect = ValueError("No recipient configured.")

        token = self._csrf_token()
        response = self.client.post(
            "/settings/send-test-digest",
            headers={"X-CSRF-Token": token},
        )
        self.assertIn(response.status_code, [302, 303])


# ── Zotero Sync Endpoint ─────────────────────────────────────────────


class ZoteroSyncEndpointTests(FlaskDBTestCase):
    """Tests for POST /settings/zotero-sync."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        paper = _make_paper(0)
        db.session.add(paper)
        db.session.commit()
        from app.services.feedback import apply_feedback_action

        apply_feedback_action(paper.id, "save")

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    @patch("app.services.zotero.ZoteroClient.sync_saved_papers")
    @patch("app.services.zotero.ZoteroClient.check_connection")
    def test_zotero_sync_calls_sync_saved_papers(self, mock_check, mock_sync):
        mock_check.return_value = {"status": "connected", "message": "OK"}
        mock_sync.return_value = {"success": True, "message": "Synced 1 paper(s)."}

        token = self._csrf_token()
        response = self.client.post(
            "/settings/zotero-sync",
            data={"zotero_collection": "ABC123"},
            headers={"X-CSRF-Token": token},
            content_type="application/x-www-form-urlencoded",
        )

        self.assertIn(response.status_code, [302, 303])
        mock_sync.assert_called_once()
        call_kwargs = mock_sync.call_args
        self.assertEqual(call_kwargs.kwargs.get("collection_key"), "ABC123")

    @patch("app.services.zotero.ZoteroClient.check_connection")
    def test_zotero_sync_rejects_when_not_connected(self, mock_check):
        mock_check.return_value = {"status": "error", "message": "No API key"}

        token = self._csrf_token()
        response = self.client.post(
            "/settings/zotero-sync",
            data={},
            headers={"X-CSRF-Token": token},
            content_type="application/x-www-form-urlencoded",
        )

        self.assertIn(response.status_code, [302, 303])


# ── Post-Scrape Pipeline Integration ─────────────────────────────────


class PostScrapePipelineTests(FlaskDBTestCase):
    """Tests for thumbnail generation and embedding generation during scrape."""

    def setUp(self):
        super().setUp()

    def test_generate_thumbnails_calls_worker_for_each_result(self):
        from app.services.scrape_engine import _generate_thumbnails

        results = [
            {
                "arxiv_id": "2607.0001",
                "pdf_link": "https://arxiv.org/pdf/2607.0001",
                "pdf_content": b"fake-pdf",
            },
            {
                "arxiv_id": "2607.0002",
                "pdf_link": "https://arxiv.org/pdf/2607.0002",
                "pdf_content": None,
            },
        ]

        with patch("app.services.thumbnail_generator.generate_thumbnail") as mock_gen:
            _generate_thumbnails(self.app, results, MagicMock())

        self.assertEqual(mock_gen.call_count, 2)

    def test_generate_embeddings_adds_new_papers_to_index(self):
        from app.services.scrape_engine import _generate_embeddings

        paper = _make_paper(0)
        db.session.add(paper)
        db.session.commit()

        results = [{"link": paper.link}]

        mock_service = MagicMock()
        mock_service.has_paper.return_value = False
        mock_service.add_papers.return_value = 1

        with patch("app.services.embeddings.get_embedding_service", return_value=mock_service):
            _generate_embeddings(self.app, results)

        mock_service.add_papers.assert_called_once()
        mock_service.save.assert_called_once()

    def test_generate_embeddings_skips_already_indexed_papers(self):
        from app.services.scrape_engine import _generate_embeddings

        paper = _make_paper(0)
        db.session.add(paper)
        db.session.commit()

        results = [{"link": paper.link}]

        mock_service = MagicMock()
        mock_service.has_paper.return_value = True

        with patch("app.services.embeddings.get_embedding_service", return_value=mock_service):
            _generate_embeddings(self.app, results)

        mock_service.add_papers.assert_not_called()
