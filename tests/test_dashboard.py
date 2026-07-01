import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from app.models import Paper, db
from app.services.feedback import apply_feedback_action
from tests.helpers import FlaskDBTestCase


class DashboardRouteTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        today = date.today()

        for idx in range(30):
            publication_dt = today if idx < 20 else today - timedelta(days=45)
            paper = Paper(
                arxiv_id=f"2602.{1000 + idx}",
                title=f"Paper {idx}",
                authors="Author A",
                link=f"https://arxiv.org/abs/2602.{1000 + idx}",
                pdf_link=f"https://arxiv.org/pdf/2602.{1000 + idx}",
                abstract_text="vision transformer segmentation",
                summary_text="Summary text",
                topic_tags=["Segmentation", "Vision"],
                categories=["cs.CV"] if idx % 2 == 0 else ["cs.RO"],
                resource_links=([{"type": "code", "url": f"https://example.com/code/{idx}"}] if idx % 3 == 0 else []),
                match_type="Title",
                matched_terms=["vision"],
                paper_score=10.0 + idx,
                feedback_score=0,
                is_hidden=False,
                publication_date=publication_dt.isoformat(),
                publication_dt=publication_dt,
                scraped_date=today.isoformat(),
                scraped_at=now,
            )
            db.session.add(paper)
        db.session.commit()

        static_root = Path(self._tmpdir.name) / "static"
        static_root.mkdir(parents=True, exist_ok=True)
        self.app.static_folder = str(static_root)
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_default_daily_timeframe_filters_old_papers(self):
        response = self.client.get("/")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Inbox", text)
        self.assertNotIn("Paper 29", text)

    def test_pagination_preserves_active_filters(self):
        # All 30 seeded papers have authors="Author A"; with timeframe=all that is two
        # pages (per_page=24). The "Next" link must keep the author filter, else page 2
        # is computed against the unfiltered query (regression: it dropped it).
        import re

        response = self.client.get("/?author=Author+A&timeframe=all&view=inbox")
        self.assertEqual(response.status_code, 200)
        text = response.get_data(as_text=True)
        next_hrefs = re.findall(r'href="([^"]*page=2[^"]*)"', text)
        self.assertTrue(next_hrefs, "expected a page=2 pagination link")
        self.assertTrue(
            any("author=Author" in href for href in next_hrefs),
            f"pagination link dropped the author filter: {next_hrefs}",
        )

    def _add_paper(self, *, title: str, arxiv_id: str, published_days_ago: int) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        publication_dt = date.today() - timedelta(days=published_days_ago)
        db.session.add(
            Paper(
                arxiv_id=arxiv_id,
                title=title,
                authors="Author B",
                link=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_link=f"https://arxiv.org/pdf/{arxiv_id}",
                abstract_text="vision transformer segmentation",
                match_type="Title",
                matched_terms=["vision"],
                paper_score=999.0,  # sort onto page 1 regardless of the seeded papers
                feedback_score=0,
                is_hidden=False,
                publication_date=publication_dt.isoformat(),
                publication_dt=publication_dt,
                scraped_date=date.today().isoformat(),
                scraped_at=now,  # scraped just now
            )
        )
        db.session.commit()

    def test_daily_inbox_shows_announcement_lagged_paper(self):
        # arXiv announces papers a few days after their publication date, so a
        # just-scraped paper routinely carries a recent-past publication_dt. It
        # must still appear in the default daily inbox (regression: it didn't).
        self._add_paper(title="Announcement Lagged Paper", arxiv_id="2602.9001", published_days_ago=3)

        response = self.client.get("/")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Announcement Lagged Paper", text)

    def test_daily_inbox_excludes_recently_scraped_stale_paper(self):
        # A genuinely old paper (well beyond the announcement-lag window) that is
        # scraped today must not flood the daily inbox.
        self._add_paper(title="Stale Backfilled Paper", arxiv_id="2602.9002", published_days_ago=30)

        response = self.client.get("/")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Stale Backfilled Paper", text)

    def test_all_time_second_page_available(self):
        response = self.client.get("/?timeframe=all&page=2")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Page 2", text)

    def test_feedback_endpoint_toggles_action(self):
        paper = Paper.query.first()
        token = self._csrf_token()
        response = self.client.post(
            f"/api/papers/{paper.id}/feedback",
            json={"action": "save"},
            headers={"X-CSRF-Token": token},
        )
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(data["active"])
        self.assertEqual(data["counts"]["save"], 1)

    def test_user_tag_with_quote_is_not_injected_into_inline_js_string(self):
        paper = Paper.query.first()
        paper.user_tags = ['bad";alert(1)//']
        db.session.commit()

        response = self.client.get("/")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('data-tag="bad&#34;;alert(1)//"', text)
        self.assertNotIn(f"removeTag({paper.id}, 'bad", text)

    def test_saved_view_lists_only_saved_papers(self):
        saved_paper = Paper.query.filter_by(title="Paper 0").first()
        apply_feedback_action(saved_paper.id, "save")

        response = self.client.get("/?view=saved")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Saved", text)
        self.assertIn("Paper 0", text)
        self.assertIn("Recently Saved", text)

    def test_category_filter_limits_results(self):
        response = self.client.get("/?timeframe=all&category=cs.RO")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("https://arxiv.org/abs/2602.1001", text)
        self.assertNotIn("https://arxiv.org/abs/2602.1000", text)
        self.assertIn("cs.RO", text)

    def test_venue_filter_and_badge(self):
        accepted = Paper.query.filter_by(title="Paper 0").one()
        accepted.arxiv_comment = "Accepted to CVPR 2026 (oral)"
        accepted.venue = "CVPR"
        accepted.venue_year = 2026
        accepted.acceptance_status = "oral"
        db.session.commit()

        response = self.client.get("/?timeframe=all&venue=CVPR")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Paper 0", text)
        self.assertNotIn("Paper 1<", text)
        self.assertIn("CVPR 2026 · Oral", text)
        self.assertIn("All venues", text)

    def test_teaser_route_serves_existing_file(self):
        paper = Paper.query.filter_by(title="Paper 0").one()
        thumbnails_dir = Path(self.app.static_folder) / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        (thumbnails_dir / f"{paper.arxiv_id}_teaser.png").write_bytes(b"\x89PNG\r\n\x1a\nteaser")

        response = self.client.get(f"/papers/{paper.id}/teaser.png")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"teaser", response.data)

    def test_teaser_route_falls_back_to_thumbnail(self):
        paper = Paper.query.filter_by(title="Paper 0").one()
        thumbnails_dir = Path(self.app.static_folder) / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        (thumbnails_dir / f"{paper.arxiv_id}.png").write_bytes(b"\x89PNG\r\n\x1a\npage-one")

        response = self.client.get(f"/papers/{paper.id}/teaser.png")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"page-one", response.data)

    def test_visual_density_toggle_switches_grid_and_teasers(self):
        response = self.client.get("/?timeframe=all&density=visual")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("lg:grid-cols-4", text)
        self.assertIn("/teaser.png", text)
        self.assertIn("List view", text)

        default = self.client.get("/?timeframe=all").get_data(as_text=True)
        self.assertIn('id="paper-list"', default)
        self.assertNotIn("/teaser.png", default)
        self.assertIn("Visual grid", default)

    def test_invalid_density_falls_back_to_list(self):
        response = self.client.get("/?timeframe=all&density=bogus")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="paper-list"', response.get_data(as_text=True))

    def test_legacy_comfortable_density_maps_to_list(self):
        response = self.client.get("/?timeframe=all&density=comfortable")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="paper-list"', response.get_data(as_text=True))

    def test_dataset_filter_and_insight_chips(self):
        paper = Paper.query.filter_by(title="Paper 0").one()
        paper.llm_insights = {
            "tasks": ["object detection"],
            "datasets": ["COCO", "LVIS"],
            "method_type": "transformer",
            "backbone": "ViT-L",
            "why_matched": "Zero-shot detection overlaps your interests.",
        }
        db.session.commit()

        response = self.client.get("/?timeframe=all&dataset=COCO")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Paper 0", text)
        self.assertNotIn("Paper 1<", text)
        self.assertIn("COCO", text)
        self.assertIn("Why for you:", text)
        self.assertIn("Zero-shot detection overlaps your interests.", text)

    def test_dataset_filter_quote_delimited_avoids_substring_collisions(self):
        coco = Paper.query.filter_by(title="Paper 0").one()
        coco.llm_insights = {"datasets": ["COCO"]}
        coco_stuff = Paper.query.filter_by(title="Paper 2").one()
        coco_stuff.llm_insights = {"datasets": ["COCO-Stuff"]}
        db.session.commit()

        response = self.client.get("/?timeframe=all&dataset=COCO")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Paper 0", text)
        self.assertNotIn("Paper 2<", text)

    def test_mentioned_venue_shows_no_badge(self):
        paper = Paper.query.filter_by(title="Paper 0").one()
        paper.venue = "ICLR"
        paper.acceptance_status = "mentioned"
        db.session.commit()

        response = self.client.get("/?timeframe=all")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("ICLR · ", text)

    def test_resource_filter_limits_results(self):
        response = self.client.get("/?timeframe=all&resource_filter=available")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("https://example.com/code/0", text)
        self.assertNotIn("https://arxiv.org/abs/2602.1001", text)
        self.assertIn("Has resources", text)

    def test_dashboard_shows_citation_provenance_tooltip_and_openalex_fallback(self):
        paper = Paper.query.filter_by(title="Paper 0").one()
        paper.citation_count = None
        paper.openalex_id = "W999"
        paper.openalex_cited_by_count = 17
        paper.citation_source = "openalex"
        paper.citation_provenance = {"source": "openalex", "updated_at": "2026-04-01T12:00:00"}
        db.session.commit()

        response = self.client.get("/")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Citation count from OpenAlex, updated 2026-04-01", text)
        self.assertIn("https://openalex.org/W999", text)
        self.assertRegex(text, re.compile(r">\s*17\s*</a>"))

    def test_dashboard_uses_app_thumbnail_route(self):
        paper = Paper.query.filter_by(title="Paper 0").one()

        response = self.client.get("/")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn(f"/papers/{paper.id}/thumbnail.png", text)
        self.assertNotIn("cdn-thumbnails.huggingface.co", text)

    @patch("app.services.mendeley.MendeleyClient.check_connection")
    def test_dashboard_shows_mendeley_button_when_connected(self, mock_check_connection):
        mock_check_connection.return_value = {"status": "connected", "message": "Mendeley is connected."}

        response = self.client.get("/")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Send to Mendeley", text)
        self.assertIn("mendeley-sync-btn", text)

    @patch("app.services.mendeley.MendeleyClient.check_connection")
    def test_dashboard_hides_mendeley_button_when_not_connected(self, mock_check_connection):
        mock_check_connection.return_value = {"status": "no_token", "message": "Mendeley not authorized."}

        response = self.client.get("/")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Send to Mendeley", text)
        self.assertNotIn("mendeley-sync-btn", text)

    def test_paper_thumbnail_route_serves_existing_thumbnail(self):
        paper = Paper.query.filter_by(title="Paper 0").one()
        thumbnails_dir = Path(self.app.static_folder) / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        thumbnail_path = thumbnails_dir / f"{paper.arxiv_id}.png"
        thumbnail_path.write_bytes(b"png-bytes")

        response = self.client.get(f"/papers/{paper.id}/thumbnail.png")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/png")
        self.assertEqual(response.get_data(), b"png-bytes")

    def test_paper_thumbnail_route_warms_in_background_when_missing(self):
        paper = Paper.query.filter_by(title="Paper 0").one()

        # A missing thumbnail must NOT be generated inline (a PDF download + render
        # would block the worker thread and freeze the UI). The route returns an
        # uncacheable placeholder and enqueues a background warm instead.
        with patch("app.routes.dashboard.THUMBNAIL_WARMER.warm") as mock_warm:
            response = self.client.get(f"/papers/{paper.id}/thumbnail.png")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        mock_warm.assert_called_once_with(paper.arxiv_id, paper.pdf_link, Path(self.app.static_folder))

    def test_teaser_route_warms_in_background_when_both_missing(self):
        paper = Paper.query.filter_by(title="Paper 0").one()

        with patch("app.routes.dashboard.THUMBNAIL_WARMER.warm") as mock_warm:
            response = self.client.get(f"/papers/{paper.id}/teaser.png")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        mock_warm.assert_called_once_with(paper.arxiv_id, paper.pdf_link, Path(self.app.static_folder))

    def test_mendeley_status_stale_refreshes_off_request_thread(self):
        import threading
        import time as _time

        from app.routes import dashboard as dash

        cache = self.app.extensions.setdefault(
            "mendeley_status_cache", {"ts": 0.0, "connected": False, "refreshing": False}
        )
        # Warm-but-stale: a connected value cached just past the TTL.
        cache["ts"] = _time.monotonic() - (dash._MENDELEY_STATUS_TTL + 1)
        cache["connected"] = True
        cache["refreshing"] = False

        # Gate the probe so it can't finish until after we observe the immediate
        # return — proving the request thread does not wait on the network.
        release = threading.Event()

        def _blocking_check():
            release.wait(timeout=5)
            return {"status": "no_token"}

        with self.app.test_request_context("/"):
            with patch(
                "app.services.mendeley.MendeleyClient.check_connection",
                side_effect=_blocking_check,
            ) as mock_check:
                result = dash._mendeley_connected()
                # Stale cached value served immediately, while the slow probe is
                # still blocked on the background thread.
                self.assertTrue(result)
                self.assertTrue(cache["refreshing"])
                release.set()
                # Drain the single-thread refresh pool (FIFO) so the refresh has
                # finished before asserting its effect.
                dash._MENDELEY_EXECUTOR.submit(lambda: None).result(timeout=5)

        mock_check.assert_called_once()
        self.assertFalse(cache["connected"])
        self.assertFalse(cache["refreshing"])

    def test_paper_thumbnail_route_rejects_traversal_arxiv_id(self):
        paper = Paper(
            arxiv_id="../../etc/passwd",
            title="Evil",
            authors="A",
            link="https://arxiv.org/abs/evil",
            pdf_link="https://arxiv.org/pdf/evil",
            abstract_text="x",
            summary_text="x",
            topic_tags=[],
            categories=["cs.CV"],
            resource_links=[],
            match_type="Title",
            matched_terms=[],
            paper_score=1.0,
            feedback_score=0,
            is_hidden=False,
            publication_date=date.today().isoformat(),
            publication_dt=date.today(),
            scraped_date=date.today().isoformat(),
            scraped_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        db.session.add(paper)
        db.session.commit()

        with patch("app.routes.dashboard.THUMBNAIL_WARMER.warm") as mock_warm:
            response = self.client.get(f"/papers/{paper.id}/thumbnail.png")

        self.assertEqual(response.status_code, 404)
        mock_warm.assert_not_called()

    def test_paper_thumbnail_route_rejects_absolute_path_arxiv_id(self):
        paper = Paper(
            arxiv_id="/etc/passwd",
            title="Evil Abs",
            authors="A",
            link="https://arxiv.org/abs/abs",
            pdf_link="https://arxiv.org/pdf/abs",
            abstract_text="x",
            summary_text="x",
            topic_tags=[],
            categories=["cs.CV"],
            resource_links=[],
            match_type="Title",
            matched_terms=[],
            paper_score=1.0,
            feedback_score=0,
            is_hidden=False,
            publication_date=date.today().isoformat(),
            publication_dt=date.today(),
            scraped_date=date.today().isoformat(),
            scraped_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        db.session.add(paper)
        db.session.commit()

        with patch("app.routes.dashboard.THUMBNAIL_WARMER.warm") as mock_warm:
            response = self.client.get(f"/papers/{paper.id}/thumbnail.png")

        self.assertEqual(response.status_code, 404)
        mock_warm.assert_not_called()

    def test_paper_thumbnail_route_accepts_legacy_arxiv_id_with_slash(self):
        paper = Paper(
            arxiv_id="cs.CV/9912345",
            title="Legacy",
            authors="A",
            link="https://arxiv.org/abs/cs.CV/9912345",
            pdf_link="https://arxiv.org/pdf/cs.CV/9912345",
            abstract_text="x",
            summary_text="x",
            topic_tags=[],
            categories=["cs.CV"],
            resource_links=[],
            match_type="Title",
            matched_terms=[],
            paper_score=1.0,
            feedback_score=0,
            is_hidden=False,
            publication_date=date.today().isoformat(),
            publication_dt=date.today(),
            scraped_date=date.today().isoformat(),
            scraped_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        db.session.add(paper)
        db.session.commit()

        thumbnails_dir = Path(self.app.static_folder) / "thumbnails" / "cs.CV"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        (thumbnails_dir / "9912345.png").write_bytes(b"legacy-png")

        response = self.client.get(f"/papers/{paper.id}/thumbnail.png")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(), b"legacy-png")
