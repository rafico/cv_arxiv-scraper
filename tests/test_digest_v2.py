"""Tests for Digest 2.0: one-tap feedback, catch-up window, controls, CID figures, alerts."""

from __future__ import annotations

import base64
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from email import message_from_bytes
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.models import DigestRun, Paper, PaperFeedback, SavedSearch, db
from app.services.email_digest import (
    DIGEST_WEEKDAY_KEYS,
    _one_tap_serializer,
    build_digest_preview,
    get_digest_config,
    load_one_tap_token,
    make_one_tap_token,
    send_digest,
)
from app.services.text import now_utc, utc_today
from tests.helpers import FlaskDBTestCase


def _make_paper(**overrides) -> Paper:
    defaults = dict(
        title="Digest V2 Paper",
        authors="Alice, Bob",
        link="https://arxiv.org/abs/2607.00001",
        pdf_link="https://arxiv.org/pdf/2607.00001",
        abstract_text="An abstract.",
        summary_text="A summary.",
        topic_tags=["vision"],
        categories=["cs.CV"],
        resource_links=[],
        match_type="Author",
        matched_terms=["Alice"],
        paper_score=42.5,
        feedback_score=0,
        is_hidden=False,
        publication_date="2026-07-01",
        scraped_date="2026-07-01",
        publication_dt=date(2026, 7, 1),
        scraped_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    defaults.update(overrides)
    return Paper(**defaults)


class OneTapTokenTests(FlaskDBTestCase):
    def test_round_trip(self):
        token = make_one_tap_token(self.app, 5, "save", 7)
        data = load_one_tap_token(self.app, token)
        self.assertEqual(data, {"p": 5, "a": "save", "r": 7})

    def test_tampered_token_rejected(self):
        from itsdangerous import BadSignature

        token = make_one_tap_token(self.app, 5, "save")
        tampered = token[:-4] + ("aaaa" if not token.endswith("aaaa") else "bbbb")
        with self.assertRaises(BadSignature):
            load_one_tap_token(self.app, tampered)

    def test_expired_token_rejected(self):
        from itsdangerous import SignatureExpired

        token = make_one_tap_token(self.app, 5, "save")
        with self.assertRaises(SignatureExpired):
            load_one_tap_token(self.app, token, max_age=-1)

    def test_malformed_payload_rejected(self):
        from itsdangerous import BadSignature

        serializer = _one_tap_serializer(self.app.config["SECRET_KEY"])
        for payload in ({"p": "not-int", "a": "save", "r": None}, {"p": 3, "a": "delete", "r": None}, ["p", "a"]):
            token = serializer.dumps(payload)
            with self.assertRaises(BadSignature):
                load_one_tap_token(self.app, token)


class OneTapEndpointTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        self.paper = _make_paper()
        db.session.add(self.paper)
        db.session.commit()

    def test_save_applies_feedback(self):
        token = make_one_tap_token(self.app, self.paper.id, "save")
        response = self.client.get(f"/api/feedback/one-tap?token={token}")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Saved", response.get_data(as_text=True))
        rows = PaperFeedback.query.filter_by(paper_id=self.paper.id, action="save").all()
        self.assertEqual(len(rows), 1)

    def test_repeat_click_is_idempotent(self):
        token = make_one_tap_token(self.app, self.paper.id, "save")
        self.client.get(f"/api/feedback/one-tap?token={token}")
        db.session.expire_all()
        score_after_first = int(db.session.get(Paper, self.paper.id).feedback_score or 0)

        response = self.client.get(f"/api/feedback/one-tap?token={token}")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Already recorded", response.get_data(as_text=True))
        db.session.expire_all()
        rows = PaperFeedback.query.filter_by(paper_id=self.paper.id, action="save").all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(db.session.get(Paper, self.paper.id).feedback_score or 0), score_after_first)

    def test_skip_hides_paper(self):
        token = make_one_tap_token(self.app, self.paper.id, "skip")
        response = self.client.get(f"/api/feedback/one-tap?token={token}")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Skipped", response.get_data(as_text=True))
        db.session.expire_all()
        self.assertTrue(db.session.get(Paper, self.paper.id).is_hidden)

    def test_missing_and_invalid_tokens_rejected(self):
        self.assertEqual(self.client.get("/api/feedback/one-tap").status_code, 400)
        self.assertEqual(self.client.get("/api/feedback/one-tap?token=garbage").status_code, 400)
        self.assertFalse(PaperFeedback.query.all())

    def test_unknown_paper_returns_404(self):
        token = make_one_tap_token(self.app, 999_999, "save")
        response = self.client.get(f"/api/feedback/one-tap?token={token}")
        self.assertEqual(response.status_code, 404)


class CatchUpWindowTests(FlaskDBTestCase):
    def test_no_previous_run_keeps_default_window(self):
        db.session.add(
            _make_paper(title="Old", link="https://arxiv.org/abs/2607.10001", scraped_at=now_utc() - timedelta(days=3))
        )
        db.session.commit()

        preview = build_digest_preview(self.app)

        self.assertFalse(preview["catch_up"])
        self.assertEqual(preview["lookback_hours"], 26)
        self.assertNotIn("Old", [p.title for p in preview["papers"]])
        self.assertNotIn("Catch-up", preview["subject"])

    def test_gap_widens_window_and_labels_email(self):
        db.session.add(DigestRun(status="success", started_at=now_utc() - timedelta(days=4)))
        db.session.add(
            _make_paper(
                title="Missed", link="https://arxiv.org/abs/2607.10002", scraped_at=now_utc() - timedelta(days=3)
            )
        )
        db.session.commit()

        preview = build_digest_preview(self.app)

        self.assertTrue(preview["catch_up"])
        self.assertGreater(preview["lookback_hours"], 26)
        self.assertIn("Missed", [p.title for p in preview["papers"]])
        self.assertIn("Catch-up digest — last", preview["subject"])
        self.assertIn("Catch-up digest", preview["html"])

    def test_gap_is_capped_at_seven_days(self):
        db.session.add(DigestRun(status="success", started_at=now_utc() - timedelta(days=30)))
        db.session.commit()

        preview = build_digest_preview(self.app)

        self.assertTrue(preview["catch_up"])
        self.assertEqual(preview["lookback_hours"], 7 * 24)
        self.assertIn("last 7 days", preview["subject"])

    def test_recent_run_does_not_trigger_catch_up(self):
        db.session.add(DigestRun(status="success", started_at=now_utc() - timedelta(hours=24)))
        db.session.commit()

        preview = build_digest_preview(self.app)
        self.assertFalse(preview["catch_up"])


class DigestControlsTests(FlaskDBTestCase):
    def test_min_score_filters_papers(self):
        self.app.config["SCRAPER_CONFIG"]["digest"] = {"min_score": 20.0}
        db.session.add_all(
            [
                _make_paper(title="High", link="https://arxiv.org/abs/2607.20001", paper_score=50.0),
                _make_paper(title="Low", link="https://arxiv.org/abs/2607.20002", paper_score=5.0),
            ]
        )
        db.session.commit()

        titles = [p.title for p in build_digest_preview(self.app)["papers"]]
        self.assertEqual(titles, ["High"])

    def test_max_papers_caps_list(self):
        self.app.config["SCRAPER_CONFIG"]["digest"] = {"max_papers": 2}
        db.session.add_all(
            _make_paper(title=f"P{i}", link=f"https://arxiv.org/abs/2607.3000{i}", paper_score=10.0 + i)
            for i in range(4)
        )
        db.session.commit()

        preview = build_digest_preview(self.app)
        self.assertEqual([p.title for p in preview["papers"]], ["P3", "P2"])

    def test_weekday_gating_skips_scheduled_send(self):
        today_key = DIGEST_WEEKDAY_KEYS[utc_today().weekday()]
        self.app.config["SCRAPER_CONFIG"]["email"] = {"recipient": "a@b.com"}
        self.app.config["SCRAPER_CONFIG"]["digest"] = {
            "weekdays": [day for day in DIGEST_WEEKDAY_KEYS if day != today_key]
        }

        result = send_digest(self.app, dry_run=True)

        self.assertEqual(result["skipped_reason"], "weekday")
        self.assertFalse(result["sent"])
        self.assertEqual(DigestRun.query.count(), 0)

    def test_force_bypasses_weekday_gating(self):
        today_key = DIGEST_WEEKDAY_KEYS[utc_today().weekday()]
        self.app.config["SCRAPER_CONFIG"]["email"] = {"recipient": "a@b.com"}
        self.app.config["SCRAPER_CONFIG"]["digest"] = {
            "weekdays": [day for day in DIGEST_WEEKDAY_KEYS if day != today_key]
        }

        result = send_digest(self.app, dry_run=True, force=True)

        self.assertNotIn("skipped_reason", result)
        self.assertEqual(DigestRun.query.count(), 1)

    def test_get_digest_config_defaults_and_bad_values(self):
        cfg = get_digest_config(self.app)
        self.assertEqual(cfg["weekdays"], list(DIGEST_WEEKDAY_KEYS))
        self.assertEqual(cfg["max_papers"], 15)
        self.assertEqual(cfg["min_score"], 0.0)
        self.assertEqual(cfg["exploration_slots"], 2)
        self.assertTrue(cfg["base_url"].startswith("http://127.0.0.1:"))

        self.app.config["SCRAPER_CONFIG"]["digest"] = {
            "weekdays": "tuesday",
            "min_score": "nan",
            "max_papers": 5000,
            "exploration_slots": "many",
            "base_url": "http://127.0.0.1:5000/",
        }
        cfg = get_digest_config(self.app)
        self.assertEqual(cfg["weekdays"], list(DIGEST_WEEKDAY_KEYS))
        self.assertEqual(cfg["min_score"], 0.0)
        self.assertEqual(cfg["max_papers"], 100)
        self.assertEqual(cfg["exploration_slots"], 2)
        self.assertEqual(cfg["base_url"], "http://127.0.0.1:5000")

        self.app.config["SCRAPER_CONFIG"]["digest"] = {"exploration_slots": 5000}
        self.assertEqual(get_digest_config(self.app)["exploration_slots"], 10)


class ExplorationSlotTests(FlaskDBTestCase):
    def test_exploration_prefers_papers_the_model_knows_least(self):
        self.app.config["SCRAPER_CONFIG"]["digest"] = {"max_papers": 2, "exploration_slots": 2}
        db.session.add_all(
            [
                _make_paper(
                    title="Main A", link="https://arxiv.org/abs/2608.1", paper_score=90.0, interest_similarity=0.9
                ),
                _make_paper(
                    title="Main B", link="https://arxiv.org/abs/2608.2", paper_score=80.0, interest_similarity=0.8
                ),
                _make_paper(
                    title="Known", link="https://arxiv.org/abs/2608.3", paper_score=70.0, interest_similarity=0.7
                ),
                _make_paper(
                    title="Unknown", link="https://arxiv.org/abs/2608.4", paper_score=10.0, interest_similarity=None
                ),
                _make_paper(
                    title="Far", link="https://arxiv.org/abs/2608.5", paper_score=20.0, interest_similarity=0.05
                ),
            ]
        )
        db.session.commit()

        preview = build_digest_preview(self.app)

        self.assertEqual([p.title for p in preview["papers"]], ["Main A", "Main B"])
        # NULL similarity first, then the lowest — never the already-selected mains.
        self.assertEqual({p.title for p in preview["exploration"]}, {"Unknown", "Far"})
        self.assertIn("Exploration", preview["html"])
        self.assertIn("Unknown", preview["html"])

    def _synthesis_payload(self, narrative=None):
        return {
            "window_days": 7,
            "topics": [{"label": "diffusion", "recent_count": 4, "delta_share": 0.12, "sample_titles": ["T1"]}],
            "narrative": narrative,
            "citations": [],
        }

    def test_synthesis_renders_on_configured_weekday(self):
        from app.services.email_digest import DIGEST_WEEKDAY_KEYS, utc_today

        today_key = DIGEST_WEEKDAY_KEYS[utc_today().weekday()]
        self.app.config["SCRAPER_CONFIG"]["digest"] = {"synthesis_weekday": today_key, "exploration_slots": 0}
        db.session.add(_make_paper())
        db.session.commit()

        payload = self._synthesis_payload(narrative="Diffusion had a big week.")
        with (
            patch("app.services.corpus_analysis.synthesize_recent_topics", return_value=payload),
            patch("app.services.rag.build_llm_client", return_value=None),
        ):
            preview = build_digest_preview(self.app)

        self.assertIn("This week in your field", preview["html"])
        self.assertIn("Diffusion had a big week.", preview["html"])

    def test_synthesis_degrades_to_topic_list_without_narrative(self):
        from app.services.email_digest import DIGEST_WEEKDAY_KEYS, utc_today

        today_key = DIGEST_WEEKDAY_KEYS[utc_today().weekday()]
        self.app.config["SCRAPER_CONFIG"]["digest"] = {"synthesis_weekday": today_key, "exploration_slots": 0}
        db.session.add(_make_paper())
        db.session.commit()

        with (
            patch("app.services.corpus_analysis.synthesize_recent_topics", return_value=self._synthesis_payload()),
            patch("app.services.rag.build_llm_client", return_value=None),
        ):
            preview = build_digest_preview(self.app)

        self.assertIn("This week in your field", preview["html"])
        self.assertIn("diffusion", preview["html"])
        self.assertIn("4 new", preview["html"])

    def test_synthesis_absent_on_other_weekdays_and_by_default(self):
        db.session.add(_make_paper())
        db.session.commit()

        preview = build_digest_preview(self.app)  # no synthesis_weekday configured
        self.assertIsNone(preview["synthesis"])
        self.assertNotIn("This week in your field", preview["html"])

    def test_zero_slots_disables_exploration(self):
        self.app.config["SCRAPER_CONFIG"]["digest"] = {"exploration_slots": 0}
        db.session.add_all(
            [
                _make_paper(title="Main", link="https://arxiv.org/abs/2608.6", paper_score=90.0),
                _make_paper(title="Other", link="https://arxiv.org/abs/2608.7", paper_score=1.0),
            ]
        )
        db.session.commit()

        preview = build_digest_preview(self.app)
        self.assertEqual(preview["exploration"], [])
        self.assertNotIn("Exploration", preview["html"])


class CidAttachmentTests(FlaskDBTestCase):
    def _send_and_parse(self):
        fake_service = MagicMock()
        fake_service.users.return_value.messages.return_value.send.return_value.execute.return_value = {"id": "m"}
        with (
            patch("app.services.email_digest._load_gmail_credentials", return_value=MagicMock()),
            patch("app.services.email_digest._build_gmail_service", return_value=fake_service),
        ):
            send_digest(self.app)
        raw = fake_service.users.return_value.messages.return_value.send.call_args.kwargs["body"]["raw"]
        return message_from_bytes(base64.urlsafe_b64decode(raw))

    def test_figures_attached_as_cid_parts(self):
        self.app.config["SCRAPER_CONFIG"]["email"] = {"recipient": "a@b.com"}
        paper = _make_paper(arxiv_id="2607.40001")
        db.session.add(paper)
        db.session.commit()

        with tempfile.TemporaryDirectory() as tmpdir:
            figure = Path(tmpdir) / "2607.40001_fig1.png"
            figure.write_bytes(b"\x89PNG-not-a-real-image")
            with patch("app.services.thumbnail_generator.figure_paths_for", return_value=[figure]):
                message = self._send_and_parse()

        self.assertEqual(message.get_content_type(), "multipart/related")
        parts = message.get_payload()
        self.assertEqual(parts[0].get_content_type(), "text/html")
        images = [p for p in parts[1:] if p.get_content_type() == "image/png"]
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["Content-ID"], f"<fig-{paper.id}>")
        self.assertEqual(images[0].get_payload(decode=True), b"\x89PNG-not-a-real-image")

        html = parts[0].get_payload(decode=True).decode("utf-8")
        self.assertIn(f"cid:fig-{paper.id}", html)
        # One-tap buttons point at the configured base URL.
        self.assertIn("/api/feedback/one-tap?token=", html)

    def test_oversized_figures_are_skipped(self):
        self.app.config["SCRAPER_CONFIG"]["email"] = {"recipient": "a@b.com"}
        db.session.add(_make_paper(arxiv_id="2607.40002", link="https://arxiv.org/abs/2607.40002"))
        db.session.commit()

        with tempfile.TemporaryDirectory() as tmpdir:
            figure = Path(tmpdir) / "2607.40002_fig1.png"
            figure.write_bytes(b"x" * 2_000_000)  # over the 1.5MB inline cap
            with patch("app.services.thumbnail_generator.figure_paths_for", return_value=[figure]):
                message = self._send_and_parse()

        self.assertEqual(message.get_content_type(), "multipart/alternative")
        self.assertNotIn("cid:", message.get_payload()[0].get_payload(decode=True).decode("utf-8"))

    def test_no_figures_keeps_alternative_structure(self):
        self.app.config["SCRAPER_CONFIG"]["email"] = {"recipient": "a@b.com"}
        db.session.add(_make_paper())
        db.session.commit()

        message = self._send_and_parse()
        self.assertEqual(message.get_content_type(), "multipart/alternative")


class NotifyOnMatchTests(FlaskDBTestCase):
    def test_alert_section_lists_new_matches_and_dedupes_main_list(self):
        # High-scoring paper lands in the main digest list; the low-scoring match
        # is cut by min_score but must surface under "Saved search alerts".
        self.app.config["SCRAPER_CONFIG"]["digest"] = {"min_score": 20.0}
        main = _make_paper(title="Tracking Transformers", link="https://arxiv.org/abs/2607.50001", paper_score=50.0)
        alert_only = _make_paper(
            title="Tracking Small Objects", link="https://arxiv.org/abs/2607.50002", paper_score=5.0
        )
        db.session.add_all([main, alert_only])
        db.session.add(
            SavedSearch(name="Tracking watch", include_keywords=["Tracking"], notify_on_match=True, is_active=True)
        )
        db.session.commit()

        preview = build_digest_preview(self.app)

        self.assertEqual([p.title for p in preview["papers"]], ["Tracking Transformers"])
        self.assertEqual(len(preview["alerts"]), 1)
        self.assertEqual(preview["alerts"][0]["name"], "Tracking watch")
        alert_titles = [p.title for p in preview["alerts"][0]["papers"]]
        self.assertEqual(alert_titles, ["Tracking Small Objects"])
        self.assertIn("Saved search alerts", preview["html"])
        self.assertIn("Tracking Small Objects", preview["html"])

    def test_no_notify_searches_means_no_alert_section(self):
        db.session.add(SavedSearch(name="Silent", include_keywords=["Tracking"], notify_on_match=False, is_active=True))
        db.session.add(_make_paper(title="Tracking Things", link="https://arxiv.org/abs/2607.50003"))
        db.session.commit()

        preview = build_digest_preview(self.app)
        self.assertEqual(preview["alerts"], [])
        self.assertNotIn("Saved search alerts", preview["html"])

    def test_only_matches_since_last_digest_run_are_alerted(self):
        db.session.add(DigestRun(status="success", started_at=now_utc() - timedelta(hours=24)))
        db.session.add(
            SavedSearch(name="Tracking watch", include_keywords=["Tracking"], notify_on_match=True, is_active=True)
        )
        # Matches the search but predates the last digest run — already alerted.
        db.session.add(
            _make_paper(
                title="Tracking Ancient",
                link="https://arxiv.org/abs/2607.50004",
                paper_score=0.0,
                scraped_at=now_utc() - timedelta(days=2),
            )
        )
        db.session.commit()

        preview = build_digest_preview(self.app)
        self.assertEqual(preview["alerts"], [])


class DigestOptionsSettingsTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_settings_page_shows_digest_options(self):
        response = self.client.get("/settings?section=automation")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("digest-options-form", body)
        self.assertIn("digest_weekday_mon", body)
        self.assertIn("digest_min_score", body)
        self.assertIn("digest_max_papers", body)

    def test_round_trip_saves_and_rereads(self):
        import yaml

        response = self.client.post(
            "/settings/digest-options",
            data={
                "csrf_token": self._csrf_token(),
                "digest_weekday_mon": "1",
                "digest_weekday_fri": "1",
                "digest_min_score": "12.5",
                "digest_max_papers": "10",
                "digest_base_url": "http://127.0.0.1:5000/",
            },
        )
        self.assertEqual(response.status_code, 302)

        saved = self.app.config["SCRAPER_CONFIG"]["digest"]
        self.assertEqual(saved["weekdays"], ["mon", "fri"])
        self.assertEqual(saved["min_score"], 12.5)
        self.assertEqual(saved["max_papers"], 10)
        self.assertEqual(saved["base_url"], "http://127.0.0.1:5000")

        on_disk = yaml.safe_load(Path(self.app.config["CONFIG_PATH"]).read_text(encoding="utf-8"))
        self.assertEqual(on_disk["digest"]["weekdays"], ["mon", "fri"])

        cfg = get_digest_config(self.app)
        self.assertEqual(cfg["weekdays"], ["mon", "fri"])
        self.assertEqual(cfg["min_score"], 12.5)
        self.assertEqual(cfg["max_papers"], 10)

    def test_no_weekdays_selected_is_rejected(self):
        response = self.client.post(
            "/settings/digest-options",
            data={"csrf_token": self._csrf_token(), "digest_min_score": "0", "digest_max_papers": "15"},
        )
        self.assertEqual(response.status_code, 302)
        self.assertNotIn("digest", self.app.config["SCRAPER_CONFIG"])

    def test_requires_csrf(self):
        response = self.client.post("/settings/digest-options", data={"digest_weekday_mon": "1"})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
