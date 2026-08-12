from __future__ import annotations

import copy
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import yaml

from app import create_app
from tests.helpers import TEST_SCRAPER_CONFIG, FlaskDBTestCase


class SchedulerSettingsRouteQaTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/settings")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_enable_persists_config_and_starts_scheduler(self):
        with patch("app.services.scheduler.SCRAPE_SCHEDULER.start") as mock_start:
            response = self.client.post(
                "/settings/scheduler",
                data={
                    "csrf_token": self._csrf_token(),
                    "scheduler_action": "enable",
                    "scheduler_daily_at": "09:30",
                    "scheduler_send_digest": "on",
                },
            )

        self.assertEqual(response.status_code, 302)
        mock_start.assert_called_once_with(self.app, daily_at="09:30", send_digest=True)
        saved = yaml.safe_load(Path(self.app.config["CONFIG_PATH"]).read_text(encoding="utf-8"))
        self.assertEqual(saved["scheduler"], {"enabled": True, "daily_at": "09:30", "send_digest": True})

    def test_disable_persists_and_stops_scheduler(self):
        with patch("app.services.scheduler.SCRAPE_SCHEDULER.stop") as mock_stop:
            response = self.client.post(
                "/settings/scheduler",
                data={
                    "csrf_token": self._csrf_token(),
                    "scheduler_action": "disable",
                    "scheduler_daily_at": "08:00",
                },
            )

        self.assertEqual(response.status_code, 302)
        mock_stop.assert_called_once_with()
        saved = yaml.safe_load(Path(self.app.config["CONFIG_PATH"]).read_text(encoding="utf-8"))
        self.assertFalse(saved["scheduler"]["enabled"])

    def test_invalid_time_is_rejected_without_writing_config(self):
        with patch("app.services.scheduler.SCRAPE_SCHEDULER.start") as mock_start:
            response = self.client.post(
                "/settings/scheduler",
                data={
                    "csrf_token": self._csrf_token(),
                    "scheduler_action": "enable",
                    "scheduler_daily_at": "25:99",
                },
            )

        self.assertEqual(response.status_code, 302)
        mock_start.assert_not_called()
        saved = yaml.safe_load(Path(self.app.config["CONFIG_PATH"]).read_text(encoding="utf-8"))
        self.assertNotIn("scheduler", saved)


class SchedulerDigestQaTests(FlaskDBTestCase):
    def test_run_sends_digest_after_scrape_completes(self):
        from app.services.scheduler import ScrapeScheduler

        class _FakeJob:
            status = "finished"
            condition = threading.Condition()

        scheduler = ScrapeScheduler()
        scheduler._app = self.app
        scheduler._enabled = True
        scheduler._send_digest = True

        with (
            patch("app.services.jobs.SCRAPE_JOB_MANAGER.start_or_get_active", return_value=_FakeJob()) as mock_job,
            patch("app.services.email_digest.send_digest", return_value={"sent": True}) as mock_digest,
            patch.object(scheduler, "_schedule_next"),
        ):
            scheduler._run()

        mock_job.assert_called_once_with(self.app)
        mock_digest.assert_called_once_with(self.app)

    def test_run_without_send_digest_skips_email(self):
        from app.services.scheduler import ScrapeScheduler

        class _FakeJob:
            status = "finished"
            condition = threading.Condition()

        scheduler = ScrapeScheduler()
        scheduler._app = self.app
        scheduler._enabled = True
        scheduler._send_digest = False

        with (
            patch("app.services.jobs.SCRAPE_JOB_MANAGER.start_or_get_active", return_value=_FakeJob()),
            patch("app.services.email_digest.send_digest") as mock_digest,
            patch.object(scheduler, "_schedule_next"),
        ):
            scheduler._run()

        mock_digest.assert_not_called()


class SchedulerStartupQaTests(FlaskDBTestCase):
    def _write_config(self, root: Path, *, scheduler_config: dict) -> Path:
        config = copy.deepcopy(TEST_SCRAPER_CONFIG)
        config["scheduler"] = scheduler_config
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return config_path

    def test_create_app_starts_scheduler_with_configured_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_config(
                root, scheduler_config={"enabled": True, "daily_at": "06:45", "send_digest": True}
            )

            with patch("app.web.scheduler.SCRAPE_SCHEDULER.start") as mock_start:
                app = create_app(
                    {
                        "TESTING": True,
                        "CONFIG_PATH": str(config_path),
                        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{root / 'test.db'}",
                        "INSTANCE_PATH": str(root / "instance"),
                        "LLM_KEY_PATH": str(root / ".llm_api_key"),
                    }
                )

            self.assertTrue(app.config["SCRAPER_CONFIG"]["scheduler"]["enabled"])
            mock_start.assert_called_once_with(app, daily_at="06:45", send_digest=True)

    def test_create_app_uses_default_scheduler_time_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_config(root, scheduler_config={"enabled": True})

            with patch("app.web.scheduler.SCRAPE_SCHEDULER.start") as mock_start:
                app = create_app(
                    {
                        "TESTING": True,
                        "CONFIG_PATH": str(config_path),
                        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{root / 'test.db'}",
                        "INSTANCE_PATH": str(root / "instance"),
                        "LLM_KEY_PATH": str(root / ".llm_api_key"),
                    }
                )

            mock_start.assert_called_once_with(app, daily_at="08:00", send_digest=False)
