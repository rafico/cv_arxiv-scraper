from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from app import create_app
from tests.helpers import FlaskDBTestCase, TEST_SCRAPER_CONFIG


class SchedulingAutomationQaTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/settings")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    @patch("app.services.cron.install_cron_job")
    def test_settings_cron_install_forwards_hour_minute_and_mode(self, mock_install):
        mock_install.return_value = {"success": True, "message": "Installed."}

        response = self.client.post(
            "/settings/cron",
            data={
                "csrf_token": self._csrf_token(),
                "cron_hour": "9",
                "cron_minute": "30",
                "cron_mode": "scrape",
                "cron_action": "install",
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_install.assert_called_once_with(9, 30, "scrape")

    @patch("app.services.cron.remove_cron_job")
    def test_settings_cron_remove_calls_remove_helper(self, mock_remove):
        mock_remove.return_value = {"success": True, "message": "Removed."}

        response = self.client.post(
            "/settings/cron",
            data={
                "csrf_token": self._csrf_token(),
                "cron_action": "remove",
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_remove.assert_called_once_with()


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
            config_path = self._write_config(root, scheduler_config={"enabled": True, "daily_at": "06:45"})

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
            mock_start.assert_called_once_with(app, daily_at="06:45")

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

            mock_start.assert_called_once_with(app, daily_at="08:00")
