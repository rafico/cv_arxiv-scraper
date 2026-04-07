"""QA tests for cron job management.

Covers: CVARX-65 (Scheduling & Automation)
- _build_cron_line for all modes (full, scrape, digest)
- _remove_our_lines filters only tagged lines
- get_cron_status parsing
- install_cron_job clamps hour/minute
- install_cron_job defaults invalid mode to 'full'
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.services.cron import (
    CRON_TAG,
    _build_cron_line,
    _remove_our_lines,
    get_cron_status,
    install_cron_job,
    remove_cron_job,
)


class BuildCronLineTests(unittest.TestCase):
    """Test cron line construction for each mode."""

    def test_full_mode_uses_digest_cli(self):
        line = _build_cron_line(8, 0, "full")
        self.assertIn("digest_cli.py", line)
        self.assertNotIn("--send-only", line)
        self.assertNotIn("scrape_cli.py", line)
        self.assertIn(CRON_TAG, line)

    def test_scrape_mode_uses_scrape_cli(self):
        line = _build_cron_line(9, 30, "scrape")
        self.assertIn("scrape_cli.py", line)
        self.assertNotIn("digest_cli.py", line)
        self.assertTrue(line.startswith("30 9 "))

    def test_digest_mode_uses_send_only(self):
        line = _build_cron_line(10, 15, "digest")
        self.assertIn("digest_cli.py --send-only", line)
        self.assertTrue(line.startswith("15 10 "))

    def test_cron_line_has_tag(self):
        for mode in ("full", "scrape", "digest"):
            line = _build_cron_line(8, 0, mode)
            self.assertIn(CRON_TAG, line)

    def test_cron_line_logs_to_cron_log(self):
        line = _build_cron_line(8, 0, "full")
        self.assertIn("cron.log", line)


class RemoveOurLinesTests(unittest.TestCase):
    """Test filtering of tagged cron lines."""

    def test_removes_tagged_lines(self):
        crontab = f"0 * * * * some-other-job\n30 8 * * * our-job {CRON_TAG}\n"
        result = _remove_our_lines(crontab)
        self.assertNotIn(CRON_TAG, result)
        self.assertIn("some-other-job", result)

    def test_preserves_unrelated_lines(self):
        crontab = "0 * * * * keep-me\n30 2 * * * also-keep-me\n"
        result = _remove_our_lines(crontab)
        self.assertIn("keep-me", result)
        self.assertIn("also-keep-me", result)

    def test_empty_crontab(self):
        result = _remove_our_lines("")
        self.assertEqual(result, "")

    def test_only_our_lines_returns_empty(self):
        crontab = f"30 8 * * * job {CRON_TAG}\n"
        result = _remove_our_lines(crontab)
        self.assertEqual(result.strip(), "")


class GetCronStatusTests(unittest.TestCase):
    """Test cron status parsing from crontab output."""

    @patch("app.services.cron._get_current_crontab")
    def test_no_cron_job_installed(self, mock_crontab):
        mock_crontab.return_value = ""
        status = get_cron_status()
        self.assertFalse(status["installed"])
        self.assertEqual(status["hour"], 8)
        self.assertEqual(status["minute"], 0)

    @patch("app.services.cron._get_current_crontab")
    def test_full_mode_detected(self, mock_crontab):
        mock_crontab.return_value = f"0 8 * * * cd /proj && python digest_cli.py >> /proj/cron.log 2>&1 {CRON_TAG}\n"
        status = get_cron_status()
        self.assertTrue(status["installed"])
        self.assertEqual(status["mode"], "full")
        self.assertEqual(status["hour"], 8)
        self.assertEqual(status["minute"], 0)

    @patch("app.services.cron._get_current_crontab")
    def test_scrape_mode_detected(self, mock_crontab):
        mock_crontab.return_value = f"30 9 * * * cd /proj && python scrape_cli.py >> /proj/cron.log 2>&1 {CRON_TAG}\n"
        status = get_cron_status()
        self.assertTrue(status["installed"])
        self.assertEqual(status["mode"], "scrape")
        self.assertEqual(status["hour"], 9)
        self.assertEqual(status["minute"], 30)

    @patch("app.services.cron._get_current_crontab")
    def test_digest_mode_detected(self, mock_crontab):
        mock_crontab.return_value = f"15 10 * * * cd /proj && python digest_cli.py --send-only >> /proj/cron.log 2>&1 {CRON_TAG}\n"
        status = get_cron_status()
        self.assertTrue(status["installed"])
        self.assertEqual(status["mode"], "digest")
        self.assertEqual(status["hour"], 10)
        self.assertEqual(status["minute"], 15)


class InstallCronJobTests(unittest.TestCase):
    """Test cron job installation edge cases."""

    @patch("app.services.cron._set_crontab")
    @patch("app.services.cron._get_current_crontab", return_value="")
    def test_hour_clamped_to_valid_range(self, _, mock_set):
        install_cron_job(25, 0, "full")
        call_content = mock_set.call_args[0][0]
        self.assertIn("0 23 ", call_content)

    @patch("app.services.cron._set_crontab")
    @patch("app.services.cron._get_current_crontab", return_value="")
    def test_minute_clamped_to_valid_range(self, _, mock_set):
        install_cron_job(8, 99, "full")
        call_content = mock_set.call_args[0][0]
        self.assertIn("59 8 ", call_content)

    @patch("app.services.cron._set_crontab")
    @patch("app.services.cron._get_current_crontab", return_value="")
    def test_invalid_mode_defaults_to_full(self, _, mock_set):
        result = install_cron_job(8, 0, "invalid")
        self.assertTrue(result["success"])
        call_content = mock_set.call_args[0][0]
        self.assertIn("digest_cli.py", call_content)
        self.assertNotIn("--send-only", call_content)

    @patch("app.services.cron._set_crontab")
    @patch("app.services.cron._get_current_crontab", return_value="")
    def test_install_returns_success(self, _, mock_set):
        result = install_cron_job(8, 30, "scrape")
        self.assertTrue(result["success"])

    @patch("app.services.cron._set_crontab", side_effect=Exception("fail"))
    @patch("app.services.cron._get_current_crontab", return_value="")
    def test_install_failure_returns_error(self, _, __):
        result = install_cron_job(8, 0, "full")
        self.assertFalse(result["success"])


class RemoveCronJobTests(unittest.TestCase):
    """Test cron job removal."""

    @patch("app.services.cron._get_current_crontab")
    def test_remove_when_no_job_installed(self, mock_crontab):
        mock_crontab.return_value = "0 * * * * other-job\n"
        result = remove_cron_job()
        self.assertTrue(result["success"])
        self.assertIn("No cron job", result["message"])

    @patch("app.services.cron._set_crontab")
    @patch("app.services.cron._get_current_crontab")
    def test_remove_existing_job(self, mock_crontab, mock_set):
        mock_crontab.return_value = f"0 * * * * other-job\n30 8 * * * our-job {CRON_TAG}\n"
        result = remove_cron_job()
        self.assertTrue(result["success"])
        call_content = mock_set.call_args[0][0]
        self.assertNotIn(CRON_TAG, call_content)
        self.assertIn("other-job", call_content)


if __name__ == "__main__":
    unittest.main()
