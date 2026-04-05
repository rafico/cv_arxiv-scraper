from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.services.scheduler import ScrapeScheduler


class SchedulerLockTests(unittest.TestCase):
    def test_only_one_scheduler_instance_owns_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app = SimpleNamespace(instance_path=tmpdir)
            primary = ScrapeScheduler()
            secondary = ScrapeScheduler()

            try:
                with patch.object(primary, "_schedule_next") as primary_schedule:
                    primary.start(app)
                with patch.object(secondary, "_schedule_next") as secondary_schedule:
                    secondary.start(app)

                self.assertTrue(primary.is_enabled)
                self.assertFalse(secondary.is_enabled)
                primary_schedule.assert_called_once()
                secondary_schedule.assert_not_called()
            finally:
                primary.stop()
                secondary.stop()

    def test_releasing_lock_allows_another_instance_to_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app = SimpleNamespace(instance_path=tmpdir)
            first = ScrapeScheduler()
            second = ScrapeScheduler()

            try:
                with patch.object(first, "_schedule_next"):
                    first.start(app)
                first.stop()

                with patch.object(second, "_schedule_next") as second_schedule:
                    second.start(app)

                self.assertTrue(second.is_enabled)
                second_schedule.assert_called_once()
            finally:
                first.stop()
                second.stop()

    def test_restarting_same_instance_cancels_prior_timer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app = SimpleNamespace(instance_path=tmpdir)
            scheduler = ScrapeScheduler()
            first_timer = Mock()
            second_timer = Mock()

            with patch("app.services.scheduler.threading.Timer", side_effect=[first_timer, second_timer]):
                scheduler.start(app, daily_at="08:00")
                scheduler.start(app, daily_at="09:00")

            self.assertTrue(scheduler.is_enabled)
            first_timer.cancel.assert_called_once()
            first_timer.start.assert_called_once()
            second_timer.start.assert_called_once()

            scheduler.stop()


if __name__ == "__main__":
    unittest.main()
