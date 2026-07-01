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

    def test_schedule_next_supersedes_existing_timer(self):
        # A reschedule must cancel the previously scheduled timer before creating a new
        # one, else a race can leave two live timers that both fire next cycle.
        scheduler = ScrapeScheduler()
        scheduler._enabled = True
        scheduler._daily_at = "08:00"
        try:
            scheduler._schedule_next()
            first = scheduler._timer
            scheduler._schedule_next()
            second = scheduler._timer

            self.assertIsNotNone(first)
            self.assertIsNot(first, second)
            # threading.Timer.cancel() sets its `finished` event.
            self.assertTrue(first.finished.is_set(), "superseded timer was not cancelled")
        finally:
            if scheduler._timer is not None:
                scheduler._timer.cancel()


class SchedulerDailyAtParsingTests(unittest.TestCase):
    def test_parse_daily_at_falls_back_on_invalid_values(self):
        # Out-of-range-but-parseable, malformed, and good inputs.
        self.assertEqual(ScrapeScheduler._parse_daily_at("25:00"), (8, 0))
        self.assertEqual(ScrapeScheduler._parse_daily_at("08:60"), (8, 0))
        self.assertEqual(ScrapeScheduler._parse_daily_at("-1:30"), (8, 0))
        self.assertEqual(ScrapeScheduler._parse_daily_at("notatime"), (8, 0))
        self.assertEqual(ScrapeScheduler._parse_daily_at("23:59"), (23, 59))

    def test_out_of_range_daily_at_does_not_crash_startup(self):
        # "25:00" parses to ints but is not a valid wall-clock time; before the
        # fix, _seconds_until's datetime.replace raised straight out of start().
        with tempfile.TemporaryDirectory() as tmpdir:
            app = SimpleNamespace(instance_path=tmpdir)
            scheduler = ScrapeScheduler()
            try:
                with patch("app.services.scheduler.threading.Timer") as timer:
                    scheduler.start(app, daily_at="25:00")
                self.assertTrue(scheduler.is_enabled)
                timer.assert_called_once()
            finally:
                scheduler.stop()

    def test_next_run_at_does_not_crash_on_invalid_daily_at(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app = SimpleNamespace(instance_path=tmpdir)
            scheduler = ScrapeScheduler()
            try:
                with patch.object(scheduler, "_schedule_next"):
                    scheduler.start(app, daily_at="25:00")
                self.assertIn("08:00", scheduler.next_run_at)
            finally:
                scheduler.stop()


if __name__ == "__main__":
    unittest.main()
