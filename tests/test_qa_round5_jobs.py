"""QA round 5 regression test — R5-con2 (S3): get_status_snapshot must never report
a finishing job as a spurious non-terminal {"running": False} (no terminal_status).

Pre-fix, _publish cleared _active_job_id (under _lock) *before* setting finished_at
(under job.condition); a snapshot interleaving between the two saw 'active cleared
but not finished' and fell to the completed-jobs branch, which excluded the
still-unfinished job. The fix clears the active id only after the job is marked
finished.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.services import jobs as jobs_module
from app.services.jobs import ScrapeJob, ScrapeJobManager
from app.services.text import now_utc


class StatusSnapshotRaceTests(unittest.TestCase):
    def test_snapshot_during_publish_is_never_spuriously_non_terminal(self):
        manager = ScrapeJobManager()
        job = ScrapeJob(id="job-1", started_at=now_utc())
        manager._jobs["job-1"] = job
        manager._active_job_id = "job-1"

        captured: dict[str, dict] = {}
        real_now = jobs_module.now_utc

        def capturing_now():
            # Called while _publish is setting finished_at — i.e. exactly the window
            # the snapshot used to read inconsistently.
            captured["snapshot"] = manager.get_status_snapshot()
            return real_now()

        with patch.object(jobs_module, "now_utc", side_effect=capturing_now):
            manager._publish("job-1", "done", {})

        snap = captured["snapshot"]
        # The interleaved snapshot must be either still-running or properly terminal,
        # never a non-terminal {"running": False} with no terminal_status.
        self.assertTrue(snap.get("running") is True or "terminal_status" in snap, snap)

    def test_snapshot_after_done_is_terminal(self):
        manager = ScrapeJobManager()
        job = ScrapeJob(id="job-2", started_at=now_utc())
        manager._jobs["job-2"] = job
        manager._active_job_id = "job-2"

        manager._publish("job-2", "done", {})

        snap = manager.get_status_snapshot()
        self.assertFalse(snap["running"])
        self.assertEqual(snap["terminal_status"], "finished")
        self.assertIsNone(manager._active_job_id)


if __name__ == "__main__":
    unittest.main()
