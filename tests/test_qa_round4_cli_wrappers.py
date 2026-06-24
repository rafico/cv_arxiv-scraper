"""QA round 4 regression tests for CLI wrapper shims.

Covers G1 (S2, silent no-op): scrape_cli.py / sync_cli.py / backfill_cli.py are
backward-compatible alias shims. Running them directly (``python scrape_cli.py``,
as app/services/cron.py writes for cron mode="scrape" and as the README
documents) must invoke the underlying ``app.cli.<X>.main()``. Before the fix the
modules lacked an ``if __name__ == "__main__"`` guard, so running them as
__main__ merely re-aliased the module and exited 0 WITHOUT scraping/syncing/
backfilling — a silent no-op that cron records as success.
"""

from __future__ import annotations

import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent

WRAPPERS = [
    ("scrape_cli.py", "app.cli.scrape"),
    ("sync_cli.py", "app.cli.sync"),
    ("backfill_cli.py", "app.cli.backfill"),
]


class CliWrapperMainGuardTests(unittest.TestCase):
    def _assert_wrapper_runs_main(self, script: str, target: str) -> None:
        calls: list[bool] = []

        def _sentinel_main(*args, **kwargs) -> int:
            calls.append(True)
            return 0

        script_path = str(REPO_ROOT / script)
        with patch(f"{target}.main", _sentinel_main):
            with self.assertRaises(SystemExit) as cm:
                runpy.run_path(script_path, run_name="__main__")

        self.assertEqual(
            calls,
            [True],
            f"{script} did not invoke {target}.main() when run as __main__",
        )
        # main() returning 0 must propagate as a 0 exit code.
        code = cm.exception.code
        self.assertIn(code, (0, None), f"{script} exited with {code!r}")

    def test_g1_scrape_cli_invokes_main(self) -> None:
        self._assert_wrapper_runs_main("scrape_cli.py", "app.cli.scrape")

    def test_g1_sync_cli_invokes_main(self) -> None:
        self._assert_wrapper_runs_main("sync_cli.py", "app.cli.sync")

    def test_g1_backfill_cli_invokes_main(self) -> None:
        self._assert_wrapper_runs_main("backfill_cli.py", "app.cli.backfill")


if __name__ == "__main__":
    unittest.main()
