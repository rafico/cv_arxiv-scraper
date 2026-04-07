from __future__ import annotations

import io
import socket
import unittest
from unittest.mock import Mock, patch

import run


class RunEntryPointTests(unittest.TestCase):
    def test_debug_mode_uses_flask_dev_server_with_requested_host_and_port(self):
        fake_app = Mock()

        exit_code = run.main(
            ["--debug", "--host", "0.0.0.0", "--port", "5123", "--no-browser"],
            app_factory=lambda: fake_app,
        )

        self.assertEqual(exit_code, 0)
        fake_app.run.assert_called_once_with(host="0.0.0.0", port=5123, debug=True)

    def test_production_mode_uses_gunicorn_with_worker_flags(self):
        fake_app = Mock()
        fake_runner = Mock()

        with patch("run._create_gunicorn_application", return_value=fake_runner) as mock_create_runner:
            exit_code = run.main(
                ["--host", "0.0.0.0", "--port", "6123", "--workers", "4", "--threads", "8", "--no-browser"],
                app_factory=lambda: fake_app,
            )

        self.assertEqual(exit_code, 0)
        mock_create_runner.assert_called_once_with(
            fake_app,
            {
                "bind": "0.0.0.0:6123",
                "workers": 4,
                "worker_class": "gthread",
                "threads": 8,
            },
        )
        fake_runner.run.assert_called_once_with()

    def test_no_browser_flag_skips_auto_open_timer(self):
        fake_app = Mock()
        timer_factory = Mock()

        run.main(
            ["--debug", "--port", "5124", "--no-browser"],
            app_factory=lambda: fake_app,
            timer_factory=timer_factory,
            browser_opener=Mock(),
        )

        timer_factory.assert_not_called()

    def test_find_free_port_skips_bound_ports(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            busy_port = sock.getsockname()[1]

            free_port = run._find_free_port(busy_port, attempts=20)

        self.assertNotEqual(free_port, busy_port)
        self.assertGreaterEqual(free_port, busy_port)
        self.assertLess(free_port, busy_port + 20)

    def test_main_reports_port_fallback_and_uses_discovered_port(self):
        fake_app = Mock()
        stdout = io.StringIO()

        with patch("run._find_free_port", return_value=5126):
            exit_code = run.main(
                ["--debug", "--port", "5125", "--no-browser"],
                app_factory=lambda: fake_app,
                stdout=stdout,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Port 5125 in use, falling back to 5126", stdout.getvalue())
        fake_app.run.assert_called_once_with(host="127.0.0.1", port=5126, debug=True)


if __name__ == "__main__":
    unittest.main()
