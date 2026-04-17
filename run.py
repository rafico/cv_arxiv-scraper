import argparse
import ipaddress
import os
import socket
import sys
import webbrowser
from threading import Timer

from app import create_app

DEFAULT_PORT = int(os.environ.get("PORT", 5000))


def _host_is_loopback(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return host in {"localhost", ""}
    return addr.is_loopback


def _find_free_port(start, attempts=10):
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found in range {start}-{start + attempts - 1}")


def build_parser(default_port=DEFAULT_PORT):
    parser = argparse.ArgumentParser(description="Run the CV arXiv scraper server")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with Flask dev server")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to listen on (default: {default_port})")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--workers", type=int, default=2, help="Number of gunicorn workers")
    parser.add_argument("--threads", type=int, default=2, help="Number of threads per worker")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser on start")
    parser.add_argument(
        "--expose",
        action="store_true",
        help="Acknowledge that the app has no authentication and bind to a non-loopback host anyway",
    )
    return parser


def _schedule_browser(host, port, *, timer_factory=Timer, browser_opener=webbrowser.open):
    timer_factory(1.0, lambda: browser_opener(f"http://{host}:{port}")).start()


def _create_gunicorn_application(app, options):
    from gunicorn.app.base import BaseApplication

    class StandaloneApplication(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            for key, value in self.options.items():
                if key in self.cfg.settings and value is not None:
                    self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

    return StandaloneApplication(app, options)


def main(argv=None, *, app_factory=create_app, timer_factory=Timer, browser_opener=webbrowser.open, stdout=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    out = stdout or sys.stdout
    if not _host_is_loopback(args.host) and not args.expose:
        print(
            f"refusing to bind to non-loopback host {args.host!r}: this app has no authentication. "
            "Pass --expose to override.",
            file=sys.stderr,
        )
        return 2
    if not _host_is_loopback(args.host) and args.expose:
        print(
            f"\033[31mWARNING: binding to {args.host} with no authentication — anyone on the network can access the app.\033[0m",
            file=sys.stderr,
        )

    port = _find_free_port(args.port)
    if port != args.port:
        print(f"Port {args.port} in use, falling back to {port}", file=out)

    app = app_factory()

    if not args.no_browser:
        _schedule_browser(args.host, port, timer_factory=timer_factory, browser_opener=browser_opener)

    if args.debug:
        app.run(host=args.host, port=port, debug=True)
        return 0

    options = {
        "bind": f"{args.host}:{port}",
        "workers": args.workers,
        "worker_class": "gthread",
        "threads": args.threads,
    }
    _create_gunicorn_application(app, options).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
