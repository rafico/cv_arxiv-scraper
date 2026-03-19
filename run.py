import argparse
import os
import socket
from threading import Timer
import webbrowser

from app import create_app

DEFAULT_PORT = int(os.environ.get("PORT", 5000))


def _find_free_port(start, attempts=10):
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found in range {start}-{start + attempts - 1}")


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CV arXiv scraper server")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with Flask dev server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--workers", type=int, default=2, help="Number of gunicorn workers")
    parser.add_argument("--threads", type=int, default=2, help="Number of threads per worker")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser on start")
    args = parser.parse_args()

    port = _find_free_port(args.port)
    if port != args.port:
        print(f"Port {args.port} in use, falling back to {port}")

    if not args.no_browser:
        Timer(1.0, lambda: webbrowser.open(f"http://{args.host}:{port}")).start()

    if args.debug:
        app.run(host=args.host, port=port, debug=True)
    else:
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

        options = {
            "bind": f"{args.host}:{port}",
            "workers": args.workers,
            "worker_class": "gthread",
            "threads": args.threads,
        }
        StandaloneApplication(app, options).run()
