"""``cv-arxiv serve`` — launch the web server against a single data directory.

A thin wrapper over ``run.main`` (the same gunicorn/Flask path ``run.py`` uses)
for ``pip``/``uvx`` installs. It resolves one *data directory* that holds the
SQLite DB, FAISS index, ``config.yaml``, and secret dotfiles, points the app's
existing path-resolution env vars at it (``CV_ARXIV_INSTANCE_PATH`` /
``CV_ARXIV_CONFIG``), seeds a config on first run, then hands off to the server.

Data-dir precedence: ``--data-dir`` > ``$CV_ARXIV_DATA_DIR`` > the default
``~/.local/share/cv-arxiv``. Loopback-only stays the default; ``--expose`` (and
every other ``run.py`` flag) is forwarded untouched.

Usage::

    cv-arxiv serve [--data-dir DIR] [--port N] [--host H] [--expose] ...
    cv-arxiv --version
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from app._version import __version__

DATA_DIR_ENV_VAR = "CV_ARXIV_DATA_DIR"
DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "cv-arxiv"

# Seeded on first run only when no ``config.example.yaml`` template can be found
# (e.g. a bare ``pip``/``uvx`` install with no source tree next to it). It must
# satisfy ``app._validate_config``: a scraper feed plus the three whitelist
# buckets. Whitelists start empty — the Settings UI configures them.
_MINIMAL_CONFIG: dict = {
    "scraper": {
        "feed_url": "https://rss.arxiv.org/rss/cs.CV",
        "rolling_window_days": 4,
    },
    "llm": {"enabled": False},
    "whitelists": {"titles": [], "authors": [], "affiliations": []},
}


def resolve_data_dir(cli_data_dir: str | None = None, *, environ: Mapping[str, str] | None = None) -> Path:
    """Resolve the data directory.

    Precedence: ``--data-dir`` flag > ``$CV_ARXIV_DATA_DIR`` > default.
    """
    env = os.environ if environ is None else environ
    raw = (cli_data_dir or "").strip() or env.get(DATA_DIR_ENV_VAR, "").strip() or str(DEFAULT_DATA_DIR)
    return Path(raw).expanduser().resolve()


def _find_example_config() -> Path | None:
    """Locate a ``config.example.yaml`` template shipped with a source checkout."""
    candidates = [
        Path.cwd() / "config.example.yaml",
        # repo root relative to app/cli/serve.py -> app/ -> app/.. (source layout)
        Path(__file__).resolve().parents[2] / "config.example.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _seed_config(config_path: Path) -> None:
    """Ensure a ``config.yaml`` exists so the app can boot on a fresh data dir."""
    if config_path.exists():
        return
    config_path.parent.mkdir(parents=True, exist_ok=True)
    template = _find_example_config()
    if template is not None:
        shutil.copyfile(template, config_path)
    else:
        config_path.write_text(yaml.safe_dump(_MINIMAL_CONFIG, sort_keys=False), encoding="utf-8")


def prepare_data_dir(data_dir: Path, *, environ: dict[str, str] | None = None) -> Path:
    """Create ``data_dir``, point the app env vars at it, and seed a config.

    Returns the resolved config path. ``CV_ARXIV_INSTANCE_PATH`` is set to the
    data dir (that is the whole point of a single dir). ``CV_ARXIV_CONFIG`` is
    honoured if the caller already set it, otherwise it defaults to
    ``<data_dir>/config.yaml``.
    """
    env = os.environ if environ is None else environ
    data_dir.mkdir(parents=True, exist_ok=True)
    env["CV_ARXIV_INSTANCE_PATH"] = str(data_dir)

    existing = env.get("CV_ARXIV_CONFIG", "").strip()
    config_path = Path(existing).expanduser().resolve() if existing else (data_dir / "config.yaml")
    if not existing:
        env["CV_ARXIV_CONFIG"] = str(config_path)
    _seed_config(config_path)
    return config_path


def _serve_help() -> str:
    import run

    return (
        "usage: cv-arxiv serve [--data-dir DIR] [server options]\n\n"
        "Launch the server against a single data directory (DB, FAISS index,\n"
        "config, and secrets all live under it).\n\n"
        f"  --data-dir DIR   Data directory. Precedence: --data-dir > ${DATA_DIR_ENV_VAR}\n"
        f"                   > {DEFAULT_DATA_DIR}\n\n"
        "Forwarded server options:\n\n" + run.build_parser().format_help()
    )


def _run_serve(argv: Sequence[str]) -> int:
    if any(a in ("-h", "--help") for a in argv):
        print(_serve_help())
        return 0

    import run

    parser = argparse.ArgumentParser(prog="cv-arxiv serve", add_help=False)
    parser.add_argument("--data-dir", default=None)
    known, rest = parser.parse_known_args(list(argv))

    data_dir = resolve_data_dir(known.data_dir)
    prepare_data_dir(data_dir)
    return run.main(rest)


def _print_help() -> None:
    print(
        "usage: cv-arxiv [--version] <command> [options]\n\n"
        "Commands:\n"
        "  serve       Run the web server (see 'cv-arxiv serve --help').\n\n"
        "Options:\n"
        "  --version   Print the version and exit.\n"
        "  -h, --help  Show this message and exit."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] in ("--version", "-V"):
        print(f"cv-arxiv-scraper {__version__}")
        return 0
    if not args or args[0] in ("-h", "--help"):
        _print_help()
        return 0

    command, rest = args[0], args[1:]
    if command == "serve":
        return _run_serve(rest)

    sys.stderr.write(f"cv-arxiv: unknown command {command!r}. Try 'cv-arxiv serve' or 'cv-arxiv --version'.\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
