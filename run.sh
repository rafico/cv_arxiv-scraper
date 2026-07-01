#!/usr/bin/env bash
# Convenience wrapper around the CV arXiv scraper, using the global venv.
# UI-first: the bare command launches the web dashboard.
#
#   ./run.sh                       # dashboard: Flask debug server, opens the browser
#   ./run.sh web                   # same as above (explicit form)
#   ./run.sh web --prod            # production-style server (in-process gunicorn, 1 worker)
#   ./run.sh --port 5099 --debug   # any run.py flag passes through to the web server
#   ./run.sh scrape ...            # CLI: run the arXiv scrape pipeline
#   ./run.sh sync ...              # CLI: sync ranked papers to a reference manager
#   ./run.sh backfill ...          # CLI: backfill historical papers
#   ./run.sh digest ...            # CLI: build/send the email digest
#   ./run.sh export ...            # CLI: export a paper report to HTML
#
# Override the interpreter with: PYTHON_BIN=/path/to/python ./run.sh
#   (or VENV=/path/to/venv ./run.sh)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-${VENV:-$HOME/venv}/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: no python at $PYTHON_BIN (set PYTHON_BIN=/path/to/python or VENV=/path/to/venv)" >&2
  exit 1
fi

usage() {
  # Reprint the header comment block above (drop the shebang, strip "# ").
  sed -n '2,/^$/{/^# /s/^# \{0,1\}//p;}' "$0"
}

# --help before anything else, so it shows this wrapper's usage (not run.py's).
case "${1:-}" in
  -h | --help)
    usage
    exit 0
    ;;
esac

# UI-first: no args, an explicit `web`, or a bare flag (--port, --prod, …) all
# launch the web server. Everything else falls through to the CLIs below.
if [[ $# -eq 0 || "${1:-}" == "web" || "${1:-}" == -* ]]; then
  [[ "${1:-}" == "web" ]] && shift
  if [[ "${1:-}" == "--prod" ]]; then
    shift
    exec "$PYTHON_BIN" run.py "$@" # production-style: in-process gunicorn, 1 worker
  fi
  exec "$PYTHON_BIN" run.py --debug "$@" # dev: Flask auto-reload, opens the browser
fi

case "$1" in
  scrape | sync | backfill | digest | export)
    exec "$PYTHON_BIN" "$1_cli.py" "${@:2}"
    ;;
  *)
    echo "error: unknown command '$1' (use: web scrape sync backfill digest export, or run.py flags)" >&2
    exit 2
    ;;
esac
