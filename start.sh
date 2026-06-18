#!/usr/bin/env bash
# Launch the CV arXiv scraper using the global venv.
#
# Usage:
#   ./start.sh                  # production-style server on http://127.0.0.1:5000
#   ./start.sh --debug          # Flask dev server with auto-reload
#   ./start.sh --port 5099      # any run.py flag is passed through
#
# Override the venv with: VENV=/path/to/venv ./start.sh
set -euo pipefail

cd "$(dirname "$0")"

VENV="${VENV:-$HOME/venv}"
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "error: no python found at $VENV/bin/python (set VENV=/path/to/venv)" >&2
    exit 1
fi

exec "$VENV/bin/python" run.py "$@"
