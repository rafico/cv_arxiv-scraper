"""``cv-arxiv-mcp`` — serve the corpus to Claude Desktop over MCP (stdio).

A thin console-script shim over :func:`app.mcp_server.run`. When the optional
``mcp`` extra is not installed, print an actionable install hint and exit
non-zero instead of dumping a traceback.

Usage::

    cv-arxiv-mcp [--data-dir DIR]
"""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from app.mcp_server import run

    try:
        return run(argv)
    except ModuleNotFoundError as exc:
        # Missing optional 'mcp' extra: message is already actionable.
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
