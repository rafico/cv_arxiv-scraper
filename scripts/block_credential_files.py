#!/usr/bin/env python3
"""Pre-commit hook: reject staging of credential filenames."""

from __future__ import annotations

import re
import sys
from pathlib import Path

BLOCKED_PATTERNS = [
    re.compile(r"(^|/)credentials\.json$"),
    re.compile(r"(^|/)token\.json$"),
    re.compile(r"(^|/)\.llm_api_key$"),
    re.compile(r"(^|/)mendeley_credentials\.json$"),
    re.compile(r"(^|/)\.mendeley_token$"),
    re.compile(r"(^|/)\.zotero_credentials$"),
    re.compile(r"(^|/)\.env$"),
    re.compile(r"(^|/)\.mcp\.json$"),
    re.compile(r"(^|/)\.flask_secret$"),
]


def main(argv: list[str]) -> int:
    offenders: list[str] = []
    for raw in argv:
        path = Path(raw).as_posix()
        if any(pattern.search(path) for pattern in BLOCKED_PATTERNS):
            offenders.append(raw)

    if offenders:
        print("Refusing to commit credential/secret files:", file=sys.stderr)
        for item in offenders:
            print(f"  - {item}", file=sys.stderr)
        print(
            "\nThese filenames are listed in .gitignore; rotate and keep them local.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
