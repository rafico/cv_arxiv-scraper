"""Atomic, 0600 writes for credential/token dotfiles.

Writing a secret with ``Path.write_text`` then ``os.chmod(0o600)`` leaves a window
where the file exists under the default umask (often world-readable). Create the
file with mode 0600 from the start instead; the trailing chmod also tightens an
already-existing file whose mode may be looser. This is the same pattern used by
``llm_client.write_api_key`` — shared here so every secret writer stays consistent.
"""

from __future__ import annotations

import os
from pathlib import Path


def write_secret_file(path: Path | str, text: str) -> Path:
    """Write ``text`` to ``path`` with mode 0600, never briefly world-readable."""
    path = Path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, 0o600)
    return path
