"""Atomic, 0600 writes for credential/token dotfiles.

Writing a secret with ``Path.write_text`` then ``os.chmod(0o600)`` leaves a window
where the file exists under the default umask (often world-readable). Even
``os.open(path, O_CREAT|O_TRUNC, 0o600)`` does NOT re-apply the mode when the target
already exists (the mode arg only applies on creation), so *overwriting* a
pre-existing looser-mode secret writes the new bytes through the old, looser inode
until the trailing chmod runs. Instead we always write to a fresh ``mkstemp`` file
(created 0600) next to the target and atomically rename it into place — the secret is
never present on disk at a looser mode, regardless of any pre-existing file. Mirrors
``preferences.save_config`` / ``backup._atomic_write``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_secret_file(path: Path | str, text: str) -> Path:
    """Write ``text`` to ``path`` with mode 0600, never briefly readable at a looser mode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")

    # mkstemp creates a brand-new O_EXCL file at mode 0600 — it can never inherit a
    # pre-existing target's looser mode.
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        try:
            os.replace(tmp_path, path)
        except OSError:
            # Rename failed (e.g. a single-file bind mount is the target). Fall back to
            # an in-place write, still forcing 0600. This is the same edge-case tradeoff
            # accepted by save_config for non-renameable destinations.
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as handle:
                handle.write(data)
            os.chmod(path, 0o600)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return path
