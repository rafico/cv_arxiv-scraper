"""Atomic, 0600 writes — and env-first reads — for credential/token dotfiles.

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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data-source API credentials: (env var, dotfile name). Env beats dotfile —
# mirrors llm_client.resolve_api_key for .llm_api_key. Dotfiles live next to
# .llm_api_key (the config root) so every secret shares one gitignored home.
DATA_SOURCE_SECRETS: dict[str, tuple[str, str]] = {
    "openalex": ("OPENALEX_API_KEY", ".openalex_api_key"),
    "semantic_scholar": ("SEMANTIC_SCHOLAR_API_KEY", ".s2_api_key"),
    "github": ("GITHUB_TOKEN", ".github_token"),
}


def _secrets_root() -> Path:
    """Directory holding secret dotfiles: the .llm_api_key root when an app is active."""
    from flask import current_app, has_app_context

    if has_app_context():
        llm_key_path = current_app.config.get("LLM_KEY_PATH")
        if llm_key_path:
            return Path(llm_key_path).parent
    return _PROJECT_ROOT


def read_secret_file(path: Path | str) -> str | None:
    """Read a secret dotfile; None when missing, unreadable, or empty."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def data_source_key_path(source: str, base_dir: Path | str | None = None) -> Path:
    """Dotfile path for a data-source key (``source`` per DATA_SOURCE_SECRETS)."""
    _env_var, filename = DATA_SOURCE_SECRETS[source]
    root = Path(base_dir) if base_dir is not None else _secrets_root()
    return root / filename


def resolve_data_source_key(source: str, key_path: Path | str | None = None) -> str | None:
    """Resolve a data-source API key: env var first, then the gitignored dotfile."""
    env_var, _filename = DATA_SOURCE_SECRETS[source]
    value = os.environ.get(env_var, "").strip()
    if value:
        return value
    return read_secret_file(key_path or data_source_key_path(source))


def has_data_source_key(source: str, key_path: Path | str | None = None) -> bool:
    return resolve_data_source_key(source, key_path=key_path) is not None


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
            # accepted by save_config for non-renameable destinations. The mode arg to
            # os.open is ignored when the file already exists, so fchmod the fd to 0600
            # BEFORE writing bytes — the secret is never present at a looser mode.
            fallback_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.fchmod(fallback_fd, 0o600)
            with os.fdopen(fallback_fd, "wb") as handle:
                handle.write(data)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return path
