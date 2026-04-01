"""Manage crontab entries for daily scrape/digest automation."""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)

CRON_TAG = "# cv-arxiv-scraper-auto"


def _project_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _python_bin() -> str:
    return sys.executable


def _build_cron_line(hour: int, minute: int, mode: str) -> str:
    project = _project_dir()
    python = _python_bin()

    if mode == "scrape":
        script = "scrape_cli.py"
    elif mode == "digest":
        script = "digest_cli.py --send-only"
    else:
        script = "digest_cli.py"

    return f"{minute} {hour} * * * cd {project} && {python} {script} >> {project}/cron.log 2>&1 {CRON_TAG}"


def _get_current_crontab() -> str:
    try:
        result = subprocess.run(
            ["crontab", "-l"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return ""
        return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _set_crontab(content: str) -> None:
    subprocess.run(
        ["crontab", "-"],
        input=content,
        text=True,
        check=True,
        timeout=5,
    )


def _remove_our_lines(crontab: str) -> str:
    lines = crontab.splitlines()
    filtered = [line for line in lines if CRON_TAG not in line]
    return "\n".join(filtered) + "\n" if filtered else ""


def install_cron_job(hour: int, minute: int, mode: str) -> dict:
    """Install or update the cron job. Returns status dict."""
    hour = max(0, min(23, int(hour)))
    minute = max(0, min(59, int(minute)))
    if mode not in ("full", "scrape", "digest"):
        mode = "full"

    cron_line = _build_cron_line(hour, minute, mode)
    existing = _get_current_crontab()
    cleaned = _remove_our_lines(existing)
    new_crontab = cleaned.rstrip("\n") + "\n" + cron_line + "\n"

    try:
        _set_crontab(new_crontab)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        LOGGER.exception("Failed to install cron job")
        return {"success": False, "message": f"Failed to install cron job: {exc}"}

    LOGGER.info("Cron job installed: %s", cron_line)
    return {"success": True, "message": "Cron job installed successfully."}


def remove_cron_job() -> dict:
    """Remove our cron job entry."""
    existing = _get_current_crontab()
    if CRON_TAG not in existing:
        return {"success": True, "message": "No cron job was installed."}

    cleaned = _remove_our_lines(existing)
    try:
        _set_crontab(cleaned)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        LOGGER.exception("Failed to remove cron job")
        return {"success": False, "message": f"Failed to remove cron job: {exc}"}

    LOGGER.info("Cron job removed")
    return {"success": True, "message": "Cron job removed."}


def get_cron_status() -> dict:
    """Return current cron job status for the UI."""
    existing = _get_current_crontab()

    for line in existing.splitlines():
        if CRON_TAG in line:
            match = re.match(r"^(\d+)\s+(\d+)\s+", line.strip())
            if match:
                minute, hour = int(match.group(1)), int(match.group(2))
            else:
                minute, hour = 0, 8

            if "scrape_cli.py" in line:
                mode = "scrape"
            elif "--send-only" in line:
                mode = "digest"
            else:
                mode = "full"

            mode_labels = {
                "full": "Scrape + Send Digest",
                "scrape": "Scrape only",
                "digest": "Send Digest only",
            }

            return {
                "installed": True,
                "hour": hour,
                "minute": minute,
                "mode": mode,
                "mode_label": mode_labels.get(mode, mode),
                "cron_line": line.strip(),
            }

    return {"installed": False, "hour": 8, "minute": 0, "mode": "full"}
