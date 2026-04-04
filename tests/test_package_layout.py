from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import _resolve_config_path
from app.ingest import run_scrape
from app.rank import get_preferences
from app.search_ import now_utc
from app.services.jobs import SCRAPE_JOB_MANAGER as LEGACY_JOB_MANAGER
from app.services.preferences import get_preferences as legacy_get_preferences
from app.services.scrape_engine import run_scrape as legacy_run_scrape
from app.services.text import now_utc as legacy_now_utc
from app.web import SCRAPE_JOB_MANAGER


class SemanticPackageTests(unittest.TestCase):
    def test_semantic_packages_reexport_existing_services(self):
        self.assertIs(run_scrape, legacy_run_scrape)
        self.assertIs(get_preferences, legacy_get_preferences)
        self.assertIs(now_utc, legacy_now_utc)
        self.assertIs(SCRAPE_JOB_MANAGER, LEGACY_JOB_MANAGER)

    def test_submodule_aliases_share_underlying_modules(self):
        self.assertIs(
            importlib.import_module("app.ingest.http_client"),
            importlib.import_module("app.services.http_client"),
        )
        self.assertIs(
            importlib.import_module("app.enrich.citations"),
            importlib.import_module("app.services.citations"),
        )
        self.assertIs(
            importlib.import_module("app.search_.embeddings"),
            importlib.import_module("app.services.embeddings"),
        )
        self.assertIs(
            importlib.import_module("app.web.email_digest"),
            importlib.import_module("app.services.email_digest"),
        )

    def test_legacy_cli_modules_alias_installable_entrypoints(self):
        self.assertIs(importlib.import_module("sync_cli"), importlib.import_module("app.cli.sync"))
        self.assertIs(importlib.import_module("backfill_cli"), importlib.import_module("app.cli.backfill"))


class ConfigPathResolutionTests(unittest.TestCase):
    def test_explicit_config_path_wins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "custom.yaml"
            config_path.write_text("scraper: {}\n", encoding="utf-8")

            resolved = _resolve_config_path(str(config_path))

            self.assertEqual(resolved, config_path.resolve())

    def test_env_var_beats_repo_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "env.yaml"
            config_path.write_text("scraper: {}\n", encoding="utf-8")
            with patch.dict(os.environ, {"CV_ARXIV_CONFIG": str(config_path)}, clear=False):
                resolved = _resolve_config_path()

            self.assertEqual(resolved, config_path.resolve())

    def test_cwd_config_is_used_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("scraper: {}\n", encoding="utf-8")
            original_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                with patch.dict(os.environ, {"CV_ARXIV_CONFIG": ""}, clear=False):
                    resolved = _resolve_config_path()
            finally:
                os.chdir(original_cwd)

            self.assertEqual(resolved, config_path.resolve())
