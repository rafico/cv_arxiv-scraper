"""Distribution & operational-trust packaging (Wave 3, ROADMAP item 14).

Covers /healthz, the `cv-arxiv serve` data-dir resolution + `--version`, and the
single-source version wiring. Everything runs against tmp dirs / the isolated
test app — nothing here touches the real repo ``instance/``.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import tomllib
import yaml

from app._version import __version__
from app.cli import serve
from tests.helpers import FlaskDBTestCase

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_INSTANCE = _REPO_ROOT / "instance"


class HealthzEndpointTests(FlaskDBTestCase):
    def test_healthz_returns_200_and_expected_keys(self):
        with self.app.test_client() as client:
            resp = client.get("/healthz")

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(set(data), {"status", "version", "db_ok", "faiss_ok", "paper_count"})
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["version"], __version__)
        self.assertIs(data["db_ok"], True)
        self.assertIsInstance(data["faiss_ok"], bool)
        self.assertIsInstance(data["paper_count"], int)
        self.assertGreaterEqual(data["paper_count"], 0)

    def test_healthz_is_top_level_not_under_api_prefix(self):
        # Must live at /healthz (probes/healthchecks target that), not /api/healthz.
        self.assertIn("health", self.app.blueprints)
        with self.app.test_client() as client:
            self.assertEqual(client.get("/api/healthz").status_code, 404)


class ServeDataDirResolutionTests(unittest.TestCase):
    def test_flag_beats_env_and_default(self):
        resolved = serve.resolve_data_dir("/tmp/flagdir", environ={"CV_ARXIV_DATA_DIR": "/tmp/envdir"})
        self.assertEqual(resolved, Path("/tmp/flagdir").resolve())

    def test_env_beats_default(self):
        resolved = serve.resolve_data_dir(None, environ={"CV_ARXIV_DATA_DIR": "/tmp/envdir"})
        self.assertEqual(resolved, Path("/tmp/envdir").resolve())

    def test_blank_flag_falls_through_to_env(self):
        resolved = serve.resolve_data_dir("", environ={"CV_ARXIV_DATA_DIR": "/tmp/envdir"})
        self.assertEqual(resolved, Path("/tmp/envdir").resolve())

    def test_default_when_nothing_set(self):
        resolved = serve.resolve_data_dir(None, environ={})
        self.assertEqual(resolved, serve.DEFAULT_DATA_DIR.resolve())

    def test_prepare_data_dir_seeds_config_and_sets_env_in_tmp(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            env: dict[str, str] = {}

            config_path = serve.prepare_data_dir(data_dir, environ=env)

            self.assertTrue(data_dir.is_dir())
            self.assertEqual(config_path, data_dir / "config.yaml")
            self.assertTrue(config_path.is_file())
            self.assertEqual(env["CV_ARXIV_INSTANCE_PATH"], str(data_dir))
            self.assertEqual(env["CV_ARXIV_CONFIG"], str(config_path))
            # Seeded config must be valid enough to load.
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertIn("scraper", loaded)
            self.assertIn("whitelists", loaded)

    def test_prepare_data_dir_honours_explicit_config_env(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "custom.yaml"
            env = {"CV_ARXIV_CONFIG": str(custom)}

            config_path = serve.prepare_data_dir(Path(tmp) / "d", environ=env)

            self.assertEqual(config_path, custom.resolve())
            self.assertEqual(env["CV_ARXIV_CONFIG"], str(custom))
            self.assertTrue(custom.is_file())

    def test_serve_forwards_remaining_args_to_run_main_without_touching_repo_instance(self):
        import os
        import tempfile

        before = sorted(p.name for p in _REPO_INSTANCE.iterdir()) if _REPO_INSTANCE.exists() else None

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "dd"
            # patch.dict restores os.environ afterwards so the CLI's env writes don't leak.
            with patch.dict(os.environ, {}, clear=False), patch("run.main", return_value=0) as mock_main:
                rc = serve.main(["serve", "--data-dir", str(data_dir), "--no-browser", "--port", "5599"])

            self.assertEqual(rc, 0)
            mock_main.assert_called_once_with(["--no-browser", "--port", "5599"])
            self.assertTrue((data_dir / "config.yaml").is_file())

        after = sorted(p.name for p in _REPO_INSTANCE.iterdir()) if _REPO_INSTANCE.exists() else None
        self.assertEqual(before, after, "serve must not create anything under the repo instance/ dir")

    def test_unknown_command_returns_2(self):
        self.assertEqual(serve.main(["bogus"]), 2)


class VersionTests(unittest.TestCase):
    def test_version_is_valid_string(self):
        self.assertIsInstance(__version__, str)
        parts = __version__.split(".")
        self.assertGreaterEqual(len(parts), 2)
        self.assertTrue(all(p.isdigit() for p in parts[:2]), __version__)

    def test_version_matches_pyproject(self):
        with (_REPO_ROOT / "pyproject.toml").open("rb") as fh:
            pyproject = tomllib.load(fh)
        project = pyproject["project"]

        declared = project.get("version")
        if declared is None:
            # Dynamic version: resolve the single-source attr exactly as setuptools does.
            self.assertIn("version", project.get("dynamic", []))
            attr = pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"]
            self.assertEqual(attr, "app._version.__version__")
            import importlib

            module_name, _, attr_name = attr.rpartition(".")
            declared = getattr(importlib.import_module(module_name), attr_name)

        self.assertEqual(declared, __version__)

    def test_cli_version_flag_prints_version(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = serve.main(["--version"])
        self.assertEqual(rc, 0)
        self.assertIn(__version__, buf.getvalue())


if __name__ == "__main__":
    unittest.main()
