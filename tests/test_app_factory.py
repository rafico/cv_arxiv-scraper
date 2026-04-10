from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import yaml
from sqlalchemy import inspect, text

from app import create_app
from app.models import Paper, db
from tests.helpers import TEST_SCRAPER_CONFIG


class CreateAppInitializationTests(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(TEST_SCRAPER_CONFIG), encoding="utf-8")
        return config_path

    def _create_app(self, root: Path):
        return create_app(
            {
                "TESTING": True,
                "CONFIG_PATH": str(self._write_config(root)),
                "INSTANCE_PATH": str(root / "instance"),
                "LLM_KEY_PATH": str(root / ".llm_api_key"),
            }
        )

    def test_first_run_creates_sqlite_database_in_instance_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = self._create_app(root)

            db_path = Path(app.instance_path) / "arxiv_papers.db"

            self.assertTrue(db_path.exists())

    def test_create_app_initializes_tables_and_indexes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = self._create_app(root)

            with app.app_context():
                inspector = inspect(db.engine)
                tables = set(inspector.get_table_names())
                expected_tables = {
                    "papers",
                    "paper_feedback",
                    "paper_sections",
                    "scrape_runs",
                    "digest_runs",
                    "saved_searches",
                    "sync_state",
                    "papers_fts",
                }
                self.assertTrue(expected_tables.issubset(tables))

                paper_indexes = {
                    row["name"] for row in db.session.execute(text("PRAGMA index_list('papers')")).mappings()
                }
                self.assertTrue(
                    {
                        "idx_papers_scraped_at",
                        "idx_papers_publication_dt",
                        "idx_papers_rank",
                        "idx_papers_hidden",
                        "uq_papers_arxiv_id",
                    }.issubset(paper_indexes)
                )

                feedback_indexes = {
                    row["name"] for row in db.session.execute(text("PRAGMA index_list('paper_feedback')")).mappings()
                }
                self.assertIn("idx_feedback_paper_action", feedback_indexes)

    def test_existing_database_is_preserved_on_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            app = self._create_app(root)
            with app.app_context():
                db.session.add(
                    Paper(
                        arxiv_id="2604.00001",
                        title="Persistent Paper",
                        authors="Author A",
                        link="https://arxiv.org/abs/2604.00001",
                        pdf_link="https://arxiv.org/pdf/2604.00001",
                        abstract_text="startup persistence",
                        match_type="Title",
                        matched_terms=["Vision"],
                        paper_score=1.0,
                        publication_date="2026-04-07",
                        scraped_date="2026-04-07",
                    )
                )
                db.session.commit()
                db.session.remove()
                db.engine.dispose()

            restarted_app = self._create_app(root)
            with restarted_app.app_context():
                stored = Paper.query.one()
                self.assertEqual(stored.title, "Persistent Paper")
                self.assertEqual(stored.arxiv_id, "2604.00001")

    def test_create_app_uses_template_defaults_without_writing_instance_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_path = root / "config.example.yaml"
            template_path.write_text(yaml.safe_dump(TEST_SCRAPER_CONFIG), encoding="utf-8")
            original_cwd = Path.cwd()
            try:
                os.chdir(root)
                app = create_app(
                    {
                        "TESTING": True,
                        "INSTANCE_PATH": str(root / "instance"),
                        "LLM_KEY_PATH": str(root / ".llm_api_key"),
                    }
                )
            finally:
                os.chdir(original_cwd)

            config_path = Path(app.config["CONFIG_PATH"])
            self.assertEqual(config_path, (root / "instance" / "config.yaml").resolve())
            self.assertFalse(config_path.exists())
            self.assertTrue(app.config["USING_DEFAULT_CONFIG"])
            self.assertEqual(app.config["SCRAPER_CONFIG"], TEST_SCRAPER_CONFIG)


if __name__ == "__main__":
    unittest.main()
