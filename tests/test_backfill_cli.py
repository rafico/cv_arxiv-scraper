from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.models import Paper, db
from app.services.embeddings import EmbeddingService, reset_embedding_service
from app.services.ranking import compute_paper_score
from backfill_cli import (
    backfill_citations,
    backfill_openalex,
    backfill_thumbnails,
    main,
    rebuild_semantic_index,
    run_all_backfills,
    run_embeddings_backfill,
)
from tests.helpers import FlaskDBTestCase


def _paper(arxiv_id: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        authors="Author A",
        link=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_link=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        match_type="title",
        matched_terms=["Vision"],
        paper_score=1.0,
        publication_date="2026-01-01",
        scraped_date="2026-01-01",
    )


class BackfillCliTests(FlaskDBTestCase):
    def tearDown(self):
        reset_embedding_service()
        super().tearDown()

    @patch("app.services.embed_backfill.backfill_embeddings", return_value=3)
    def test_run_embeddings_backfill_wraps_service(self, mock_backfill):
        messages: list[str] = []

        count = run_embeddings_backfill(self.app, batch_size=32, emit=messages.append)

        self.assertEqual(count, 3)
        mock_backfill.assert_called_once_with(self.app, batch_size=32)
        self.assertTrue(messages[-1].startswith("Embeddings backfill complete:"))

    @patch("app.services.citations.fetch_citations_batch")
    def test_backfill_citations_updates_missing_papers(self, mock_fetch):
        db.session.add(_paper("2601.00001"))
        db.session.commit()
        mock_fetch.return_value = {
            "2601.00001": {
                "citation_count": 12,
                "influential_citation_count": 4,
                "semantic_scholar_id": "abc123",
            }
        }
        messages: list[str] = []

        updated = backfill_citations(self.app, batch_size=10, delay_seconds=0, emit=messages.append)

        stored = Paper.query.filter_by(arxiv_id="2601.00001").one()
        self.assertEqual(updated, 1)
        self.assertEqual(stored.citation_count, 12)
        self.assertEqual(stored.influential_citation_count, 4)
        self.assertEqual(stored.semantic_scholar_id, "abc123")
        self.assertEqual(stored.citation_source, "semantic_scholar")
        self.assertEqual(stored.citation_provenance["source"], "semantic_scholar")
        self.assertIsNotNone(stored.citation_updated_at)
        self.assertEqual(
            stored.paper_score,
            compute_paper_score(
                match_types=[part.strip() for part in (stored.match_type or "").split("+") if part.strip()],
                matched_terms_count=len(stored.matched_terms_list),
                publication_dt=stored.publication_dt,
                resource_count=len(stored.resource_links_list),
                llm_relevance_score=stored.llm_relevance_score,
                citation_count=stored.citation_count,
                config=self.app.config["SCRAPER_CONFIG"],
            ),
        )
        self.assertTrue(messages[-1].startswith("Citations batch"))

    @patch("app.services.openalex.fetch_openalex_batch")
    def test_backfill_openalex_updates_missing_papers(self, mock_fetch):
        db.session.add(_paper("2601.00002"))
        db.session.commit()
        mock_fetch.return_value = {
            "2601.00002": {
                "openalex_id": "W123",
                "openalex_topics": [{"name": "Vision", "score": 0.9}],
                "oa_status": "green",
                "openalex_cited_by_count": 7,
                "referenced_works_count": 2,
            }
        }

        updated = backfill_openalex(self.app, batch_size=10, delay_seconds=0, emit=lambda _: None)

        stored = Paper.query.filter_by(arxiv_id="2601.00002").one()
        self.assertEqual(updated, 1)
        self.assertEqual(stored.openalex_id, "W123")
        self.assertEqual(stored.oa_status, "green")
        self.assertEqual(stored.citation_count, 7)
        self.assertEqual(stored.openalex_cited_by_count, 7)
        self.assertEqual(stored.referenced_works_count, 2)
        self.assertEqual(stored.citation_source, "openalex")
        self.assertEqual(stored.citation_provenance["source"], "openalex")
        self.assertIsNotNone(stored.citation_updated_at)
        self.assertEqual(
            stored.paper_score,
            compute_paper_score(
                match_types=[part.strip() for part in (stored.match_type or "").split("+") if part.strip()],
                matched_terms_count=len(stored.matched_terms_list),
                publication_dt=stored.publication_dt,
                resource_count=len(stored.resource_links_list),
                llm_relevance_score=stored.llm_relevance_score,
                citation_count=stored.citation_count,
                config=self.app.config["SCRAPER_CONFIG"],
            ),
        )

    @patch("app.services.thumbnail_generator.generate_thumbnail", return_value=True)
    def test_backfill_thumbnails_only_generates_missing_files(self, mock_generate):
        static_dir = Path(self._tmpdir.name) / "static"
        self.app.static_folder = str(static_dir)
        db.session.add_all([_paper("2601.00003"), _paper("2601.00004")])
        db.session.commit()

        existing_dir = static_dir / "thumbnails"
        existing_dir.mkdir(parents=True, exist_ok=True)
        (existing_dir / "2601.00003.png").write_bytes(b"png")

        generated = backfill_thumbnails(self.app, batch_size=10, delay_seconds=0, emit=lambda _: None)

        self.assertEqual(generated, 1)
        mock_generate.assert_called_once()
        self.assertEqual(mock_generate.call_args.args[:2], ("2601.00004", "https://arxiv.org/pdf/2601.00004.pdf"))

    @patch("app.services.embeddings.EmbeddingService.encode", autospec=True)
    def test_rebuild_semantic_index_replaces_existing_files(self, mock_encode):
        index_dir = Path(self._tmpdir.name) / "faiss_index"
        index_dir.mkdir(parents=True, exist_ok=True)
        self.app.config["FAISS_INDEX_DIR"] = str(index_dir)
        (index_dir / "papers.index").write_bytes(b"old-index")
        (index_dir / "id_map.json").write_text("[999]", encoding="utf-8")

        db.session.add_all([_paper("2601.00005"), _paper("2601.00006")])
        db.session.commit()

        def fake_encode(_service, texts):
            vectors = np.zeros((len(texts), 768), dtype=np.float32)
            for idx in range(len(texts)):
                vectors[idx, idx % 768] = 1.0
            return vectors

        mock_encode.side_effect = fake_encode
        messages: list[str] = []

        indexed = rebuild_semantic_index(self.app, batch_size=1, emit=messages.append)

        self.assertEqual(indexed, 2)
        self.assertTrue(messages[0].startswith("Rebuilding semantic index"))
        self.assertTrue(any(message.startswith("Index rebuild batch 1:") for message in messages))
        self.assertTrue(messages[-1].startswith("Semantic index rebuild complete:"))

        service = EmbeddingService(index_dir)
        self.assertEqual(service.index_count(), 2)
        self.assertTrue(service.has_paper(Paper.query.filter_by(arxiv_id="2601.00005").one().id))
        self.assertTrue(service.has_paper(Paper.query.filter_by(arxiv_id="2601.00006").one().id))

    @patch("backfill_cli.backfill_thumbnails", return_value=4)
    @patch("backfill_cli.backfill_openalex", return_value=3)
    @patch("backfill_cli.backfill_citations", return_value=2)
    @patch("backfill_cli.run_embeddings_backfill", return_value=1)
    def test_run_all_backfills_runs_each_task(self, mock_embeddings, mock_citations, mock_openalex, mock_thumbnails):
        result = run_all_backfills(self.app, batch_size=25, delay_seconds=0, emit=lambda _: None)

        self.assertEqual(
            result,
            {
                "embeddings": 1,
                "citations": 2,
                "openalex": 3,
                "thumbnails": 4,
            },
        )
        mock_embeddings.assert_called_once()
        mock_citations.assert_called_once()
        mock_openalex.assert_called_once()
        mock_thumbnails.assert_called_once()

    @patch("backfill_cli.rebuild_semantic_index", return_value=2)
    @patch("backfill_cli.create_app")
    def test_main_routes_index_rebuild_command(self, mock_create_app, mock_rebuild):
        mock_create_app.return_value = self.app

        exit_code = main(["index-rebuild", "--batch-size", "16"])

        self.assertEqual(exit_code, 0)
        mock_rebuild.assert_called_once_with(self.app, batch_size=16)


if __name__ == "__main__":
    unittest.main()
