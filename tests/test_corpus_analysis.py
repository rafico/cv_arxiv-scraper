from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np

from app.models import Paper, db
from app.services.corpus_analysis import analyze_topic_clusters, detect_emerging_topics, find_neighbor_papers
from tests.helpers import FlaskDBTestCase


def _unit(vector: list[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    return array if norm == 0 else array / norm


class FakeEmbeddingService:
    def __init__(self, vectors: dict[int, np.ndarray], neighbors: dict[int, list[tuple[int, float]]] | None = None):
        self.vectors = {paper_id: np.asarray(vector, dtype=np.float32) for paper_id, vector in vectors.items()}
        self.neighbors = neighbors or {}

    def get_paper_vectors(self, paper_ids: list[int]) -> tuple[list[int], np.ndarray]:
        found_ids = [paper_id for paper_id in paper_ids if paper_id in self.vectors]
        if not found_ids:
            return [], np.empty((0, 3), dtype=np.float32)
        return found_ids, np.vstack([self.vectors[paper_id] for paper_id in found_ids])

    def search_by_id(self, paper_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        return self.neighbors.get(paper_id, [])[:top_k]


class CorpusAnalysisTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.reference_time = datetime(2026, 4, 3, 12, 0, 0)

    def _add_paper(
        self,
        *,
        arxiv_id: str,
        title: str,
        abstract: str,
        authors: str = "Author A",
        publication_dt: date | None = None,
        scraped_at: datetime | None = None,
        paper_score: float = 1.0,
        is_hidden: bool = False,
    ) -> Paper:
        paper = Paper(
            arxiv_id=arxiv_id,
            title=title,
            authors=authors,
            link=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_link=f"https://arxiv.org/pdf/{arxiv_id}",
            abstract_text=abstract,
            match_type="Title",
            matched_terms=["Vision"],
            paper_score=paper_score,
            publication_date=publication_dt.isoformat() if publication_dt else None,
            scraped_date=(scraped_at or self.reference_time).date().isoformat(),
            publication_dt=publication_dt,
            scraped_at=scraped_at or self.reference_time,
            is_hidden=is_hidden,
        )
        db.session.add(paper)
        db.session.commit()
        return paper

    def test_analyze_topic_clusters_groups_recent_papers_by_embedding_similarity(self):
        recent_date = self.reference_time.date() - timedelta(days=2)
        old_date = self.reference_time.date() - timedelta(days=20)
        papers = [
            self._add_paper(
                arxiv_id="2604.0001",
                title="Satellite segmentation with transformers",
                abstract="satellite segmentation remote sensing transformer",
                publication_dt=recent_date,
            ),
            self._add_paper(
                arxiv_id="2604.0002",
                title="Remote sensing segmentation for imagery",
                abstract="segmentation model for satellite imagery",
                publication_dt=recent_date,
            ),
            self._add_paper(
                arxiv_id="2604.0003",
                title="Video diffusion for motion synthesis",
                abstract="video diffusion temporal motion synthesis",
                publication_dt=recent_date,
            ),
            self._add_paper(
                arxiv_id="2604.0004",
                title="Motion generation with diffusion models",
                abstract="video generation with diffusion",
                publication_dt=recent_date,
            ),
        ]
        self._add_paper(
            arxiv_id="2604.0099",
            title="Old paper outside the window",
            abstract="legacy segmentation work",
            publication_dt=old_date,
        )

        service = FakeEmbeddingService(
            {
                papers[0].id: _unit([1.0, 0.0, 0.0]),
                papers[1].id: _unit([0.98, 0.05, 0.0]),
                papers[2].id: _unit([0.0, 1.0, 0.0]),
                papers[3].id: _unit([0.05, 0.98, 0.0]),
            }
        )

        result = analyze_topic_clusters(
            window_days=7,
            cluster_count=2,
            reference_time=self.reference_time,
            embedding_service=service,
        )

        self.assertEqual(result["paper_count"], 4)
        self.assertEqual(result["indexed_paper_count"], 4)
        self.assertEqual(result["cluster_count"], 2)
        self.assertEqual(sorted(cluster["size"] for cluster in result["clusters"]), [2, 2])
        labels = " | ".join(cluster["label"] for cluster in result["clusters"])
        self.assertIn("segmentation", labels)
        self.assertIn("diffusion", labels)

    def test_detect_emerging_topics_compares_recent_distribution_to_baseline(self):
        recent_date = self.reference_time.date() - timedelta(days=2)
        baseline_date = self.reference_time.date() - timedelta(days=14)
        recent_papers = [
            self._add_paper(
                arxiv_id="2604.0101",
                title="Video diffusion for motion synthesis",
                abstract="video diffusion temporal control",
                publication_dt=recent_date,
            ),
            self._add_paper(
                arxiv_id="2604.0102",
                title="Diffusion models for video generation",
                abstract="video diffusion generation pipeline",
                publication_dt=recent_date,
            ),
        ]
        baseline_papers = [
            self._add_paper(
                arxiv_id="2603.0201",
                title="Satellite segmentation with transformers",
                abstract="satellite segmentation remote sensing transformer",
                publication_dt=baseline_date,
            ),
            self._add_paper(
                arxiv_id="2603.0202",
                title="Segmentation for aerial imagery",
                abstract="remote sensing segmentation for aerial imagery",
                publication_dt=baseline_date,
            ),
            self._add_paper(
                arxiv_id="2603.0203",
                title="Instance segmentation for sensors",
                abstract="segmentation for multimodal sensing",
                publication_dt=baseline_date,
            ),
            self._add_paper(
                arxiv_id="2603.0204",
                title="Video diffusion baseline",
                abstract="diffusion model for video",
                publication_dt=baseline_date,
            ),
        ]

        service = FakeEmbeddingService(
            {
                recent_papers[0].id: _unit([0.0, 1.0, 0.0]),
                recent_papers[1].id: _unit([0.05, 0.98, 0.0]),
                baseline_papers[0].id: _unit([1.0, 0.0, 0.0]),
                baseline_papers[1].id: _unit([0.98, 0.04, 0.0]),
                baseline_papers[2].id: _unit([0.96, 0.08, 0.0]),
                baseline_papers[3].id: _unit([0.02, 0.99, 0.0]),
            }
        )

        result = detect_emerging_topics(
            recent_days=7,
            baseline_days=28,
            cluster_count=2,
            reference_time=self.reference_time,
            embedding_service=service,
        )

        self.assertEqual(result["indexed_recent_paper_count"], 2)
        self.assertEqual(result["indexed_baseline_paper_count"], 4)
        self.assertGreaterEqual(len(result["topics"]), 1)
        top_topic = result["topics"][0]
        self.assertEqual(top_topic["recent_count"], 2)
        self.assertEqual(top_topic["baseline_count"], 1)
        self.assertGreater(top_topic["delta_share"], 0)
        self.assertIn("diffusion", top_topic["label"])

    def test_find_neighbor_papers_excludes_tracked_authors_and_merges_seed_hits(self):
        seed_a = self._add_paper(
            arxiv_id="2604.0301",
            title="Seed A",
            abstract="seed paper",
            authors="Seed Author",
            publication_dt=self.reference_time.date(),
        )
        seed_b = self._add_paper(
            arxiv_id="2604.0302",
            title="Seed B",
            abstract="seed paper",
            authors="Seed Author",
            publication_dt=self.reference_time.date(),
        )
        tracked = self._add_paper(
            arxiv_id="2604.0303",
            title="Tracked neighbor",
            abstract="neighbor paper",
            authors="Tracked Author, Collaborator",
            publication_dt=self.reference_time.date(),
        )
        untracked = self._add_paper(
            arxiv_id="2604.0304",
            title="Untracked neighbor",
            abstract="neighbor paper",
            authors="Independent Researcher",
            publication_dt=self.reference_time.date(),
        )
        hidden = self._add_paper(
            arxiv_id="2604.0305",
            title="Hidden neighbor",
            abstract="neighbor paper",
            authors="Independent Researcher",
            publication_dt=self.reference_time.date(),
            is_hidden=True,
        )

        service = FakeEmbeddingService(
            vectors={},
            neighbors={
                seed_a.id: [(untracked.id, 0.91), (tracked.id, 0.88), (hidden.id, 0.7)],
                seed_b.id: [(untracked.id, 0.87)],
            },
        )

        result = find_neighbor_papers(
            [seed_a.id, seed_b.id],
            limit=10,
            tracked_authors=["Tracked Author"],
            exclude_tracked_authors=True,
            embedding_service=service,
        )

        self.assertEqual(result["seed_paper_ids"], [seed_a.id, seed_b.id])
        self.assertEqual(len(result["results"]), 1)
        neighbor = result["results"][0]
        self.assertEqual(neighbor["id"], untracked.id)
        self.assertEqual(neighbor["matched_seed_ids"], [seed_a.id, seed_b.id])
        self.assertEqual(neighbor["similarity_score"], 0.91)
