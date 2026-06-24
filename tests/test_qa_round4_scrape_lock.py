from __future__ import annotations

import threading
import time
import unittest
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from app.models import Paper, db
from app.services import scrape_engine
from tests.helpers import FlaskDBTestCase


def _make_paper(link: str) -> Paper:
    return Paper(
        arxiv_id=link.rsplit("/", 1)[-1],
        title="Lock Test Paper",
        authors="Alice Example",
        link=link,
        pdf_link=link.replace("/abs/", "/pdf/"),
        abstract_text="An abstract.",
        summary_text="A summary.",
        topic_tags=[],
        categories=["cs.CV"],
        match_type="Title",
        matched_terms=["Vision"],
        paper_score=1.0,
        publication_date="2026-04-07",
        publication_dt=date(2026, 4, 7),
        scraped_date="2026-04-07",
        scraped_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )


class ScrapeIndexWriteLockQaTests(FlaskDBTestCase):
    """G9: FAISS read-append-rename inside _generate_embeddings must be serialized
    so a scheduled scrape and a concurrent historical search can't both load the
    same index snapshot and have the later os.replace silently drop the other's
    vectors.
    """

    def test_g9_lock_exists(self):
        self.assertTrue(
            hasattr(scrape_engine, "_INDEX_WRITE_LOCK"),
            "scrape_engine must expose a module-level lock guarding FAISS writes",
        )
        # A non-reentrant Lock (so each stage acquires once and releases).
        lock = scrape_engine._INDEX_WRITE_LOCK
        self.assertTrue(lock.acquire(blocking=False))
        lock.release()

    def test_g9_generate_embeddings_serializes_index_write(self):
        link_a = "https://arxiv.org/abs/2604.10001"
        link_b = "https://arxiv.org/abs/2604.10002"
        db.session.add(_make_paper(link_a))
        db.session.add(_make_paper(link_b))
        db.session.commit()

        # State shared between the two concurrent run_isolated invocations.
        active = 0
        max_active = 0
        counter_lock = threading.Lock()

        def fake_run_isolated(func, *args, **kwargs):
            nonlocal active, max_active
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
            # Hold the critical section long enough that, if unguarded, the
            # second thread would enter and bump active to 2.
            time.sleep(0.2)
            with counter_lock:
                active -= 1
            return 1

        # A fake embedding service: index_dir + has_paper(False) so each call
        # finds work to do and reaches run_isolated.
        service = SimpleNamespace(
            index_dir="/tmp/does-not-matter",
            has_paper=lambda _pid: False,
        )

        results_a = [{"link": link_a, "embedding": None}]
        results_b = [{"link": link_b, "embedding": None}]

        with (
            patch("app.services.embeddings.get_embedding_service", return_value=service),
            patch("app.services.embeddings.reset_embedding_service"),
            patch("app.services.subprocess_runner.run_isolated", side_effect=fake_run_isolated),
        ):
            t1 = threading.Thread(target=scrape_engine._generate_embeddings, args=(self.app, results_a))
            t2 = threading.Thread(target=scrape_engine._generate_embeddings, args=(self.app, results_b))
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

        self.assertEqual(
            max_active,
            1,
            "Concurrent _generate_embeddings calls overlapped the FAISS index write "
            "(read-append-rename is unguarded -> silent vector loss)",
        )


if __name__ == "__main__":
    unittest.main()
