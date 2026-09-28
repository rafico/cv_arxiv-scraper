from __future__ import annotations

import csv
import io
from datetime import date, datetime, timezone
from unittest.mock import patch

from app.models import Collection, Paper, PaperCollection, ScrapeRun, db
from tests.helpers import FlaskDBTestCase


def _make_paper(idx: int, **overrides) -> Paper:
    today = date.today()
    defaults = dict(
        arxiv_id=f"2607.{3000 + idx:04d}",
        title=f"Collection Dashboard Paper {idx}",
        authors="Author A",
        link=f"https://arxiv.org/abs/2607.{3000 + idx:04d}",
        pdf_link=f"https://arxiv.org/pdf/2607.{3000 + idx:04d}",
        abstract_text="abstract",
        match_type="Title",
        is_hidden=False,
        publication_date=today.isoformat(),
        publication_dt=today,
        scraped_date=today.isoformat(),
        scraped_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    defaults.update(overrides)
    return Paper(**defaults)


class CollectionDashboardTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_empty_collection_filter_shows_empty_state_message(self):
        collection = Collection(name="Empty Collection")
        db.session.add(collection)
        db.session.add(ScrapeRun(status="success"))
        db.session.commit()

        response = self.client.get(f"/?collection={collection.id}&timeframe=all")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No papers for this filter", response.data)

    def test_collection_view_renders_manager_controls(self):
        collection = Collection(name="Survey Seeds")
        paper = _make_paper(1)
        db.session.add_all([collection, paper])
        db.session.flush()
        db.session.add(PaperCollection(paper_id=paper.id, collection_id=collection.id))
        db.session.commit()

        text = self.client.get(f"/?collection={collection.id}&timeframe=all").get_data(as_text=True)

        self.assertIn('id="collection-rename-btn"', text)
        self.assertIn('id="collection-delete-btn"', text)
        self.assertIn('id="collection-import-ids"', text)
        self.assertIn(f'href="/api/collections/{collection.id}/table.csv"', text)
        self.assertIn(f'href="/graph?collection={collection.id}"', text)
        self.assertIn("data-remove-from-collection", text)
        self.assertIn(f"removeFromCollection({paper.id}, {collection.id}", text)
        self.assertIn('const collectionName = "Survey Seeds";', text)

    def test_collection_actions_explain_failures(self):
        # A new (empty) collection from the sidebar "+": Suggest similar 400s, and a
        # stale-token/already-deleted Delete 4xxs; neither may silently no-op.
        collection = Collection(name="Fresh")
        db.session.add(collection)
        db.session.commit()

        text = self.client.get(f"/?collection={collection.id}&timeframe=all").get_data(as_text=True)

        self.assertIn("Add a paper to this collection first, then try Suggest similar.", text)
        self.assertIn("Could not delete the collection.", text)
        self.assertIn("un-skip to see them here", text)

    def test_search_inside_collection_keeps_members_outside_global_top_hits(self):
        collection = Collection(name="Review")
        member = _make_paper(1, title="Sparse Voxel Occupancy")
        outsider = _make_paper(2, title="Unrelated Paper")
        db.session.add_all([collection, member, outsider])
        db.session.flush()
        db.session.add(PaperCollection(paper_id=member.id, collection_id=collection.id))
        db.session.commit()

        # The corpus-wide hybrid top-k misses the member entirely.
        hits = [{"paper_id": outsider.id}]
        with patch("app.services.search.search_hybrid", return_value=hits):
            scoped = self.client.get(f"/?collection={collection.id}&timeframe=all&q=voxel")
            inbox = self.client.get("/?timeframe=all&q=voxel")

        self.assertIn(f'data-paper-id="{member.id}"', scoped.get_data(as_text=True))
        self.assertNotIn(f'data-paper-id="{outsider.id}"', scoped.get_data(as_text=True))
        # Unscoped views still trust the ranked ids alone.
        self.assertIn(f'data-paper-id="{outsider.id}"', inbox.get_data(as_text=True))
        self.assertNotIn(f'data-paper-id="{member.id}"', inbox.get_data(as_text=True))

    def test_collection_csv_export(self):
        collection = Collection(name="Review")
        member = _make_paper(
            1,
            title="=HYPERLINK(1)",
            authors="Ada Lovelace, Alan Turing",
            venue="CVPR",
            acceptance_status="accepted",
            openalex_cited_by_count=7,
            user_tags=["screen-in", "3d"],
            user_notes="@SUM(A1)\nsecond line",
        )
        outsider = _make_paper(2)
        hidden = _make_paper(3, is_hidden=True)  # out of the view and .bib, so out of the CSV too
        db.session.add_all([collection, member, outsider, hidden])
        db.session.flush()
        db.session.add(PaperCollection(paper_id=member.id, collection_id=collection.id))
        db.session.add(PaperCollection(paper_id=hidden.id, collection_id=collection.id))
        db.session.commit()

        response = self.client.get(f"/api/collections/{collection.id}/table.csv")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.content_type)
        self.assertIn(f"collection-{collection.id}.csv", response.headers["Content-Disposition"])
        rows = list(csv.DictReader(io.StringIO(response.get_data(as_text=True))))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["arxiv_id"], member.arxiv_id)
        # Formula-looking cells are neutralised with a leading apostrophe.
        self.assertEqual(row["title"], "'=HYPERLINK(1)")
        self.assertEqual(row["user_notes"], "'@SUM(A1)\nsecond line")
        self.assertEqual(row["first_author"], "Ada Lovelace")
        self.assertEqual(row["venue"], "CVPR")
        self.assertEqual(row["citation_count"], "7")  # falls back to OpenAlex
        self.assertEqual(row["readiness"], "none")
        self.assertEqual(row["user_tags"], "screen-in; 3d")

        self.assertEqual(self.client.get("/api/collections/9999/table.csv").status_code, 404)
