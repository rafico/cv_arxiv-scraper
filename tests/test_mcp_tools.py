"""Tests for the MCP tool logic layer (:mod:`app.services.mcp_tools`) and the
import-hygiene guarantees of the protocol wrapper (:mod:`app.mcp_server`).

The logic layer is exercised against a seeded ``FlaskDBTestCase`` DB with **no
``mcp`` SDK installed** — that is the whole point of splitting logic from the
protocol wrapper. Separate tests assert that importing ``app.mcp_server`` never
imports the ``mcp`` SDK and that ``cv-arxiv-mcp`` fails cleanly when the extra is
absent.
"""

from __future__ import annotations

import importlib
import sys
import unittest
from datetime import date, datetime, timezone
from unittest.mock import patch

from app.models import Collection, Paper, PaperCollection, PaperSection, db
from app.services import mcp_tools
from tests.helpers import FlaskDBTestCase


def _make_paper(idx: int = 0, **overrides) -> Paper:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today = date.today()
    defaults = dict(
        arxiv_id=f"2607.{3000 + idx:04d}",
        title=f"MCP Test Paper {idx}",
        authors="Ada Lovelace, Alan Turing",
        link=f"https://arxiv.org/abs/2607.{3000 + idx:04d}",
        pdf_link=f"https://arxiv.org/pdf/2607.{3000 + idx:04d}",
        abstract_text=f"An abstract about vision transformers number {idx}.",
        summary_text=f"TL;DR summary {idx}.",
        llm_insights={"tasks": ["detection"], "datasets": ["COCO"]},
        llm_relevance_score=8.0,
        topic_tags=["vision", "transformers"],
        categories=["cs.CV"],
        match_type="Title",
        matched_terms=["Vision"],
        paper_score=float(10 + idx),
        feedback_score=0,
        is_hidden=False,
        citation_count=idx * 5,
        github_repo="https://github.com/example/repo" if idx % 2 == 0 else None,
        publication_date=today.isoformat(),
        publication_dt=today,
        scraped_date=today.isoformat(),
        scraped_at=now,
    )
    defaults.update(overrides)
    return Paper(**defaults)


class SearchPapersTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        for i in range(3):
            db.session.add(_make_paper(i, title=f"Vision Transformer study {i}"))
        db.session.add(
            _make_paper(
                9,
                title="Completely unrelated topic on quantum chemistry",
                abstract_text="A study of molecular orbital energies.",
            )
        )
        db.session.commit()

    def test_keyword_search_returns_hits(self):
        result = mcp_tools.search_papers("Vision Transformer", mode="keyword", limit=10)
        self.assertEqual(result["mode"], "keyword")
        self.assertGreaterEqual(result["count"], 3)
        titles = [row["title"] for row in result["results"]]
        self.assertTrue(all("Vision" in t or "vision" in t.lower() for t in titles))
        # Compact brief shape.
        first = result["results"][0]
        for key in ("id", "arxiv_id", "title", "score", "abstract"):
            self.assertIn(key, first)

    def test_unknown_mode_falls_back_to_hybrid(self):
        result = mcp_tools.search_papers("Vision", mode="bogus", limit=5)
        self.assertEqual(result["mode"], "hybrid")

    def test_empty_query_returns_no_hits(self):
        result = mcp_tools.search_papers("   ", mode="keyword")
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["results"], [])

    def test_limit_is_respected(self):
        result = mcp_tools.search_papers("Vision", mode="keyword", limit=2)
        self.assertLessEqual(len(result["results"]), 2)

    def test_ranked_modes_skip_hidden_papers_like_keyword_mode(self):
        hidden = _make_paper(5, title="Vision Transformer skipped", is_hidden=True)
        db.session.add(hidden)
        db.session.commit()
        visible = Paper.query.filter_by(arxiv_id="2607.3000").one()
        hits = [(hidden.id, 1.0), (visible.id, 0.5)]

        with (
            patch("app.services.search.search_bm25", return_value=hits),
            patch("app.services.search.search_semantic", return_value=hits),
        ):
            for mode in ("hybrid", "semantic"):
                result = mcp_tools.search_papers("Vision", mode=mode)
                self.assertEqual([r["id"] for r in result["results"]], [visible.id], mode)


class GetPaperTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.paper = _make_paper(0, arxiv_id="2607.12345")
        db.session.add(self.paper)
        db.session.commit()
        self.paper_id = self.paper.id

    def test_get_paper_by_numeric_id(self):
        result = mcp_tools.get_paper(self.paper_id)
        self.assertEqual(result["id"], self.paper_id)
        self.assertEqual(result["arxiv_id"], "2607.12345")
        self.assertIn("enrichment", result)
        self.assertIn("readiness", result["enrichment"])

    def test_get_paper_by_numeric_string(self):
        result = mcp_tools.get_paper(str(self.paper_id))
        self.assertEqual(result["id"], self.paper_id)

    def test_get_paper_by_arxiv_id(self):
        result = mcp_tools.get_paper("2607.12345")
        self.assertEqual(result["id"], self.paper_id)

    def test_get_paper_by_arxiv_id_with_prefix_and_version(self):
        result = mcp_tools.get_paper("arXiv:2607.12345v2")
        self.assertEqual(result["id"], self.paper_id)

    def test_unknown_id_is_graceful(self):
        result = mcp_tools.get_paper("9999.99999")
        self.assertEqual(result["error"], "not_found")

    def test_unknown_numeric_id_is_graceful(self):
        result = mcp_tools.get_paper(424242)
        self.assertEqual(result["error"], "not_found")


class GetSummaryTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.paper = _make_paper(0)
        db.session.add(self.paper)
        db.session.commit()
        self.paper_id = self.paper.id

    def test_get_summary_returns_stored_tldr_and_insights(self):
        result = mcp_tools.get_summary(self.paper_id)
        self.assertEqual(result["id"], self.paper_id)
        self.assertIn("TL;DR", result["summary"])
        self.assertEqual(result["insights"]["tasks"], ["detection"])
        self.assertEqual(result["llm_relevance_score"], 8.0)

    def test_unknown_id_is_graceful(self):
        result = mcp_tools.get_summary(999999)
        self.assertEqual(result["error"], "not_found")


class TopRankedTodayTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        # Higher idx => higher paper_score, so ordering is deterministic.
        for i in range(5):
            db.session.add(_make_paper(i))
        # A hidden paper must never surface.
        db.session.add(_make_paper(8, is_hidden=True, paper_score=1000.0))
        db.session.commit()

    def test_returns_ranked_and_respects_limit(self):
        result = mcp_tools.top_ranked_today(limit=3)
        self.assertEqual(len(result["results"]), 3)
        scores = [row["score"] for row in result["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # Hidden paper excluded despite its huge score.
        self.assertTrue(all(row["score"] < 1000.0 for row in result["results"]))

    def test_reports_active_profile_by_default(self):
        result = mcp_tools.top_ranked_today(limit=2)
        self.assertTrue(result["profile"]["is_active"])
        self.assertEqual(result["profile"]["slug"], "default")

    def test_selects_profile_by_slug(self):
        from app.services.profiles import create_profile

        created = create_profile("My Focus")
        result = mcp_tools.top_ranked_today(limit=2, profile=created.slug)
        self.assertEqual(result["profile"]["slug"], created.slug)
        self.assertEqual(result["profile"]["name"], "My Focus")

    def test_selects_profile_by_id(self):
        from app.services.profiles import create_profile

        created = create_profile("By Id")
        result = mcp_tools.top_ranked_today(limit=2, profile=created.id)
        self.assertEqual(result["profile"]["id"], created.id)

    def test_unknown_profile_falls_back_to_active(self):
        result = mcp_tools.top_ranked_today(limit=2, profile="does-not-exist")
        self.assertTrue(result["profile"]["is_active"])


class ListCollectionsTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.paper = _make_paper(0)
        db.session.add(self.paper)
        empty = Collection(name="Empty")
        filled = Collection(name="Filled")
        db.session.add_all([empty, filled])
        db.session.commit()
        db.session.add(PaperCollection(paper_id=self.paper.id, collection_id=filled.id))
        db.session.commit()

    def test_lists_collections_with_paper_counts(self):
        result = mcp_tools.list_collections()
        by_name = {c["name"]: c for c in result["collections"]}
        self.assertEqual(result["count"], 2)
        self.assertEqual(by_name["Empty"]["paper_count"], 0)
        self.assertEqual(by_name["Filled"]["paper_count"], 1)


class AddToCollectionTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.paper = _make_paper(0)
        db.session.add(self.paper)
        self.collection = Collection(name="Reading List")
        db.session.add(self.collection)
        db.session.commit()
        self.paper_id = self.paper.id
        self.collection_id = self.collection.id

    def test_add_links_paper_to_existing_collection(self):
        result = mcp_tools.add_to_collection(self.collection_id, self.paper_id)
        self.assertTrue(result["added"])
        self.assertFalse(result["created_collection"])
        link = PaperCollection.query.filter_by(paper_id=self.paper_id, collection_id=self.collection_id).first()
        self.assertIsNotNone(link)

    def test_add_is_idempotent(self):
        first = mcp_tools.add_to_collection(self.collection_id, self.paper_id)
        second = mcp_tools.add_to_collection(self.collection_id, self.paper_id)
        self.assertTrue(first["added"])
        self.assertFalse(second["added"])
        count = PaperCollection.query.filter_by(paper_id=self.paper_id, collection_id=self.collection_id).count()
        self.assertEqual(count, 1)

    def test_add_by_collection_name_creates_it(self):
        result = mcp_tools.add_to_collection("Brand New Collection", self.paper_id)
        self.assertTrue(result["added"])
        self.assertTrue(result["created_collection"])
        self.assertIsNotNone(Collection.query.filter_by(name="Brand New Collection").first())

    def test_unknown_paper_is_graceful(self):
        result = mcp_tools.add_to_collection(self.collection_id, 999999)
        self.assertEqual(result["error"], "not_found")
        self.assertEqual(result["resource"], "paper")

    def test_unknown_numeric_collection_is_graceful(self):
        result = mcp_tools.add_to_collection(999999, self.paper_id)
        self.assertEqual(result["error"], "not_found")
        self.assertEqual(result["resource"], "collection")


class GetCollectionTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        papers = [_make_paper(i, user_notes="key baseline" if i == 2 else "") for i in range(3)]
        hidden = _make_paper(7, is_hidden=True)
        self.collection = Collection(name="Review")
        db.session.add_all([*papers, hidden, self.collection])
        db.session.commit()
        for paper in [*papers, hidden]:
            db.session.add(PaperCollection(paper_id=paper.id, collection_id=self.collection.id))
        db.session.commit()
        self.collection_id = self.collection.id

    def test_by_name_returns_members_with_cite_keys_and_notes(self):
        result = mcp_tools.get_collection("Review")
        self.assertEqual(result["id"], self.collection_id)
        self.assertEqual(result["count"], 3)  # hidden member excluded, like Export .bib
        by_arxiv = {row["arxiv_id"]: row for row in result["papers"]}
        self.assertEqual(by_arxiv["2607.3002"]["cite_key"], "2607_3002")
        self.assertEqual(by_arxiv["2607.3002"]["user_notes"], "key baseline")
        self.assertNotIn("2607.3007", by_arxiv)
        self.assertIsNone(result["next_offset"])

    def test_by_id_pages_with_next_offset(self):
        first = mcp_tools.get_collection(str(self.collection_id), limit=2)
        self.assertEqual(len(first["papers"]), 2)
        self.assertEqual(first["next_offset"], 2)
        rest = mcp_tools.get_collection(self.collection_id, offset=first["next_offset"], limit=2)
        self.assertEqual(len(rest["papers"]), 1)
        self.assertIsNone(rest["next_offset"])

    def test_unknown_name_is_not_created(self):
        result = mcp_tools.get_collection("Nope")
        self.assertEqual(result["error"], "not_found")
        self.assertIsNone(Collection.query.filter_by(name="Nope").first())


class GetPaperTextTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.paper = _make_paper(0)
        self.bare = _make_paper(1)
        db.session.add_all([self.paper, self.bare])
        db.session.commit()
        self.long_text = "".join(str(i % 10) for i in range(25_000))
        db.session.add_all(
            [
                PaperSection(paper_id=self.paper.id, section_type="introduction", text="Intro text.", order_index=0),
                PaperSection(paper_id=self.paper.id, section_type="method", text=self.long_text, order_index=1),
            ]
        )
        db.session.commit()

    def test_contents_lists_sections_and_abstract(self):
        result = mcp_tools.get_paper_text(self.paper.id)
        self.assertTrue(result["has_full_text"])
        self.assertIn("vision transformers", result["abstract"])
        self.assertEqual(
            result["sections"],
            [
                {"order_index": 0, "section_type": "introduction", "chars": 11},
                {"order_index": 1, "section_type": "method", "chars": 25_000},
            ],
        )

    def test_section_pages_verbatim_and_clamps(self):
        page = mcp_tools.get_paper_text(self.paper.id, order_index=1, offset=10, max_chars=100)
        self.assertEqual(page["text"], self.long_text[10:110])
        self.assertEqual(page["next_offset"], 110)
        big = mcp_tools.get_paper_text(self.paper.id, order_index=1, max_chars=999_999)
        self.assertEqual(len(big["text"]), 20_000)
        tail = mcp_tools.get_paper_text(self.paper.id, order_index=1, offset=big["next_offset"])
        self.assertEqual(tail["text"], self.long_text[20_000:])
        self.assertIsNone(tail["next_offset"])

    def test_no_sections_falls_back(self):
        result = mcp_tools.get_paper_text(self.bare.id)
        self.assertFalse(result["has_full_text"])
        self.assertEqual(result["sections"], [])
        self.assertEqual(result["link"], self.bare.link)

    def test_unknown_section_is_graceful(self):
        result = mcp_tools.get_paper_text(self.paper.id, order_index=9)
        self.assertEqual(result["resource"], "section")


class BuildServerTests(FlaskDBTestCase):
    def test_registers_read_tools_and_instructions(self):
        from app.mcp_server import build_server

        class FakeFastMCP:
            def __init__(self, name, **kwargs):
                self.kwargs, self.tools = kwargs, {}

            def tool(self, name, description):
                def register(fn):
                    self.tools[name] = fn
                    return fn

                return register

        server = build_server(self.app, FakeFastMCP)
        self.assertIn("get_paper_text", server.kwargs["instructions"])
        self.assertIn("get_collection", server.tools)
        self.assertEqual(server.tools["get_paper_text"](str(424242))["error"], "not_found")

    @unittest.skipUnless(importlib.util.find_spec("mcp"), "mcp extra not installed")
    def test_builds_with_the_installed_sdk(self):
        # Guards SDK API drift (mcp 2.x renamed FastMCP -> MCPServer).
        import asyncio

        from app.mcp_server import _load_fastmcp, build_server

        server = build_server(self.app, _load_fastmcp())
        names = {tool.name for tool in asyncio.run(server.list_tools())}
        self.assertTrue({"get_collection", "get_paper_text", "search_papers"} <= names)


class AskPaperTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.paper = _make_paper(0)
        db.session.add(self.paper)
        db.session.commit()
        self.paper_id = self.paper.id

    def test_ask_degrades_without_llm(self):
        # No LLM configured in the test config: answer is None, degraded True,
        # but the grounded sections are still returned.
        result = mcp_tools.ask_paper(self.paper_id, "What datasets are used?")
        self.assertIsNone(result["answer"])
        self.assertTrue(result["degraded"])
        self.assertIn("sections", result)

    def test_empty_question_is_graceful(self):
        result = mcp_tools.ask_paper(self.paper_id, "   ")
        self.assertEqual(result["error"], "empty_question")

    def test_unknown_paper_is_graceful(self):
        result = mcp_tools.ask_paper(999999, "anything?")
        self.assertEqual(result["error"], "not_found")


class McpServerImportHygieneTests(unittest.TestCase):
    """The protocol wrapper must stay installable without the optional extra."""

    def test_importing_mcp_server_does_not_import_mcp_sdk(self):
        # Drop any cached copies so the import is observed fresh.
        for name in list(sys.modules):
            if name == "mcp" or name.startswith("mcp."):
                del sys.modules[name]
        sys.modules.pop("app.mcp_server", None)

        importlib.import_module("app.mcp_server")

        leaked = [name for name in sys.modules if name == "mcp" or name.startswith("mcp.")]
        self.assertEqual(leaked, [], f"app.mcp_server imported the mcp SDK at module load: {leaked}")

    def test_load_fastmcp_raises_actionable_error_when_absent(self):
        import app.mcp_server as mcp_server

        # Simulate the extra being absent regardless of the real environment.
        blocked = {"mcp": None, "mcp.server": None, "mcp.server.fastmcp": None, "mcp.server.mcpserver": None}
        original = {name: sys.modules.get(name) for name in blocked}
        try:
            sys.modules.update(blocked)
            with self.assertRaises(ModuleNotFoundError) as ctx:
                mcp_server._load_fastmcp()
            self.assertIn("pip install", str(ctx.exception))
        finally:
            for name, value in original.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

    def test_cli_main_errors_cleanly_when_sdk_absent(self):
        from app.cli import mcp as mcp_cli

        blocked = {"mcp": None, "mcp.server": None, "mcp.server.fastmcp": None, "mcp.server.mcpserver": None}
        original = {name: sys.modules.get(name) for name in blocked}
        try:
            sys.modules.update(blocked)
            code = mcp_cli.main([])
            self.assertEqual(code, 1)
        finally:
            for name, value in original.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value


if __name__ == "__main__":
    unittest.main()
