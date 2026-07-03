"""Tests for the citation-verification layer (app.services.citation_verifier)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import patch

from sqlalchemy import event

# Import for side-effect: registers the chat routes on the shared blueprint.
import app.routes.api.chat  # noqa: F401
from app.models import Paper, db
from app.services import citation_verifier as cv
from tests.helpers import FlaskDBTestCase


def _make_paper(idx: int, **overrides) -> Paper:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today = date.today()
    defaults = dict(
        arxiv_id=f"2401.{1000 + idx:04d}",
        title=f"Corpus Paper {idx}",
        authors="Author A, Author B",
        link=f"https://arxiv.org/abs/2401.{1000 + idx:04d}",
        pdf_link=f"https://arxiv.org/pdf/2401.{1000 + idx:04d}",
        abstract_text=f"Abstract {idx}.",
        summary_text=f"Summary {idx}",
        topic_tags=["Segmentation"],
        categories=["cs.CV"],
        match_type="Title",
        matched_terms=["Vision"],
        paper_score=10.0,
        feedback_score=0,
        is_hidden=False,
        publication_date=today.isoformat(),
        publication_dt=today,
        scraped_date=today.isoformat(),
        scraped_at=now,
    )
    defaults.update(overrides)
    return Paper(**defaults)


class ExtractionTests(FlaskDBTestCase):
    def test_extracts_new_style_arxiv_id_with_and_without_version(self):
        refs = cv.extract_references("See arXiv:2401.01234 and also 2312.05678v3 for details.")
        arxiv = [r for r in refs if r["kind"] == "arxiv"]
        values = {r["value"] for r in arxiv}
        self.assertEqual(values, {"2401.01234", "2312.05678"})

    def test_extracts_old_style_arxiv_id_and_strips_subject_class(self):
        refs = cv.extract_references("The classic math.GT/0309136 and cond-mat/0311021 papers.")
        values = {r["value"] for r in refs if r["kind"] == "arxiv"}
        self.assertIn("math/0309136", values)
        self.assertIn("cond-mat/0311021", values)

    def test_extracts_doi(self):
        refs = cv.extract_references("Reported in 10.1109/CVPR.2016.90, a landmark result.")
        dois = [r for r in refs if r["kind"] == "doi"]
        self.assertEqual(len(dois), 1)
        self.assertEqual(dois[0]["value"], "10.1109/cvpr.2016.90")

    def test_arxiv_doi_resolves_to_arxiv_kind(self):
        refs = cv.extract_references("Preprint 10.48550/arXiv.2401.01234 is on arXiv.")
        self.assertEqual([r["kind"] for r in refs], ["arxiv"])
        self.assertEqual(refs[0]["value"], "2401.01234")

    def test_extracts_quoted_title(self):
        refs = cv.extract_references('The paper "Deep Residual Learning for Image Recognition" is seminal.')
        titles = [r for r in refs if r["kind"] == "title"]
        self.assertEqual(len(titles), 1)
        self.assertEqual(titles[0]["raw"], "Deep Residual Learning for Image Recognition")

    def test_extracts_title_case_run_of_four_plus_words(self):
        refs = cv.extract_references("We build on Deep Residual Learning for Image Recognition in this work.")
        titles = [r for r in refs if r["kind"] == "title"]
        self.assertTrue(titles)
        self.assertEqual(titles[0]["value"], "deep residual learning for image recognition")

    def test_ignores_ordinary_prose(self):
        prose = "We propose a new method that works well on several benchmarks and improves accuracy overall."
        refs = cv.extract_references(prose)
        self.assertEqual(refs, [])

    def test_deduplicates_repeated_reference(self):
        refs = cv.extract_references("arXiv:2401.01234 is great. Again, 2401.01234 is great.")
        self.assertEqual(len([r for r in refs if r["kind"] == "arxiv"]), 1)

    def test_empty_text_returns_empty(self):
        self.assertEqual(cv.extract_references(""), [])
        self.assertEqual(cv.extract_references(None), [])  # type: ignore[arg-type]


class ResolutionTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.p0 = _make_paper(0, arxiv_id="2401.01234", title="Deep Residual Learning for Image Recognition")
        self.p1 = _make_paper(1, arxiv_id="cond-mat/0311021", title="Attention Is All You Need")
        db.session.add_all([self.p0, self.p1])
        db.session.commit()

    def test_exact_arxiv_match_is_verified(self):
        refs = cv.extract_references("As shown in arXiv:2401.01234, ...")
        [v] = cv.verify_references(refs)
        self.assertEqual(v["status"], "verified")
        self.assertEqual(v["paper_id"], self.p0.id)
        self.assertEqual(v["matched_title"], "Deep Residual Learning for Image Recognition")

    def test_old_style_arxiv_match_is_verified(self):
        refs = cv.extract_references("See cond-mat/0311021 for background.")
        [v] = cv.verify_references(refs)
        self.assertEqual(v["status"], "verified")
        self.assertEqual(v["paper_id"], self.p1.id)

    def test_arxiv_doi_matches_local_paper(self):
        refs = cv.extract_references("Preprint 10.48550/arXiv.2401.01234.")
        [v] = cv.verify_references(refs)
        self.assertEqual(v["status"], "verified")
        self.assertEqual(v["paper_id"], self.p0.id)

    def test_exact_title_match_is_verified(self):
        refs = cv.extract_references('The paper "Attention Is All You Need" introduced transformers.')
        [v] = cv.verify_references(refs)
        self.assertEqual(v["status"], "verified")
        self.assertEqual(v["paper_id"], self.p1.id)

    def test_fuzzy_title_match_tolerates_punctuation_and_case(self):
        # Different casing/punctuation but > 0.9 similarity.
        refs = cv.extract_references('Cited "deep residual learning for image recognition." here.')
        [v] = cv.verify_references(refs)
        self.assertEqual(v["status"], "verified")
        self.assertEqual(v["paper_id"], self.p0.id)

    def test_made_up_arxiv_id_is_unverified(self):
        refs = cv.extract_references("The (nonexistent) arXiv:2199.99999 result.")
        [v] = cv.verify_references(refs)
        self.assertEqual(v["status"], "unverified")
        self.assertIsNone(v["paper_id"])

    def test_unrelated_title_is_unverified(self):
        refs = cv.extract_references('The imaginary "Quantum Banana Splitting Networks Everywhere" paper.')
        titles = [v for v in cv.verify_references(refs) if v["kind"] == "title"]
        self.assertTrue(titles)
        self.assertTrue(all(v["status"] == "unverified" for v in titles))

    def test_non_arxiv_doi_is_unverified(self):
        # No DOI column exists, so a real non-arXiv DOI is honestly unverified.
        refs = cv.extract_references("Published under 10.1109/CVPR.2016.90.")
        [v] = cv.verify_references(refs)
        self.assertEqual(v["kind"], "doi")
        self.assertEqual(v["status"], "unverified")

    def test_batched_lookups_no_n_plus_one(self):
        # Many references must not fan out into a query per reference.
        text = " ".join(f"arXiv:2401.0{i:04d}" for i in range(20))
        text += ' plus "Attention Is All You Need" and "Some Made Up Title Here".'
        refs = cv.extract_references(text)
        self.assertGreater(len(refs), 20)

        counter = {"n": 0}

        def _count(conn, cursor, statement, params, context, executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                counter["n"] += 1

        event.listen(db.engine, "before_cursor_execute", _count)
        try:
            verifications = cv.verify_references(refs)
        finally:
            event.remove(db.engine, "before_cursor_execute", _count)

        # One IN query for ids + one title scan = 2, regardless of ref count.
        self.assertLessEqual(counter["n"], 2)
        self.assertEqual(len(verifications), len(refs))

    def test_db_failure_degrades_to_empty(self):
        refs = cv.extract_references("arXiv:2401.01234")
        with patch.object(cv, "_resolve_arxiv", side_effect=RuntimeError("boom")):
            self.assertEqual(cv.verify_references(refs), [])


class AnnotateTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        db.session.add(_make_paper(0, arxiv_id="2401.01234", title="Deep Residual Learning for Image Recognition"))
        db.session.commit()

    def test_annotate_returns_structured_data_without_html(self):
        annotation = cv.verify_text('See arXiv:2401.01234 and the fake arXiv:2199.00000 in "Made Up Title Words".')
        self.assertEqual(annotation["total"], 3)
        self.assertEqual(annotation["verified_count"], 1)
        self.assertIn("references verified", annotation["summary_flag"])
        # The service must never inject HTML — only structured data.
        for verification in annotation["citations"]:
            self.assertNotIn("<", verification["ref"])
            self.assertIn(verification["status"], {"verified", "unverified"})

    def test_summary_flag_empty_when_no_references(self):
        annotation = cv.verify_text("Plain prose with no citations of any kind here.")
        self.assertEqual(annotation["total"], 0)
        self.assertEqual(annotation["summary_flag"], "")
        self.assertEqual(annotation["citations"], [])

    def test_verify_summary_text_alias(self):
        annotation = cv.verify_summary_text("Referenced in arXiv:2401.01234.")
        self.assertEqual(annotation["verified_count"], 1)


class PaperChatIntegrationTests(FlaskDBTestCase):
    def test_no_llm_returns_verifications_key_as_none(self):
        from app.services import paper_chat

        paper = _make_paper(0)
        db.session.add(paper)
        db.session.commit()
        # No LLM configured in the test config → graceful degradation.
        result = paper_chat.answer_paper_question(paper.id, "what is this about?")
        self.assertIn("verifications", result)
        self.assertIsNone(result["verifications"])

    def test_answer_gets_verifications_populated(self):
        from app.services import paper_chat

        target = _make_paper(0, arxiv_id="2401.09999", title="A Referenced Local Paper Title")
        paper = _make_paper(1)
        db.session.add_all([target, paper])
        db.session.commit()
        db.session.add(
            __import__("app.models", fromlist=["PaperSection"]).PaperSection(
                paper_id=paper.id, section_type="method", text="We use a magic decoder.", order_index=1
            )
        )
        db.session.commit()

        class FakeClient:
            def complete_text(self, *, system_prompt, user_prompt, max_tokens, temperature=0.2, **extra):
                if system_prompt == paper_chat._EVIDENCE_SYSTEM_PROMPT:
                    return '{"score": 9, "quote": "We use a magic decoder."}'
                # Synthesis names a local paper (verified) and a fake id (unverified).
                return "It builds on arXiv:2401.09999 and the bogus arXiv:2100.00000 [1]."

        with (
            patch("app.services.paper_chat._build_client", return_value=FakeClient()),
            patch("app.services.embeddings.get_embedding_service", side_effect=RuntimeError("no model")),
        ):
            result = paper_chat.answer_paper_question(paper.id, "what does it build on?")

        verifications = result["verifications"]
        self.assertIsNotNone(verifications)
        self.assertEqual(verifications["total"], 2)
        self.assertEqual(verifications["verified_count"], 1)
        statuses = {v["ref"]: v["status"] for v in verifications["citations"]}
        self.assertEqual(statuses["2401.09999"], "verified")
        self.assertEqual(statuses["2100.00000"], "unverified")


class RagIntegrationTests(FlaskDBTestCase):
    def test_synthesis_answer_carries_verifications(self):
        from types import SimpleNamespace

        from app.enums import FeedbackAction
        from app.models import PaperFeedback
        from app.services import rag

        target = _make_paper(0, arxiv_id="2401.07777", title="A Saved Reference Paper")
        db.session.add(target)
        db.session.commit()
        db.session.add(PaperFeedback(paper_id=target.id, action=FeedbackAction.SAVE.value))
        db.session.commit()

        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="Grounded in arXiv:2401.07777 and fake 2100.11111."))
            ]
        )
        fake_client = SimpleNamespace(complete=lambda **kwargs: fake_response)

        with (
            patch("app.services.rag.search_hybrid", return_value=[]),
            patch("app.services.rag._build_client", return_value=fake_client),
        ):
            result = rag.answer_query("what did I save?", app=self.app)

        self.assertIsNotNone(result["verifications"])
        self.assertEqual(result["verifications"]["verified_count"], 1)
        self.assertEqual(result["verifications"]["total"], 2)

    def test_no_synthesis_leaves_verifications_none(self):
        from app.enums import FeedbackAction
        from app.models import PaperFeedback
        from app.services import rag

        target = _make_paper(0)
        db.session.add(target)
        db.session.commit()
        db.session.add(PaperFeedback(paper_id=target.id, action=FeedbackAction.SAVE.value))
        db.session.commit()

        # LLM disabled in test config → synthesis None → verifications None.
        with patch("app.services.rag.search_hybrid", return_value=[]):
            result = rag.answer_query("anything", app=self.app)
        self.assertIsNone(result["synthesis"])
        self.assertIsNone(result["verifications"])
