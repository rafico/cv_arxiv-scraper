"""Pre-release QA round 3 — regression tests.

Each test below was verified to fail on the pre-fix code and pass on the fix.
Findings are labelled F1..F19 to match QA_FINDINGS.md. Grouped by subsystem; pure
units are pytest functions, anything needing the app/DB uses ``FlaskDBTestCase``.
"""

from __future__ import annotations

import copy
import json
import tempfile
import threading
import time
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import yaml

from app import create_app
from app.models import Paper, PaperFeedback, SavedSearch, db
from app.services.feedback import apply_feedback_action
from tests.helpers import TEST_SCRAPER_CONFIG, FlaskDBTestCase


# --------------------------------------------------------------------------- #
# F2 — venue qualifier must sit near the matched venue, like the workshop cue. #
# --------------------------------------------------------------------------- #
def test_distant_oral_qualifier_does_not_flip_status():
    from app.services.venues import parse_venue

    # "oral" belongs to the ICCV clause, not the CVPR acceptance.
    match = parse_venue("Accepted to CVPR 2024. Extended version of our ICCV 2024 oral paper.")
    assert match is not None
    assert match.venue == "CVPR"
    assert match.status == "accepted"


def test_distant_spotlight_qualifier_does_not_flip_status():
    from app.services.venues import parse_venue

    match = parse_venue("Accepted to CVPR 2026. Our earlier NeurIPS spotlight covered the basics.")
    assert match is not None
    assert match.status == "accepted"


def test_adjacent_oral_qualifier_still_honoured():
    from app.services.venues import parse_venue

    match = parse_venue("Accepted to CVPR 2024 (Oral).")
    assert match is not None
    assert match.status == "oral"


# --------------------------------------------------------------------------- #
# F13 — BibTeX escaper must escape ``$`` (math delimiter) like ^ and _.        #
# --------------------------------------------------------------------------- #
def test_bibtex_escapes_dollar_sign():
    from app.services.bibtex import _escape_latex

    out = _escape_latex("Cost is $5 and the bound is $O(n^2)$")
    # No bare dollar may survive once the escaped ones are removed.
    assert "$" not in out.replace(r"\$", "")
    assert r"\$" in out


# --------------------------------------------------------------------------- #
# F14 — DB ranking-config path must allow a 0 weight (disable a signal),       #
#       matching the preferences path; >1000 / NaN / inf stay rejected.        #
# --------------------------------------------------------------------------- #
def test_ranking_weight_zero_is_kept():
    from app.services.ranking import _normalize_ranking_weights

    assert _normalize_ranking_weights({"ai_weight": 0})["ai_weight"] == 0.0


def test_ranking_weight_out_of_range_still_rejected():
    from app.services.ranking import _normalize_ranking_weights

    assert "ai_weight" not in _normalize_ranking_weights({"ai_weight": 1001})
    assert "ai_weight" not in _normalize_ranking_weights({"ai_weight": float("inf")})
    assert "ai_weight" not in _normalize_ranking_weights({"ai_weight": float("nan")})


# --------------------------------------------------------------------------- #
# F12 — request_with_backoff must make >=1 attempt; attempts<=0 used to        #
#       ``raise None`` -> TypeError without ever issuing a request.            #
# --------------------------------------------------------------------------- #
@patch("app.services.http_client.requests.request")
def test_request_with_backoff_clamps_non_positive_attempts(mock_request):
    from app.services.http_client import request_with_backoff

    resp = Mock()
    resp.raise_for_status = Mock()
    mock_request.return_value = resp

    out = request_with_backoff("GET", "https://example.invalid/x", attempts=0)
    assert out is resp
    mock_request.assert_called_once()


# --------------------------------------------------------------------------- #
# F15 — hybrid RRF must break score ties deterministically (by paper_id),      #
#       since all_pids is a set and the top_k cut depends on order.            #
# --------------------------------------------------------------------------- #
def test_hybrid_rrf_tie_break_is_deterministic():
    from app.services import search

    # Equal weights + each paper at rank 1 in exactly one system => identical rrf.
    with (
        patch.object(search, "search_bm25", return_value=[(2, 9.0)]),
        patch.object(search, "search_semantic", return_value=[(1, 0.9)]),
    ):
        first = search.search_hybrid("q", top_k=10, bm25_weight=0.5, semantic_weight=0.5)
        second = search.search_hybrid("q", top_k=10, bm25_weight=0.5, semantic_weight=0.5)

    assert [r["paper_id"] for r in first] == [r["paper_id"] for r in second]
    # Tie resolved by ascending paper_id.
    assert [r["paper_id"] for r in first] == [1, 2]


# --------------------------------------------------------------------------- #
# F3 — embedding-based related papers that fall outside the candidate pool     #
#      must not shadow the in-pool TF-IDF fallback.                            #
# --------------------------------------------------------------------------- #
def test_related_falls_back_to_tfidf_when_neighbours_out_of_pool(monkeypatch):
    from app.services import related

    vectors = {
        1: related.build_vector("deep learning for image segmentation"),
        2: related.build_vector("deep learning image segmentation networks"),
        3: related.build_vector("quantum chromodynamics lattice gauge theory"),
    }
    # FAISS returns a globally-nearest neighbour (999) that is not renderable.
    monkeypatch.setattr(related, "top_related_papers_embedding", lambda pid, top_k=3: [999])

    result = related.top_related_papers(1, vectors, top_k=3)
    assert 999 not in result
    assert 2 in result  # the in-pool TF-IDF match is surfaced instead of nothing


def test_related_keeps_only_in_pool_embedding_neighbours(monkeypatch):
    from app.services import related

    vectors = {1: related.build_vector("a"), 2: related.build_vector("b"), 5: related.build_vector("c")}
    monkeypatch.setattr(related, "top_related_papers_embedding", lambda pid, top_k=3: [5, 999])

    assert related.top_related_papers(1, vectors, top_k=3) == [5]


# --------------------------------------------------------------------------- #
# F11 — the section FAISS index must not re-add an already-indexed paper.      #
# --------------------------------------------------------------------------- #
def _section_service(index_dir, dim=768):
    from app.services.embeddings import EmbeddingService

    service = EmbeddingService(index_dir)
    model = MagicMock()

    def fake_encode(texts, **kwargs):
        vecs = np.random.default_rng(3).random((len(texts), dim)).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    model.encode = fake_encode
    service._model = model
    return service


def test_add_sections_dedups_already_indexed_paper(tmp_path):
    from app.services.embeddings import reset_embedding_service

    reset_embedding_service()
    service = _section_service(str(tmp_path / "idx"))
    first = service.add_sections([(1, "intro", "alpha"), (1, "method", "beta")])
    second = service.add_sections([(1, "intro", "alpha"), (1, "method", "beta")])
    reset_embedding_service()

    assert first == 2
    assert second == 0  # same paper -> nothing re-added
    assert service._section_index.ntotal == 2


# --------------------------------------------------------------------------- #
# F7 — Mendeley add_document must not raise when a mid-batch 401 hits a token   #
#      that cannot be refreshed (RuntimeError), it should return a failure dict.#
# --------------------------------------------------------------------------- #
def test_mendeley_add_document_survives_unrefreshable_token(tmp_path):
    from app.services.mendeley import MendeleyClient

    creds = tmp_path / "mendeley_credentials.json"
    creds.write_text(json.dumps({"client_id": "i", "client_secret": "s"}), encoding="utf-8")
    token = tmp_path / ".mendeley_token"
    token.write_text(json.dumps({"access_token": "a"}), encoding="utf-8")  # no refresh_token

    client = MendeleyClient(credentials_path=creds, token_path=token)

    paper = Mock()
    paper.title = "T"
    paper.authors = "Alice Smith"
    paper.arxiv_id = "2601.00001"
    paper.link = "https://arxiv.org/abs/2601.00001"
    paper.pdf_link = "https://arxiv.org/pdf/2601.00001"
    paper.abstract_text = "abstract"
    paper.publication_dt = date(2026, 1, 1)

    with patch("app.services.mendeley.requests.post", return_value=Mock(status_code=401)):
        result = client.add_document(paper)

    assert result["success"] is False


def _make_paper(idx: int = 0, **overrides) -> Paper:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today = date.today()
    defaults = dict(
        arxiv_id=f"2608.{5000 + idx:04d}",
        title=f"Transformer Study {idx}",
        authors="Author A",
        link=f"https://arxiv.org/abs/2608.{5000 + idx:04d}",
        pdf_link=f"https://arxiv.org/pdf/2608.{5000 + idx:04d}",
        abstract_text="abstract about transformers",
        summary_text="summary",
        match_type="Title",
        matched_terms=["transformer"],
        paper_score=10.0,
        publication_date=today.isoformat(),
        publication_dt=today,
        scraped_date=today.isoformat(),
        scraped_at=now,
    )
    defaults.update(overrides)
    return Paper(**defaults)


class FeedbackApiValidationTests(FlaskDBTestCase):
    """F4 / F5 — feedback endpoints must 4xx (not 500) on wrong-typed input."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _hdr(self) -> dict:
        self.client.get("/")
        with self.client.session_transaction() as sess:
            return {"X-CSRF-Token": sess["settings_csrf_token"]}

    def _paper_id(self) -> int:
        paper = _make_paper()
        db.session.add(paper)
        db.session.commit()
        return paper.id

    def test_feedback_non_string_reason_returns_400(self):
        pid = self._paper_id()
        r = self.client.post(
            f"/api/papers/{pid}/feedback", json={"action": "save", "reason": ["x"]}, headers=self._hdr()
        )
        self.assertEqual(r.status_code, 400)

    def test_feedback_non_string_note_returns_400(self):
        pid = self._paper_id()
        r = self.client.post(
            f"/api/papers/{pid}/feedback", json={"action": "save", "note": {"x": 1}}, headers=self._hdr()
        )
        self.assertEqual(r.status_code, 400)

    def test_feedback_valid_string_reason_ok(self):
        pid = self._paper_id()
        r = self.client.post(
            f"/api/papers/{pid}/feedback", json={"action": "save", "reason": "interesting"}, headers=self._hdr()
        )
        self.assertEqual(r.status_code, 200)

    def test_bulk_feedback_non_scalar_ids_does_not_500(self):
        self._paper_id()
        r = self.client.post(
            "/api/papers/bulk-feedback", json={"action": "save", "paper_ids": [{"x": 1}]}, headers=self._hdr()
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["processed"], 0)


class SavedSearchLimitClampTests(FlaskDBTestCase):
    """F6 — a negative ``limit`` must not become an unlimited SQLite query."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _hdr(self) -> dict:
        self.client.get("/")
        with self.client.session_transaction() as sess:
            return {"X-CSRF-Token": sess["settings_csrf_token"]}

    def test_run_clamps_negative_limit(self):
        for i in range(3):
            db.session.add(_make_paper(i))
        search = SavedSearch(name="t", include_keywords=["transformer"])
        db.session.add(search)
        db.session.commit()

        r = self.client.post(f"/api/saved-searches/{search.id}/run", json={"limit": -1}, headers=self._hdr())
        self.assertEqual(r.status_code, 200)
        # Pre-fix LIMIT -1 returned all 3; the clamp now bounds it.
        self.assertLessEqual(r.get_json()["count"], 1)


class FeedbackInvariantTests(FlaskDBTestCase):
    """F9 / F18 — the priority-implies-save invariant and the private sentinel."""

    def _paper_id(self) -> int:
        paper = _make_paper()
        db.session.add(paper)
        db.session.commit()
        return paper.id

    def test_explicit_save_on_implied_save_promotes_not_deletes(self):
        pid = self._paper_id()
        apply_feedback_action(pid, "priority")  # creates priority + implied save
        result = apply_feedback_action(pid, "save")  # click save

        self.assertTrue(result["active"])
        actions = {row.action for row in PaperFeedback.query.filter_by(paper_id=pid).all()}
        self.assertIn("save", actions)
        self.assertIn("priority", actions)
        save_row = PaperFeedback.query.filter_by(paper_id=pid, action="save").first()
        self.assertIsNone(save_row.reason)  # marker cleared -> now an explicit save

    def test_client_reason_cannot_impersonate_internal_sentinel(self):
        pid = self._paper_id()
        apply_feedback_action(pid, "save", reason="implied_by_priority")

        save_row = PaperFeedback.query.filter_by(paper_id=pid, action="save").first()
        self.assertNotEqual(save_row.reason, "implied_by_priority")

        # The explicit save must survive a prioritise/un-prioritise cycle.
        apply_feedback_action(pid, "priority")
        apply_feedback_action(pid, "priority")  # toggle off
        self.assertIsNotNone(PaperFeedback.query.filter_by(paper_id=pid, action="save").first())


class SchedulerStartupConfigTests(unittest.TestCase):
    """F1 / F19 — a null/non-dict/"false" scheduler section must not crash or
    accidentally enable the scheduler at startup."""

    def _create_app(self, scheduler_value):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        cfg = copy.deepcopy(TEST_SCRAPER_CONFIG)
        cfg["scheduler"] = scheduler_value
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return create_app(
            {
                "TESTING": True,
                "SQLALCHEMY_DATABASE_URI": f"sqlite:///{root / 'test.db'}",
                "CONFIG_PATH": str(config_path),
                "SCRAPER_CONFIG": cfg,
                "LLM_KEY_PATH": str(root / ".llm_api_key"),
            }
        )

    @patch("app.services.scheduler.SCRAPE_SCHEDULER.start")
    def test_null_scheduler_section_does_not_crash(self, mock_start):
        self.assertIsNotNone(self._create_app(None))
        mock_start.assert_not_called()

    @patch("app.services.scheduler.SCRAPE_SCHEDULER.start")
    def test_non_dict_scheduler_section_does_not_crash(self, mock_start):
        self.assertIsNotNone(self._create_app(True))
        mock_start.assert_not_called()

    @patch("app.services.scheduler.SCRAPE_SCHEDULER.start")
    def test_string_false_disables_scheduler(self, mock_start):
        self._create_app({"enabled": "false"})
        mock_start.assert_not_called()

    @patch("app.services.scheduler.SCRAPE_SCHEDULER.start")
    def test_real_true_starts_scheduler(self, mock_start):
        self._create_app({"enabled": True})
        mock_start.assert_called_once()


class SchedulerSingleFlightTests(FlaskDBTestCase):
    """F10 — a scheduled scrape must go through the job manager's single-flight
    gate instead of calling execute_scrape directly (concurrent FAISS writers)."""

    def test_scheduled_run_uses_job_manager(self):
        from app.services import jobs
        from app.services.scheduler import ScrapeScheduler

        sched = ScrapeScheduler()
        sched._enabled = True
        sched._app = self.app

        with (
            patch.object(jobs.SCRAPE_JOB_MANAGER, "start_or_get_active") as mock_start,
            patch("app.services.scrape_engine.execute_scrape") as mock_exec,
            patch.object(sched, "_schedule_next"),
        ):
            sched._run()

        mock_start.assert_called_once_with(self.app)
        mock_exec.assert_not_called()


class ConfigWriteRaceTests(FlaskDBTestCase):
    """F8 — concurrent settings writes to different sections must not clobber
    each other (read-modify-write is now serialized)."""

    def test_concurrent_writes_preserve_both_sections(self):
        from app.routes import settings as settings_mod

        real_load = settings_mod._load_full_config

        def slow_load():
            cfg = real_load()
            time.sleep(0.05)  # widen the read->write window so the race is reliable
            return cfg

        errors: list[Exception] = []

        def writer(key, value):
            with self.app.app_context():
                try:
                    settings_mod._save_config_key(key, value)
                except Exception as exc:  # pragma: no cover - failure path
                    errors.append(exc)

        with patch.object(settings_mod, "_load_full_config", side_effect=slow_load):
            threads = [
                threading.Thread(target=writer, args=("qa3_alpha", 1)),
                threading.Thread(target=writer, args=("qa3_beta", 2)),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertEqual(errors, [])
        final = yaml.safe_load(Path(self.app.config["CONFIG_PATH"]).read_text(encoding="utf-8"))
        self.assertEqual(final.get("qa3_alpha"), 1)
        self.assertEqual(final.get("qa3_beta"), 2)
