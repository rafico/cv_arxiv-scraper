"""QA round 2 — schema migration & config validation robustness.

- A legacy DB with two versions of one paper (NULL arxiv_id) must not crash the
  arxiv_id backfill (and thus create_app) on the partial unique index.
- The scraped_at backfill must parse a real created_at instead of discarding it.
- A NaN ranking weight must be rejected by config validation (it slips past the
  positive/<=1000 bounds and corrupts every paper_score).
- Redundant ix_papers_* duplicate indexes are dropped on already-provisioned DBs.
"""

from __future__ import annotations

import math

from app import _validate_config
from app.models import Paper, db
from app.schema import _backfill_arxiv_ids, _try_parse_datetime
from tests.helpers import TEST_SCRAPER_CONFIG, FlaskDBTestCase


def _legacy_paper(link: str, **overrides) -> Paper:
    defaults = dict(
        arxiv_id=None,
        title="Legacy",
        authors="A",
        link=link,
        pdf_link=link.replace("/abs/", "/pdf/"),
        abstract_text="x",
        summary_text="y",
        match_type="Title",
        paper_score=1.0,
        scraped_date="2025-01-01",
    )
    defaults.update(overrides)
    return Paper(**defaults)


class ArxivIdBackfillTests(FlaskDBTestCase):
    def test_two_versions_of_same_paper_do_not_crash_backfill(self):
        # v1 and v2 strip to the same arxiv_id; the partial unique index would
        # reject the second. The backfill must dedupe instead of raising.
        db.session.add(_legacy_paper("https://arxiv.org/abs/2501.00001v1"))
        db.session.add(_legacy_paper("https://arxiv.org/abs/2501.00001v2"))
        db.session.commit()

        _backfill_arxiv_ids()  # must not raise IntegrityError

        arxiv_ids = [p.arxiv_id for p in Paper.query.order_by(Paper.id).all()]
        # Exactly one row owns the id; the other is left NULL (deduped, no crash).
        self.assertEqual(arxiv_ids.count("2501.00001"), 1)
        self.assertEqual(arxiv_ids.count(None), 1)


class ScrapedAtBackfillTests(FlaskDBTestCase):
    """The scraped_at backfill now parses created_at via _try_parse_datetime.

    The buggy branch (``isinstance(created_at, datetime)``) was always False because
    a raw text() SELECT of a SQLite DATETIME yields a str. The branch only fires on
    a legacy row whose scraped_at is NULL — impossible to insert under the current
    NOT NULL schema — so we guard the helper the fix now relies on directly.
    """

    def test_parses_iso_datetime_string(self):
        parsed = _try_parse_datetime("2025-01-15T10:00:00")
        self.assertIsNotNone(parsed)
        self.assertEqual((parsed.year, parsed.month, parsed.day), (2025, 1, 15))

    def test_parses_date_only_string(self):
        parsed = _try_parse_datetime("2025-01-15")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.year, 2025)

    def test_empty_value_is_none(self):
        self.assertIsNone(_try_parse_datetime(""))
        self.assertIsNone(_try_parse_datetime(None))


class RankingConfigValidationTests(FlaskDBTestCase):
    def _config_with_weight(self, value: float) -> dict:
        import copy

        config = copy.deepcopy(TEST_SCRAPER_CONFIG)
        config["preferences"]["ranking"]["author_weight"] = value
        return config

    def test_nan_ranking_weight_rejected(self):
        with self.assertRaises(ValueError):
            _validate_config(self._config_with_weight(float("nan")))

    def test_inf_ranking_weight_rejected(self):
        with self.assertRaises(ValueError):
            _validate_config(self._config_with_weight(float("inf")))

    def test_finite_ranking_weight_accepted(self):
        # Sanity: a normal value still validates.
        _validate_config(self._config_with_weight(42.0))
        self.assertTrue(math.isfinite(42.0))
