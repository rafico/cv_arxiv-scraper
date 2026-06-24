from __future__ import annotations

import unittest
from unittest.mock import patch

from tests.helpers import FlaskDBTestCase


class HistoricalSearchCategoriesValidationTests(FlaskDBTestCase):
    """Regression tests for G10: POST /api/search/historical must validate the
    ``categories`` field before executing the scrape, so a wrong-typed value
    yields a clean 400 instead of a silent garbage query (string iterated
    char-by-char) or a misleading 502 (non-iterable raising TypeError)."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    @patch("app.services.scrape_engine.execute_historical_scrape")
    def test_g10_categories_string_returns_400(self, mock_exec):
        response = self.client.post(
            "/api/search/historical",
            json={
                "categories": "cs.CV",
                "start_date": "2026-01-01",
                "end_date": "2026-01-31",
            },
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        # A bare string must not silently coerce into a char-by-char query.
        self.assertEqual(response.status_code, 400)
        mock_exec.assert_not_called()

    @patch("app.services.scrape_engine.execute_historical_scrape")
    def test_g10_categories_non_iterable_returns_400_not_502(self, mock_exec):
        response = self.client.post(
            "/api/search/historical",
            json={
                "categories": 5,
                "start_date": "2026-01-01",
                "end_date": "2026-01-31",
            },
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 400)
        mock_exec.assert_not_called()

    @patch("app.services.scrape_engine.execute_historical_scrape")
    def test_g10_categories_list_with_non_string_returns_400(self, mock_exec):
        response = self.client.post(
            "/api/search/historical",
            json={
                "categories": ["cs.CV", 7],
                "start_date": "2026-01-01",
                "end_date": "2026-01-31",
            },
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 400)
        mock_exec.assert_not_called()

    @patch("app.services.scrape_engine.execute_historical_scrape")
    def test_g10_valid_categories_list_proceeds(self, mock_exec):
        mock_exec.return_value = {"new_papers": 0, "total_found": 0}
        response = self.client.post(
            "/api/search/historical",
            json={
                "categories": ["cs.CV", "cs.LG"],
                "start_date": "2026-01-01",
                "end_date": "2026-01-31",
            },
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        mock_exec.assert_called_once()
        # The validated category list reaches the engine unchanged.
        self.assertEqual(mock_exec.call_args.args[1], ["cs.CV", "cs.LG"])

    @patch("app.services.scrape_engine.execute_historical_scrape")
    def test_g10_missing_categories_defaults_to_cs_cv(self, mock_exec):
        mock_exec.return_value = {"new_papers": 0, "total_found": 0}
        response = self.client.post(
            "/api/search/historical",
            json={
                "start_date": "2026-01-01",
                "end_date": "2026-01-31",
            },
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        mock_exec.assert_called_once()
        self.assertEqual(mock_exec.call_args.args[1], ["cs.CV"])

    @patch("app.services.scrape_engine.execute_historical_scrape")
    def test_g10_empty_categories_list_normalizes_to_default(self, mock_exec):
        mock_exec.return_value = {"new_papers": 0, "total_found": 0}
        response = self.client.post(
            "/api/search/historical",
            json={
                "categories": [],
                "start_date": "2026-01-01",
                "end_date": "2026-01-31",
            },
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        mock_exec.assert_called_once()
        self.assertEqual(mock_exec.call_args.args[1], ["cs.CV"])


if __name__ == "__main__":
    unittest.main()
