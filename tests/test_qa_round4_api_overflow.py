"""QA round 4 — G8: oversized integer params must not 500.

Integer query/path params larger than SQLite's signed-64-bit range raise an
uncaught ``OverflowError`` (an ``ArithmeticError``) inside SQLAlchemy when used
in ``db.session.get(...)`` or an ``IN`` list. The api_bp safety-net handler only
caught TypeError/ValueError/KeyError/AttributeError, so these surfaced as opaque
500s. They must now be turned into a clean 400.
"""

from __future__ import annotations

from app.routes.api import api_bp
from tests.helpers import FlaskDBTestCase

# Just past SQLite's signed 64-bit integer range; SQLAlchemy raises OverflowError.
HUGE_INT = 2**63


class ApiOverflowTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_g8_overflow_handler_registered_on_api_bp(self):
        handlers = api_bp.error_handler_spec.get(None, {}).get(None, {})
        self.assertIn(OverflowError, handlers)

    def test_g8_neighbors_oversized_collection_id_returns_400(self):
        r = self.client.get(f"/api/corpus/neighbors?paper_ids=1&collection_id={HUGE_INT}")
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.get_json())

    def test_g8_bulk_bibtex_oversized_id_returns_400(self):
        r = self.client.get(f"/api/papers/bulk-bibtex?ids={HUGE_INT}")
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.get_json())
