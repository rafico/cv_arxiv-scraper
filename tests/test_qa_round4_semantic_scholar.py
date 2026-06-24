"""Regression tests for QA round 4 finding G2.

G2 (Semantic Scholar, silent data loss): SemanticScholarProvider.fetch_batch
sent ALL missing ids in a single POST to /paper/batch. The Semantic Scholar
batch endpoint caps at 500 ids and errors beyond that, and the broad except
returned only the cached payloads, so a >500 paper batch lost all citation data.

The fix chunks missing ids into slices of <=500 and maps each response item by
its position WITHIN the current chunk (``batch[idx]``), not the global
``missing_ids[idx]``.
"""

from __future__ import annotations

import unittest

from app.services.enrichment_providers.semantic_scholar import (
    SemanticScholarProvider,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class SemanticScholarChunkingTests(unittest.TestCase):
    """Run without an app context so get_cached_payloads treats all ids as missing."""

    def test_g2_batch_over_500_is_chunked_and_mapped_per_slice(self):
        ids = [f"2301.{i:05d}" for i in range(600)]

        posted_payloads: list[list[str]] = []

        def fake_request_fn(method, url, *, json, params, session, timeout):
            assert method == "POST"
            sent_ids = json["ids"]
            posted_payloads.append(sent_ids)
            # Echo a positional list: one item per id in this POST, with the
            # citationCount encoding the global index so we can verify mapping.
            data = []
            for raw in sent_ids:
                aid = raw.split("ARXIV:", 1)[1]
                global_idx = int(aid.split(".")[1])
                data.append(
                    {
                        "citationCount": global_idx,
                        "influentialCitationCount": global_idx * 2,
                        "paperId": f"PID-{global_idx}",
                    }
                )
            return _FakeResponse(data)

        provider = SemanticScholarProvider(request_fn=fake_request_fn)
        result = provider.fetch_batch(ids)

        # Two POSTs, neither exceeding the 500-id cap.
        self.assertEqual(len(posted_payloads), 2)
        for chunk in posted_payloads:
            self.assertLessEqual(len(chunk), 500)
        self.assertEqual(sum(len(c) for c in posted_payloads), 600)

        # All 600 ids enriched (no silent data loss).
        self.assertEqual(len(result), 600)

        # Per-id mapping is correct, ESPECIALLY for ids in the 2nd chunk
        # (index >= 500), which the pre-fix missing_ids[idx] logic mishandled.
        for global_idx in (0, 1, 499, 500, 501, 599):
            aid = f"2301.{global_idx:05d}"
            self.assertIn(aid, result)
            self.assertEqual(result[aid]["citation_count"], global_idx)
            self.assertEqual(result[aid]["influential_citation_count"], global_idx * 2)
            self.assertEqual(result[aid]["semantic_scholar_id"], f"PID-{global_idx}")

    def test_g2_one_failed_chunk_does_not_abandon_the_rest(self):
        ids = [f"2401.{i:05d}" for i in range(600)]

        call_count = {"n": 0}

        def fake_request_fn(method, url, *, json, params, session, timeout):
            call_count["n"] += 1
            # Fail the first chunk; the second must still be fetched.
            if call_count["n"] == 1:
                raise RuntimeError("413 Payload Too Large")
            data = []
            for raw in json["ids"]:
                aid = raw.split("ARXIV:", 1)[1]
                global_idx = int(aid.split(".")[1])
                data.append(
                    {
                        "citationCount": global_idx,
                        "influentialCitationCount": global_idx,
                        "paperId": f"PID-{global_idx}",
                    }
                )
            return _FakeResponse(data)

        provider = SemanticScholarProvider(request_fn=fake_request_fn)
        result = provider.fetch_batch(ids)

        self.assertEqual(call_count["n"], 2)
        # 2nd chunk (ids 500..599) survived despite the 1st chunk failing.
        self.assertIn("2401.00500", result)
        self.assertEqual(result["2401.00599"]["citation_count"], 599)


if __name__ == "__main__":
    unittest.main()
