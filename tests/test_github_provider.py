"""Tests for the GitHub repository metadata enrichment provider."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, Mock

import requests

from app.models import EnrichmentCache, Paper, db
from app.services.enrichment_providers import GitHubProvider, extract_github_repo
from tests.helpers import FlaskDBTestCase


def _paper(arxiv_id: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        authors="Author A",
        link=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_link=f"https://arxiv.org/pdf/{arxiv_id}",
        match_type="Title",
        matched_terms=["Vision"],
        paper_score=1.0,
        publication_date="2026-01-01",
        scraped_date="2026-01-01",
    )


def _repo_response(full_name: str, stars: int = 100) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {
        "full_name": full_name,
        "stargazers_count": stars,
        "license": {"spdx_id": "MIT", "name": "MIT License"},
        "archived": False,
        "pushed_at": "2026-06-01T00:00:00Z",
    }
    return response


def _http_error(status_code: int) -> requests.HTTPError:
    return requests.HTTPError(response=Mock(status_code=status_code))


class ExtractGithubRepoTests(unittest.TestCase):
    def test_parses_owner_repo_from_code_link(self):
        links = [{"type": "code", "label": "Code", "url": "https://github.com/lab/model/tree/main"}]
        self.assertEqual(extract_github_repo(links), "lab/model")

    def test_strips_git_suffix_and_accepts_www(self):
        links = [{"type": "code", "label": "Code", "url": "https://www.github.com/lab/model.git"}]
        self.assertEqual(extract_github_repo(links), "lab/model")

    def test_ignores_non_github_and_incomplete_urls(self):
        self.assertIsNone(extract_github_repo([{"url": "https://gitlab.com/lab/model"}]))
        self.assertIsNone(extract_github_repo([{"url": "https://github.com/lab"}]))
        self.assertIsNone(extract_github_repo(None))

    def test_rejects_dot_segments_that_traverse_to_other_endpoints(self):
        # `..` must not slip through: requests would normalize /repos/../user -> /user,
        # hitting unintended (token-authenticated) api.github.com endpoints.
        self.assertIsNone(extract_github_repo([{"url": "https://github.com/../user"}]))
        self.assertIsNone(extract_github_repo([{"url": "https://github.com/../settings"}]))
        self.assertIsNone(extract_github_repo([{"url": "https://github.com/lab/.."}]))
        # A single-dot owner/repo is likewise not a real repo.
        self.assertIsNone(extract_github_repo([{"url": "https://github.com/./model"}]))
        # Legitimate repos whose names merely *contain* dots still work.
        self.assertEqual(extract_github_repo([{"url": "https://github.com/lab/model.js"}]), "lab/model.js")


class GitHubProviderTests(FlaskDBTestCase):
    def test_fetch_caches_after_first_fetch(self):
        paper = _paper("2606.00001")
        db.session.add(paper)
        db.session.commit()

        request_fn = MagicMock(return_value=_repo_response("lab/model", stars=321))
        provider = GitHubProvider(request_fn=request_fn)
        repos = {"2606.00001": "lab/model"}

        first = provider.fetch_batch(["2606.00001"], repos_by_arxiv_id=repos)
        second = provider.fetch_batch(["2606.00001"], repos_by_arxiv_id=repos)

        self.assertEqual(first["2606.00001"]["github_stars"], 321)
        self.assertEqual(second["2606.00001"]["github_license"], "MIT")
        self.assertEqual(request_fn.call_count, 1)

        cache_row = EnrichmentCache.query.filter_by(paper_id=paper.id, source="github").one()
        self.assertEqual(cache_row.data["github_repo"], "lab/model")

    def test_per_run_fetch_cap_respected(self):
        db.session.add_all([_paper("2606.00002"), _paper("2606.00003")])
        db.session.commit()

        request_fn = MagicMock(return_value=_repo_response("lab/model"))
        provider = GitHubProvider(request_fn=request_fn, max_fetches=1)
        repos = {"2606.00002": "lab/a", "2606.00003": "lab/b"}

        payloads = provider.fetch_batch(["2606.00002", "2606.00003"], repos_by_arxiv_id=repos)

        self.assertEqual(request_fn.call_count, 1)
        self.assertEqual(len(payloads), 1)

    def test_rate_limit_aborts_remaining_fetches(self):
        db.session.add_all([_paper("2606.00004"), _paper("2606.00005")])
        db.session.commit()

        request_fn = MagicMock(side_effect=_http_error(403))
        provider = GitHubProvider(request_fn=request_fn)
        repos = {"2606.00004": "lab/a", "2606.00005": "lab/b"}

        payloads = provider.fetch_batch(["2606.00004", "2606.00005"], repos_by_arxiv_id=repos)

        self.assertEqual(request_fn.call_count, 1)
        self.assertEqual(payloads, {})
        # The flag lets the CLI backfill stop advancing its cursor past unfetched papers.
        self.assertTrue(provider.rate_limited)

    def test_non_json_response_skips_repo_without_aborting(self):
        db.session.add_all([_paper("2606.00040"), _paper("2606.00041")])
        db.session.commit()

        bad = MagicMock()
        bad.json.side_effect = ValueError("not json")
        good = _repo_response("lab/good", stars=12)
        request_fn = MagicMock(side_effect=[bad, good])
        provider = GitHubProvider(request_fn=request_fn)
        repos = {"2606.00040": "lab/bad", "2606.00041": "lab/good"}

        payloads = provider.fetch_batch(["2606.00040", "2606.00041"], repos_by_arxiv_id=repos)

        # The non-JSON 200 skips its repo; the next repo is still processed.
        self.assertNotIn("2606.00040", payloads)
        self.assertEqual(payloads["2606.00041"]["github_stars"], 12)
        self.assertFalse(provider.rate_limited)

    def test_not_found_is_cached_as_miss(self):
        paper = _paper("2606.00006")
        db.session.add(paper)
        db.session.commit()

        request_fn = MagicMock(side_effect=_http_error(404))
        provider = GitHubProvider(request_fn=request_fn)
        repos = {"2606.00006": "lab/gone"}

        first = provider.fetch_batch(["2606.00006"], repos_by_arxiv_id=repos)
        second = provider.fetch_batch(["2606.00006"], repos_by_arxiv_id=repos)

        self.assertIsNone(first["2606.00006"]["github_stars"])
        self.assertEqual(second["2606.00006"]["github_repo"], "lab/gone")
        self.assertEqual(request_fn.call_count, 1)

    def test_papers_without_repo_are_skipped(self):
        db.session.add(_paper("2606.00007"))
        db.session.commit()

        request_fn = MagicMock()
        provider = GitHubProvider(request_fn=request_fn)

        payloads = provider.fetch_batch(["2606.00007"], repos_by_arxiv_id={})

        self.assertEqual(payloads, {})
        request_fn.assert_not_called()

    def test_token_sent_as_bearer_header(self):
        db.session.add(_paper("2606.00008"))
        db.session.commit()

        request_fn = MagicMock(return_value=_repo_response("lab/model"))
        provider = GitHubProvider(request_fn=request_fn, token="tok-123")

        provider.fetch_batch(["2606.00008"], repos_by_arxiv_id={"2606.00008": "lab/model"})

        headers = request_fn.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer tok-123")


if __name__ == "__main__":
    unittest.main()
