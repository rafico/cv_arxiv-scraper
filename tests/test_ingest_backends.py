from __future__ import annotations

import time
from datetime import date, datetime
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import requests

from app.services.arxiv_adapter import result_to_entry
from app.services.enrichment import parse_feed_entries, query_arxiv_api
from app.services.ingest import ArxivApiBackend, PaperCandidate, RssFeedBackend, arxiv_api_backend
from app.services.ingest.arxiv_api_backend import (
    ArxivRefused,
    _build_query,
    _oai_set,
    fetch_oai_records,
    list_oai_candidates,
    request_arxiv_api,
)
from app.services.ingest.base import clean_abstract, parse_publication_dt
from tests.helpers import OAI_RECORD_2609_12871, OAI_RECORD_2609_22706, oai_response

RSS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test feed</title>
    <item>
      <title>Vision Paper</title>
      <link>https://arxiv.org/abs/2604.00001v1</link>
      <author>Alice Example and Bob Example</author>
      <description><![CDATA[<p>Useful abstract.</p>]]></description>
      <pubDate>Wed, 01 Apr 2026 12:34:56 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

ARXIV_API_XML_PAGE_ONE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>https://arxiv.org/abs/2604.00002v1</id>
    <published>2026-04-01T08:00:00Z</published>
    <title>API Paper</title>
    <summary>Abstract from API</summary>
    <author><name>Carol Example</name></author>
    <category term="cs.CV" />
    <arxiv:comment>Project page: https://example.com/project</arxiv:comment>
    <arxiv:doi>10.48550/arXiv.2604.00002</arxiv:doi>
  </entry>
</feed>
"""

ARXIV_API_XML_RESUME_FIRST_PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2604.00003v1</id>
    <published>2026-04-01T08:00:00Z</published>
    <title>Paper 3</title>
    <summary>Abstract 3</summary>
    <author><name>Author A</name></author>
    <category term="cs.CV" />
  </entry>
  <entry>
    <id>https://arxiv.org/abs/2604.00004v1</id>
    <published>2026-04-01T08:00:00Z</published>
    <title>Paper 4</title>
    <summary>Abstract 4</summary>
    <author><name>Author A</name></author>
    <category term="cs.CV" />
  </entry>
</feed>
"""

ARXIV_API_XML_RESUME_SECOND_PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2604.00005v1</id>
    <published>2026-04-01T08:00:00Z</published>
    <title>Paper 5</title>
    <summary>Abstract 5</summary>
    <author><name>Author A</name></author>
    <category term="cs.CV" />
  </entry>
</feed>
"""


class IngestBaseHelperTests(TestCase):
    def test_parse_publication_dt_handles_rfc2822_rss_dates(self):
        self.assertEqual(
            parse_publication_dt("Wed, 01 Apr 2026 12:34:56 GMT"),
            (date(2026, 4, 1), "2026-04-01"),
        )

    def test_parse_publication_dt_handles_iso8601_api_dates(self):
        # The arXiv API (Atom) uses ISO-8601, which email.utils cannot parse.
        self.assertEqual(parse_publication_dt("2024-01-01T18:59:59Z"), (date(2024, 1, 1), "2024-01-01"))
        self.assertEqual(parse_publication_dt("2024-01-01T18:59:59-05:00"), (date(2024, 1, 1), "2024-01-01"))

    def test_parse_publication_dt_returns_unknown_for_garbage(self):
        self.assertEqual(parse_publication_dt("not a date"), (None, "Date Unknown"))
        self.assertEqual(parse_publication_dt(None), (None, "Date Unknown"))

    def test_clean_abstract_strips_arxiv_rss_announce_prefix(self):
        raw = "arXiv:2511.20302v3 Announce Type: replace Abstract: The real abstract body."
        self.assertEqual(clean_abstract(raw), "The real abstract body.")
        for announce in ("new", "cross", "replace-cross"):
            self.assertEqual(
                clean_abstract(f"arXiv:2604.00001v1 Announce Type: {announce} Abstract: Body text."),
                "Body text.",
            )

    def test_clean_abstract_leaves_normal_abstract_and_strips_html(self):
        self.assertEqual(clean_abstract("<p>Plain abstract.</p>"), "Plain abstract.")
        self.assertEqual(clean_abstract("A study of arXiv usage patterns."), "A study of arXiv usage patterns.")


class PaperCandidateTests(TestCase):
    def test_round_trips_through_legacy_entry_dict(self):
        candidate = PaperCandidate(
            arxiv_id="2604.00001",
            link="https://arxiv.org/abs/2604.00001",
            title="Vision Paper",
            author="Alice Example",
            authors_list=["Alice Example"],
            abstract="Useful abstract.",
            publication_date="2026-04-01",
            categories=["cs.CV"],
        )

        restored = PaperCandidate.from_entry_dict(candidate.to_entry_dict())

        self.assertEqual(restored, candidate)


class RssFeedBackendTests(TestCase):
    @patch("app.services.ingest.rss_backend.request_with_backoff")
    def test_fetch_parses_feed_entries_into_candidates(self, mock_request):
        mock_request.return_value = Mock(content=RSS_XML)

        backend = RssFeedBackend(["https://rss.arxiv.org/rss/cs.CV"])
        candidates = backend.fetch()

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].arxiv_id, "2604.00001")
        self.assertEqual(candidates[0].title, "Vision Paper")
        self.assertEqual(candidates[0].authors_list, ["Alice Example and Bob Example"])
        self.assertEqual(candidates[0].publication_date, "2026-04-01")

    @patch("app.services.ingest.rss_backend.RssFeedBackend.fetch")
    def test_parse_feed_entries_preserves_legacy_dict_shape(self, mock_fetch):
        mock_fetch.return_value = [
            PaperCandidate(
                arxiv_id="2604.00001",
                link="https://arxiv.org/abs/2604.00001",
                title="Vision Paper",
                author="Alice Example",
                authors_list=["Alice Example"],
            )
        ]

        entries = parse_feed_entries("https://rss.arxiv.org/rss/cs.CV")

        self.assertEqual(entries[0]["title"], "Vision Paper")
        self.assertEqual(entries[0]["authors_list"], ["Alice Example"])
        self.assertIn("resource_links", entries[0])


class ArxivApiBackendTests(TestCase):
    @patch("app.services.ingest.arxiv_api_backend.request_with_backoff")
    def test_fetch_builds_submitted_date_query_and_returns_candidates(self, mock_request):
        mock_request.return_value = Mock(text=ARXIV_API_XML_PAGE_ONE)
        backend = ArxivApiBackend()
        candidates = backend.fetch(
            categories=["cs.CV", "cs.LG"],
            start_dt=date(2026, 4, 1),
            end_dt=date(2026, 4, 2),
            max_results=25,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].arxiv_id, "2604.00002")
        self.assertEqual(candidates[0].doi, "10.48550/arXiv.2604.00002")
        # The API returns ISO-8601 <published>; it must still yield a real date.
        self.assertEqual(candidates[0].publication_dt, date(2026, 4, 1))
        self.assertEqual(candidates[0].publication_date, "2026-04-01")
        self.assertEqual(
            mock_request.call_args.kwargs["params"]["search_query"],
            "(cat:cs.CV OR cat:cs.LG) AND submittedDate:[202604010000 TO 202604022359]",
        )
        self.assertEqual(mock_request.call_args.kwargs["params"]["max_results"], 25)
        self.assertEqual(mock_request.call_args.kwargs["params"]["start"], 0)

    def test_build_query_ands_optional_categories_and_topic_query(self):
        start, end = date(2019, 1, 1), date(2019, 1, 2)
        dates = "submittedDate:[201901010000 TO 201901022359]"
        self.assertEqual(_build_query(["cs.CV"], start, end), f"(cat:cs.CV) AND {dates}")
        self.assertEqual(_build_query([], start, end, 'abs:"ovs"'), f'(abs:"ovs") AND {dates}')
        self.assertEqual(
            _build_query(["cs.CV", "cs.LG"], start, end, "ti:seg OR abs:seg"),
            f"(cat:cs.CV OR cat:cs.LG) AND (ti:seg OR abs:seg) AND {dates}",
        )

    @patch("app.services.ingest.arxiv_api_backend.request_with_backoff")
    def test_fetch_runs_a_query_without_categories(self, mock_request):
        mock_request.return_value = Mock(text=ARXIV_API_XML_PAGE_ONE)
        window = {"start_dt": date(2019, 1, 1), "end_dt": date(2026, 4, 2)}

        self.assertEqual(ArxivApiBackend().fetch(categories=[], **window), [])  # nothing to search
        candidates = ArxivApiBackend().fetch(categories=[], query='abs:"ovs"', **window)

        self.assertEqual([c.arxiv_id for c in candidates], ["2604.00002"])
        mock_request.assert_called_once()
        self.assertTrue(mock_request.call_args.kwargs["params"]["search_query"].startswith('(abs:"ovs") AND '))

    @patch("app.services.ingest.arxiv_api_backend.request_with_backoff", side_effect=RuntimeError("network down"))
    def test_fetch_propagates_request_errors(self, mock_request):
        backend = ArxivApiBackend()

        with self.assertRaisesRegex(RuntimeError, "network down"):
            backend.fetch(
                categories=["cs.CV"],
                start_dt=date(2026, 4, 1),
                end_dt=date(2026, 4, 2),
                max_results=25,
            )

        mock_request.assert_called_once()

    @patch("app.services.ingest.arxiv_api_backend.ArxivApiBackend.fetch")
    def test_query_arxiv_api_preserves_legacy_dict_shape(self, mock_fetch):
        mock_fetch.return_value = [
            PaperCandidate(
                arxiv_id="2604.00002",
                link="https://arxiv.org/abs/2604.00002",
                title="API Paper",
                author="Carol Example",
                authors_list=["Carol Example"],
                comment="Demo comment",
                doi="10.48550/arXiv.2604.00002",
            )
        ]

        entries = query_arxiv_api(["cs.CV"], date(2026, 4, 1), date(2026, 4, 2), max_results=25)

        self.assertEqual(entries[0]["title"], "API Paper")
        self.assertEqual(entries[0]["doi"], "10.48550/arXiv.2604.00002")
        self.assertIn("categories", entries[0])

    @patch("app.services.ingest.arxiv_api_backend.request_with_backoff")
    def test_fetch_resumes_from_offset_and_skips_processed_cursor(self, mock_request):
        mock_request.side_effect = [
            Mock(text=ARXIV_API_XML_RESUME_FIRST_PAGE),
            Mock(text=ARXIV_API_XML_RESUME_SECOND_PAGE),
        ]
        progress: list[tuple[int, str | None]] = []

        backend = ArxivApiBackend(page_size=2)
        candidates = backend.fetch(
            categories=["cs.CV"],
            start_dt=date(2026, 4, 1),
            end_dt=date(2026, 4, 2),
            max_results=25,
            offset=2,
            resume_after_arxiv_id="2604.00004",
            progress_callback=lambda page_number, candidate: progress.append((page_number, candidate.arxiv_id)),
        )

        self.assertEqual([candidate.arxiv_id for candidate in candidates], ["2604.00005"])
        self.assertEqual(progress, [(3, "2604.00005")])
        self.assertEqual(mock_request.call_args_list[0].kwargs["params"]["start"], 2)
        self.assertEqual(mock_request.call_args_list[1].kwargs["params"]["start"], 4)

    @patch("app.services.ingest.arxiv_api_backend.request_with_backoff")
    def test_fetch_does_not_drop_page_when_cursor_absent(self, mock_request):
        # The saved cursor was pushed off its page between runs (new submissions /
        # withdrawals). The resumed page must still be returned, not silently
        # skipped while waiting to re-find a cursor that is no longer there.
        mock_request.side_effect = [
            Mock(text=ARXIV_API_XML_RESUME_FIRST_PAGE),
            Mock(text=ARXIV_API_XML_RESUME_SECOND_PAGE),
        ]

        backend = ArxivApiBackend(page_size=2)
        candidates = backend.fetch(
            categories=["cs.CV"],
            start_dt=date(2026, 4, 1),
            end_dt=date(2026, 4, 2),
            max_results=25,
            offset=2,
            resume_after_arxiv_id="2604.09999",  # not present on the resumed page
        )

        self.assertEqual(
            [candidate.arxiv_id for candidate in candidates],
            ["2604.00003", "2604.00004", "2604.00005"],
        )


class ArxivAdapterTests(TestCase):
    def test_result_to_entry_uses_candidate_shape(self):
        result = SimpleNamespace(
            entry_id="https://arxiv.org/abs/2604.00003v2",
            title="Adapter Paper",
            authors=[SimpleNamespace(name="Dana Example")],
            published=datetime(2026, 4, 3, 10, 0, 0),
            summary="Adapter abstract",
            categories=["cs.CV"],
            comment="Code: https://example.com/code",
            doi="10.48550/arXiv.2604.00003",
        )

        entry = result_to_entry(result)

        self.assertEqual(entry["arxiv_id"], "2604.00003")
        self.assertEqual(entry["author"], "Dana Example")
        self.assertEqual(entry["categories"], ["cs.CV"])


# More live arXivRaw records (abstracts trimmed) that a cs.CV harvest from 2026-09-11 returns: an old
# paper revised since (2303.15533), a 2508 paper whose v2 landed in the window (2508.11450), and a
# paper submitted in the window and revised after it (2609.12825). 1706.03762's seven versions show
# the v1 date wins (the arXiv format's <created> says 2023); 2609.28194 has numbered affiliations.
OAI_RECORD_2303_15533 = """<record>
                <header>
        <identifier>oai:arXiv.org:2303.15533</identifier>
        <datestamp>2026-09-24</datestamp>
            <setSpec>cs:cs:LG</setSpec>
            <setSpec>cs:cs:CV</setSpec>
    </header>
            <metadata>
                        <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <id>2303.15533</id>
        <submitter>Arkanath Pathak</submitter>
            <version version="v1">
                <date>Mon, 27 Mar 2023 18:18:15 GMT</date>
                <size>3279kb</size>
                    <source_type>D</source_type>
            </version>
        <title>Sequential training of GANs against GAN-classifiers reveals correlated &#34;knowledge gaps&#34; present among independently trained GAN instances</title>
        <authors>Arkanath Pathak, Nicholas Dufour</authors>
        <categories>cs.LG cs.CV</categories>
            <journal-ref>2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</journal-ref>
            <doi>10.1109/CVPR52729.2023.02343</doi>
            <license>http://creativecommons.org/licenses/by/4.0/</license>
            <abstract>Modern Generative Adversarial Networks (GANs) generate realistic images remarkably well.</abstract>
    </arXivRaw>
            </metadata>
        </record>"""

OAI_RECORD_2508_11450 = """<record>
                <header>
        <identifier>oai:arXiv.org:2508.11450</identifier>
        <datestamp>2026-09-14</datestamp>
            <setSpec>eess:eess:IV</setSpec>
            <setSpec>cs:cs:CV</setSpec>
    </header>
            <metadata>
                        <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <id>2508.11450</id>
        <submitter>Augustine Lee</submitter>
            <version version="v1">
                <date>Fri, 15 Aug 2025 12:57:35 GMT</date>
                <size>857kb</size>
                    <source_type>A</source_type>
            </version>
            <version version="v2">
                <date>Fri, 11 Sep 2026 06:31:45 GMT</date>
                <size>1312kb</size>
            </version>
        <title>Subcortical Masks Generation in CT Images via Ensemble-Based Cross-Domain Label Transfer</title>
        <authors>Augustine X. W. Lee, Pak-Hei Yeung and Jagath C. Rajapakse</authors>
        <categories>eess.IV cs.CV</categories>
            <comments>Accepted by Annual Conference on Medical Image Understanding and Analysis (MIUA) 2025 (Oral)</comments>
            <doi>10.1007/978-3-031-98694-9_12</doi>
            <license>http://creativecommons.org/licenses/by-nc-sa/4.0/</license>
            <abstract>Subcortical segmentation in neuroimages plays an important role in understanding brain anatomy and facilitating computer-aided diagnosis of traumatic brain injuries and neurodegenerative disorders.</abstract>
    </arXivRaw>
            </metadata>
        </record>"""

OAI_RECORD_2609_12825 = r"""<record>
                <header>
        <identifier>oai:arXiv.org:2609.12825</identifier>
        <datestamp>2026-09-24</datestamp>
            <setSpec>cs:cs:CV</setSpec>
    </header>
            <metadata>
                        <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <id>2609.12825</id>
        <submitter>Mustafa Bora Celik</submitter>
            <version version="v1">
                <date>Fri, 11 Sep 2026 13:21:59 GMT</date>
                <size>3041kb</size>
            </version>
            <version version="v2">
                <date>Wed, 23 Sep 2026 11:01:12 GMT</date>
                <size>3041kb</size>
            </version>
        <title>SCDM: Spatial-Contextual Disentanglement Mamba via Differential Inference for Efficient Image Classification</title>
        <authors>Mustafa Bora \c{C}elik, Hayriye Akta\c{s} Din\c{c}er, Ayse Keles</authors>
        <categories>cs.CV</categories>
            <comments>9 pages, 5 figures</comments>
            <license>http://creativecommons.org/licenses/by/4.0/</license>
            <abstract>State Space Models (SSMs), particularly VMamba, have emerged as efficient alternatives for modeling long-range dependencies in medical image analysis.</abstract>
    </arXivRaw>
            </metadata>
        </record>"""

OAI_RECORD_1706_03762 = """<record>
                <header>
        <identifier>oai:arXiv.org:1706.03762</identifier>
        <datestamp>2023-08-03</datestamp>
            <setSpec>cs:cs:CL</setSpec>
            <setSpec>cs:cs:LG</setSpec>
    </header>
            <metadata>
                        <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <id>1706.03762</id>
        <submitter>Llion Jones</submitter>
            <version version="v1">
                <date>Mon, 12 Jun 2017 17:57:34 GMT</date>
                <size>1102kb</size>
                    <source_type>D</source_type>
            </version>
            <version version="v2">
                <date>Mon, 19 Jun 2017 16:49:45 GMT</date>
                <size>1124kb</size>
                    <source_type>D</source_type>
            </version>
            <version version="v3">
                <date>Tue, 20 Jun 2017 05:20:02 GMT</date>
                <size>1124kb</size>
                    <source_type>D</source_type>
            </version>
            <version version="v4">
                <date>Fri, 30 Jun 2017 17:29:30 GMT</date>
                <size>1124kb</size>
                    <source_type>D</source_type>
            </version>
            <version version="v5">
                <date>Wed, 06 Dec 2017 03:30:32 GMT</date>
                <size>1123kb</size>
                    <source_type>D</source_type>
            </version>
            <version version="v6">
                <date>Mon, 24 Jul 2023 00:48:54 GMT</date>
                <size>1124kb</size>
                    <source_type>D</source_type>
            </version>
            <version version="v7">
                <date>Wed, 02 Aug 2023 00:41:18 GMT</date>
                <size>1124kb</size>
                    <source_type>D</source_type>
            </version>
        <title>Attention Is All You Need</title>
        <authors>Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin</authors>
        <categories>cs.CL cs.LG</categories>
            <comments>15 pages, 5 figures</comments>
            <license>http://arxiv.org/licenses/nonexclusive-distrib/1.0/</license>
            <abstract>The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration.</abstract>
    </arXivRaw>
            </metadata>
        </record>"""

OAI_RECORD_2609_28194 = """<record>
                <header>
        <identifier>oai:arXiv.org:2609.28194</identifier>
        <datestamp>2026-09-24</datestamp>
            <setSpec>cs:cs:LG</setSpec>
            <setSpec>cs:cs:CV</setSpec>
            <setSpec>eess:eess:IV</setSpec>
    </header>
            <metadata>
                        <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <id>2609.28194</id>
        <submitter>Thomas Ratsakatika</submitter>
            <version version="v1">
                <date>Wed, 23 Sep 2026 14:33:19 GMT</date>
                <size>4057kb</size>
            </version>
        <title>Geospatial embeddings detect old-growth forests but buffered spatial validation narrows their advantage over Sentinel features</title>
        <authors>Thomas Ratsakatika (1), Mihai Zotta (2), Srinivasan Keshav (3), Emily R. Lines (1) ((1) Department of Geography, University of Cambridge, Cambridge, UK, (2) Fundatia Conservation Carpathia, Brasov, Romania, (3) Department of Computer Science and Technology, University of Cambridge, Cambridge, UK)</authors>
        <categories>cs.LG cs.CV eess.IV</categories>
            <comments>34 pages, including supplementary material (19-page main article with 7 figures and 3 tables; 15-page supplement with 9 figures and 21 tables). Submitted for publication. Data: https://doi.org/10.5281/zenodo.22693148 (embargoed until publication); code: https://github.com/ratsakatika/detecting-old-growth-forests</comments>
            <license>http://creativecommons.org/licenses/by/4.0/</license>
            <abstract>Old-growth forests develop over centuries under minimal anthropogenic disturbance, producing structurally complex and biodiverse stands.</abstract>
    </arXivRaw>
            </metadata>
        </record>"""


def _http_error(status: int) -> requests.HTTPError:
    return requests.HTTPError(response=Mock(status_code=status))


@patch.object(arxiv_api_backend, "_refused", (0.0, 0))
class ArxivRefusalTests(TestCase):
    @patch("app.services.ingest.arxiv_api_backend.request_with_backoff")
    def test_refusal_is_remembered_and_short_circuits_later_calls(self, mock_request):
        mock_request.side_effect = [_http_error(500), _http_error(406)]

        with self.assertRaises(requests.HTTPError) as caught:
            request_arxiv_api({"id_list": "2609.22706"})
        self.assertNotIsInstance(caught.exception, ArxivRefused)  # a 5xx is not a refusal
        for _ in range(2):
            with self.assertRaises(ArxivRefused) as refused:
                request_arxiv_api({"id_list": "2609.22706"})
            self.assertEqual(refused.exception.status, 406)

        self.assertEqual(mock_request.call_count, 2)  # the remembered refusal never reached arXiv

    @patch("app.services.ingest.arxiv_api_backend.request_with_backoff", side_effect=_http_error(429))
    def test_429_that_outlives_the_retries_is_a_refusal(self, mock_request):
        # Otherwise a 429 would fall into _fetch_api_metadata_batch's halving recursion.
        for _ in range(2):
            with self.assertRaises(ArxivRefused) as refused:
                request_arxiv_api({"id_list": "2609.22706"})
            self.assertEqual(refused.exception.status, 429)
        self.assertEqual(mock_request.call_count, 1)

    @patch("app.services.ingest.arxiv_api_backend.request_with_backoff")
    def test_fetch_oai_records_parses_the_live_arxivraw_shape(self, mock_request):
        mock_request.side_effect = [
            Mock(content=oai_response("GetRecord", OAI_RECORD_1706_03762)),
            Mock(
                content=b'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><error code="idDoesNotExist"/></OAI-PMH>'
            ),
            Mock(content=oai_response("GetRecord", OAI_RECORD_2609_28194)),
        ]

        records, deferred = fetch_oai_records(["1706.03762", "2609.99999", "2609.28194"])

        self.assertEqual((list(records), deferred), (["1706.03762", "2609.28194"], []))  # unknown id absent
        attention = records["1706.03762"]
        self.assertEqual(
            (attention.title, attention.authors_list[:2], len(attention.authors_list), attention.publication_dt),
            ("Attention Is All You Need", ["Ashish Vaswani", "Noam Shazeer"], 8, date(2017, 6, 12)),  # v1, not v7
        )
        self.assertEqual(
            (attention.categories, attention.comment, attention.doi), (["cs.CL", "cs.LG"], "15 pages, 5 figures", "")
        )
        forests = records["2609.28194"]
        self.assertEqual(
            forests.authors_list, ["Thomas Ratsakatika", "Mihai Zotta", "Srinivasan Keshav", "Emily R. Lines"]
        )
        self.assertTrue(forests.api_affiliations.startswith("(1) Department of Geography, University of Cambridge"))
        self.assertIn("https://github.com/ratsakatika/detecting-old-growth-forests", forests.comment)
        self.assertTrue(forests.has_api_metadata)
        self.assertEqual(
            mock_request.call_args_list[0].kwargs["params"],
            {"verb": "GetRecord", "identifier": "oai:arXiv.org:1706.03762", "metadataPrefix": "arXivRaw"},
        )
        # arXiv's 1 request / 3 s (the limiter the export API shares), full retries without a deadline.
        self.assertEqual({c.kwargs["rate_limit_profile"] for c in mock_request.call_args_list}, {"bulk"})
        self.assertEqual({c.kwargs["attempts"] for c in mock_request.call_args_list}, {4})

    @patch("app.services.ingest.arxiv_api_backend.request_with_backoff")
    def test_fetch_oai_records_defers_what_the_deadline_or_a_failure_cuts_off(self, mock_request):
        ids = ["2609.22706", "2609.12871", "2609.28194"]
        self.assertEqual(fetch_oai_records(ids, deadline=time.monotonic() - 1), ({}, ids))
        mock_request.assert_not_called()

        mock_request.side_effect = [Mock(content=oai_response("GetRecord", OAI_RECORD_2609_22706)), _http_error(503)]
        records, deferred = fetch_oai_records(ids, deadline=time.monotonic() + 60)

        self.assertEqual((list(records), deferred), (["2609.22706"], ["2609.12871", "2609.28194"]))
        self.assertEqual({c.kwargs["attempts"] for c in mock_request.call_args_list}, {1})  # interactive: no retries

    @patch("app.services.ingest.arxiv_api_backend.utc_today", return_value=date(2026, 9, 28))
    @patch("app.services.ingest.arxiv_api_backend.request_with_backoff")
    def test_refused_listing_falls_back_to_an_oai_window(self, mock_request, _today):
        mock_request.side_effect = [
            _http_error(406),
            Mock(content=oai_response("ListRecords", OAI_RECORD_2303_15533, OAI_RECORD_2508_11450, token="tok1")),
            Mock(
                content=oai_response(
                    "ListRecords", OAI_RECORD_2609_22706, OAI_RECORD_2609_12871, OAI_RECORD_2609_12825, token=""
                )
            ),
        ]
        window = {"start_dt": date(2026, 9, 11), "end_dt": date(2026, 9, 18)}

        candidates = ArxivApiBackend().fetch(categories=["cs.CV"], max_results=25, user_agent="MyApp/9.9", **window)

        # By v1 date: 2303.15533 is from 2023 and 2508.11450 from 2025 (only their revisions are recent),
        # 2609.22706 is the day after the window, and 2609.12825 counts although its v2 came after it.
        self.assertEqual([c.arxiv_id for c in candidates], ["2609.12871", "2609.12825"])
        self.assertEqual(candidates[0].comment, 'Paper accompanying the dataset "7V-Scanario"')
        self.assertTrue(all(c.has_api_metadata for c in candidates))
        self.assertEqual(
            [call.kwargs["params"] for call in mock_request.call_args_list[1:]],
            [
                {"verb": "ListRecords", "metadataPrefix": "arXivRaw", "set": "cs:cs:CV", "from": "2026-09-11"},
                {"verb": "ListRecords", "resumptionToken": "tok1"},
            ],
        )
        self.assertEqual({call.kwargs["user_agent"] for call in mock_request.call_args_list}, {"MyApp/9.9"})
        with self.assertRaises(ArxivRefused):  # a search has no OAI equivalent; the caller decides
            ArxivApiBackend().fetch(categories=["cs.CV"], query="ti:x", **window)
        self.assertEqual(mock_request.call_count, 3)

    @patch("app.services.ingest.arxiv_api_backend.utc_today", return_value=date(2026, 9, 28))
    @patch("app.services.ingest.arxiv_api_backend.request_with_backoff")
    def test_oai_listing_refuses_a_deep_window_before_downloading(self, mock_request, _today):
        # The listing covers every record revised since the start, whatever the window's width.
        with self.assertRaisesRegex(RuntimeError, "use a later start date"):
            list_oai_candidates(["cs.CV"], date(2026, 6, 1), date(2026, 6, 1), 25)
        mock_request.assert_not_called()

    def test_oai_set_names_follow_the_archive_group(self):
        self.assertEqual(
            [_oai_set(c) for c in ("cs.CV", "math.GT", "astro-ph.CO", "hep-th", "cs")],
            ["cs:cs:CV", "math:math:GT", "physics:astro-ph:CO", "physics:hep-th", "cs:cs"],
        )
