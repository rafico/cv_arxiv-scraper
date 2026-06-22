from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import Mock, patch

from app.services.enrichment import (
    _fetch_api_metadata,
    extract_affiliation_text_batch,
    extract_pdf_resource_links,
    extract_pdf_resource_links_batch,
    fetch_recent_papers,
    merge_resource_links,
)


def _make_pdf(pages: list[list[str]]) -> bytes:
    """Create a minimal PDF with the given text lines per page using fpdf."""
    import os
    import pathlib
    import tempfile

    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_font("Helvetica", size=11)
    for lines in pages:
        pdf.add_page()
        for line in lines:
            pdf.cell(0, 6, txt=line, ln=1)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf.output(tmp.name)
    try:
        return pathlib.Path(tmp.name).read_bytes()
    finally:
        os.unlink(tmp.name)


class ExtractPdfResourceLinksBatchTests(unittest.TestCase):
    def test_batch_aligns_with_inputs_and_handles_none(self):
        with patch(
            "app.services.enrichment.extract_pdf_resource_links",
            side_effect=lambda pdf, max_pages=2: [{"url": pdf.decode()}] if pdf else [],
        ):
            out = extract_pdf_resource_links_batch([b"a", None, b"b"])
        self.assertEqual(out, [[{"url": "a"}], [], [{"url": "b"}]])


class ExtractAffiliationTextBatchTests(unittest.TestCase):
    def test_batch_aligns_and_maps_none_to_empty(self):
        with patch(
            "app.services.enrichment.extract_affiliation_text",
            side_effect=lambda pdf, **kw: pdf.decode(),
        ):
            out = extract_affiliation_text_batch([b"MIT", None, b"Stanford"])
        self.assertEqual(out, ["MIT", "", "Stanford"])

    def test_kwargs_are_forwarded(self):
        captured = {}

        def fake(pdf, *, lines_start, max_header_lines, smart_header):
            captured.update(lines_start=lines_start, max_header_lines=max_header_lines, smart_header=smart_header)
            return "x"

        with patch("app.services.enrichment.extract_affiliation_text", side_effect=fake):
            extract_affiliation_text_batch([b"a"], lines_start=3, max_header_lines=10, smart_header=False)
        self.assertEqual(captured, {"lines_start": 3, "max_header_lines": 10, "smart_header": False})


class FetchRecentPapersTests(unittest.TestCase):
    @patch("app.services.enrichment.utc_today", return_value=date(2026, 3, 20))
    @patch("app.services.enrichment.request_with_backoff")
    def test_uses_utc_today_for_query_window(self, mock_request, _mock_today):
        mock_request.return_value = Mock(text='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>')

        fetch_recent_papers(2, "https://rss.arxiv.org/rss/cs.CV")

        params = mock_request.call_args.kwargs["params"]
        self.assertEqual(
            params["search_query"],
            "cat:cs.CV AND submittedDate:[202603170000 TO 202603202359]",
        )
        self.assertEqual(mock_request.call_args.kwargs["rate_limit_profile"], "bulk")

    @patch("app.services.enrichment.time.sleep")
    @patch("app.services.enrichment.request_with_backoff")
    def test_fetch_api_metadata_splits_failed_batch(self, mock_request, _mock_sleep):
        def _xml_for_id(arxiv_id: str) -> str:
            return f"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}v1</id>
    <author><name>Author</name><arxiv:affiliation>Test Lab</arxiv:affiliation></author>
    <category term="cs.CV" />
    <arxiv:comment>Has code</arxiv:comment>
    <arxiv:doi>10.1000/test</arxiv:doi>
  </entry>
</feed>"""

        def _side_effect(_method, _url, **kwargs):
            ids = kwargs["params"]["id_list"].split(",")
            if len(ids) > 1:
                raise RuntimeError("429")
            return Mock(text=_xml_for_id(ids[0]))

        mock_request.side_effect = _side_effect

        metadata = _fetch_api_metadata(["2604.00001", "2604.00002"])

        self.assertEqual(set(metadata), {"2604.00001", "2604.00002"})
        self.assertEqual(
            [len(call.kwargs["params"]["id_list"].split(",")) for call in mock_request.call_args_list],
            [2, 1, 1],
        )
        self.assertTrue(all(call.kwargs["rate_limit_profile"] == "bulk" for call in mock_request.call_args_list))


class ExtractPdfResourceLinksTests(unittest.TestCase):
    def test_keeps_code_and_project_links_and_drops_generic_web(self):
        pdf = _make_pdf(
            [
                [
                    "A Great Paper",
                    "Code: https://github.com/lab/great-paper",
                    "Project page: https://lab.github.io/great-paper",
                    "See https://example.com/blog for details",
                ]
            ]
        )

        links = extract_pdf_resource_links(pdf)

        urls = {link["url"]: link["type"] for link in links}
        self.assertEqual(urls["https://github.com/lab/great-paper"], "code")
        self.assertEqual(urls["https://lab.github.io/great-paper"], "project")
        self.assertNotIn("https://example.com/blog", urls)

    def test_drops_arxiv_and_doi_self_links(self):
        pdf = _make_pdf([["https://arxiv.org/abs/2606.00001", "https://doi.org/10.1000/x"]])

        self.assertEqual(extract_pdf_resource_links(pdf), [])

    def test_ignores_pages_beyond_max_pages(self):
        pdf = _make_pdf(
            [
                ["First page"],
                ["Second page"],
                ["https://github.com/lab/late-mention"],
            ]
        )

        self.assertEqual(extract_pdf_resource_links(pdf, max_pages=2), [])

    def test_handles_none_and_invalid_bytes(self):
        self.assertEqual(extract_pdf_resource_links(None), [])
        self.assertEqual(extract_pdf_resource_links(b"not a pdf"), [])


class MergeResourceLinksTests(unittest.TestCase):
    def test_dedupes_on_url_keeping_existing_entry(self):
        existing = [{"type": "code", "label": "Code", "url": "https://github.com/a/b"}]
        new = [
            {"type": "code", "label": "Other", "url": "https://github.com/a/b"},
            {"type": "project", "label": "Project", "url": "https://a.github.io/b"},
        ]

        merged = merge_resource_links(existing, new)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0], existing[0])
        self.assertEqual(merged[1]["url"], "https://a.github.io/b")

    def test_handles_none_inputs(self):
        self.assertEqual(merge_resource_links(None, None), [])
        link = [{"type": "code", "label": "Code", "url": "https://github.com/a/b"}]
        self.assertEqual(merge_resource_links(None, link), link)


if __name__ == "__main__":
    unittest.main()
