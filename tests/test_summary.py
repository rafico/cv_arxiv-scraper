import unittest
from unittest.mock import Mock

from app.services.summary import extract_topic_tags, generate_llm_summary, generate_summary


class SummaryTests(unittest.TestCase):
    def test_generate_summary_uses_abstract_content(self):
        summary = generate_summary(
            "A Vision Model",
            "We introduce a transformer for dense prediction tasks. "
            "The method improves segmentation and detection benchmarks.",
        )
        self.assertIn("transformer", summary.lower())

    def test_extract_topic_tags_detects_known_topics(self):
        tags = extract_topic_tags(
            "Zero-shot remote sensing segmentation",
            "We study satellite imagery and propose zero shot segmentation approach.",
        )
        self.assertIn("Remote Sensing", tags)
        self.assertIn("Segmentation", tags)

    def test_generate_llm_summary_uses_llm_result(self):
        llm_client = Mock()
        llm_client.generate_tldr.return_value = "A concise model summary."

        summary = generate_llm_summary(llm_client, "Title", "Abstract")

        self.assertEqual(summary, "A concise model summary.")

    def test_generate_llm_summary_falls_back_to_extractive(self):
        llm_client = Mock()
        llm_client.generate_tldr.return_value = None

        summary = generate_llm_summary(
            llm_client,
            "A Vision Model",
            "We introduce a transformer for dense prediction tasks.",
        )

        self.assertIn("transformer", summary.lower())


if __name__ == "__main__":
    unittest.main()
