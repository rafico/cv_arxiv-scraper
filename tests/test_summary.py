import unittest

from app.services.summary import extract_topic_tags, generate_summary


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


if __name__ == "__main__":
    unittest.main()
