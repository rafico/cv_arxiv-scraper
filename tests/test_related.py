import unittest

from app.services.related import build_vector, top_related_papers


class RelatedTests(unittest.TestCase):
    def test_related_selection_prefers_similar_documents(self):
        vectors = {
            1: build_vector("vision transformer segmentation remote sensing"),
            2: build_vector("transformer segmentation for satellite imagery"),
            3: build_vector("language model for code generation"),
        }

        related = top_related_papers(1, vectors, top_k=2)
        self.assertIn(2, related)
        self.assertNotIn(3, related)


if __name__ == "__main__":
    unittest.main()
