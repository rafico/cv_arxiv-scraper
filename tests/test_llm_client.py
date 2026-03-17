from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.services.llm_client import LLMClient, resolve_api_key


class LLMClientTests(unittest.TestCase):
    @patch("app.services.llm_client.OpenAI")
    def test_generate_tldr_returns_content(self, mock_openai):
        response = Mock()
        response.choices = [Mock(message=Mock(content="Useful TLDR"))]
        mock_openai.return_value.chat.completions.create.return_value = response

        client = LLMClient("test-key", "model", "https://example.com")

        self.assertEqual(client.generate_tldr("Title", "Abstract"), "Useful TLDR")

    @patch("app.services.llm_client.OpenAI")
    def test_generate_tldr_returns_none_on_failure(self, mock_openai):
        mock_openai.return_value.chat.completions.create.side_effect = RuntimeError("boom")

        client = LLMClient("test-key", "model", "https://example.com")

        self.assertIsNone(client.generate_tldr("Title", "Abstract"))

    @patch("app.services.llm_client.OpenAI")
    def test_rate_relevance_parses_numeric_response(self, mock_openai):
        response = Mock()
        response.choices = [Mock(message=Mock(content="8.5"))]
        mock_openai.return_value.chat.completions.create.return_value = response

        client = LLMClient("test-key", "model", "https://example.com")

        self.assertEqual(client.rate_relevance("Title", "Abstract", "Vision"), 8.5)

    def test_resolve_api_key_reads_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / ".llm_api_key"
            key_path.write_text("abc123", encoding="utf-8")
            self.assertEqual(resolve_api_key(key_path), "abc123")
