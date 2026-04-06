from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.services.llm_client import (
    LLMClient,
    has_api_key,
    resolve_api_key,
    write_api_key,
)


class ResolveApiKeyTests(unittest.TestCase):
    def test_resolve_api_key_reads_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / ".llm_api_key"
            key_path.write_text("abc123", encoding="utf-8")
            self.assertEqual(resolve_api_key(key_path), "abc123")

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"})
    def test_resolve_api_key_prefers_env_var(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / ".llm_api_key"
            key_path.write_text("file-key", encoding="utf-8")
            self.assertEqual(resolve_api_key(key_path), "env-key")

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False)
    def test_resolve_api_key_returns_none_when_no_source(self):
        key_path = Path("/nonexistent/path/.llm_api_key")
        self.assertIsNone(resolve_api_key(key_path))

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False)
    def test_resolve_api_key_returns_none_for_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / ".llm_api_key"
            key_path.write_text("   \n  ", encoding="utf-8")
            self.assertIsNone(resolve_api_key(key_path))

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False)
    def test_resolve_api_key_returns_none_for_unreadable_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / ".llm_api_key"
            key_path.write_text("secret", encoding="utf-8")
            key_path.chmod(0o000)
            try:
                self.assertIsNone(resolve_api_key(key_path))
            finally:
                key_path.chmod(0o600)


class HasApiKeyTests(unittest.TestCase):
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False)
    def test_has_api_key_true_with_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / ".llm_api_key"
            key_path.write_text("key123", encoding="utf-8")
            self.assertTrue(has_api_key(key_path))

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False)
    def test_has_api_key_false_without_file(self):
        key_path = Path("/nonexistent/.llm_api_key")
        self.assertFalse(has_api_key(key_path))

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"})
    def test_has_api_key_true_with_env_var(self):
        key_path = Path("/nonexistent/.llm_api_key")
        self.assertTrue(has_api_key(key_path))


class WriteApiKeyTests(unittest.TestCase):
    def test_write_api_key_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / ".llm_api_key"
            write_api_key("my-key", key_path)
            self.assertEqual(key_path.read_text(encoding="utf-8"), "my-key")

    def test_write_api_key_sets_permissions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / ".llm_api_key"
            write_api_key("my-key", key_path)
            mode = key_path.stat().st_mode & 0o777
            self.assertEqual(mode, 0o600)

    def test_write_api_key_strips_whitespace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / ".llm_api_key"
            write_api_key("  spaced-key  \n", key_path)
            self.assertEqual(key_path.read_text(encoding="utf-8"), "spaced-key")


class LLMClientInitTests(unittest.TestCase):
    @patch("app.services.llm_client.OpenAI")
    def test_empty_api_key_raises_value_error(self, mock_openai):
        with self.assertRaises(ValueError):
            LLMClient("", "model", "https://example.com")

    @patch("app.services.llm_client.OpenAI")
    def test_whitespace_api_key_raises_value_error(self, mock_openai):
        with self.assertRaises(ValueError):
            LLMClient("   ", "model", "https://example.com")

    def test_missing_openai_raises_runtime_error(self):
        with patch("app.services.llm_client.OpenAI", None):
            with self.assertRaises(RuntimeError):
                LLMClient("key", "model", "https://example.com")

    @patch("app.services.llm_client.OpenAI")
    def test_max_concurrent_minimum_is_one(self, mock_openai):
        client = LLMClient("key", "model", "https://example.com", max_concurrent=0)
        # Semaphore should allow at least 1 concurrent call
        self.assertTrue(client._semaphore.acquire(blocking=False))
        client._semaphore.release()


class GenerateTLDRTests(unittest.TestCase):
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
    def test_generate_tldr_returns_none_on_empty_content(self, mock_openai):
        response = Mock()
        response.choices = [Mock(message=Mock(content=""))]
        mock_openai.return_value.chat.completions.create.return_value = response

        client = LLMClient("test-key", "model", "https://example.com")
        self.assertIsNone(client.generate_tldr("Title", "Abstract"))

    @patch("app.services.llm_client.OpenAI")
    def test_generate_tldr_returns_none_on_whitespace_content(self, mock_openai):
        response = Mock()
        response.choices = [Mock(message=Mock(content="   \n  "))]
        mock_openai.return_value.chat.completions.create.return_value = response

        client = LLMClient("test-key", "model", "https://example.com")
        self.assertIsNone(client.generate_tldr("Title", "Abstract"))

    @patch("app.services.llm_client.OpenAI")
    def test_generate_tldr_returns_none_on_no_choices(self, mock_openai):
        response = Mock()
        response.choices = []
        mock_openai.return_value.chat.completions.create.return_value = response

        client = LLMClient("test-key", "model", "https://example.com")
        self.assertIsNone(client.generate_tldr("Title", "Abstract"))

    @patch("app.services.llm_client.OpenAI")
    def test_generate_tldr_strips_result(self, mock_openai):
        response = Mock()
        response.choices = [Mock(message=Mock(content="  Stripped TLDR  \n"))]
        mock_openai.return_value.chat.completions.create.return_value = response

        client = LLMClient("test-key", "model", "https://example.com")
        self.assertEqual(client.generate_tldr("Title", "Abstract"), "Stripped TLDR")


class RateRelevanceTests(unittest.TestCase):
    @patch("app.services.llm_client.OpenAI")
    def test_rate_relevance_parses_numeric_response(self, mock_openai):
        response = Mock()
        response.choices = [Mock(message=Mock(content="8.5"))]
        mock_openai.return_value.chat.completions.create.return_value = response

        client = LLMClient("test-key", "model", "https://example.com")
        self.assertEqual(client.rate_relevance("Title", "Abstract", "Vision"), 8.5)

    @patch("app.services.llm_client.OpenAI")
    def test_rate_relevance_returns_none_on_non_numeric_response(self, mock_openai):
        response = Mock()
        response.choices = [Mock(message=Mock(content="Not a number at all"))]
        mock_openai.return_value.chat.completions.create.return_value = response

        client = LLMClient("test-key", "model", "https://example.com")
        self.assertIsNone(client.rate_relevance("Title", "Abstract", "Vision"))

    @patch("app.services.llm_client.OpenAI")
    def test_rate_relevance_clamps_high_scores(self, mock_openai):
        response = Mock()
        response.choices = [Mock(message=Mock(content="15"))]
        mock_openai.return_value.chat.completions.create.return_value = response

        client = LLMClient("test-key", "model", "https://example.com")
        self.assertEqual(client.rate_relevance("Title", "Abstract", "Vision"), 10.0)

    @patch("app.services.llm_client.OpenAI")
    def test_rate_relevance_clamps_low_scores(self, mock_openai):
        response = Mock()
        response.choices = [Mock(message=Mock(content="-3"))]
        mock_openai.return_value.chat.completions.create.return_value = response

        client = LLMClient("test-key", "model", "https://example.com")
        self.assertEqual(client.rate_relevance("Title", "Abstract", "Vision"), 1.0)

    @patch("app.services.llm_client.OpenAI")
    def test_rate_relevance_returns_none_on_exception(self, mock_openai):
        mock_openai.return_value.chat.completions.create.side_effect = Exception("API error")

        client = LLMClient("test-key", "model", "https://example.com")
        self.assertIsNone(client.rate_relevance("Title", "Abstract", "Vision"))

    @patch("app.services.llm_client.OpenAI")
    def test_rate_relevance_returns_none_on_empty_content(self, mock_openai):
        response = Mock()
        response.choices = [Mock(message=Mock(content=""))]
        mock_openai.return_value.chat.completions.create.return_value = response

        client = LLMClient("test-key", "model", "https://example.com")
        self.assertIsNone(client.rate_relevance("Title", "Abstract", "Vision"))

    @patch("app.services.llm_client.OpenAI")
    def test_rate_relevance_returns_none_on_none_content(self, mock_openai):
        response = Mock()
        response.choices = [Mock(message=Mock(content=None))]
        mock_openai.return_value.chat.completions.create.return_value = response

        client = LLMClient("test-key", "model", "https://example.com")
        self.assertIsNone(client.rate_relevance("Title", "Abstract", "Vision"))

    @patch("app.services.llm_client.OpenAI")
    def test_rate_relevance_extracts_from_text_with_number(self, mock_openai):
        response = Mock()
        response.choices = [Mock(message=Mock(content="I'd rate this 7.5 out of 10"))]
        mock_openai.return_value.chat.completions.create.return_value = response

        client = LLMClient("test-key", "model", "https://example.com")
        self.assertEqual(client.rate_relevance("Title", "Abstract", "Vision"), 7.5)


class CreateCompletionTests(unittest.TestCase):
    @patch("app.services.llm_client.OpenAI")
    def test_passes_all_parameters(self, mock_openai):
        mock_create = mock_openai.return_value.chat.completions.create
        response = Mock()
        response.choices = [Mock(message=Mock(content="TLDR"))]
        mock_create.return_value = response

        client = LLMClient("test-key", "test-model", "https://example.com")
        client.generate_tldr("Title", "Abstract")

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "test-model")
        self.assertEqual(call_kwargs["max_tokens"], 150)
        self.assertEqual(call_kwargs["temperature"], 0.3)

    @patch("app.services.llm_client.OpenAI")
    def test_uses_correct_message_format(self, mock_openai):
        mock_create = mock_openai.return_value.chat.completions.create
        response = Mock()
        response.choices = [Mock(message=Mock(content="TLDR"))]
        mock_create.return_value = response

        client = LLMClient("test-key", "model", "https://example.com")
        client.generate_tldr("Title", "Abstract")

        messages = mock_create.call_args.kwargs["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Title", messages[1]["content"])
        self.assertIn("Abstract", messages[1]["content"])

    @patch("app.services.llm_client.OpenAI")
    def test_passes_reasoning_effort_when_configured(self, mock_openai):
        mock_create = mock_openai.return_value.chat.completions.create
        response = Mock()
        response.choices = [Mock(message=Mock(content="TLDR"))]
        mock_create.return_value = response

        client = LLMClient("test-key", "model", "https://example.com", reasoning_effort="none")
        client.generate_tldr("Title", "Abstract")

        self.assertEqual(mock_create.call_args.kwargs["reasoning_effort"], "none")


class ConcurrencySemaphoreTests(unittest.TestCase):
    @patch("app.services.llm_client.OpenAI")
    def test_semaphore_limits_concurrent_calls(self, mock_openai):
        max_concurrent = 2
        concurrent_count = []
        lock = threading.Lock()

        def slow_create(**kwargs):
            with lock:
                concurrent_count.append(threading.active_count())
            time.sleep(0.1)
            response = Mock()
            response.choices = [Mock(message=Mock(content="TLDR"))]
            return response

        mock_openai.return_value.chat.completions.create.side_effect = slow_create

        client = LLMClient("test-key", "model", "https://example.com", max_concurrent=max_concurrent)

        threads = []
        for _ in range(5):
            t = threading.Thread(target=client.generate_tldr, args=("Title", "Abstract"))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 5 calls should complete successfully
        self.assertEqual(mock_openai.return_value.chat.completions.create.call_count, 5)
