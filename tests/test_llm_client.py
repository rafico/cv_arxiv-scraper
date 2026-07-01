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

    def test_write_api_key_creates_0600_without_relying_on_chmod(self):
        # Neutralize the trailing chmod so the assertion proves the *creation*
        # mode — i.e. the key is never briefly world-readable (no TOCTOU window),
        # even under a permissive umask.
        import os
        import stat

        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / ".llm_api_key"
            old_umask = os.umask(0)
            try:
                with patch("app.services.llm_client.os.chmod"):
                    write_api_key("my-key", key_path)
                mode = stat.S_IMODE(key_path.stat().st_mode)
                self.assertEqual(mode, 0o600)
            finally:
                os.umask(old_umask)


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


_VALID_ANALYSIS = (
    '{"tldr": "A new detector.", "relevance": 8, "tasks": ["object detection"], '
    '"datasets": ["COCO", "LVIS"], "method_type": "transformer", "backbone": "ViT-L", '
    '"why_matched": "Zero-shot detection overlaps your interests."}'
)


class AnalyzePaperTests(unittest.TestCase):
    def _client_with_content(self, mock_openai, content: str) -> LLMClient:
        response = Mock()
        response.choices = [Mock(message=Mock(content=content))]
        mock_openai.return_value.chat.completions.create.return_value = response
        return LLMClient("test-key", "model", "https://example.com")

    @patch("app.services.llm_client.OpenAI")
    def test_analyze_paper_parses_valid_json(self, mock_openai):
        client = self._client_with_content(mock_openai, _VALID_ANALYSIS)

        result = client.analyze_paper("Title", "Abstract", "Vision", matched_terms=["Zero Shot"])

        self.assertEqual(result["tldr"], "A new detector.")
        self.assertEqual(result["relevance"], 8.0)
        self.assertEqual(result["datasets"], ["COCO", "LVIS"])
        self.assertEqual(result["backbone"], "ViT-L")
        self.assertIn("Zero-shot", result["why_matched"])
        # Single combined call.
        self.assertEqual(mock_openai.return_value.chat.completions.create.call_count, 1)

    @patch("app.services.llm_client.OpenAI")
    def test_analyze_paper_strips_code_fences(self, mock_openai):
        client = self._client_with_content(mock_openai, f"```json\n{_VALID_ANALYSIS}\n```")

        result = client.analyze_paper("Title", "Abstract", "Vision")

        self.assertEqual(result["tasks"], ["object detection"])

    @patch("app.services.llm_client.OpenAI")
    def test_analyze_paper_returns_none_on_malformed_json(self, mock_openai):
        client = self._client_with_content(mock_openai, "this is not json {")
        self.assertIsNone(client.analyze_paper("Title", "Abstract", "Vision"))

    @patch("app.services.llm_client.OpenAI")
    def test_analyze_paper_returns_none_on_non_dict_json(self, mock_openai):
        client = self._client_with_content(mock_openai, '["a", "list"]')
        self.assertIsNone(client.analyze_paper("Title", "Abstract", "Vision"))

    @patch("app.services.llm_client.OpenAI")
    def test_analyze_paper_clamps_relevance(self, mock_openai):
        client = self._client_with_content(mock_openai, '{"tldr": "x", "relevance": 42}')

        result = client.analyze_paper("Title", "Abstract", "Vision")

        self.assertEqual(result["relevance"], 10.0)
        self.assertEqual(result["datasets"], [])
        self.assertIsNone(result["backbone"])

    @patch("app.services.llm_client.OpenAI")
    def test_complete_acquires_semaphore_around_call(self, mock_openai):
        # The public complete() wrapper must hold the concurrency semaphore for the whole
        # call so external callers (chat) respect max_concurrent.
        client = LLMClient("test-key", "model", "https://example.com")
        events = []

        class _TrackingSem:
            def __enter__(self):
                events.append("acquire")
                return self

            def __exit__(self, *exc):
                events.append("release")
                return False

        client._semaphore = _TrackingSem()
        mock_openai.return_value.chat.completions.create.return_value = Mock()

        client.complete(system_prompt="s", user_prompt="u", max_tokens=10, temperature=0.0)

        self.assertEqual(events, ["acquire", "release"])

    @patch("app.services.llm_client.OpenAI")
    def test_analyze_paper_rejects_non_finite_relevance(self, mock_openai):
        # A NaN survives float() and the min/max clamp silently becomes 10.0 (top
        # relevance); it must fall back to None instead of polluting ranking.
        client = self._client_with_content(mock_openai, '{"tldr": "x", "relevance": NaN}')

        result = client.analyze_paper("Title", "Abstract", "Vision")

        self.assertIsNone(result["relevance"])

    @patch("app.services.llm_client.OpenAI")
    def test_analyze_paper_retries_without_response_format(self, mock_openai):
        mock_create = mock_openai.return_value.chat.completions.create
        ok_response = Mock()
        ok_response.choices = [Mock(message=Mock(content=_VALID_ANALYSIS))]
        mock_create.side_effect = [TypeError("response_format unsupported"), ok_response]

        client = LLMClient("test-key", "model", "https://example.com")
        result = client.analyze_paper("Title", "Abstract", "Vision")

        self.assertIsNotNone(result)
        self.assertEqual(mock_create.call_count, 2)
        self.assertIn("response_format", mock_create.call_args_list[0].kwargs)
        self.assertNotIn("response_format", mock_create.call_args_list[1].kwargs)

    @patch("app.services.llm_client.OpenAI")
    def test_analyze_paper_returns_none_when_all_attempts_fail(self, mock_openai):
        mock_openai.return_value.chat.completions.create.side_effect = RuntimeError("boom")

        client = LLMClient("test-key", "model", "https://example.com")
        self.assertIsNone(client.analyze_paper("Title", "Abstract", "Vision"))


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
