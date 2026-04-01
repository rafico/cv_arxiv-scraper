"""OpenAI-compatible LLM helpers."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

try:  # pragma: no cover - exercised indirectly in integration paths
    from openai import OpenAI
except ImportError:  # pragma: no cover - depends on local environment
    OpenAI = None  # type: ignore[assignment]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_KEY_PATH = _PROJECT_ROOT / ".llm_api_key"
_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def resolve_api_key(key_path: Path | None = None) -> str | None:
    """Resolve API key from env var first, then from a gitignored file."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key

    path = key_path or _DEFAULT_KEY_PATH
    if not path.is_file():
        return None

    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def has_api_key(key_path: Path | None = None) -> bool:
    return resolve_api_key(key_path) is not None


def write_api_key(api_key: str, key_path: Path | None = None) -> Path:
    path = key_path or _DEFAULT_KEY_PATH
    path.write_text(api_key.strip(), encoding="utf-8")
    os.chmod(path, 0o600)
    return path


class LLMClient:
    def __init__(self, api_key: str, model: str, base_url: str, max_concurrent: int = 4):
        if not api_key.strip():
            raise ValueError("LLM API key is required")
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self._semaphore = threading.Semaphore(max(1, int(max_concurrent)))

    def _create_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate_tldr(self, title: str, abstract: str) -> str | None:
        system_prompt = "Produce a specific 1-2 sentence TLDR for a research paper. Keep it under 280 characters."
        user_prompt = f"Title: {title}\n\nAbstract: {abstract}"
        try:
            with self._semaphore:
                response = self._create_completion(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=150,
                    temperature=0.3,
                )
        except Exception:
            return None

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError):
            return None
        return content.strip() if isinstance(content, str) and content.strip() else None

    def rate_relevance(self, title: str, abstract: str, interests: str) -> float | None:
        system_prompt = (
            "Rate this paper's relevance to the provided research interests from 1 to 10. Respond with ONLY a number."
        )
        user_prompt = (
            f"Research interests: {interests or 'General computer vision'}\n\nTitle: {title}\n\nAbstract: {abstract}"
        )
        try:
            with self._semaphore:
                response = self._create_completion(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=5,
                    temperature=0.0,
                )
        except Exception:
            return None

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError):
            return None
        if not isinstance(content, str):
            return None

        match = _NUMERIC_RE.search(content)
        if not match:
            return None

        try:
            score = float(match.group(0))
        except ValueError:
            return None
        return max(1.0, min(10.0, score))
