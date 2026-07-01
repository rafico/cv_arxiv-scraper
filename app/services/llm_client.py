"""OpenAI-compatible LLM helpers."""

from __future__ import annotations

import math
import os
import re
import threading
from pathlib import Path

try:  # pragma: no cover - exercised indirectly in integration paths
    from openai import OpenAI
except ImportError:  # pragma: no cover - depends on local environment
    OpenAI = None

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_KEY_PATH = _PROJECT_ROOT / ".llm_api_key"
_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _strip_code_fences(content: str) -> str:
    """Remove a wrapping ```json ... ``` fence that some models emit."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


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
    # Create with 0600 from the start so the key is never briefly world-readable
    # (avoids a write-then-chmod TOCTOU window). Shared with the OAuth/reference-
    # manager secret writers so the safe pattern can't drift.
    from app.services.secret_files import write_secret_file

    return write_secret_file(key_path or _DEFAULT_KEY_PATH, api_key.strip())


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        max_concurrent: int = 4,
        reasoning_effort: str | None = None,
    ):
        if not api_key.strip():
            raise ValueError("LLM API key is required")
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self._semaphore = threading.Semaphore(max(1, int(max_concurrent)))
        self.reasoning_effort = reasoning_effort

    def _create_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        **extra,
    ):
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **extra,
        }
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort

        return self.client.chat.completions.create(
            **request,
        )

    def complete(self, *, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float, **extra):
        """Throttled public wrapper around ``_create_completion``.

        Acquires the concurrency semaphore so external callers (e.g. the corpus chat)
        respect ``max_concurrent`` instead of reaching into the private helper and
        issuing an extra in-flight request beyond the configured cap.
        """
        with self._semaphore:
            return self._create_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **extra,
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

    def analyze_paper(
        self,
        title: str,
        abstract: str,
        interests: str,
        matched_terms: list[str] | None = None,
    ) -> dict | None:
        """One structured call combining TLDR, relevance, and paper facts.

        Returns a dict with tldr/relevance/tasks/datasets/method_type/backbone/
        why_matched, or None on any failure (caller falls back to the legacy
        two-call path).
        """
        import json

        system_prompt = (
            "You analyze computer-vision research papers. Respond with STRICT JSON only, no prose, "
            "matching exactly this schema: "
            '{"tldr": string, "relevance": number, "tasks": [string], "datasets": [string], '
            '"method_type": string, "backbone": string or null, "why_matched": string}. '
            "tldr: specific 1-2 sentence summary, under 280 characters. "
            "relevance: 1-10 relevance to the reader's interests. "
            "tasks: vision tasks addressed (e.g. object detection). "
            "datasets: benchmark/dataset names evaluated on (e.g. COCO, ADE20K); [] if unclear. "
            "method_type: one short phrase (e.g. diffusion model, transformer). "
            "backbone: main architecture/backbone, or null. "
            "why_matched: one line (under 120 chars) tying the paper to the reader's interests."
        )
        matched = ", ".join(matched_terms or []) or "none"
        user_prompt = (
            f"Reader interests: {interests or 'General computer vision'}\n"
            f"Matched interest terms: {matched}\n\n"
            f"Title: {title}\n\nAbstract: {abstract}"
        )

        content = None
        # Some OpenAI-compatible servers reject response_format; retry without it.
        for extra in ({"response_format": {"type": "json_object"}}, {}):
            try:
                with self._semaphore:
                    response = self._create_completion(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=400,
                        temperature=0.2,
                        **extra,
                    )
                content = response.choices[0].message.content
                break
            except Exception:  # noqa: S112 — retry without response_format, then give up
                continue

        if not isinstance(content, str) or not content.strip():
            return None
        try:
            data = json.loads(_strip_code_fences(content))
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None

        relevance = None
        try:
            if data.get("relevance") is not None:
                candidate = float(data["relevance"])
                # A NaN survives float() and the min/max clamp silently becomes 10.0
                # (top relevance); reject non-finite values so they fall back to None.
                if math.isfinite(candidate):
                    relevance = max(1.0, min(10.0, candidate))
        except (TypeError, ValueError):
            relevance = None

        def _str_list(value) -> list[str]:
            if not isinstance(value, list):
                return []
            return [str(item).strip() for item in value if str(item).strip()][:8]

        backbone = data.get("backbone")
        return {
            "tldr": str(data.get("tldr") or "").strip(),
            "relevance": relevance,
            "tasks": _str_list(data.get("tasks")),
            "datasets": _str_list(data.get("datasets")),
            "method_type": str(data.get("method_type") or "").strip(),
            "backbone": str(backbone).strip() if backbone else None,
            "why_matched": str(data.get("why_matched") or "").strip()[:160],
        }

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
