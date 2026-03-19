"""Shared text normalization and tokenization utilities."""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timezone

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{2,}")

STOP_WORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "also",
    "among",
    "an",
    "and",
    "any",
    "are",
    "around",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "data",
    "deep",
    "during",
    "each",
    "for",
    "from",
    "further",
    "has",
    "have",
    "having",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "learning",
    "may",
    "method",
    "methods",
    "model",
    "models",
    "more",
    "most",
    "new",
    "of",
    "on",
    "or",
    "our",
    "paper",
    "propose",
    "proposed",
    "results",
    "show",
    "shows",
    "system",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "to",
    "towards",
    "using",
    "we",
    "with",
    "within",
    "without",
}


def normalize(text: str | None) -> str:
    """Strip accents for robust text matching."""
    nfkd = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in nfkd if not unicodedata.combining(char))


def clean_whitespace(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    normalized = normalize(text).lower()
    return _TOKEN_RE.findall(normalized)


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def now_utc():
    """Return the current UTC time as a naive datetime (for SQLite storage)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
