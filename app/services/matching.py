"""Whitelist matchers for author/affiliation/title signals."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

from app.services.text import normalize

MATCH_PRIORITY = {
    "Author": 1,
    "Affiliation": 2,
    "Title": 3,
}


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _build_pattern(term: str) -> tuple[str, int]:
    """
    Build regex source for a whitelist term.

    - ALL-CAPS short terms (<=4 chars) are case-sensitive.
    - Multi-word terms accept hyphen/space/newline separators.
    - Other terms are case-insensitive.
    """
    normalized = normalize(term)
    escaped = re.escape(normalized)
    is_short_acronym = len(term) <= 4 and term.isupper()
    flags = 0 if is_short_acronym else re.IGNORECASE

    if " " in term:
        flexible_spacing = escaped.replace(r"\ ", r"[-\s]+")
        return rf"\b{flexible_spacing}\b", flags

    return rf"\b{escaped}\b", flags


@lru_cache(maxsize=64)
def _compile_patterns(terms: tuple[str, ...], mode: str) -> tuple[tuple[str, re.Pattern[str]], ...]:
    compiled: list[tuple[str, re.Pattern[str]]] = []

    if mode == "author":
        for term in terms:
            normalized_term = normalize(term)
            pattern = re.compile(rf"\b{re.escape(normalized_term)}\b", re.IGNORECASE)
            compiled.append((term, pattern))
        return tuple(compiled)

    for term in terms:
        source, flags = _build_pattern(term)
        compiled.append((term, re.compile(source, flags)))

    return tuple(compiled)


def check_whitelist_match(texts: Iterable[str], whitelist: list[str]) -> list[str]:
    """Return deduplicated whitelist terms found in provided texts."""
    normalized_texts = [normalize(text) for text in texts if text]
    matches: list[str] = []
    patterns = _compile_patterns(tuple(whitelist), mode="general")

    for term, pattern in patterns:
        if any(pattern.search(text) for text in normalized_texts):
            matches.append(term)

    return dedupe_preserve_order(matches)


def check_author_match(author_names: Iterable[str], whitelist: list[str]) -> list[str]:
    normalized_names = [normalize(name.strip()) for name in author_names if name]
    matches: list[str] = []
    patterns = _compile_patterns(tuple(whitelist), mode="author")

    for term, pattern in patterns:
        if any(pattern.search(name) for name in normalized_names):
            matches.append(term)

    return dedupe_preserve_order(matches)
