"""Auto-summary and topic tag extraction."""

from __future__ import annotations

import re
from collections import Counter

from app.services.text import STOP_WORDS, clean_whitespace, tokenize

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

TOPIC_HINTS = {
    "Segmentation": ("segmentation", "segment"),
    "Detection": ("detection", "detector", "object"),
    "Tracking": ("tracking", "tracker", "mot"),
    "3D Vision": ("3d", "reconstruction", "depth", "neural radiance"),
    "Generative Models": ("diffusion", "generative", "gan", "synthesis"),
    "Medical Imaging": ("medical", "clinical", "ct", "mri", "xray"),
    "Remote Sensing": ("remote sensing", "satellite", "sar", "aerial"),
    "Vision-Language": ("vision-language", "multimodal", "caption", "vlm"),
    "Few-Shot": ("few-shot", "few shot", "zero-shot", "zero shot"),
}


def generate_summary(title: str, abstract: str, max_chars: int = 260) -> str:
    """
    Build a compact, readable summary from title/abstract.

    This is extractive rather than LLM-generated, but it is deterministic and fast.
    """
    clean_title = clean_whitespace(title)
    clean_abstract = clean_whitespace(abstract)

    if not clean_abstract:
        return clean_title

    sentences = [clean_whitespace(sentence) for sentence in _SENTENCE_SPLIT_RE.split(clean_abstract)]
    sentences = [sentence for sentence in sentences if sentence]
    if not sentences:
        return clean_abstract[:max_chars].rstrip()

    first_substantive = next((s for s in sentences if len(s.split()) >= 8), sentences[0])
    candidate = first_substantive

    if len(candidate) > max_chars:
        candidate = candidate[: max_chars - 1].rstrip() + "…"

    return candidate


def generate_llm_summary(llm_client, title: str, abstract: str, max_chars: int = 280) -> str:
    result = llm_client.generate_tldr(title, abstract)
    if result:
        trimmed = clean_whitespace(result)
        return trimmed[:max_chars].rstrip()
    return generate_summary(title, abstract)


def extract_topic_tags(title: str, abstract: str, limit: int = 5) -> list[str]:
    full_text = f"{title} {abstract}".lower()
    tags: list[str] = []

    for label, hints in TOPIC_HINTS.items():
        if any(hint in full_text for hint in hints):
            tags.append(label)

    if len(tags) >= limit:
        return tags[:limit]

    token_counts = Counter(
        token
        for token in tokenize(full_text)
        if token not in STOP_WORDS and len(token) >= 4
    )
    for token, _ in token_counts.most_common(limit * 2):
        cleaned = token.replace("-", " ").title()
        if cleaned not in tags:
            tags.append(cleaned)
        if len(tags) >= limit:
            break

    return tags[:limit]
