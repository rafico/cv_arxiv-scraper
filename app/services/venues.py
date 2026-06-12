"""Venue/acceptance detection from arXiv comment metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# Canonical venue -> aliases, most specific aliases first across the dict
# (e.g. "SIGGRAPH Asia" must be checked before "SIGGRAPH").
KNOWN_VENUES: dict[str, tuple[str, ...]] = {
    "CVPR": ("CVPR",),
    "ICCV": ("ICCV",),
    "ECCV": ("ECCV",),
    "WACV": ("WACV",),
    "BMVC": ("BMVC",),
    "ACCV": ("ACCV",),
    "3DV": ("3DV",),
    "NeurIPS": ("NeurIPS", "NIPS"),
    "ICML": ("ICML",),
    "ICLR": ("ICLR",),
    "AAAI": ("AAAI",),
    "IJCAI": ("IJCAI",),
    "ICRA": ("ICRA",),
    "IROS": ("IROS",),
    "CoRL": ("CoRL",),
    "RSS": ("RSS", "Robotics: Science and Systems"),
    "SIGGRAPH Asia": ("SIGGRAPH Asia",),
    "SIGGRAPH": ("SIGGRAPH",),
    "MICCAI": ("MICCAI",),
    "TPAMI": ("TPAMI", "T-PAMI", "IEEE Transactions on Pattern Analysis and Machine Intelligence"),
    "IJCV": ("IJCV", "International Journal of Computer Vision"),
    "ICIP": ("ICIP",),
    "ICASSP": ("ICASSP",),
    "ACM MM": ("ACM MM", "ACM Multimedia"),
}

# Score multipliers applied to the venue ranking weight. A bare venue mention
# ("submitted to CVPR") must not score — only confirmed acceptance counts.
VENUE_STATUS_MULTIPLIERS = {
    "oral": 1.5,
    "spotlight": 1.5,
    "highlight": 1.5,
    "accepted": 1.0,
    "workshop": 0.5,
    "mentioned": 0.0,
}

_ACCEPT_RE = re.compile(
    r"\b(accept\w*|to appear|camera[- ]ready|appear(s|ing)? (at|in)|published (at|in)|presented at|proceedings of)\b",
    re.IGNORECASE,
)
_SUBMIT_RE = re.compile(r"\b(submitted|under review|in submission)\b", re.IGNORECASE)
_QUALIFIER_RE = re.compile(r"\b(oral|spotlight|highlight)\b", re.IGNORECASE)
_WORKSHOP_RE = re.compile(r"\bworkshops?\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass(slots=True)
class VenueMatch:
    venue: str
    year: int | None
    status: str


@lru_cache(maxsize=128)
def _alias_pattern(alias: str) -> re.Pattern[str]:
    """Build a word-boundary pattern for a venue alias.

    Mirrors the matching.py acronym discipline: short ALL-CAPS aliases are
    case-sensitive so "RSS" the conference does not match "rss feed". A
    trailing digit is allowed ("CVPR2026") but trailing letters are not.
    """
    escaped = re.escape(alias).replace(r"\ ", r"[-\s]+")
    flags = 0 if len(alias) <= 4 and alias.isupper() else re.IGNORECASE
    return re.compile(rf"(?<![A-Za-z]){escaped}(?![A-Za-z])", flags)


def _find_year(comment: str, *, search_from: int) -> int | None:
    """Year adjacent to the venue mention, falling back to anywhere in the comment."""
    window = comment[search_from : search_from + 12]
    match = _YEAR_RE.search(window) or _YEAR_RE.search(comment)
    return int(match.group()) if match else None


def parse_venue(comment: str | None) -> VenueMatch | None:
    """Detect a known venue and acceptance status in an arXiv comment string."""
    if not comment or not comment.strip():
        return None

    found: tuple[str, re.Match[str]] | None = None
    for canonical, aliases in KNOWN_VENUES.items():
        for alias in aliases:
            match = _alias_pattern(alias).search(comment)
            if match:
                found = (canonical, match)
                break
        if found:
            break
    if not found:
        return None

    canonical, match = found
    year = _find_year(comment, search_from=match.end())

    accepted = bool(_ACCEPT_RE.search(comment))
    if _SUBMIT_RE.search(comment) and not accepted:
        status = "mentioned"
    elif _WORKSHOP_RE.search(comment):
        status = "workshop"
    else:
        qualifier = _QUALIFIER_RE.search(comment)
        if qualifier:
            status = qualifier.group(1).lower()
        elif accepted:
            status = "accepted"
        else:
            status = "mentioned"

    return VenueMatch(venue=canonical, year=year, status=status)


def venue_bonus(acceptance_status: str | None, weight: float) -> float:
    """Ranking bonus for a confirmed venue acceptance."""
    if not acceptance_status:
        return 0.0
    return weight * VENUE_STATUS_MULTIPLIERS.get(acceptance_status, 0.0)
