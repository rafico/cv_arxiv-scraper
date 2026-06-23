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


def _nearest_distance(pattern: re.Pattern[str], comment: str, venue: re.Match[str]) -> int | None:
    """Distance from the venue mention to the closest occurrence of ``pattern``."""
    best: int | None = None
    for m in pattern.finditer(comment):
        distance = abs(m.start() - venue.start())
        if best is None or distance < best:
            best = distance
    return best


def _signal_near_venue(pattern: re.Pattern[str], comment: str, venue: re.Match[str], *, window: int = 30) -> bool:
    """Whether ``pattern`` occurs within ``window`` characters of the venue mention."""
    lo = max(0, venue.start() - window)
    hi = min(len(comment), venue.end() + window)
    return bool(pattern.search(comment, lo, hi))


def _qualifier_near_venue(comment: str, venue: re.Match[str], *, window: int = 30) -> str | None:
    """The oral/spotlight/highlight qualifier within ``window`` chars of the venue, if any.

    Mirrors :func:`_signal_near_venue` so a qualifier that belongs to a *different*
    venue elsewhere in the comment (e.g. "Accepted to CVPR 2024. Extended version of
    our ICCV oral paper.") does not flip the matched venue's status to oral.
    """
    lo = max(0, venue.start() - window)
    hi = min(len(comment), venue.end() + window)
    match = _QUALIFIER_RE.search(comment, lo, hi)
    return match.group(1).lower() if match else None


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

    # Weigh acceptance signals by how close they sit to the venue mention rather
    # than first-match-wins, so unrelated words elsewhere in the comment don't
    # flip the status. "Submitted to CVPR; we hope it is accepted" is a
    # submission (the stray "accepted" is farther away), and "Accepted to CVPR
    # (Oral). Extended version of our ICCV workshop paper" is an oral (the
    # "workshop" token belongs to the other venue, not CVPR).
    submit_dist = _nearest_distance(_SUBMIT_RE, comment, match)
    accept_dist = _nearest_distance(_ACCEPT_RE, comment, match)
    qualifier = _qualifier_near_venue(comment, match)

    if submit_dist is not None and (accept_dist is None or submit_dist <= accept_dist):
        # A submission cue at least as close as any acceptance cue → not confirmed.
        status = "mentioned"
    elif _signal_near_venue(_WORKSHOP_RE, comment, match):
        # "workshop" adjacent to the venue → a workshop acceptance.
        status = "workshop"
    elif qualifier is not None:
        # An oral/spotlight/highlight qualifier *near the venue* outranks a bare
        # acceptance; a stray qualifier about another venue does not (see above).
        status = qualifier
    elif accept_dist is not None:
        status = "accepted"
    else:
        status = "mentioned"

    return VenueMatch(venue=canonical, year=year, status=status)


def venue_bonus(acceptance_status: str | None, weight: float) -> float:
    """Ranking bonus for a confirmed venue acceptance."""
    if not acceptance_status:
        return 0.0
    return weight * VENUE_STATUS_MULTIPLIERS.get(acceptance_status, 0.0)
