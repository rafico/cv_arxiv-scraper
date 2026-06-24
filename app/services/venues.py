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


def _span_distance(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Edge distance between two spans — 0 if they overlap.

    Measuring against a venue's start only mis-ranks long-form aliases
    ("International Journal of Computer Vision"): a genuine acceptance phrase that
    *follows* the venue sits far from its start, so a preceding submission word looks
    closer. Compare against the nearest edge instead.
    """
    if a_start >= b_end:
        return a_start - b_end
    if a_end <= b_start:
        return b_start - a_end
    return 0


def _all_venue_spans(comment: str) -> list[tuple[str, int, int]]:
    """Every ``(canonical, start, end)`` known-venue mention in the comment."""
    spans: list[tuple[str, int, int]] = []
    for canonical, aliases in KNOWN_VENUES.items():
        for alias in aliases:
            for m in _alias_pattern(alias).finditer(comment):
                spans.append((canonical, m.start(), m.end()))
    return spans


def _venue_owns_signal(
    s_start: int, s_end: int, venue: re.Match[str], canonical: str, venue_spans: list[tuple[str, int, int]]
) -> bool:
    """Whether the matched venue is the nearest venue mention to a signal occurrence.

    A submit/accept/qualifier/workshop cue that sits closer to a *different*
    co-mentioned venue belongs to that venue, not the matched one (e.g. "Submitted to
    CVPR. Accepted to ICCV 2024 oral" — the "Accepted"/"oral" cues are ICCV's). Ties
    go to the matched venue; with a single venue in the comment this is always True.
    """
    own = _span_distance(s_start, s_end, venue.start(), venue.end())
    for other_canonical, vs, ve in venue_spans:
        if other_canonical == canonical:
            continue
        if _span_distance(s_start, s_end, vs, ve) < own:
            return False
    return True


def _nearest_distance(
    pattern: re.Pattern[str],
    comment: str,
    venue: re.Match[str],
    canonical: str,
    venue_spans: list[tuple[str, int, int]],
) -> int | None:
    """Edge distance to the closest ``pattern`` match the matched venue *owns*."""
    best: int | None = None
    for m in pattern.finditer(comment):
        if not _venue_owns_signal(m.start(), m.end(), venue, canonical, venue_spans):
            continue
        distance = _span_distance(m.start(), m.end(), venue.start(), venue.end())
        if best is None or distance < best:
            best = distance
    return best


def _signal_near_venue(
    pattern: re.Pattern[str],
    comment: str,
    venue: re.Match[str],
    canonical: str,
    venue_spans: list[tuple[str, int, int]],
    *,
    window: int = 30,
) -> bool:
    """Whether ``pattern`` occurs within ``window`` chars of the venue *and* is owned by it."""
    lo = max(0, venue.start() - window)
    hi = min(len(comment), venue.end() + window)
    return any(
        _venue_owns_signal(m.start(), m.end(), venue, canonical, venue_spans) for m in pattern.finditer(comment, lo, hi)
    )


def _qualifier_near_venue(
    comment: str, venue: re.Match[str], canonical: str, venue_spans: list[tuple[str, int, int]], *, window: int = 30
) -> str | None:
    """The oral/spotlight/highlight qualifier within ``window`` chars of, and owned by, the venue."""
    lo = max(0, venue.start() - window)
    hi = min(len(comment), venue.end() + window)
    for m in _QUALIFIER_RE.finditer(comment, lo, hi):
        if _venue_owns_signal(m.start(), m.end(), venue, canonical, venue_spans):
            return m.group(1).lower()
    return None


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
    # Attribute each acceptance/submission cue only when the matched venue is the
    # nearest venue mention to it, so cues that belong to a co-mentioned venue are
    # not stolen by the (dict-order) first-matched venue.
    venue_spans = _all_venue_spans(comment)
    submit_dist = _nearest_distance(_SUBMIT_RE, comment, match, canonical, venue_spans)
    accept_dist = _nearest_distance(_ACCEPT_RE, comment, match, canonical, venue_spans)
    qualifier = _qualifier_near_venue(comment, match, canonical, venue_spans)

    if submit_dist is not None and (accept_dist is None or submit_dist <= accept_dist):
        # A submission cue at least as close as any acceptance cue → not confirmed.
        status = "mentioned"
    elif _signal_near_venue(_WORKSHOP_RE, comment, match, canonical, venue_spans):
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
