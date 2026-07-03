"""Implementation-readiness scoring: how *runnable* a paper's code is.

A pure, dependency-free heuristic over already-extracted metadata — the paper's
``github_repo`` / ``github_stars`` / ``github_license`` columns, its
``resource_links`` (project pages), ``hf_upvotes``, and repo freshness (the
GitHub ``pushed_at`` timestamp when present, else ``scraped_at``). No DB writes,
no network, so it is safe to call per-row while rendering.

The 0-100 score maps to a coarse tier (``runnable`` / ``partial`` / ``none``)
that drives a card badge, a dashboard filter, and a small additive ranking bonus
(``app/services/ranking.py``). Having runnable code dominates; stars, license,
freshness and secondary signals (project page, HF upvotes) are modifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import exp, log1p

# Point budget. The contributions sum to at most 100 after the final cap; the
# repo itself is the dominant signal, the rest are quality/freshness modifiers.
REPO_POINTS = 35.0
STARS_MAX_POINTS = 25.0
# Stars at/above this saturate the (log-scaled) star contribution.
STARS_LOG_CAP = 1000
LICENSE_PERMISSIVE_POINTS = 15.0
LICENSE_KNOWN_POINTS = 8.0
NOT_ARCHIVED_POINTS = 8.0
FRESHNESS_MAX_POINTS = 12.0
# Exponential decay constant (points at age 0 = FRESHNESS_MAX_POINTS, ~37% of it
# at this many days). Not a strict half-life.
FRESHNESS_DECAY_DAYS = 180.0
# A repo push within this many days earns a "recently updated" reason (only when
# the age came from the repo's own pushed_at, not the scraped_at fallback).
RECENT_PUSH_DAYS = 45
PROJECT_PAGE_POINTS = 5.0
HF_MAX_POINTS = 5.0
HF_SATURATION_UPVOTES = 50

RUNNABLE_THRESHOLD = 60.0
PARTIAL_THRESHOLD = 30.0

# SPDX ids / license names treated as permissive (green-light for reuse). Lower-cased.
_PERMISSIVE_LICENSES = {
    "mit",
    "mit license",
    "apache-2.0",
    "apache license 2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "bsd 3-clause",
    "bsd 2-clause",
    "isc",
    "unlicense",
    "the unlicense",
    "0bsd",
    "mpl-2.0",
    "zlib",
    "bsl-1.0",
    "cc0-1.0",
    "cc-by-4.0",
}

# GitHub reports these when it cannot map the repo to a recognised license; treat
# them as "no license detected" rather than a known one.
_UNKNOWN_LICENSES = {"", "noassertion", "other", "none", "null"}

# resource_links "type" values that denote a project/demo page (case-insensitive).
_PROJECT_LINK_TYPES = {"project", "project_page", "page", "website", "homepage", "demo"}


@dataclass(frozen=True)
class ReadinessResult:
    """A paper's implementation-readiness verdict."""

    score: int  # 0-100
    tier: str  # "runnable" | "partial" | "none"
    reasons: list[str]


def _utc_today() -> date:
    from app.services.text import utc_today

    return utc_today()


def _coerce_int(value: object) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return 0


def _clean_str(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _resource_links(paper: object) -> list:
    links = getattr(paper, "resource_links_list", None)
    if links is None:
        links = getattr(paper, "resource_links", None)
    return links if isinstance(links, list) else []


def _to_date(value: object) -> date | None:
    """Coerce a datetime / date / ISO-8601 string to a plain date, else None."""
    if isinstance(value, datetime):  # must precede `date` — datetime subclasses date
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Python 3.10's fromisoformat rejects a trailing 'Z' (GitHub's pushed_at).
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text).date()
        except ValueError:
            try:
                return date.fromisoformat(text[:10])
            except ValueError:
                return None
    return None


def _star_points(stars: int) -> float:
    if stars <= 0:
        return 0.0
    return min(STARS_MAX_POINTS, log1p(stars) / log1p(STARS_LOG_CAP) * STARS_MAX_POINTS)


def _license_points(license_: object) -> tuple[float, str | None]:
    norm = _clean_str(license_).lower()
    if norm in _UNKNOWN_LICENSES:
        return 0.0, None
    if norm in _PERMISSIVE_LICENSES:
        return LICENSE_PERMISSIVE_POINTS, f"Permissive license ({_clean_str(license_)})"
    return LICENSE_KNOWN_POINTS, f"Licensed ({_clean_str(license_)})"


def _repo_age(paper: object, today: date) -> tuple[int | None, bool]:
    """Return ``(age_days, from_push)``.

    ``from_push`` is True when the age came from the repo's own ``pushed_at``
    timestamp; False when it fell back to ``scraped_at`` / ``publication_dt``.
    """
    pushed = _to_date(getattr(paper, "pushed_at", None))
    if pushed is not None:
        return max(0, (today - pushed).days), True
    for attr in ("scraped_at", "publication_dt"):
        stamp = _to_date(getattr(paper, attr, None))
        if stamp is not None:
            return max(0, (today - stamp).days), False
    return None, False


def _freshness_points(age_days: int | None) -> float:
    if age_days is None:
        return 0.0
    return FRESHNESS_MAX_POINTS * exp(-age_days / FRESHNESS_DECAY_DAYS)


def _has_project_page(paper: object) -> bool:
    for link in _resource_links(paper):
        if not isinstance(link, dict):
            continue
        if str(link.get("type") or "").strip().lower() in _PROJECT_LINK_TYPES:
            return True
    return False


def _hf_points(upvotes: int) -> float:
    if upvotes <= 0:
        return 0.0
    return min(HF_MAX_POINTS, upvotes / HF_SATURATION_UPVOTES * HF_MAX_POINTS)


def _tier(score: float) -> str:
    if score >= RUNNABLE_THRESHOLD:
        return "runnable"
    if score >= PARTIAL_THRESHOLD:
        return "partial"
    return "none"


def implementation_readiness(paper: object, *, today: date | None = None) -> ReadinessResult:
    """Score how runnable ``paper``'s implementation is (0-100 + tier + reasons).

    Pure: reads only already-populated attributes (via ``getattr`` so lightweight
    query rows / mocks work too), never touches the DB or network.
    """
    today = today or _utc_today()
    reasons: list[str] = []
    score = 0.0

    repo = _clean_str(getattr(paper, "github_repo", None))
    if repo:
        score += REPO_POINTS
        reasons.append("Has a code repository")

        stars = _coerce_int(getattr(paper, "github_stars", None))
        star_points = _star_points(stars)
        if star_points > 0:
            score += star_points
            reasons.append(f"{stars:,} GitHub stars")

        license_points, license_reason = _license_points(getattr(paper, "github_license", None))
        if license_points > 0 and license_reason:
            score += license_points
            reasons.append(license_reason)

        if getattr(paper, "archived", None):
            reasons.append("Repository archived")
        else:
            score += NOT_ARCHIVED_POINTS

        age_days, from_push = _repo_age(paper, today)
        freshness = _freshness_points(age_days)
        if freshness > 0:
            score += freshness
            if from_push and age_days is not None and age_days <= RECENT_PUSH_DAYS:
                reasons.append("Recently updated")

    # Secondary signals count even without a GitHub repo (but can never, alone,
    # reach the runnable tier — that requires a repo).
    if _has_project_page(paper):
        score += PROJECT_PAGE_POINTS
        reasons.append("Project page")

    hf_upvotes = _coerce_int(getattr(paper, "hf_upvotes", None))
    hf_points = _hf_points(hf_upvotes)
    if hf_points > 0:
        score += hf_points
        reasons.append(f"{hf_upvotes} Hugging Face upvotes")

    score = min(100.0, score)
    return ReadinessResult(score=int(round(score)), tier=_tier(score), reasons=reasons)


def readiness_badge(paper: object, *, today: date | None = None) -> dict[str, str] | None:
    """Return ``{"label", "title"}`` for a runnable paper, else None."""
    result = implementation_readiness(paper, today=today)
    if result.tier != "runnable":
        return None
    detail = ", ".join(result.reasons) or "code available"
    return {"label": "Runnable", "title": f"Runnable — {detail}"}


def readiness_bonus(readiness_score: float | None, weight: float) -> float:
    """Additive ranking bonus: ``weight`` scaled by the 0-100 readiness score.

    Bounded to ``[0, weight]`` so the (intentionally small) weight caps how much
    implementation readiness can ever move a paper's rank.
    """
    if not readiness_score or readiness_score <= 0 or weight <= 0:
        return 0.0
    return weight * min(1.0, readiness_score / 100.0)
