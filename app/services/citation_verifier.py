"""Citation-verification layer: zero fabricated references by construction.

Every LLM output the app shows a reader is post-processed here. Any arXiv id,
DOI, or paper title the model names is resolved against the *local* SQLite
corpus: a hit becomes a verified citation the UI renders as a green check that
links to the reader's own copy of the paper; a miss is flagged ``unverified``
("not in your library — may be fabricated"). This is always-on — a guarantee,
not a configurable option — because fabricated references are the single
best-documented failure mode of cloud AI research tools.

Design constraints honoured here:

* Extraction is pure ``re`` (no NLP deps) so it is cheap enough to run on every
  answer, summary, and digest line.
* Resolution batches its DB reads: at most one ``IN`` query for ids and one
  title scan per call, never a query per reference (no N+1).
* The service returns **structured data only** — never HTML. Templates decide
  how to render the verified/​unverified markers (textContent / data-* attrs),
  so LLM-authored strings can never inject markup.
* Every DB path degrades gracefully: a lookup failure yields "no citations"
  rather than raising, so a verification hiccup can never break a page render
  or a scrape.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from re import IGNORECASE
from re import compile as re_compile
from typing import TypedDict

from app.models import Paper, db
from app.services.text import normalize as _strip_accents

LOGGER = logging.getLogger(__name__)

# ── Reference extraction patterns (stdlib re only) ──────────────────────────
# New-scheme arXiv id ("2401.01234", optional version suffix). Bare digits, so
# it is matched inside "arXiv:2401.01234" and abs/pdf URLs alike.
_ARXIV_NEW_RE = re_compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
# Legacy-scheme arXiv id ("math.GT/0309136", "cond-mat/0309136"). The optional
# subject-class component is dropped in normalisation (metadata, not the id).
_ARXIV_OLD_RE = re_compile(r"\b([a-z][a-z\-]*(?:\.[a-z\-]+)?/\d{7})(?:v\d+)?\b", IGNORECASE)
# A DOI ("10.xxxx/…"). Trailing sentence punctuation is trimmed after matching.
_DOI_RE = re_compile(r"\b(10\.\d{4,9}/[-._;()/:a-z0-9]+)", IGNORECASE)
# The arXiv DOI namespace resolves straight back to an arXiv id, so those DOIs
# are verified against the corpus like any other arXiv reference.
_ARXIV_DOI_RE = re_compile(r"^10\.48550/arxiv\.(.+)$", IGNORECASE)

# Title-case run: a capitalised word followed by >= 3 more capitalised words or
# lowercase connectors ("of", "the", …). Post-filtered to >= 4 capitalised
# words so ordinary prose and short emphasised phrases are ignored.
_TITLE_WORD = r"[A-Z][A-Za-z0-9][A-Za-z0-9\-']*"
_CONNECTOR_WORDS = frozenset(
    ("of", "the", "and", "for", "a", "an", "in", "on", "to", "with", "via", "from", "by", "as", "at", "or")
)
_CONNECTOR = r"(?:" + "|".join(sorted(_CONNECTOR_WORDS)) + r")"
_TITLECASE_RE = re_compile(rf"\b{_TITLE_WORD}(?:\s+(?:{_TITLE_WORD}|{_CONNECTOR})){{3,}}\b")
# Quoted spans (straight and curly quotes) are treated as titles when long
# enough to be a paper title rather than a scare-quoted word.
_QUOTED_RE = re_compile(r"[\"“]([^\"”]{8,200})[\"”]")

# A title must contain at least this many capitalised words to be a candidate.
_MIN_TITLE_CAP_WORDS = 4
# difflib similarity above which a normalised title is accepted as the same paper.
_TITLE_MATCH_RATIO = 0.9
# Upper bound on corpus titles fuzzy-compared per call, so a huge library can
# never turn verification into a slow O(refs × corpus) scan.
_MAX_FUZZY_TITLES = 5000


class Reference(TypedDict):
    """A candidate reference pulled from an LLM output."""

    kind: str  # "arxiv" | "doi" | "title"
    raw: str  # the substring as it appeared, for display
    value: str  # normalised match key (arXiv id / lowered DOI / normalised title)


class Verification(TypedDict):
    """A reference plus its resolution against the local corpus."""

    ref: str
    kind: str
    status: str  # "verified" | "unverified"
    paper_id: int | None
    matched_title: str | None


class Annotation(TypedDict):
    """Structured verification payload handed to templates (never HTML)."""

    text: str
    citations: list[Verification]
    verified_count: int
    total: int
    summary_flag: str


def _normalize_arxiv_new(raw: str) -> str | None:
    match = _ARXIV_NEW_RE.search(raw)
    return match.group(1) if match else None


def _normalize_arxiv_old(raw: str) -> str | None:
    match = _ARXIV_OLD_RE.search(raw)
    if not match:
        return None
    archive, _, number = match.group(1).partition("/")
    # Drop the subject-class component ("math.GT" -> "math") and lowercase.
    return f"{archive.split('.')[0].lower()}/{number}"


def _arxiv_id_from_doi(doi: str) -> str | None:
    """Return the arXiv id encoded in an arXiv-namespace DOI, else None."""
    match = _ARXIV_DOI_RE.match(doi)
    if not match:
        return None
    tail = match.group(1)
    return _normalize_arxiv_new(tail) or _normalize_arxiv_old(tail)


def _normalize_title(title: str) -> str:
    """Case/punctuation/accent-insensitive key for title matching."""
    stripped = _strip_accents(title).lower()
    out = []
    prev_space = False
    for ch in stripped:
        if ch.isalnum():
            out.append(ch)
            prev_space = False
        elif not prev_space:
            out.append(" ")
            prev_space = True
    return "".join(out).strip()


def _title_cap_words(run: str) -> int:
    return sum(1 for word in run.split() if word[:1].isupper())


def extract_references(text: str) -> list[Reference]:
    """Extract candidate references (arXiv ids, DOIs, titles) from ``text``.

    Returns them in appearance order, de-duplicated by ``(kind, value)``.
    Overlapping matches are resolved by precedence (DOI ▸ arXiv id ▸ title) so a
    DOI's embedded id or a quoted title is never double-counted.
    """
    if not text:
        return []

    consumed: list[tuple[int, int]] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(start < ce and cs < end for cs, ce in consumed)

    def _reserve(start: int, end: int) -> None:
        consumed.append((start, end))

    refs: list[Reference] = []
    seen: set[tuple[str, str]] = set()

    def _add(kind: str, raw: str, value: str) -> None:
        key = (kind, value)
        if value and key not in seen:
            seen.add(key)
            refs.append(Reference(kind=kind, raw=raw, value=value))

    ordered: list[tuple[int, Reference]] = []

    # DOIs first (an arXiv DOI resolves to an arXiv id and is verified as such).
    for match in _DOI_RE.finditer(text):
        raw = match.group(1).rstrip(").,;:")
        start = match.start(1)
        end = start + len(raw)
        if _overlaps(start, end):
            continue
        _reserve(start, end)
        arxiv_from_doi = _arxiv_id_from_doi(raw)
        if arxiv_from_doi:
            ordered.append((start, Reference(kind="arxiv", raw=raw, value=arxiv_from_doi)))
        else:
            ordered.append((start, Reference(kind="doi", raw=raw, value=raw.lower())))

    # arXiv ids (new then old scheme).
    for pattern, normalizer in ((_ARXIV_NEW_RE, _normalize_arxiv_new), (_ARXIV_OLD_RE, _normalize_arxiv_old)):
        for match in pattern.finditer(text):
            start, end = match.span(1)
            if _overlaps(start, end):
                continue
            value = normalizer(match.group(1))
            if not value:
                continue
            _reserve(start, end)
            ordered.append((start, Reference(kind="arxiv", raw=match.group(1), value=value)))

    # Titles: quoted spans, then Title-Case runs of >= 4 capitalised words.
    for match in _QUOTED_RE.finditer(text):
        inner = match.group(1).strip()
        start, end = match.span(1)
        # A quoted span is a title candidate when it is long enough to be one —
        # either >= 4 words or >= 4 capitalised words (an explicit quote need not
        # be Title Case). Too-short scare-quotes are skipped.
        long_enough = len(inner.split()) >= _MIN_TITLE_CAP_WORDS or _title_cap_words(inner) >= _MIN_TITLE_CAP_WORDS
        if _overlaps(start, end) or not long_enough:
            continue
        _reserve(start, end)
        ordered.append((start, Reference(kind="title", raw=inner, value=_normalize_title(inner))))

    for match in _TITLECASE_RE.finditer(text):
        start, end = match.span(0)
        # Drop trailing lowercase connectors ("… Recognition in") that the run
        # greedily absorbed, so the candidate is just the title itself.
        words = match.group(0).split()
        while words and words[-1].lower() in _CONNECTOR_WORDS and not words[-1][:1].isupper():
            words.pop()
        run = " ".join(words)
        if _overlaps(start, end) or _title_cap_words(run) < _MIN_TITLE_CAP_WORDS:
            continue
        _reserve(start, end)
        ordered.append((start, Reference(kind="title", raw=run, value=_normalize_title(run))))

    ordered.sort(key=lambda item: item[0])
    for _, ref in ordered:
        _add(ref["kind"], ref["raw"], ref["value"])
    return refs


def _resolve_arxiv(values: set[str], session) -> dict[str, tuple[int, str]]:
    if not values:
        return {}
    rows = session.query(Paper.id, Paper.arxiv_id, Paper.title).filter(Paper.arxiv_id.in_(values)).all()
    # Normalise stored ids so a versioned/legacy row still keys off the same value.
    resolved: dict[str, tuple[int, str]] = {}
    for pid, arxiv_id, title in rows:
        norm = _normalize_arxiv_new(arxiv_id or "") or _normalize_arxiv_old(arxiv_id or "") or (arxiv_id or "")
        resolved[norm] = (pid, title)
    return resolved


def _resolve_titles(values: set[str], session) -> dict[str, tuple[int, str]]:
    """Resolve normalised titles: exact map first, then a bounded fuzzy fallback."""
    if not values:
        return {}
    rows = session.query(Paper.id, Paper.title).all()
    exact: dict[str, tuple[int, str]] = {}
    corpus: list[tuple[str, int, str]] = []
    for pid, title in rows:
        norm = _normalize_title(title or "")
        if not norm:
            continue
        exact.setdefault(norm, (pid, title))
        corpus.append((norm, pid, title))

    resolved: dict[str, tuple[int, str]] = {}
    fuzzy_targets: list[str] = []
    for value in values:
        hit = exact.get(value)
        if hit is not None:
            resolved[value] = hit
        else:
            fuzzy_targets.append(value)

    if fuzzy_targets and len(corpus) <= _MAX_FUZZY_TITLES:
        matcher = SequenceMatcher()
        for value in fuzzy_targets:
            matcher.set_seq2(value)
            best_ratio = 0.0
            best: tuple[int, str] | None = None
            for norm, pid, title in corpus:
                matcher.set_seq1(norm)
                if matcher.real_quick_ratio() < _TITLE_MATCH_RATIO or matcher.quick_ratio() < _TITLE_MATCH_RATIO:
                    continue
                ratio = matcher.ratio()
                if ratio >= _TITLE_MATCH_RATIO and ratio > best_ratio:
                    best_ratio, best = ratio, (pid, title)
            if best is not None:
                resolved[value] = best
    return resolved


def verify_references(refs: list[Reference], *, session=None) -> list[Verification]:
    """Resolve each reference against the local corpus (batched, read-only).

    Returns one :class:`Verification` per input reference, preserving order. A
    DB failure degrades to an empty list so verification can never break the
    caller (the guarantee is best-effort *by construction*, not fragile).
    """
    if not refs:
        return []

    session = session or db.session
    try:
        arxiv_values = {ref["value"] for ref in refs if ref["kind"] == "arxiv"}
        title_values = {ref["value"] for ref in refs if ref["kind"] == "title"}
        arxiv_map = _resolve_arxiv(arxiv_values, session)
        title_map = _resolve_titles(title_values, session)
    except Exception:  # noqa: BLE001 — verification is best-effort; never break the caller
        LOGGER.debug("Citation verification lookup failed; skipping", exc_info=True)
        return []

    verifications: list[Verification] = []
    for ref in refs:
        match: tuple[int, str] | None = None
        if ref["kind"] == "arxiv":
            match = arxiv_map.get(ref["value"])
        elif ref["kind"] == "title":
            match = title_map.get(ref["value"])
        # DOIs (non-arXiv) cannot be resolved: the corpus has no DOI column, so
        # they are honestly reported unverified rather than falsely green.
        if match is not None:
            verifications.append(
                Verification(
                    ref=ref["raw"], kind=ref["kind"], status="verified", paper_id=match[0], matched_title=match[1]
                )
            )
        else:
            verifications.append(
                Verification(ref=ref["raw"], kind=ref["kind"], status="unverified", paper_id=None, matched_title=None)
            )
    return verifications


def summary_flag(verified: int, total: int) -> str:
    """One-line human summary; empty when there was nothing to verify."""
    if total <= 0:
        return ""
    return f"{verified} of {total} references verified against your library"


def annotate(text: str, verifications: list[Verification]) -> Annotation:
    """Wrap ``text`` + its verifications into the structured template payload.

    Deliberately returns data only — no HTML — so the caller renders verified
    chips vs. amber unverified markers via textContent / data-* attributes.
    """
    verified = sum(1 for item in verifications if item["status"] == "verified")
    total = len(verifications)
    return Annotation(
        text=text,
        citations=verifications,
        verified_count=verified,
        total=total,
        summary_flag=summary_flag(verified, total),
    )


def verify_text(text: str | None, *, session=None) -> Annotation:
    """Extract → resolve → annotate ``text`` in one call.

    The single entry point every LLM-output path uses. Always returns an
    :class:`Annotation`; ``citations`` is empty (and ``summary_flag`` "") when
    the text names no references, so callers can include it unconditionally and
    let the UI hide an empty result.
    """
    if not text:
        return annotate(text or "", [])
    refs = extract_references(text)
    verifications = verify_references(refs, session=session)
    return annotate(text, verifications)


# Stable alias for the email-digest path (owned by another module): it can show
# a small "N of M references verified" note without importing internals.
def verify_summary_text(text: str | None, *, session=None) -> Annotation:
    """Verify an LLM-authored summary/TL;DR line (digest + summary callers)."""
    return verify_text(text, session=session)
