"""BibTeX generation utilities for paper export."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models import Paper

_LATEX_SPECIAL = re.compile(r"([&%#$_{}~^\\])")

_LATEX_REPLACEMENTS = {
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "$": r"\$",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in text."""
    return _LATEX_SPECIAL.sub(lambda m: _LATEX_REPLACEMENTS[m.group(1)], text)


def _split_authors(authors_str: str) -> list[str]:
    """Split an authors string into individual names.

    Prefer unambiguous separators when present — ``" and "`` (already BibTeX-style) or
    ``";"`` — because a bare comma is ambiguous: it separates authors in a
    ``"First Last, First Last"`` list but ALSO appears inside a single
    ``"Last, First"`` / ``"Last, Jr."`` name. We only fall back to comma-splitting
    when neither unambiguous separator is present, which is the common arXiv format.
    """
    if " and " in authors_str:
        return [a.strip() for a in authors_str.split(" and ") if a.strip()]
    if ";" in authors_str:
        return [a.strip() for a in authors_str.split(";") if a.strip()]
    return [a.strip() for a in authors_str.split(",") if a.strip()]


def _format_bibtex_authors(authors_str: str) -> str:
    """Convert an authors string to BibTeX 'and'-separated ``Last, First`` format.

    ``"Alice Smith, Bob Jones"`` -> ``"Smith, Alice and Jones, Bob"``. Names that
    already contain a comma (``"Smith, Alice"``) or are mononyms are passed through.
    """
    if not authors_str:
        return ""

    formatted = []
    for author in _split_authors(authors_str):
        if "," in author:
            # Already in "Last, First" form (came via a ';'/' and ' separated list).
            formatted.append(author)
            continue
        parts = author.rsplit(None, 1)
        if len(parts) == 2:
            formatted.append(f"{parts[1]}, {parts[0]}")
        else:
            formatted.append(author)
    return " and ".join(formatted)


def _make_cite_key(paper: Paper) -> str:
    """Generate a BibTeX cite key from the arxiv_id or paper link."""
    if paper.arxiv_id:
        return paper.arxiv_id.replace(".", "_").replace("/", "_")
    # Fallback to last segment of link, sanitized: a non-arXiv link can contain
    # characters illegal in a BibTeX key (spaces, '?', '&', '#', …) — or be empty —
    # which would emit a malformed/un-citable @article{...} entry.
    segment = paper.link.rstrip("/").split("/")[-1] if paper.link else ""
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", segment).strip("_")
    return sanitized or f"paper_{paper.id}"


# Venue statuses that mean the paper was actually accepted (see app/services/venues.py).
_ACCEPTED_STATUSES = ("accepted", "oral", "spotlight", "highlight")
_JOURNAL_VENUES = ("TPAMI", "IJCV")


def paper_to_bibtex(paper: Paper) -> str:
    """Convert a Paper to BibTeX: an arXiv ``@misc`` preprint by default, or
    ``@inproceedings``/``@article`` once an accepted venue was detected."""
    cite_key = _make_cite_key(paper)
    authors = _escape_latex(_format_bibtex_authors(paper.authors))
    title = _escape_latex(paper.title)

    fields = [
        f"  author = {{{authors}}}",
        f"  title = {{{title}}}",
    ]

    entry_type = "misc"
    year = paper.publication_dt.year if paper.publication_dt else None
    # ponytail: the venue is regex-parsed from the arXiv comment, so a false positive
    # lands in the bibliography; add a per-paper override if that bites.
    if paper.venue and paper.acceptance_status in _ACCEPTED_STATUSES:
        entry_type, venue_field = (
            ("article", "journal") if paper.venue in _JOURNAL_VENUES else ("inproceedings", "booktitle")
        )
        fields.append(f"  {venue_field} = {{{_escape_latex(paper.venue)}}}")
        year = paper.venue_year or year

    if paper.abstract_text:
        fields.append(f"  abstract = {{{_escape_latex(paper.abstract_text)}}}")

    if year:
        fields.append(f"  year = {{{year}}}")
    if paper.publication_dt and year == paper.publication_dt.year:
        fields.append(f"  month = {{{paper.publication_dt.month}}}")

    if paper.link:
        # Escape the URL too: an unescaped '}'/'\'/'%' in a link closes the field early
        # or comments out the rest of the line, corrupting this and following entries.
        fields.append(f"  url = {{{_escape_latex(paper.link)}}}")

    if paper.arxiv_id:
        arxiv_id = _escape_latex(paper.arxiv_id)
        fields.append(f"  eprint = {{{arxiv_id}}}")
        fields.append("  archiveprefix = {arXiv}")
        if paper.categories_list:
            fields.append(f"  primaryclass = {{{_escape_latex(paper.categories_list[0])}}}")
        # The arXiv DOI only on the preprint entry: on a venue entry it's a mixed citation.
        if entry_type == "misc":
            fields.append(f"  doi = {{10.48550/arXiv.{arxiv_id}}}")
        fields.append(f"  note = {{arXiv:{arxiv_id}}}")

    if paper.pdf_link:
        fields.append(f"  pdf = {{{_escape_latex(paper.pdf_link)}}}")

    field_str = ",\n".join(fields)
    return f"@{entry_type}{{{cite_key},\n{field_str}\n}}"


def papers_to_bibtex(papers: list[Paper]) -> str:
    """Convert multiple Paper objects to a joined BibTeX string."""
    return "\n\n".join(paper_to_bibtex(p) for p in papers)
