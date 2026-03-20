"""BibTeX generation utilities for paper export."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models import Paper

_LATEX_SPECIAL = re.compile(r"([&%#_{}~^\\])")

_LATEX_REPLACEMENTS = {
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
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


def _format_bibtex_authors(authors_str: str) -> str:
    """Convert comma-separated authors to BibTeX 'and'-separated format.

    ``"Alice Smith, Bob Jones"`` -> ``"Smith, Alice and Jones, Bob"``
    """
    if not authors_str:
        return ""

    authors = [a.strip() for a in authors_str.split(",") if a.strip()]
    formatted = []
    for author in authors:
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
    # Fallback to last segment of link
    return paper.link.rstrip("/").split("/")[-1].replace(".", "_")


def paper_to_bibtex(paper: Paper) -> str:
    """Convert a Paper object to a BibTeX @article entry."""
    cite_key = _make_cite_key(paper)
    authors = _escape_latex(_format_bibtex_authors(paper.authors))
    title = _escape_latex(paper.title)

    fields = [
        f"  author = {{{authors}}}",
        f"  title = {{{title}}}",
    ]

    if paper.abstract_text:
        fields.append(f"  abstract = {{{_escape_latex(paper.abstract_text)}}}")

    if paper.publication_dt:
        fields.append(f"  year = {{{paper.publication_dt.year}}}")
        fields.append(f"  month = {{{paper.publication_dt.month}}}")

    if paper.link:
        fields.append(f"  url = {{{paper.link}}}")

    if paper.arxiv_id:
        fields.append(f"  eprint = {{{paper.arxiv_id}}}")
        fields.append("  archiveprefix = {arXiv}")

    if paper.pdf_link:
        fields.append(f"  pdf = {{{paper.pdf_link}}}")

    field_str = ",\n".join(fields)
    return f"@article{{{cite_key},\n{field_str}\n}}"


def papers_to_bibtex(papers: list[Paper]) -> str:
    """Convert multiple Paper objects to a joined BibTeX string."""
    return "\n\n".join(paper_to_bibtex(p) for p in papers)
