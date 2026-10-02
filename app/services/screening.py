"""Writing screening decisions: the one place a membership's decision and its agent mark change."""

from __future__ import annotations

from app.models import PaperCollection, db


def apply_decision(
    collection_id: int,
    paper_ids: list[int],
    decision: str | None,
    *,
    note: str | None = None,
    create: bool = False,
) -> dict[int, str | None]:
    """Set ``decision`` on the memberships of ``paper_ids`` in a collection; the caller commits.

    ``decision_note`` is written with it, so the two never drift apart: an agent passes
    its reason as ``note``, and an owner write leaves it at None, which clears the mark
    and so confirms (or overrules) what an agent did.

    Papers without a membership are skipped, or filed with ``decision`` when ``create``
    is set (the caller has checked that they exist). Returns the previous decision of
    every paper that already had a row: its length is the number of rows updated, and
    a paper missing from it was filed (or skipped).
    """
    rows = PaperCollection.query.filter(
        PaperCollection.collection_id == collection_id, PaperCollection.paper_id.in_(paper_ids)
    ).all()
    previous = {row.paper_id: row.decision for row in rows}
    if create:
        filed = [
            PaperCollection(paper_id=paper_id, collection_id=collection_id)
            for paper_id in dict.fromkeys(paper_ids)
            if paper_id not in previous
        ]
        db.session.add_all(filed)
        rows += filed
    for row in rows:
        row.decision, row.decision_note = decision, note
    return previous
