"""Screening decisions on collection memberships: the one writer of the agent mark (``decision_note``)."""

from __future__ import annotations

from sqlalchemy.orm.attributes import flag_modified

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
    for row in rows:
        row.decision, row.decision_note = decision, note
        # Both columns go into the UPDATE, also the one that did not change here. The
        # other process (web or MCP) may have rewritten the row since it was read, and
        # an UPDATE of the changed column alone would pair this value with theirs.
        flag_modified(row, "decision")
        flag_modified(row, "decision_note")
    if create:
        db.session.add_all(
            PaperCollection(paper_id=paper_id, collection_id=collection_id, decision=decision, decision_note=note)
            for paper_id in dict.fromkeys(paper_ids)
            if paper_id not in previous
        )
    return previous
