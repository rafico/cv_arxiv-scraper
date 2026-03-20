"""Feedback action handling for local paper triage loop."""

from __future__ import annotations

from collections import defaultdict

from app.models import Paper, PaperFeedback, db
from app.services.ranking import combined_rank_score, compute_feedback_delta

ALLOWED_ACTIONS = {"upvote", "save", "skip"}


def _load_feedback_rows(paper_id: int) -> list[PaperFeedback]:
    return PaperFeedback.query.filter_by(paper_id=paper_id).all()


def apply_feedback_action(paper_id: int, action: str) -> dict:
    """Toggle a feedback action and return updated ranking metadata."""
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Unsupported action '{action}'")

    paper = db.session.get(Paper, paper_id)
    if not paper:
        raise LookupError(f"Paper {paper_id} not found")

    rows_by_action = {row.action: row for row in _load_feedback_rows(paper_id)}
    existing = rows_by_action.get(action)
    delta = 0
    active = False

    if existing:
        db.session.delete(existing)
        delta -= compute_feedback_delta(action)
    else:
        db.session.add(PaperFeedback(paper_id=paper_id, action=action))
        delta += compute_feedback_delta(action)
        active = True

        # "Not interested" should be exclusive with positive signals so papers
        # don't end up simultaneously hidden and saved/upvoted.
        if action == "skip":
            for conflicting_action in ("upvote", "save"):
                row = rows_by_action.get(conflicting_action)
                if row:
                    db.session.delete(row)
                    delta -= compute_feedback_delta(conflicting_action)
        elif action in {"upvote", "save"}:
            skip_row = rows_by_action.get("skip")
            if skip_row:
                db.session.delete(skip_row)
                delta -= compute_feedback_delta("skip")

    paper.feedback_score = max(-100, min(100, int(paper.feedback_score or 0) + delta))

    db.session.commit()

    rows = _load_feedback_rows(paper_id)
    counts = {"upvote": 0, "save": 0, "skip": 0}
    active_actions = []
    for row in rows:
        counts[row.action] = counts.get(row.action, 0) + 1
        active_actions.append(row.action)

    paper.is_hidden = "skip" in active_actions
    db.session.commit()

    return {
        "paper_id": paper.id,
        "action": action,
        "active": active,
        "counts": counts,
        "active_actions": active_actions,
        "feedback_score": int(paper.feedback_score or 0),
        "rank_score": combined_rank_score(float(paper.paper_score or 0.0), int(paper.feedback_score or 0)),
    }


def get_feedback_snapshot(paper_ids: list[int]) -> dict[int, dict]:
    """Return aggregated counts and active actions for paper cards."""
    if not paper_ids:
        return {}

    rows = (
        db.session.query(PaperFeedback.paper_id, PaperFeedback.action, db.func.count(PaperFeedback.id))
        .filter(PaperFeedback.paper_id.in_(paper_ids))
        .group_by(PaperFeedback.paper_id, PaperFeedback.action)
        .all()
    )

    snapshot: dict[int, dict] = defaultdict(
        lambda: {
            "counts": {"upvote": 0, "save": 0, "skip": 0},
            "active_actions": set(),
        }
    )
    for paper_id, action, count in rows:
        snapshot[paper_id]["counts"][action] = int(count)
        snapshot[paper_id]["active_actions"].add(action)

    return dict(snapshot)
