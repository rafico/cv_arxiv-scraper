"""Feedback action handling for local paper triage loop."""

from __future__ import annotations

from collections import defaultdict

from app.models import Paper, PaperFeedback, db
from app.services.ranking import combined_rank_score, compute_feedback_delta
from app.enums import FeedbackAction

ALLOWED_ACTIONS = {action.value for action in FeedbackAction}


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

        # "Not interested" and "Save" are mutually exclusive.
        if action == FeedbackAction.SKIP.value:
            save_row = rows_by_action.get(FeedbackAction.SAVE.value)
            if save_row:
                db.session.delete(save_row)
                delta -= compute_feedback_delta(FeedbackAction.SAVE.value)
        elif action == FeedbackAction.SAVE.value:
            skip_row = rows_by_action.get(FeedbackAction.SKIP.value)
            if skip_row:
                db.session.delete(skip_row)
                delta -= compute_feedback_delta(FeedbackAction.SKIP.value)

    paper.feedback_score = max(-100, min(100, int(paper.feedback_score or 0) + delta))

    db.session.flush()

    rows = _load_feedback_rows(paper_id)
    counts = {FeedbackAction.SAVE.value: 0, FeedbackAction.SKIP.value: 0}
    active_actions = []
    for row in rows:
        counts[row.action] = counts.get(row.action, 0) + 1
        active_actions.append(row.action)

    paper.is_hidden = FeedbackAction.SKIP.value in active_actions
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
            "counts": {FeedbackAction.SAVE.value: 0, FeedbackAction.SKIP.value: 0},
            "active_actions": set(),
        }
    )
    for paper_id, action, count in rows:
        snapshot[paper_id]["counts"][action] = int(count)
        snapshot[paper_id]["active_actions"].add(action)

    return dict(snapshot)
