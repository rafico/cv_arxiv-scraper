"""Feedback action handling for local paper triage loop."""

from __future__ import annotations

from collections import defaultdict

from app.models import Paper, PaperFeedback, db
from app.services.ranking import combined_rank_score, compute_feedback_delta

ALLOWED_ACTIONS = {"upvote", "save", "skip"}



def apply_feedback_action(paper_id: int, action: str) -> dict:
    """Toggle a feedback action and return updated ranking metadata."""
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Unsupported action '{action}'")

    paper = db.session.get(Paper, paper_id)
    if not paper:
        raise LookupError(f"Paper {paper_id} not found")

    existing = PaperFeedback.query.filter_by(paper_id=paper_id, action=action).first()
    delta = compute_feedback_delta(action)
    active = False

    if existing:
        db.session.delete(existing)
        delta *= -1
    else:
        db.session.add(PaperFeedback(paper_id=paper_id, action=action))
        active = True

    paper.feedback_score = max(-100, min(100, int(paper.feedback_score or 0) + delta))

    if action == "skip":
        paper.is_hidden = active
    elif not PaperFeedback.query.filter_by(paper_id=paper_id, action="skip").first():
        paper.is_hidden = False

    db.session.commit()

    rows = PaperFeedback.query.filter_by(paper_id=paper_id).all()
    counts = {"upvote": 0, "save": 0, "skip": 0}
    active_actions = []
    for row in rows:
        counts[row.action] = counts.get(row.action, 0) + 1
        active_actions.append(row.action)
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
