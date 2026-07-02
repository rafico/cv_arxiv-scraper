"""Feedback action handling for local paper triage loop."""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy.exc import IntegrityError

from app.enums import FeedbackAction
from app.models import Paper, PaperFeedback, db
from app.services.ranking import compute_feedback_delta

ALLOWED_ACTIONS = {action.value for action in FeedbackAction}

# Negative actions that clear positive ones
_NEGATIVE_ACTIONS = {FeedbackAction.SKIP.value, FeedbackAction.IGNORE.value}
# Strong positive actions that clear negative ones
_POSITIVE_ACTIONS = {FeedbackAction.SAVE.value, FeedbackAction.PRIORITY.value}

# Marker stored on a SAVE row that was auto-added because the user prioritized a
# paper. Lets un-prioritizing remove that implied save while leaving a save the
# user added explicitly (which carries no such marker).
_IMPLIED_BY_PRIORITY = "implied_by_priority"

# Bounded retries for the read-modify-write commit when a concurrent request
# collides on the (paper_id, action) unique constraint.
_MAX_COMMIT_RETRIES = 3


def _load_feedback_rows(paper_id: int) -> list[PaperFeedback]:
    return PaperFeedback.query.filter_by(paper_id=paper_id).all()


def apply_feedback_action(
    paper_id: int,
    action: str,
    *,
    reason: str | None = None,
    note: str | None = None,
) -> dict:
    """Toggle a feedback action and return updated ranking metadata.

    The toggle is a read-modify-write on rows guarded by a unique (paper_id,
    action) constraint plus paper.feedback_score. Concurrent requests (gthread,
    e.g. a double-clicked "save") can race: two inserts of the same row collide
    on commit. We retry on IntegrityError — after rollback the operation re-reads
    all state fresh, so the retry sees the now-existing row and toggles
    idempotently, and the score delta is recomputed against the current value
    (composing instead of clobbering the other request's update).
    """
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Unsupported action '{action}'")

    for attempt in range(_MAX_COMMIT_RETRIES):
        try:
            return _apply_feedback_action_once(paper_id, action, reason=reason, note=note)
        except IntegrityError:
            db.session.rollback()
            if attempt == _MAX_COMMIT_RETRIES - 1:
                raise
    # Unreachable: the loop either returns or raises on the final attempt.
    raise RuntimeError("apply_feedback_action retry loop exited unexpectedly")


def _apply_feedback_action_once(
    paper_id: int,
    action: str,
    *,
    reason: str | None = None,
    note: str | None = None,
) -> dict:
    """Perform one attempt of the feedback toggle; may raise IntegrityError on commit."""
    paper = db.session.get(Paper, paper_id)
    if not paper:
        raise LookupError(f"Paper {paper_id} not found")

    rows_by_action = {row.action: row for row in _load_feedback_rows(paper_id)}
    existing = rows_by_action.get(action)
    delta = 0
    active = False

    if existing and action == FeedbackAction.SAVE.value and existing.reason == _IMPLIED_BY_PRIORITY:
        # Clicking "save" on a paper whose save was auto-added by prioritizing it
        # promotes the implied save to an explicit one (clears the marker) instead
        # of toggling it off — otherwise a still-prioritized paper would be left
        # with no save row, breaking the "priority implies save" invariant.
        existing.reason = None
        active = True
    elif existing:
        db.session.delete(existing)
        delta -= compute_feedback_delta(action)
        # Un-prioritizing removes the save that priority implied, but never a save
        # the user made explicitly (only the implied one carries the marker).
        if action == FeedbackAction.PRIORITY.value:
            implied_save = rows_by_action.get(FeedbackAction.SAVE.value)
            if implied_save is not None and implied_save.reason == _IMPLIED_BY_PRIORITY:
                db.session.delete(implied_save)
                delta -= compute_feedback_delta(FeedbackAction.SAVE.value)
    else:
        fb = PaperFeedback(paper_id=paper_id, action=action)
        # Never let a client-supplied reason masquerade as the internal
        # priority-implied marker — it drives cascade-deletes on un-prioritize, so
        # an explicit save carrying that string would be silently removed later.
        if reason and reason != _IMPLIED_BY_PRIORITY:
            fb.reason = reason
        if note:
            fb.note = note
        db.session.add(fb)
        delta += compute_feedback_delta(action)
        active = True

        # Mutual exclusivity rules:
        if action in _NEGATIVE_ACTIONS:
            # Skip/ignore clear save/priority
            for pos_action in _POSITIVE_ACTIONS:
                pos_row = rows_by_action.get(pos_action)
                if pos_row:
                    db.session.delete(pos_row)
                    delta -= compute_feedback_delta(pos_action)
        elif action in _POSITIVE_ACTIONS:
            # Save/priority clear skip/ignore
            for neg_action in _NEGATIVE_ACTIONS:
                neg_row = rows_by_action.get(neg_action)
                if neg_row:
                    db.session.delete(neg_row)
                    delta -= compute_feedback_delta(neg_action)
            # Priority implies save
            if action == FeedbackAction.PRIORITY.value:
                if FeedbackAction.SAVE.value not in rows_by_action:
                    db.session.add(
                        PaperFeedback(
                            paper_id=paper_id,
                            action=FeedbackAction.SAVE.value,
                            reason=_IMPLIED_BY_PRIORITY,
                        )
                    )
                    delta += compute_feedback_delta(FeedbackAction.SAVE.value)
        # Additive actions (shared, skimmed) don't conflict with anything

    paper.feedback_score = max(-100, min(100, int(paper.feedback_score or 0) + delta))

    db.session.flush()

    rows = _load_feedback_rows(paper_id)
    counts = {a.value: 0 for a in FeedbackAction}
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
        "rank_score": paper.rank_score,
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
            "counts": {a.value: 0 for a in FeedbackAction},
            "active_actions": set(),
        }
    )
    for paper_id, action, count in rows:
        snapshot[paper_id]["counts"][action] = int(count)
        snapshot[paper_id]["active_actions"].add(action)

    # Convert sets to lists for JSON serialization safety.
    for entry in snapshot.values():
        entry["active_actions"] = list(entry["active_actions"])

    return dict(snapshot)
