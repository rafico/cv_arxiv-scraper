"""Feature-liveness diagnostics computed from local data."""

from __future__ import annotations

from app.models import Paper, PaperFeedback, db


def feature_liveness() -> dict:
    """Which shipped features actually have data flowing. Never raises: {} on failure.

    Health checks answered "is it up", never "is anything using it", so the feed ran
    for weeks with zero full-text sections, zero feedback and zero delivered digests
    while every probe stayed green. These counters make inert-but-healthy visible;
    they are diagnostics only and must not gate liveness.

    ponytail: plain COUNTs over a single-user corpus (hundreds to low thousands of
    rows), so no caching. Past ~100k papers the match_type scan is the first thing
    to index or memoize.
    """
    try:
        from sqlalchemy import distinct, func

        from app.models import DigestRun, PaperSection
        from app.services.interest_model import MIN_POSITIVE_FEEDBACK, POSITIVE_ACTIONS

        positive = int(PaperFeedback.query.filter(PaperFeedback.action.in_(POSITIVE_ACTIONS)).count())
        latest_digest = DigestRun.query.order_by(DigestRun.started_at.desc()).first()
        return {
            "papers": int(Paper.query.count()),
            # Full text behind per-paper chat, corpus chat and citation verification.
            "sections_papers": int(db.session.query(func.count(distinct(PaperSection.paper_id))).scalar() or 0),
            # The gate on the centroid profile, the learned ranker and whitelist-free
            # dense-retrieval admission: all three stay inert below the threshold.
            "positive_feedback": positive,
            "positive_feedback_needed": max(0, MIN_POSITIVE_FEEDBACK - positive),
            "interest_signal_ready": positive >= MIN_POSITIVE_FEEDBACK,
            # Papers admitted by the interest model rather than a whitelist match.
            "dense_retrieval_papers": int(Paper.query.filter(Paper.match_type.like("%Interest%")).count()),
            "last_digest_status": getattr(latest_digest, "status", None),
            "last_digest_error": getattr(latest_digest, "error_message", None),
            "citations_papers": int(Paper.query.filter(Paper.citation_count.isnot(None)).count()),
            "openalex_papers": int(Paper.query.filter(Paper.openalex_id.isnot(None)).count()),
        }
    except Exception:  # noqa: BLE001 — diagnostics must never break their caller
        return {}
