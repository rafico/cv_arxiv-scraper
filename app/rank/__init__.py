"""Semantic package for ranking, feedback, and preference logic."""

from app.services.feedback import apply_feedback_action, get_feedback_snapshot
from app.services.matching import MATCH_PRIORITY, check_author_match
from app.services.metrics import (
    compute_author_follow_hit_rate,
    compute_mean_time_to_first_open_hours,
    compute_precision_at_k,
)
from app.services.pipeline import (
    DefaultFeatureExtractor,
    FeatureExtractor,
    FeatureVector,
    RankedPaper,
    Ranker,
    ScoredCandidate,
    WeightedSumRanker,
    WhitelistCandidateGenerator,
)
from app.services.preferences import (
    DEFAULT_PREFERENCES,
    first_author_name,
    get_preferences,
    save_config,
    update_preferences_from_form,
)
from app.services.ranking import (
    FEEDBACK_BOOST,
    combined_rank_score,
    compute_feedback_delta,
    compute_paper_score,
    explain_score,
    generate_ranking_explanation,
    recompute_all_paper_scores,
    resolve_ranking_preferences,
)

__all__ = [
    "DEFAULT_PREFERENCES",
    "DefaultFeatureExtractor",
    "FEEDBACK_BOOST",
    "FeatureExtractor",
    "FeatureVector",
    "MATCH_PRIORITY",
    "RankedPaper",
    "Ranker",
    "ScoredCandidate",
    "WeightedSumRanker",
    "WhitelistCandidateGenerator",
    "apply_feedback_action",
    "check_author_match",
    "combined_rank_score",
    "compute_author_follow_hit_rate",
    "compute_feedback_delta",
    "compute_paper_score",
    "compute_precision_at_k",
    "compute_mean_time_to_first_open_hours",
    "explain_score",
    "first_author_name",
    "generate_ranking_explanation",
    "get_feedback_snapshot",
    "get_preferences",
    "recompute_all_paper_scores",
    "resolve_ranking_preferences",
    "save_config",
    "update_preferences_from_form",
]
