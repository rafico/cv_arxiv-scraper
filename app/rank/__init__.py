"""Semantic package for ranking, feedback, and preference logic."""

from app.services.feedback import apply_feedback_action, get_feedback_snapshot
from app.services.interest_model import (
    InterestProfile,
    build_interest_profile,
    get_cached_interest_profile,
    recompute_interest_similarities,
    score_vector,
)
from app.services.learned_ranker import (
    LearnedModel,
    evaluate_learned_ranker,
    model_status,
    score_vectors,
    train_learned_ranker,
)
from app.services.matching import MATCH_PRIORITY, check_author_match
from app.services.pipeline import (
    DefaultFeatureExtractor,
    FeatureVector,
    RankedPaper,
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
    "FeatureVector",
    "InterestProfile",
    "LearnedModel",
    "MATCH_PRIORITY",
    "RankedPaper",
    "ScoredCandidate",
    "WeightedSumRanker",
    "WhitelistCandidateGenerator",
    "apply_feedback_action",
    "build_interest_profile",
    "check_author_match",
    "combined_rank_score",
    "evaluate_learned_ranker",
    "compute_feedback_delta",
    "compute_paper_score",
    "explain_score",
    "first_author_name",
    "generate_ranking_explanation",
    "get_cached_interest_profile",
    "get_feedback_snapshot",
    "get_preferences",
    "model_status",
    "recompute_all_paper_scores",
    "recompute_interest_similarities",
    "resolve_ranking_preferences",
    "save_config",
    "score_vectors",
    "score_vector",
    "train_learned_ranker",
    "update_preferences_from_form",
]
