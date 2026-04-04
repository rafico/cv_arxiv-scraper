"""Semantic package for search, semantic indexing, and corpus analysis."""

from app.services.corpus_analysis import analyze_topic_clusters, detect_emerging_topics, find_neighbor_papers
from app.services.embed_backfill import backfill_embeddings
from app.services.embeddings import EmbeddingService, get_embedding_service, reset_embedding_service
from app.services.export import generate_html_report
from app.services.pdf_extraction import extract_and_store_sections
from app.services.related import build_vector, cosine_similarity, find_duplicates, top_related_papers
from app.services.saved_search import execute_saved_search, validate_saved_search
from app.services.search import RRF_K, search_bm25, search_hybrid, search_semantic
from app.services.summary import extract_topic_tags, generate_llm_summary, generate_summary
from app.services.text import STOP_WORDS, clean_whitespace, normalize, now_utc, tokenize, utc_today
from app.services.thumbnail_generator import generate_thumbnail

__all__ = [
    "EmbeddingService",
    "RRF_K",
    "STOP_WORDS",
    "analyze_topic_clusters",
    "backfill_embeddings",
    "build_vector",
    "clean_whitespace",
    "cosine_similarity",
    "detect_emerging_topics",
    "execute_saved_search",
    "extract_and_store_sections",
    "extract_topic_tags",
    "find_duplicates",
    "find_neighbor_papers",
    "generate_html_report",
    "generate_llm_summary",
    "generate_summary",
    "generate_thumbnail",
    "get_embedding_service",
    "normalize",
    "now_utc",
    "reset_embedding_service",
    "search_bm25",
    "search_hybrid",
    "search_semantic",
    "tokenize",
    "top_related_papers",
    "utc_today",
    "validate_saved_search",
]
