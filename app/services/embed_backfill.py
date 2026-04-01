"""Backfill SPECTER2 embeddings for existing papers missing from the FAISS index."""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


def backfill_embeddings(app, batch_size: int = 64) -> int:
    """Generate embeddings for all papers not yet in the FAISS index. Returns count added."""
    from app.models import Paper
    from app.services.embeddings import get_embedding_service

    service = get_embedding_service(app)
    total_added = 0

    with app.app_context():
        offset = 0
        while True:
            papers = Paper.query.order_by(Paper.id).offset(offset).limit(batch_size).all()
            if not papers:
                break

            paper_ids = []
            texts = []
            for paper in papers:
                if not service.has_paper(paper.id):
                    paper_ids.append(paper.id)
                    texts.append(f"{paper.title} {paper.abstract_text or ''}")

            if paper_ids:
                added = service.add_papers(paper_ids, texts)
                total_added += added
                LOGGER.info("Backfill batch: indexed %d papers (total so far: %d)", added, total_added)

            offset += batch_size

    if total_added > 0:
        service.save()
        LOGGER.info("Backfill complete: %d new embeddings saved", total_added)

    return total_added
