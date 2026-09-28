"""Local citation graph: resolve stored reference ids into PaperRelation
"cites" edges, and rank papers with PageRank.

referenced_works holds ids from two namespaces — OpenAlex "W…" ids and
Semantic Scholar 40-char hex paperIds (S2 covers fresh preprints months
before OpenAlex parses their references) — resolved against Paper.openalex_id
and Paper.semantic_scholar_id respectively; the namespaces cannot collide.

Edges only ever connect papers already in the local DB — references to
papers we don't track are simply skipped (and picked up automatically on a
later sync once the cited paper is scraped, since referenced_works persists
the full id list).
"""

from __future__ import annotations

import logging
import re

LOGGER = logging.getLogger(__name__)

CITES = "cites"
_S2_PAPER_ID = re.compile(r"[0-9a-f]{40}")


def sync_citation_edges() -> int:
    """Recompute "cites" edges from every paper's referenced_works list.

    Idempotent: inserts only missing pairs, never duplicates (and never
    deletes — edges are derived data, but removal only happens when a paper
    row is deleted, via the FK cascade).

    Returns the number of edges inserted.

    # ponytail: full-corpus recompute per call; scope to new papers if it
    # ever shows up in scrape time.
    """
    from app.models import Paper, PaperRelation, db

    rows = db.session.query(Paper.id, Paper.openalex_id, Paper.semantic_scholar_id, Paper.referenced_works).all()
    ref_to_paper = {oa_id: pid for pid, oa_id, _, _ in rows if oa_id}
    ref_to_paper.update({s2_id: pid for pid, _, s2_id, _ in rows if s2_id})
    existing = {
        (citing, cited)
        for citing, cited in db.session.query(PaperRelation.paper_id, PaperRelation.related_paper_id).filter(
            PaperRelation.relation_type == CITES
        )
    }

    inserted = 0
    for pid, _, _, refs in rows:
        for ref in refs or []:
            cited_id = ref_to_paper.get(ref)
            if cited_id is None or cited_id == pid or (pid, cited_id) in existing:
                continue
            db.session.add(PaperRelation(paper_id=pid, related_paper_id=cited_id, relation_type=CITES))
            existing.add((pid, cited_id))
            inserted += 1
    if inserted:
        db.session.commit()
    return inserted


def missing_references(paper_ids: list[int], limit: int = 15, request_fn=None) -> dict:
    """Outside works cited by >= 2 of ``paper_ids``: the prior works they build on.

    Counts stored Semantic Scholar paperIds only (OpenAlex W-ids aren't S2
    resolvable), resolves the top ids in one S2 /paper/batch call, and drops
    ones already local (by S2 id, then by the resolved arXiv id). Any failure
    degrades to an error payload, never raises.

    # ponytail: uncached live S2 call per click; lru_cache it if clicked a lot.
    """
    from collections import Counter

    from app.models import Paper, db
    from app.services.enrichment_providers.semantic_scholar import SEMANTIC_SCHOLAR_BATCH_URL
    from app.services.http_client import request_with_backoff
    from app.services.secret_files import resolve_data_source_key

    rows = db.session.query(Paper.referenced_works).filter(Paper.id.in_(paper_ids)).all()
    counts = Counter(ref for (refs,) in rows for ref in set(refs or []) if _S2_PAPER_ID.fullmatch(ref))
    local = {s2 for (s2,) in db.session.query(Paper.semantic_scholar_id).filter(Paper.semantic_scholar_id.isnot(None))}
    # ponytail: resolve the top 100 so the citationCount tie-break is exact for
    # typical collections (count >= 2 leaves ~1-10); raise toward 500 if cut.
    top = [ref for ref, n in counts.most_common() if n >= 2 and ref not in local][:100]
    if not top:
        return {"results": []}

    try:
        api_key = resolve_data_source_key("semantic_scholar")
        # Single web worker: one attempt, short timeout.
        response = (request_fn or request_with_backoff)(
            "POST",
            SEMANTIC_SCHOLAR_BATCH_URL,
            json={"ids": top},
            params={"fields": "title,year,externalIds,citationCount"},
            attempts=1,
            timeout=10,
            headers={"x-api-key": api_key} if api_key else None,
        )
        results = []
        for ref, item in zip(top, response.json()):
            if not item:
                continue
            arxiv_id = (item.get("externalIds") or {}).get("ArXiv")
            results.append(
                {
                    "s2_id": ref,
                    "title": item.get("title") or "",
                    "year": item.get("year"),
                    "arxiv_id": arxiv_id,
                    "citation_count": item.get("citationCount"),
                    "cited_by": counts[ref],
                    # Built from the validated id, not S2's url field: no untrusted hrefs.
                    "link": f"https://arxiv.org/abs/{arxiv_id}"
                    if arxiv_id
                    else f"https://www.semanticscholar.org/paper/{ref}",
                }
            )
        # Papers seeded by arXiv id (import-ids, sync --query, Add) have no S2 id
        # until a later scrape, so the S2-id filter above misses them.
        arxiv_ids = [r["arxiv_id"] for r in results if r["arxiv_id"]]
        have = {a for (a,) in db.session.query(Paper.arxiv_id).filter(Paper.arxiv_id.in_(arxiv_ids))}
        results = [r for r in results if r["arxiv_id"] not in have]
        results.sort(key=lambda r: (r["cited_by"], r["citation_count"] or 0), reverse=True)
    except Exception as exc:
        LOGGER.warning("Prior-works lookup failed: %s", exc)
        return {"results": [], "error": "Semantic Scholar unavailable"}
    return {"results": results[:limit]}


def pagerank(
    node_ids: list[int],
    edges: list[tuple[int, int]],
    damping: float = 0.85,
    iters: int = 30,
) -> dict[int, float]:
    """PageRank over a citation subgraph via sparse power iteration.

    O(E) per iteration using bincount — no adjacency matrix, no networkx.
    Dangling nodes' mass is redistributed uniformly, the standard treatment.
    """
    import numpy as np

    n = len(node_ids)
    if n == 0:
        return {}
    index = {pid: i for i, pid in enumerate(node_ids)}
    pairs = [(index[a], index[b]) for a, b in edges if a in index and b in index and a != b]

    rank = np.full(n, 1.0 / n)
    if pairs:
        src = np.array([p[0] for p in pairs])
        dst = np.array([p[1] for p in pairs])
        outdeg = np.bincount(src, minlength=n).astype(float)
        for _ in range(iters):
            contrib = np.bincount(dst, weights=rank[src] / outdeg[src], minlength=n)
            dangling = rank[outdeg == 0].sum() / n
            rank = (1 - damping) / n + damping * (contrib + dangling)
    return {pid: float(rank[i]) for pid, i in index.items()}
