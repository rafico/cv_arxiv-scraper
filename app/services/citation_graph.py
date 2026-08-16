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

CITES = "cites"


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

    rows = db.session.query(
        Paper.id, Paper.openalex_id, Paper.semantic_scholar_id, Paper.referenced_works
    ).all()
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
