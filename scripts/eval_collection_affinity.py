#!/usr/bin/env python3
"""Offline evaluation of the collection-affinity scorer on an instance's own data.

The scorer makes the literature collections the interest model: one mean-centred
centroid per collection, and a paper's affinity is its best z over the collections
(``app.services.interest_model``). This script fits and scores with those same pure
functions (``fit_collection_profile`` / ``affinity_scores``) and reports three views:

``holdout``
    The newest 20% of each collection (by publication date) are held out and taken
    out of every centroid. The papers in no collection are split in half (seeded): one
    half is the scorer's background, the other the negatives. Per collection it prints
    the AUC of the held-out members against the negatives, ranked by affinity (the
    maximum z over collections, as the feed uses it), the macro average, and at
    z = 2.0 / 2.5 / 3.0 the recall (held-out members with affinity at or above z)
    and the background pass rate (negatives at or above z).

``replay``
    For every past scrape day: is a feed paper that is now in a collection inside
    that day's top 10? A day's candidates are its feed rows (match_type other than
    'import' / 'Bootstrap'); the positives are fitted out of every centroid. Prints
    hit@10 and the positives' ranks for the scorer and, for tuning the interest
    weight only, for the stored ``paper_score`` (rescored since, so not the score of
    that day). ``--until`` keeps to scrape days before a date: once the scorer admits
    papers itself, later days are self-fulfilling. The model and the positives are
    always today's, so the count moves when collections are edited.

``checkpoint``
    Of the papers added to collections since a date, how many the feed had already
    stored. A paper imported before the feed reached it stays an import, so this
    undercounts. Rows an agent wrote over MCP count only once the owner has confirmed
    them.

``--collections`` restricts the model and the papers counted to some collection ids;
papers of every collection stay out of the background either way.

Read-only: the database is opened with ``mode=ro`` and the vectors are read with NumPy
from ``faiss_index/papers.npy`` and ``id_map.json`` (row i belongs to ``id_map[i]``).
No app is created and no embedding model is loaded. No database page is written, but
the app's database is in WAL mode, so SQLite may leave an empty ``-wal`` and a ``-shm``
file next to it when nothing else has it open.

No instance at hand? ``--self-test`` runs all three views on a synthetic three-cluster
corpus and asserts that the scorer separates it.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.services.interest_model import MIN_BACKGROUND, affinity_scores, fit_collection_profile  # noqa: E402
from app.services.learned_ranker import _auc_score, _recall_at_k  # noqa: E402

Z_LEVELS = (2.0, 2.5, 3.0)
NOT_FEED = ("import", "Bootstrap")  # match types of rows the feed did not store


class Corpus(NamedTuple):
    """An instance as plain rows; dates are the ISO text SQLite stores."""

    papers: dict[int, tuple]  # id -> (match_type, scraped_at, publication_dt, paper_score)
    memberships: list[tuple]  # (collection_id, paper_id, added_at, decision, decision_note)
    rows: dict[int, int]  # paper id -> row of ``vectors``
    vectors: np.ndarray


def load(data_dir: Path) -> Corpus:
    index_dir = data_dir / "faiss_index"
    matrix = np.load(index_dir / "papers.npy")
    id_map = json.loads((index_dir / "id_map.json").read_text())
    conn = sqlite3.connect(f"{(data_dir / 'arxiv_papers.db').resolve().as_uri()}?mode=ro", uri=True)
    try:
        papers = {
            row[0]: row[1:]
            for row in conn.execute("SELECT id, match_type, scraped_at, publication_dt, paper_score FROM papers")
        }
        # decision_note marks a row an agent wrote and the owner has not confirmed; a
        # database from before the MCP write tools has no such column.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(paper_collections)")}
        note = "decision_note" if "decision_note" in columns else "NULL"
        member_rows = conn.execute(f"SELECT collection_id, paper_id, added_at, decision, {note} FROM paper_collections")
        memberships = [row for row in member_rows if row[1] in papers]
    finally:
        conn.close()
    return _corpus(papers, memberships, id_map, matrix)


def _corpus(papers: dict[int, tuple], memberships: list[tuple], id_map: list[int], matrix: np.ndarray) -> Corpus:
    # As in the app, only papers that still exist count, and a non-finite row is left out
    # (the corpus mean runs over every row). zip() stops at the shorter of a torn matrix /
    # id map pair.
    finite = np.isfinite(matrix).all(axis=1)
    kept = [(paper_id, row) for row, paper_id in zip(range(len(matrix)), id_map) if paper_id in papers and finite[row]]
    rows = {paper_id: index for index, (paper_id, _row) in enumerate(kept)}
    return Corpus(papers, memberships, rows, matrix[[row for _paper_id, row in kept]])


def _members(corpus: Corpus, collection_ids: set[int] | None) -> dict[int, list[int]]:
    """Indexed in-review members of the selected collections."""
    members: dict[int, list[int]] = {}
    for collection_id, paper_id, _added_at, decision, _note in corpus.memberships:
        selected = collection_ids is None or collection_id in collection_ids
        if selected and decision != "exclude" and paper_id in corpus.rows:
            members.setdefault(collection_id, []).append(paper_id)
    return members


def _background(corpus: Corpus) -> list[int]:
    """The app's background: papers with no row in any collection that its gate did not admit."""
    filed = {paper_id for _collection_id, paper_id, *_rest in corpus.memberships}
    return [pid for pid in corpus.rows if pid not in filed and corpus.papers[pid][0] != "Interest"]


def _take(corpus: Corpus, paper_ids) -> np.ndarray:
    return corpus.vectors[[corpus.rows[pid] for pid in paper_ids]]


def _fit(corpus: Corpus, members: dict[int, list[int]], background: list[int]):
    # ponytail: the corpus mean runs over every paper, held-out members and negatives
    # included (label-free, as in the experiment and in the app). Upgrade: the mean of
    # the fit rows only.
    model = fit_collection_profile(
        {collection_id: _take(corpus, paper_ids) for collection_id, paper_ids in members.items()},
        _take(corpus, background),
        corpus.vectors.mean(axis=0),
    )
    if model is None:
        sys.exit(
            f"no collection profile: the fit had {len(background)} background papers (it needs {MIN_BACKGROUND}; "
            f"holdout fits on half of the papers outside every collection) and {len(members)} collections"
        )
    return model


def _split_holdout(
    corpus: Corpus, collection_ids: set[int] | None
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """(fitted, held out) members per collection; a held-out paper leaves every centroid."""
    members = _members(corpus, collection_ids)
    held = {}
    for collection_id, paper_ids in members.items():
        paper_ids.sort(key=lambda pid: (corpus.papers[pid][2] or "", pid))  # oldest first
        held[collection_id] = paper_ids[int(len(paper_ids) * 0.8) :]
    held_out = {pid for paper_ids in held.values() for pid in paper_ids}
    return {cid: [pid for pid in paper_ids if pid not in held_out] for cid, paper_ids in members.items()}, held


def holdout(corpus: Corpus, collection_ids: set[int] | None = None, seed: int = 0) -> dict:
    fitted, held = _split_holdout(corpus, collection_ids)
    background = _background(corpus)
    order = np.random.default_rng(seed).permutation(len(background))
    half = len(background) // 2
    model = _fit(corpus, fitted, [background[i] for i in order[:half]])
    affinity = affinity_scores(model, corpus.vectors).max(axis=1)  # every paper's best z
    negatives = affinity[[corpus.rows[background[i]] for i in order[half:]]]

    print("collection  fitted  held-out    AUC  " + "  ".join(f"z>={z}" for z in Z_LEVELS))
    aucs, recalls = [], []
    for collection_id in model.collection_ids:
        positives = affinity[[corpus.rows[pid] for pid in held[collection_id]]]
        labels = np.r_[np.ones(len(positives)), np.zeros(len(negatives))]
        aucs.append(_auc_score(labels, np.r_[positives, negatives]))
        recalls.append([float((positives >= z).mean()) for z in Z_LEVELS])
        shares = "  ".join(f"{share:6.2f}" for share in recalls[-1])
        print(f"{collection_id:10d}  {len(fitted[collection_id]):6d}  {len(positives):8d}  {aucs[-1]:.3f}  {shares}")
    macro_auc = float(np.mean(aucs))
    macro_recall = np.mean(recalls, axis=0)
    print(f"macro AUC {macro_auc:.3f} over {len(aucs)} collections, {len(negatives)} negatives (seed {seed})")
    print("recall of held-out members (macro): " + _levels(macro_recall))
    print("background pass rate (negatives):   " + _levels([(negatives >= z).mean() for z in Z_LEVELS]))
    # "recall" is the one at Z_LEVELS[0], the app's admission floor.
    return {"macro_auc": round(macro_auc, 4), "recall": round(float(macro_recall[0]), 4), "collections": len(aucs)}


def _levels(shares) -> str:
    return "  ".join(f"z>={z} {share:.2f}" for z, share in zip(Z_LEVELS, shares))


def replay(corpus: Corpus, collection_ids: set[int] | None = None, until: str | None = None) -> dict:
    feed = [
        pid
        for pid, (match_type, scraped_at, _published, _score) in corpus.papers.items()
        if match_type not in NOT_FEED and pid in corpus.rows and (until is None or scraped_at[:10] < until)
    ]
    members = _members(corpus, collection_ids)
    positives = set(feed) & {pid for paper_ids in members.values() for pid in paper_ids}
    fitted = {cid: [pid for pid in paper_ids if pid not in positives] for cid, paper_ids in members.items()}
    # ponytail: today's members, positives and background, whatever --until says, so
    # the count is not reproducible once collections are edited. Upgrade: fit as of
    # --until, leaving out memberships added and papers scraped on or after it.
    model = _fit(corpus, fitted, _background(corpus))
    scores = {
        "affinity": affinity_scores(model, _take(corpus, feed)).max(axis=1),
        "paper_score": np.asarray([corpus.papers[pid][3] or 0.0 for pid in feed], dtype=np.float64),
    }
    days: dict[str, list[int]] = {}
    for index, pid in enumerate(feed):
        days.setdefault(corpus.papers[pid][1][:10], []).append(index)

    hits = dict.fromkeys(scores, 0)
    for day, indices in sorted(days.items()):
        labels = np.asarray([feed[i] in positives for i in indices], dtype=int)
        if not labels.any():
            continue
        row = f"{day}  {len(indices):4d} rows"
        for name, values in scores.items():
            day_hits = int(round(_recall_at_k(labels, values[indices], k=10) * labels.sum()))
            hits[name] += day_hits
            order = np.argsort(-values[indices], kind="mergesort")  # the order _recall_at_k ranks by
            ranks = ", ".join(str(rank + 1) for rank in np.flatnonzero(labels[order]))
            row += f"  {name} {day_hits}/{labels.sum()} (ranks {ranks})"
        print(row)
    summary = ", ".join(f"{name} {count} of {len(positives)}" for name, count in hits.items())
    print(f"hit@10: {summary}" + (f" (scrape days before {until})" if until else ""))
    return {"positives": len(positives), **hits}


def checkpoint(
    corpus: Corpus, since: str, collection_ids: set[int] | None = None, published_since: str | None = None
) -> dict:
    # A row an agent wrote and the owner has not confirmed (decision_note set) does not
    # count: the agent files what whats_new offers, which the feed stored by construction.
    added = {
        paper_id
        for collection_id, paper_id, added_at, decision, note in corpus.memberships
        if (collection_ids is None or collection_id in collection_ids)
        and decision != "exclude"
        and note is None
        and (added_at or "")[:10] >= since
        and (published_since is None or (corpus.papers[paper_id][2] or "") >= published_since)
    }
    stored = sum(corpus.papers[pid][0] not in NOT_FEED for pid in added)
    published = f", published since {published_since}" if published_since else ""
    print(
        f"{stored} of {len(added)} papers added to collections since {since}{published} were already stored by the feed"
    )
    return {"feed_stored": stored, "added": len(added)}


def self_test(seed: int = 0) -> dict:
    """All three views on synthetic data: three topic clusters inside one common direction."""
    rng = np.random.default_rng(seed)
    dim = 64
    common = rng.normal(size=dim)  # in every paper, each to its own degree: un-centred it drowns the topics
    papers: dict[int, tuple] = {}
    memberships: list[tuple] = []
    vectors = []

    def add(match_type: str, scraped: str, published: str, topic=None) -> int:
        own = rng.normal(size=dim) if topic is None else topic + 0.3 * rng.normal(size=dim)
        vec = rng.uniform(0.5, 4.0) * common + own
        vectors.append((vec / np.linalg.norm(vec)).astype(np.float32))
        papers[len(papers) + 1] = (match_type, f"{scraped} 05:00:00.000000", published, float(rng.uniform(0, 50)))
        return len(papers)

    days = ("2026-02-01", "2026-02-02", "2026-02-03")
    for collection_id in (1, 2, 3):
        topic = rng.normal(size=dim)
        for index in range(24):  # imported members; the last four joined late
            pid = add("import", "2026-01-05", f"2025-{1 + index // 2:02d}-01", topic)
            memberships.append((collection_id, pid, "2026-03-01" if index >= 20 else "2026-01-10", None, None))
        if collection_id == 2:
            # One paper in two collections: this newest import is held out of collection 1,
            # so the split has to take it out of collection 2's centroid as well.
            memberships.append((1, pid, "2026-01-10", None, None))
        for index in range(6):  # members the feed caught, two per scrape day, filed later
            pid = add("Title", days[index % 3], "2026-01-20", topic)
            memberships.append((collection_id, pid, "2026-03-01", "include" if index else None, None))
    # The last of them was an agent's decision, not confirmed by the owner: a member like
    # the others, but the checkpoint leaves it out.
    memberships[-1] = (*memberships[-1][:4], "on-topic")
    # An off-topic feed paper excluded from a collection: neither a member nor background.
    memberships.append((1, add("Title", days[0], "2026-01-20"), "2026-03-01", "exclude", None))
    for day in days:
        for _ in range(150):
            add("Title", day, "2026-01-20")
    # A non-finite row in the index: left out, or it would turn every z into NaN.
    vectors[add("Title", days[0], "2026-01-20") - 1][:] = np.nan
    corpus = _corpus(papers, memberships, list(papers), np.asarray(vectors))

    fitted, held = _split_holdout(corpus, None)
    leaked = {pid for ids in fitted.values() for pid in ids} & {pid for ids in held.values() for pid in ids}
    assert not leaked, f"held-out papers left in the fit set: {sorted(leaked)}"
    result = {
        "holdout": holdout(corpus, seed=seed),
        "replay": replay(corpus, until="2026-02-03"),
        "checkpoint": checkpoint(corpus, "2026-02-15"),
    }
    assert result["holdout"]["macro_auc"] > 0.95, f"scorer failed to separate synthetic clusters: {result}"
    assert result["holdout"]["recall"] > 0.9, f"held-out members below the floor, centring or z lost: {result}"
    assert result["replay"]["positives"] == result["replay"]["affinity"] == 12, f"replay missed positives: {result}"
    assert result["checkpoint"] == {"feed_stored": 17, "added": 29}, f"checkpoint miscounted: {result}"
    return result


def _ids(spec: str) -> set[int]:
    """'1-3' or '1,4,7-9' -> collection ids."""
    ids: set[int] = set()
    for part in spec.split(","):
        first, _, last = part.partition("-")
        ids.update(range(int(first), int(last or first) + 1))
    return ids


def _iso(text: str) -> str:
    return date.fromisoformat(text).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="run on synthetic data, no instance needed")
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--data-dir", type=Path, default=Path("instance"), help="instance directory (default: ./instance)"
    )
    shared.add_argument("--collections", type=_ids, help="collection ids, e.g. 1-3 or 1,4,7-9 (default: all)")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("holdout", parents=[shared]).add_argument("--seed", type=int, default=0)
    commands.add_parser("replay", parents=[shared]).add_argument(
        "--until", type=_iso, help="only scrape days before this date (YYYY-MM-DD)"
    )
    check = commands.add_parser("checkpoint", parents=[shared])
    check.add_argument("--since", type=_iso, required=True, help="memberships added on or after this date")
    check.add_argument("--published-since", type=_iso, help="only papers published on or after this date")
    args = parser.parse_args()

    if args.self_test:
        print(f"self-test OK: {self_test()}")
        return
    if not args.command:
        parser.error("a command is required (or use --self-test)")
    corpus = load(args.data_dir)
    if args.command == "holdout":
        holdout(corpus, args.collections, args.seed)
    elif args.command == "replay":
        replay(corpus, args.collections, args.until)
    else:
        checkpoint(corpus, args.since, args.collections, args.published_since)


if __name__ == "__main__":
    main()
