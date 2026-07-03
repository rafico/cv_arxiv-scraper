"""Grounded per-paper Q&A ("Ask this paper").

Agentic retrieval loop over ONE paper's own extracted sections (PaperQA2
pattern, in-house): retrieve the paper's chunks ranked against the question,
score each surviving chunk's evidence with a bounded LLM call, then synthesize
an answer that may cite ONLY the supplied excerpts as ``[1]``, ``[2]``…

Grounding guarantee: every ``[n]`` in the final answer is post-processed —
markers that do not map to a supplied excerpt are stripped and reported.

Degradation mirrors :mod:`app.services.rag`: with no LLM client (disabled, no
key) the top-ranked sections plus similarity scores are returned with
``answer=None``; an upstream LLM failure is reported via ``llm_error`` so the
route can answer with an honest 502. Papers without extracted sections fall
back to abstract-only answering (``abstract_only=True``).
"""

from __future__ import annotations

import logging
import math
import re

from app.models import Paper, PaperSection, db
from app.services.rag import build_llm_client as _build_client

LOGGER = logging.getLogger(__name__)

# Retrieval: chunks considered for evidence scoring (the abstract is always kept).
_TOP_CHUNKS = 8
# Evidence: chunks surviving the per-chunk LLM relevance gate.
_EVIDENCE_KEEP = 5
_EVIDENCE_MIN_SCORE = 3.0
_EVIDENCE_WORKERS = 4

# Char budgets keep prompts bounded regardless of section length.
_ENCODE_CHAR_BUDGET = 2000
_PROMPT_CHUNK_CHARS = 1500
_QUOTE_CHAR_BUDGET = 300
_SNIPPET_CHAR_BUDGET = 240
_HISTORY_TURNS = 4
_HISTORY_CHAR_BUDGET = 400

_NOT_ADDRESSED = "The paper does not address this."

_EVIDENCE_SYSTEM_PROMPT = (
    "You judge whether an excerpt from a research paper is relevant to a reader's question. "
    'Respond with STRICT JSON only, no prose: {"score": number, "quote": string}. '
    "score: 0-10 relevance of the excerpt to answering the question (0 = irrelevant, 10 = directly answers it). "
    "quote: ONE short verbatim sentence copied from the excerpt that best supports a potential answer, "
    'or "" if nothing in the excerpt is relevant.'
)

_SYNTHESIS_SYSTEM_PROMPT = (
    "You answer a reader's question about ONE research paper using ONLY the numbered excerpts provided. "
    "After each claim, cite the supporting excerpt number in square brackets, e.g. [1] or [2][3]. "
    "Use only citation numbers that exist in the excerpt list, and no outside knowledge. "
    f'If the excerpts do not contain enough information to answer, reply exactly: "{_NOT_ADDRESSED}" '
    "Keep the answer under 200 words."
)

_CITATION_RE = re.compile(r"\[(\d{1,3})\]")


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _paper_chunks(paper: Paper) -> list[dict]:
    """The paper's retrievable chunks: abstract first, then each extracted section."""
    chunks: list[dict] = []
    abstract = (paper.abstract_text or paper.summary_text or "").strip()
    if abstract:
        chunks.append({"section_type": "abstract", "order_index": -1, "text": abstract, "score": None})
    for section in paper.sections.order_by(PaperSection.order_index).all():
        text = (section.text or "").strip()
        if text:
            chunks.append(
                {
                    "section_type": section.section_type,
                    "order_index": section.order_index,
                    "text": text,
                    "score": None,
                }
            )
    return chunks


def _rank_chunks(question: str, chunks: list[dict], top_k: int = _TOP_CHUNKS) -> list[dict]:
    """Rank the paper's chunks against the question by embedding cosine similarity.

    The section FAISS index has no per-paper id filtering, so the handful of
    chunks are embedded on the fly (vectors are L2-normalized, dot = cosine).
    Any failure (model unavailable, encode error) degrades to document order.
    The abstract is always included so there is context even when section
    retrieval misfires.
    """
    if not chunks:
        return []
    try:
        from app.services.embeddings import get_embedding_service

        service = get_embedding_service()
        vectors = service.encode([question] + [_truncate(c["text"], _ENCODE_CHAR_BUDGET) for c in chunks])
        query_vec = vectors[0]
        for chunk, vec in zip(chunks, vectors[1:]):
            chunk["score"] = float(vec @ query_vec)
        ranked = sorted(chunks, key=lambda c: c["score"], reverse=True)
    except Exception:  # noqa: BLE001 — retrieval is best-effort; degrade to document order
        LOGGER.debug("Section ranking unavailable; falling back to document order", exc_info=True)
        ranked = list(chunks)

    top = ranked[:top_k]
    abstract = next((c for c in chunks if c["section_type"] == "abstract"), None)
    if abstract is not None and abstract not in top:
        top = top[: top_k - 1] + [abstract]
    return top


def _score_one_chunk(client, question: str, chunk: dict) -> tuple[float, str] | None:
    """One evidence-scoring LLM call. Returns (score, quote) or None on failure."""
    import json

    from app.services.llm_client import _strip_code_fences

    user_prompt = (
        f"Question: {question}\n\n"
        f"Excerpt from the paper's {chunk['section_type']} section:\n"
        f"{_truncate(chunk['text'], _PROMPT_CHUNK_CHARS)}"
    )
    content = client.complete_text(
        system_prompt=_EVIDENCE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=150,
        temperature=0.0,
    )
    if not content:
        return None
    try:
        data = json.loads(_strip_code_fences(content))
        score = float(data.get("score"))
    except (ValueError, TypeError, KeyError, AttributeError):
        # Malformed / non-JSON model output; treat as a failed call.
        return None
    if math.isnan(score):
        # A NaN would survive the min/max clamp below as fully "relevant".
        return None
    score = max(0.0, min(10.0, score))
    quote = data.get("quote") if isinstance(data, dict) else ""
    quote = _truncate(quote, _QUOTE_CHAR_BUDGET) if isinstance(quote, str) else ""
    return score, quote


def _score_evidence(client, question: str, chunks: list[dict]) -> tuple[list[dict], bool]:
    """The agentic step: per-chunk LLM relevance gate.

    Returns ``(evidence, degraded)``. Each evidence dict is the chunk plus
    ``evidence_score``/``quote``. When every call fails, falls back to
    retrieval order (``degraded=True``) instead of losing the question.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not chunks:
        return [], False

    # Bounded fan-out; the client's own max_concurrent semaphore still caps
    # actual in-flight requests across the whole process.
    workers = max(1, min(len(chunks), _EVIDENCE_WORKERS))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda chunk: _score_one_chunk(client, question, chunk), chunks))

    if all(result is None for result in results):
        fallback = [{**chunk, "evidence_score": None, "quote": ""} for chunk in chunks[:_EVIDENCE_KEEP]]
        return fallback, True

    scored = [
        {**chunk, "evidence_score": result[0], "quote": result[1]}
        for chunk, result in zip(chunks, results)
        if result is not None and result[0] >= _EVIDENCE_MIN_SCORE
    ]
    scored.sort(key=lambda item: item["evidence_score"], reverse=True)
    return scored[:_EVIDENCE_KEEP], False


def _sanitize_history(history) -> list[dict]:
    """Keep only well-formed recent turns; the payload is caller-controlled JSON."""
    if not isinstance(history, list):
        return []
    turns = []
    for item in history:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        answer = item.get("answer")
        if isinstance(question, str) and question.strip() and isinstance(answer, str) and answer.strip():
            turns.append(
                {
                    "question": _truncate(question, _HISTORY_CHAR_BUDGET),
                    "answer": _truncate(answer, _HISTORY_CHAR_BUDGET),
                }
            )
    return turns[-_HISTORY_TURNS:]


def _synthesize_answer(client, paper: Paper, question: str, evidence: list[dict], history: list[dict]) -> str | None:
    """Single synthesis call over the surviving evidence. None on any failure."""
    lines = [f"Paper title: {paper.title}", ""]
    if history:
        lines.append("Earlier conversation about this paper:")
        for turn in history:
            lines.append(f"Q: {turn['question']}")
            lines.append(f"A: {turn['answer']}")
        lines.append("")
    lines.append("Excerpts:")
    for n, item in enumerate(evidence, start=1):
        header = f"[{n}] ({item['section_type']})"
        if item.get("quote"):
            header += f' key sentence: "{item["quote"]}"'
        lines.append(header)
        lines.append(_truncate(item["text"], _PROMPT_CHUNK_CHARS))
        lines.append("")
    lines.append(f"Question: {question}")
    return client.complete_text(
        system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
        user_prompt="\n".join(lines),
        max_tokens=500,
        temperature=0.2,
    )


def _ground_answer(answer: str, valid_count: int) -> tuple[str, list[int], list[int]]:
    """Enforce the grounding guarantee on ``[n]`` markers.

    Returns ``(clean_answer, cited, stripped)`` where ``cited`` is the sorted
    set of citation numbers that map to a supplied excerpt and ``stripped`` the
    fabricated ones removed from the text.
    """
    stripped: list[int] = []

    def _replace(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= valid_count:
            return match.group(0)
        stripped.append(n)
        return ""

    clean = _CITATION_RE.sub(_replace, answer)
    clean = re.sub(r"[ \t]{2,}", " ", clean).strip()
    cited = sorted({int(n) for n in _CITATION_RE.findall(clean) if 1 <= int(n) <= valid_count})
    return clean, cited, sorted(set(stripped))


def _sections_payload(chunks: list[dict]) -> list[dict]:
    return [
        {
            "section_type": chunk["section_type"],
            "order_index": chunk["order_index"],
            "score": round(chunk["score"], 6) if chunk["score"] is not None else None,
            "snippet": _truncate(chunk["text"], _SNIPPET_CHAR_BUDGET),
        }
        for chunk in chunks
    ]


def answer_paper_question(paper_id: int, question: str, history=None, *, app=None) -> dict:
    """Answer ``question`` grounded in one paper's own indexed text.

    Returns ``{paper_id, question, answer, citations, sections, llm_used,
    degraded, llm_error, abstract_only, evidence_degraded, stripped_citations}``.
    ``answer`` is None (``degraded=True``) when no LLM client is available;
    ``llm_error=True`` additionally signals a client that existed but failed,
    so the API can surface an honest 502.
    """
    paper = db.session.get(Paper, paper_id)
    if paper is None:
        raise ValueError("Paper not found")

    chunks = _paper_chunks(paper)
    abstract_only = not any(chunk["order_index"] >= 0 for chunk in chunks)
    top_chunks = _rank_chunks(question, chunks)

    result = {
        "paper_id": paper.id,
        "question": question,
        "answer": None,
        "citations": [],
        "sections": _sections_payload(top_chunks),
        "llm_used": False,
        "degraded": True,
        "llm_error": False,
        "abstract_only": abstract_only,
        "evidence_degraded": False,
        "stripped_citations": [],
    }

    if not top_chunks:
        # Nothing to ground in (no abstract, no sections): honest empty result.
        return result

    client = _build_client(app=app)
    if client is None:
        return result

    evidence, evidence_degraded = _score_evidence(client, question, top_chunks)
    result["evidence_degraded"] = evidence_degraded

    if not evidence:
        # Every chunk was confidently scored irrelevant — answer deterministically
        # rather than inviting the model to speculate without evidence.
        result.update({"answer": _NOT_ADDRESSED, "llm_used": True, "degraded": False})
        return result

    raw_answer = _synthesize_answer(client, paper, question, evidence, _sanitize_history(history))
    if not raw_answer:
        result["llm_error"] = True
        return result

    answer, cited, stripped = _ground_answer(raw_answer, len(evidence))
    result.update(
        {
            "answer": answer,
            "llm_used": True,
            "degraded": False,
            "stripped_citations": stripped,
            "citations": [
                {
                    "n": n,
                    "section_type": item["section_type"],
                    "order_index": item["order_index"],
                    "quote": item.get("quote") or "",
                    "snippet": _truncate(item["text"], _SNIPPET_CHAR_BUDGET),
                    "evidence_score": item.get("evidence_score"),
                    "cited": n in cited,
                }
                for n, item in enumerate(evidence, start=1)
            ],
        }
    )
    return result
