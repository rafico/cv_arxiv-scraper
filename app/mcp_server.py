"""Thin MCP (Model Context Protocol) wrapper exposing the corpus as a backend.

This module is deliberately import-light: **the ``mcp`` SDK is imported lazily
inside :func:`run`** (via :func:`_load_fastmcp`), never at module load. That keeps
the core app installable without the optional ``mcp`` extra — ``import
app.mcp_server`` pulls in zero third-party MCP code — and lets
:mod:`app.services.mcp_tools` (all the real logic) stay unit-testable on its own.

Serve it over stdio (the Claude Desktop default) with the ``cv-arxiv-mcp``
console script. Data-directory resolution mirrors ``cv-arxiv serve`` exactly
(:mod:`app.cli.serve`) so the MCP server and the web server address the very same
SQLite DB / FAISS index / config, and never touch ``instance/`` unsafely.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from app.services import mcp_tools

if TYPE_CHECKING:  # import only for type checkers; never at runtime
    from flask import Flask

SERVER_NAME = "cv-arxiv"

_MISSING_SDK_MESSAGE = (
    "The MCP server requires the optional 'mcp' extra, which is not installed.\n"
    "Install it with:\n\n    pip install 'cv-arxiv-scraper[mcp]'\n\n"
    "(from a source checkout: pip install '.[mcp]')"
)


def _load_fastmcp() -> Any:
    """Import ``FastMCP`` lazily, re-raising a clear, actionable error if missing."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError as exc:  # the extra isn't installed
        raise ModuleNotFoundError(_MISSING_SDK_MESSAGE) from exc
    return FastMCP


def build_server(app: Flask, fastmcp_cls: Any | None = None) -> Any:
    """Build a ``FastMCP`` server exposing the corpus tools, bound to ``app``.

    Each tool runs inside ``app.app_context()`` so the logic layer can hit the DB.
    ``fastmcp_cls`` is injectable for tests; production loads it lazily.
    """
    fastmcp = fastmcp_cls or _load_fastmcp()
    server = fastmcp(SERVER_NAME)

    @server.tool(
        name="search_papers",
        description=(
            "Search your local arXiv corpus. mode: 'hybrid' (BM25+semantic, default), "
            "'semantic' (embeddings), or 'keyword' (substring). Returns compact hits."
        ),
    )
    def search_papers(query: str, mode: str = "hybrid", limit: int = 10) -> dict[str, Any]:
        with app.app_context():
            return mcp_tools.search_papers(query, mode=mode, limit=limit)

    @server.tool(
        name="get_paper",
        description="Fetch one paper's full metadata plus its citation/readiness/enrichment "
        "summary, by numeric id or arXiv id.",
    )
    def get_paper(arxiv_id_or_id: str) -> dict[str, Any]:
        with app.app_context():
            return mcp_tools.get_paper(arxiv_id_or_id)

    @server.tool(
        name="get_summary",
        description="Return the stored TL;DR summary and structured LLM insights for one paper.",
    )
    def get_summary(paper_id: str) -> dict[str, Any]:
        with app.app_context():
            return mcp_tools.get_summary(paper_id)

    @server.tool(
        name="top_ranked_today",
        description="Today's top-ranked fresh papers, ordered like the dashboard feed. "
        "Optional 'profile' (id/slug/name) selects the interest-profile lens.",
    )
    def top_ranked_today(limit: int = 10, profile: str | None = None) -> dict[str, Any]:
        with app.app_context():
            return mcp_tools.top_ranked_today(limit=limit, profile=profile)

    @server.tool(
        name="list_collections",
        description="List all collections with their paper counts.",
    )
    def list_collections() -> dict[str, Any]:
        with app.app_context():
            return mcp_tools.list_collections()

    @server.tool(
        name="ask_paper",
        description="Grounded question answering over one paper's own text, with section "
        "citations. Degrades to top sections when no LLM is configured.",
    )
    def ask_paper(paper_id: str, question: str) -> dict[str, Any]:
        with app.app_context():
            return mcp_tools.ask_paper(paper_id, question)

    # The single mutation, clearly separated from the read tools above.
    @server.tool(
        name="add_to_collection",
        description="Add a paper to a collection (created on first use if a name). Idempotent: "
        "returns added=false when the paper is already a member.",
    )
    def add_to_collection(collection_name_or_id: str, paper_id: str) -> dict[str, Any]:
        with app.app_context():
            return mcp_tools.add_to_collection(collection_name_or_id, paper_id)

    return server


def run(argv: Sequence[str] | None = None) -> int:
    """Bootstrap the Flask app against the resolved data dir and serve over stdio.

    Mirrors ``cv-arxiv serve``: resolves the data directory (``--data-dir`` >
    ``$CV_ARXIV_DATA_DIR`` > default), points the app env vars at it, seeds a
    config on first run, then serves the MCP tools. Raises ``ModuleNotFoundError``
    with an actionable message when the ``mcp`` extra is absent.
    """
    import argparse

    from app.cli.serve import prepare_data_dir, resolve_data_dir

    parser = argparse.ArgumentParser(prog="cv-arxiv-mcp", add_help=True)
    parser.add_argument("--data-dir", default=None, help="Data directory (DB, FAISS index, config, secrets).")
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Load the SDK first so a missing extra fails fast, before app bootstrap.
    fastmcp_cls = _load_fastmcp()

    data_dir = resolve_data_dir(args.data_dir)
    prepare_data_dir(data_dir)

    from app import create_app

    app = create_app()
    server = build_server(app, fastmcp_cls)
    server.run()  # stdio transport (Claude Desktop default)
    return 0
