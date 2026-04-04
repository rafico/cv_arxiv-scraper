"""Compatibility alias for ingest orchestration helpers."""

from app._module_alias import alias_module as _alias_module

_alias_module(__name__, "app.services.ingest.orchestrator")
