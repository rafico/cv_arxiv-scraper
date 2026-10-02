"""Backward-compatible wrapper for the web server launcher (``cv-arxiv serve``)."""

from app._module_alias import alias_module as _alias_module

_mod = _alias_module(__name__, "app.cli.webserver")

if __name__ == "__main__":
    raise SystemExit(_mod.main())
