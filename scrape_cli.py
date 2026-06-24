"""Backward-compatible wrapper for the installable scrape CLI."""

from app._module_alias import alias_module as _alias_module

_mod = _alias_module(__name__, "app.cli.scrape")

if __name__ == "__main__":
    raise SystemExit(_mod.main())
