"""Backward-compatible wrapper for the installable digest CLI."""

from app._module_alias import alias_module as _alias_module

_alias_module(__name__, "app.cli.digest")

if __name__ == "__main__":
    from app.cli.digest import main

    main()
