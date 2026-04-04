"""Helpers for lightweight backwards-compatible module aliases."""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType


def alias_module(alias_name: str, target_name: str) -> ModuleType:
    module = import_module(target_name)
    sys.modules[alias_name] = module
    return module
