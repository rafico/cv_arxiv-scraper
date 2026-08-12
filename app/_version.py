"""Single source of truth for the package version.

Imported by ``/healthz`` and the ``cv-arxiv --version`` CLI, and read statically
by setuptools (``[tool.setuptools.dynamic]`` in ``pyproject.toml``) so the wheel
metadata and the running app never drift. Keep this module import-free (a bare
string assignment) so setuptools can parse it without importing the ``app``
package (whose runtime deps are absent in an isolated build environment).
"""

__version__ = "0.5.0"
