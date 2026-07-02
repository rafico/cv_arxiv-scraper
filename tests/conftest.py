"""Shared pytest configuration."""

import atexit
import os
import shutil
import tempfile

# Native-isolation targets (embeddings, PDF rendering) run inline during the suite so
# tests can keep mocking the in-process functions — mocks don't cross a spawned
# process — and to avoid multiprocessing flakiness. Tests that exercise the isolation
# itself opt back in by setting CV_ARXIV_NATIVE_ISOLATION=1.
os.environ["CV_ARXIV_NATIVE_ISOLATION"] = "0"

# Sandbox the default instance dir for the whole test session so NO test can read or
# write the developer's real instance/arxiv_papers.db (+ FAISS index / secrets). This
# matters because entrypoints like wsgi.py call create_app() with no arguments — e.g.
# test_wsgi imports wsgi — which would otherwise bind to the real instance dir and run
# create_all()/migrations against real data. create_app() honours this env var when no
# explicit INSTANCE_PATH override is passed; explicit overrides still win.
_SANDBOX_INSTANCE = tempfile.mkdtemp(prefix="cv_arxiv_test_instance_")
os.environ.setdefault("CV_ARXIV_INSTANCE_PATH", _SANDBOX_INSTANCE)
atexit.register(shutil.rmtree, _SANDBOX_INSTANCE, ignore_errors=True)
