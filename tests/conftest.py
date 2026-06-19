"""Shared pytest configuration."""

import os

# Native-isolation targets (embeddings, PDF rendering) run inline during the suite so
# tests can keep mocking the in-process functions — mocks don't cross a spawned
# process — and to avoid multiprocessing flakiness. Tests that exercise the isolation
# itself opt back in by setting CV_ARXIV_NATIVE_ISOLATION=1.
os.environ["CV_ARXIV_NATIVE_ISOLATION"] = "0"
