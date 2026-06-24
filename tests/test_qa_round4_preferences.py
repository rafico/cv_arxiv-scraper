"""QA round 4 regression tests for app.services.preferences.

Covers finding G6: get_preferences() coerced the per-key muted value with a bare
list(...) call, which (1) exploded a scalar string into single characters and
(2) raised an uncaught TypeError on a non-iterable scalar (e.g. int), crashing
app startup via _validate_config. The fix routes the value through _as_str_list.
"""

from __future__ import annotations

import unittest

from app.services.preferences import get_preferences


class MutedListCoercionTests(unittest.TestCase):
    """Finding G6: muted per-key values must be coerced like whitelists."""

    def test_g6_scalar_string_muted_author_is_one_element_not_chars(self):
        config = {"preferences": {"muted": {"authors": "Jane Doe"}}}
        prefs = get_preferences(config)
        # Before the fix list("Jane Doe") exploded into ['J', 'a', 'n', ...].
        self.assertEqual(prefs["muted"]["authors"], ["Jane Doe"])

    def test_g6_non_iterable_scalar_muted_topic_does_not_raise(self):
        config = {"preferences": {"muted": {"topics": 5}}}
        # Before the fix list(5) raised TypeError, crashing get_preferences
        # (and therefore _validate_config / create_app at startup).
        prefs = get_preferences(config)
        self.assertEqual(prefs["muted"]["topics"], [])

    def test_g6_proper_list_muted_value_is_preserved(self):
        config = {"preferences": {"muted": {"affiliations": ["MIT", "MIT", " Stanford ", "", 7]}}}
        prefs = get_preferences(config)
        self.assertEqual(prefs["muted"]["affiliations"], ["MIT", "Stanford"])


if __name__ == "__main__":
    unittest.main()
