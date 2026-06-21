"""Tests for ``pear.verifiers``.

Pure-function answer verifiers; should run in milliseconds.
"""

from __future__ import annotations

import pytest

from pear.verifiers import (
    verify_anls,
    verify_exact,
    verify_mc,
    verify_numeric,
    verify,
)


# ---------------------------------------------------------------- mc

class TestMC:
    @pytest.mark.parametrize("pred,gold,want", [
        ("answer: A", "A", True),
        ("(C)",       "C", True),
        ("The answer is B.", "B", True),
        ("answer = D",        "D", True),
        ("E",                 "E", True),
        ("a",                 "A", True),
        ("",                  "A", False),
        ("None of the above", "A", False),
        ("answer: A", "B", False),
    ])
    def test_letter_extraction(self, pred, gold, want):
        assert verify_mc(pred, gold) is want

    def test_handles_gold_with_punctuation(self):
        assert verify_mc("answer: C", "(C)") is True


# ----------------------------------------------------------------- numeric

class TestNumeric:
    @pytest.mark.parametrize("pred,gold,want", [
        ("42",         "42",    True),
        ("42.0",       "42",    True),
        ("about 41.9", "42",    True),  # within 5% rel tol
        ("100,000",    "100000", True),
        ("50%",        "50",    True),
        ("123",        "200",   False),
        ("",           "1",     False),
        ("no number",  "5",     False),
    ])
    def test_basic(self, pred, gold, want):
        assert verify_numeric(pred, gold) is want

    def test_zero_gold_tight_tolerance(self):
        assert verify_numeric("0.01", "0") is True
        assert verify_numeric("0.5",  "0") is False

    def test_last_number_wins(self):
        # The verifier picks the *last* number in the prediction.
        assert verify_numeric("first 10, then 42", "42") is True


# ------------------------------------------------------------------- exact

class TestExact:
    @pytest.mark.parametrize("pred,gold,want", [
        ("New York",     "new york",  True),
        ("the answer is Paris.", "paris", True),
        ("Berlin",       "Paris",     False),
        ("",             "x",         False),
        ("Apple, Inc.",  "apple inc", True),
    ])
    def test_normalization(self, pred, gold, want):
        assert verify_exact(pred, gold) is want


# ------------------------------------------------------------------- anls

class TestANLS:
    def test_exact_match(self):
        assert verify_anls("hello", "hello") is True

    def test_close_match_above_threshold(self):
        assert verify_anls("hello!", "hello") is True

    def test_far_match_below_threshold(self):
        assert verify_anls("zzzzz", "hello") is False

    def test_list_of_refs(self):
        assert verify_anls("paris", ["london", "Paris", "rome"]) is True

    def test_empty_pred(self):
        assert verify_anls("", "anything") is False


# ----------------------------------------------------------------- dispatch

class TestDispatch:
    def test_routes_to_mc(self):
        assert verify("answer: A", "A", "mc") is True

    def test_routes_to_numeric(self):
        assert verify("42", "42", "numeric") is True

    def test_routes_to_exact(self):
        assert verify("paris", "paris", "exact") is True

    def test_routes_to_anls(self):
        assert verify("paris", ["paris"], "anls") is True

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown ans_type"):
            verify("foo", "bar", "no-such-type")
