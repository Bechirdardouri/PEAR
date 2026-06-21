"""Answer verifiers for the four answer types we use.

All verifiers are pure functions over strings — no model state, no IO.
"""

from __future__ import annotations

import re
import string
import unicodedata
from typing import Iterable, Literal

AnswerType = Literal["mc", "numeric", "exact", "anls"]

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_MC_LETTER_RE = re.compile(r"\b([A-Ea-e])\b")
_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


# ---------------------------------------------------------------------- mc

def _extract_mc_letter(text: str) -> str | None:
    """Return the first standalone A–E letter (case-insensitive), or None.

    Also handles "Answer: B", "(C)", "the answer is d", etc.
    """
    if not text:
        return None
    # Prefer "answer: X" patterns first.
    m = re.search(r"answer\s*[:=]\s*\(?([A-Ea-e])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Otherwise first standalone letter.
    m = _MC_LETTER_RE.search(text)
    return m.group(1).upper() if m else None


def verify_mc(pred: str, gold: str) -> bool:
    g = _extract_mc_letter(gold) or gold.strip().upper()[:1]
    p = _extract_mc_letter(pred)
    return p is not None and p == g


# ----------------------------------------------------------------- numeric

def _extract_last_number(text: str) -> float | None:
    if not text:
        return None
    matches = _NUMERIC_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def verify_numeric(pred: str, gold: str, rel_tol: float = 0.05) -> bool:
    g = _extract_last_number(gold)
    p = _extract_last_number(pred)
    if g is None or p is None:
        return False
    if g == 0.0:
        return abs(p) <= rel_tol
    return abs(p - g) <= max(rel_tol, rel_tol * abs(g))


# ------------------------------------------------------------------- exact

def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = s.translate(_PUNCT_TABLE)
    s = re.sub(r"\s+", " ", s)
    return s


def verify_exact(pred: str, gold: str) -> bool:
    p, g = _normalize(pred), _normalize(gold)
    if not g:
        return False
    return g == p or g in p


# -------------------------------------------------------------------- anls

def _levenshtein(a: str, b: str) -> int:
    try:
        import Levenshtein  # python-Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:  # pure-Python fallback
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (ca != cb),
                ))
            prev = cur
        return prev[-1]


def _anls_pair(pred: str, gold: str, threshold: float = 0.5) -> float:
    p, g = _normalize(pred), _normalize(gold)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    dist = _levenshtein(p, g)
    nls = 1.0 - dist / max(len(p), len(g))
    return nls if nls >= threshold else 0.0


def verify_anls(pred: str, gold: str | Iterable[str], threshold: float = 0.5) -> bool:
    """ANLS-style verification.

    ``gold`` may be a single string or an iterable of references; we
    take the best score across references and threshold at 0.5.
    """
    if isinstance(gold, str):
        # Some loaders pack multiple refs as "a|b|c"; split conservatively.
        refs = [r for r in re.split(r"\|", gold) if r]
        if not refs:
            refs = [gold]
    else:
        refs = list(gold)
    score = max((_anls_pair(pred, r, threshold) for r in refs), default=0.0)
    return score >= threshold


# --------------------------------------------------------------- dispatcher

def verify(pred: str, gold, ans_type: AnswerType) -> bool:
    if ans_type == "mc":
        return verify_mc(pred, gold if isinstance(gold, str) else str(gold))
    if ans_type == "numeric":
        return verify_numeric(pred, gold if isinstance(gold, str) else str(gold))
    if ans_type == "exact":
        return verify_exact(pred, gold if isinstance(gold, str) else str(gold))
    if ans_type == "anls":
        return verify_anls(pred, gold)
    raise ValueError(f"Unknown ans_type: {ans_type!r}")
