"""Probe-set assembly.

Each loader returns a list of unified records:

    {
        "id":          str,                 # globally unique
        "image":       PIL.Image.Image,     # RGB
        "question":    str,
        "answer":      str | list[str],     # references; list for anls
        "answer_type": "mc" | "numeric" | "exact" | "anls",
        "source":      str,                 # dataset name
    }

`build_probe_set(cfg)` calls them in order and caches the assembled
list to a JSONL of metadata-only rows (image bytes are kept in memory
only during the run; cache stores HF datasets' (split, index) pointers
so a re-run is cheap and reproducible).
"""

from __future__ import annotations

import io
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image

from .config import Config


Record = dict[str, Any]


# ----------------------------------------------------- HF dataset helpers

def _safe_load_dataset(path: str, split: str, **kwargs):
    """Lazy-import wrapper around ``datasets.load_dataset``."""
    from datasets import load_dataset
    return load_dataset(path, split=split, **kwargs)


def _ensure_rgb(img) -> Image.Image:
    if isinstance(img, dict) and "bytes" in img and img["bytes"]:
        img = Image.open(io.BytesIO(img["bytes"]))
    if not isinstance(img, Image.Image):
        raise TypeError(f"Expected PIL or bytes-dict, got {type(img).__name__}")
    return img.convert("RGB")


def _question_str(x) -> str:
    if isinstance(x, list) and x:
        return str(x[0])
    return str(x)


# -------------------------------------------------------------- loaders

def load_chartqa(n: int, seed: int = 0) -> list[Record]:
    """ChartQA (HuggingFaceM4/ChartQA) — numeric and short-string answers.

    Heuristic: if the gold answer parses as a number → 'numeric',
    otherwise 'exact'.
    """
    ds = _safe_load_dataset("HuggingFaceM4/ChartQA", split="val")
    idxs = _stable_sample(len(ds), n, seed)
    out: list[Record] = []
    for k, i in enumerate(idxs):
        ex = ds[int(i)]
        ans = ex.get("label") or ex.get("answer") or ex.get("answers")
        if isinstance(ans, list) and ans:
            ans = ans[0]
        if ans is None:
            continue
        ans_str = str(ans).strip()
        ans_type = "numeric" if _looks_numeric(ans_str) else "exact"
        out.append({
            "id": f"chartqa-{i}-{k}",
            "image": _ensure_rgb(ex["image"]),
            "question": _question_str(ex.get("query") or ex.get("question")),
            "answer": ans_str,
            "answer_type": ans_type,
            "source": "chartqa",
        })
    return out


def load_ai2d(n: int, seed: int = 0) -> list[Record]:
    """AI2D (lmms-lab/ai2d) — multiple choice (A/B/C/D)."""
    ds = _safe_load_dataset("lmms-lab/ai2d", split="test")
    idxs = _stable_sample(len(ds), n, seed)
    out: list[Record] = []
    for k, i in enumerate(idxs):
        ex = ds[int(i)]
        options = ex.get("options") or []
        if not options:
            continue
        # Gold may be index ("0".."3") or letter ("A".."D") or text.
        gold = ex.get("answer")
        if isinstance(gold, str) and gold.isdigit():
            gold_letter = chr(ord("A") + int(gold))
        elif isinstance(gold, int):
            gold_letter = chr(ord("A") + gold)
        elif isinstance(gold, str) and len(gold) == 1 and gold.upper() in "ABCDE":
            gold_letter = gold.upper()
        else:
            # Match by text.
            try:
                gold_letter = chr(ord("A") + [o.strip() for o in options].index(str(gold).strip()))
            except ValueError:
                continue
        q = _question_str(ex.get("question"))
        choices = "\n".join(f"{chr(ord('A') + j)}. {opt}" for j, opt in enumerate(options))
        question = (
            f"{q}\n{choices}\n"
            "Please show your choice in the `answer` field with only the choice letter, "
            'e.g., `"answer": "C"`.'
        )
        out.append({
            "id": f"ai2d-{i}-{k}",
            "image": _ensure_rgb(ex["image"]),
            "question": question,
            "answer": gold_letter,
            "answer_type": "mc",
            "source": "ai2d",
        })
    return out


def load_textvqa(n: int, seed: int = 0) -> list[Record]:
    """TextVQA (lmms-lab/textvqa) — short OCR answers, ANLS scored."""
    ds = _safe_load_dataset("lmms-lab/textvqa", split="validation")
    idxs = _stable_sample(len(ds), n, seed)
    out: list[Record] = []
    for k, i in enumerate(idxs):
        ex = ds[int(i)]
        answers = ex.get("answers") or []
        if not answers:
            continue
        out.append({
            "id": f"textvqa-{i}-{k}",
            "image": _ensure_rgb(ex["image"]),
            "question": _question_str(ex.get("question")),
            "answer": list(answers),
            "answer_type": "anls",
            "source": "textvqa",
        })
    return out


# ------------------------------------------------------------- aggregator

_LOADERS: dict[str, Callable[[int, int], list[Record]]] = {
    "chartqa": load_chartqa,
    "ai2d": load_ai2d,
    "textvqa": load_textvqa,
}


def build_probe_set(cfg: Config) -> list[Record]:
    """Assemble the probe set per ``cfg.n_per_source``.

    The probe set is shuffled deterministically by ``cfg.seed`` so the
    parquet's row order is stable across runs (useful for the resume
    logic in `run.py`).
    """
    records: list[Record] = []
    for source, n in cfg.n_per_source.items():
        if source not in _LOADERS:
            raise KeyError(f"No loader for source {source!r}. Register it in pear/data.py.")
        if n <= 0:
            continue
        print(f"[data] loading {source}: target {n} examples...")
        records.extend(_LOADERS[source](n, cfg.seed))
    rng = random.Random(cfg.seed)
    rng.shuffle(records)
    return records


def summarize_probe_set(records: Iterable[Record]) -> None:
    records = list(records)
    by_src = Counter(r["source"] for r in records)
    by_type = Counter(r["answer_type"] for r in records)
    print(f"\nProbe set: {len(records)} examples")
    print("  by source:", dict(by_src))
    print("  by answer_type:", dict(by_type))


# ------------------------------------------------------------- synthetic

def synthetic_smoke_set(n: int = 8) -> list[Record]:
    """Cheap, dependency-free synthetic records for the smoke test."""
    out: list[Record] = []
    for i in range(n):
        img = Image.new("RGB", (224, 224), color=(i * 30 % 255, 80, 150))
        out.append({
            "id": f"smoke-{i}",
            "image": img,
            "question": f"What number is this? Answer with a single digit.",
            "answer": str(i % 10),
            "answer_type": "exact",
            "source": "smoke",
        })
    return out


# --------------------------------------------------------------- helpers

def _stable_sample(total: int, n: int, seed: int) -> list[int]:
    n = min(n, total)
    rng = random.Random(seed)
    return rng.sample(range(total), n)


def _looks_numeric(s: str) -> bool:
    try:
        float(s.replace(",", "").replace("%", ""))
        return True
    except ValueError:
        return False
