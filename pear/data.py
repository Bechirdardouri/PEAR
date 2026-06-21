"""Probe-set assembly. Multi-dataset loaders return unified records.

A record is::

    {
        "id":          str,                 # globally unique
        "image":       PIL.Image.Image,     # RGB
        "question":    str,
        "answer":      str | list[str],     # references; list for anls
        "answer_type": "mc" | "numeric" | "exact" | "anls",
        "source":      str,                 # dataset name
    }

Currently registered loaders: ``chartqa``, ``ai2d``, ``textvqa``.
Stubs for ``mathvista``, ``realworldqa``, ``hallusionbench`` raise
``NotImplementedError`` with a TODO note; wire them in as we extend
the VEST grid (E1 of the proposal).
"""

from __future__ import annotations

import io
import random
from collections import Counter
from typing import Any, Callable, Iterable

from PIL import Image

from .config import Config

Record = dict[str, Any]


def _safe_load_dataset(path: str, split: str, **kwargs):
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


# -------------------------------------------------------------- loaders

def load_chartqa(n: int, seed: int = 0) -> list[Record]:
    ds = _safe_load_dataset("HuggingFaceM4/ChartQA", split="val")
    out: list[Record] = []
    for k, i in enumerate(_stable_sample(len(ds), n, seed)):
        ex = ds[int(i)]
        ans = ex.get("label") or ex.get("answer") or ex.get("answers")
        if isinstance(ans, list) and ans:
            ans = ans[0]
        if ans is None:
            continue
        ans_str = str(ans).strip()
        out.append({
            "id": f"chartqa-{i}-{k}",
            "image": _ensure_rgb(ex["image"]),
            "question": _question_str(ex.get("query") or ex.get("question")),
            "answer": ans_str,
            "answer_type": "numeric" if _looks_numeric(ans_str) else "exact",
            "source": "chartqa",
        })
    return out


def load_ai2d(n: int, seed: int = 0) -> list[Record]:
    ds = _safe_load_dataset("lmms-lab/ai2d", split="test")
    out: list[Record] = []
    for k, i in enumerate(_stable_sample(len(ds), n, seed)):
        ex = ds[int(i)]
        options = ex.get("options") or []
        if not options:
            continue
        gold = ex.get("answer")
        if isinstance(gold, str) and gold.isdigit():
            gold_letter = chr(ord("A") + int(gold))
        elif isinstance(gold, int):
            gold_letter = chr(ord("A") + gold)
        elif isinstance(gold, str) and len(gold) == 1 and gold.upper() in "ABCDE":
            gold_letter = gold.upper()
        else:
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
    ds = _safe_load_dataset("lmms-lab/textvqa", split="validation")
    out: list[Record] = []
    for k, i in enumerate(_stable_sample(len(ds), n, seed)):
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


def load_mathvista(n: int, seed: int = 0) -> list[Record]:
    """MathVista (lmms-lab/MathVista) -- gated on HF, TODO."""
    raise NotImplementedError(
        "MathVista is gated; set HF_TOKEN and implement when ready."
    )


def load_realworldqa(n: int, seed: int = 0) -> list[Record]:
    """RealWorldQA (xai-org/RealworldQA) -- natural-image VQA, 765 rows.

    Schema (test split): ``{image: Image, question: str, answer: str}``.
    Questions self-describe their format ("answer with only the letter",
    "single word or number"); we infer ``answer_type`` from the answer.
    """
    ds = _safe_load_dataset("xai-org/RealworldQA", split="test")
    out: list[Record] = []
    for k, i in enumerate(_stable_sample(len(ds), n, seed)):
        ex = ds[int(i)]
        ans = ex.get("answer")
        if ans is None:
            continue
        ans_str = str(ans).strip()
        if len(ans_str) == 1 and ans_str.upper() in "ABCDE":
            ans_type = "mc"
            ans_str = ans_str.upper()
        elif _looks_numeric(ans_str):
            ans_type = "numeric"
        else:
            ans_type = "exact"
        out.append({
            "id": f"realworldqa-{i}-{k}",
            "image": _ensure_rgb(ex["image"]),
            "question": _question_str(ex.get("question")),
            "answer": ans_str,
            "answer_type": ans_type,
            "source": "realworldqa",
        })
    return out


def load_hallusionbench(n: int, seed: int = 0) -> list[Record]:
    """HallusionBench -- gated on HF, TODO."""
    raise NotImplementedError(
        "HallusionBench is gated; set HF_TOKEN and implement when ready."
    )


# ------------------------------------------------------------- aggregator

_LOADERS: dict[str, Callable[[int, int], list[Record]]] = {
    "chartqa":         load_chartqa,
    "ai2d":            load_ai2d,
    "textvqa":         load_textvqa,
    "mathvista":       load_mathvista,
    "realworldqa":     load_realworldqa,
    "hallusionbench":  load_hallusionbench,
}


def build_probe_set(cfg: Config) -> list[Record]:
    records: list[Record] = []
    for source, n in cfg.n_per_source.items():
        if source not in _LOADERS:
            raise KeyError(
                f"No loader for source {source!r}. "
                f"Available: {list(_LOADERS)}."
            )
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


def synthetic_smoke_set(n: int = 8) -> list[Record]:
    """Dependency-free synthetic records for the smoke test."""
    out: list[Record] = []
    for i in range(n):
        img = Image.new("RGB", (224, 224), color=(i * 30 % 255, 80, 150))
        out.append({
            "id": f"smoke-{i}",
            "image": img,
            "question": "What number is this? Answer with a single digit.",
            "answer": str(i % 10),
            "answer_type": "exact",
            "source": "smoke",
        })
    return out
