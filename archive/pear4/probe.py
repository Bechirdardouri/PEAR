"""PEAR-4 probe.

Two operations:

1. ``visual_axes(...)`` — the new probe. Two teacher-forced forward passes
   per example (image, grey blank). Returns m_img, m_blank, Δ.

2. ``image_only_resample(...)`` — re-samples G answers with the *real*
   image (no blank-side sampling) to sharpen pass@k / mean_logprob.
   Uses the existing `pear.scoring.sample_answers` and `pear.verifiers.verify`.

Run mode (``run_resample_chartqa``): joins PEAR-1 and PEAR-3 parquets,
filters to the requested subset (default chartqa hard ∧ ¬blank-solvable),
re-samples G=64 image-only on that subset, writes
``outputs/results_pear4.parquet``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from pear.config import Config
from pear.data import build_probe_set
from pear.model import load_model_and_processor
from pear.scoring import sample_answers, teacher_forced_logprob
from pear.verifiers import verify


GREY_RGB = (127, 127, 127)


@dataclass(frozen=True)
class VisualAxes:
    m_img: float
    m_blank: float

    @property
    def delta(self) -> float:
        return self.m_img - self.m_blank


def visual_axes(model, processor, image: Image.Image,
                question: str, gold: str, cfg: Config) -> VisualAxes:
    """The whole PEAR-4 probe: two TF passes."""
    m_img = teacher_forced_logprob(model, processor, image, question, gold, cfg)
    blank = Image.new("RGB", image.size, GREY_RGB)
    m_blank = teacher_forced_logprob(model, processor, blank, question, gold, cfg)
    return VisualAxes(m_img=float(m_img), m_blank=float(m_blank))


@dataclass(frozen=True)
class ImageOnlyResult:
    pass_at_1: bool
    pass_at_k: bool
    pass_rate: float
    mean_logprob: float
    n_samples: int


def image_only_resample(model, processor, record: dict, cfg: Config,
                        seed: int) -> ImageOnlyResult:
    """Sample ``cfg.G`` answers with the real image; verify against gold."""
    gold = record["answer"]
    # textvqa carries a list of reference strings; verify() handles list,
    # but sample_answers is image+question only so no change needed here.
    samples = sample_answers(
        model, processor, record["image"], record["question"],
        cfg, n=cfg.G, seed=seed,
    )
    correct = [verify(s.text, gold, record["answer_type"]) for s in samples]
    return ImageOnlyResult(
        pass_at_1=bool(correct[0]) if correct else False,
        pass_at_k=any(correct),
        pass_rate=(sum(correct) / len(correct)) if correct else 0.0,
        mean_logprob=(sum(s.mean_logprob for s in samples) / max(len(samples), 1)),
        n_samples=len(samples),
    )


# ---------------------------------------------------------------------------
# Resample run: reuse PEAR-1 + PEAR-3, only do the expensive thing
# (G=64 image sampling) on the subset we actually need.
# ---------------------------------------------------------------------------

def _resolve_blank_pass_at_k(pear1: pd.DataFrame, pear3: pd.DataFrame) -> pd.DataFrame:
    """Join PEAR-3 (m_img, m_blank from teacher-forced) with PEAR-1
    (pass@1, pass@k, blank_pass@k from G=16 sampling).

    We rename `m_inf` → `m_img` so downstream uses the new vocabulary.
    """
    keep1 = pear1[["id", "source", "answer_type", "pass_at_1", "pass_at_k",
                   "pass_rate", "mean_logprob", "blank_pass_at_k",
                   "blank_pass_rate"]].copy()
    keep3 = pear3[["id", "m_inf", "m_blank", "delta_attr"]].rename(
        columns={"m_inf": "m_img", "delta_attr": "delta"}
    ).copy()
    return keep1.merge(keep3, on="id", how="inner")


def run_resample_chartqa(
    cfg: Config, *,
    in_pear1: Path,
    in_pear3: Path,
    out_parquet: Path,
    source: str | None,
    hard_only: bool,
    drop_blank_solvable: bool,
    seed_offset: int = 1000,
) -> int:
    p1 = pd.read_parquet(in_pear1)
    p3 = pd.read_parquet(in_pear3)
    base = _resolve_blank_pass_at_k(p1, p3)
    print(f"[pear4] joined {len(base)} rows from PEAR-1 + PEAR-3")

    if source is not None:
        base = base[base["source"] == source]
        print(f"[pear4] filtered to source='{source}': {len(base)} rows")

    sub = base
    if hard_only:
        sub = sub[sub["pass_at_1"] == False]
    if drop_blank_solvable:
        sub = sub[sub["blank_pass_at_k"] == False]
    sub = sub.reset_index(drop=True)
    print(f"[pear4] resample subset: {len(sub)} rows  "
          f"(hard_only={hard_only}, drop_blank_solvable={drop_blank_solvable})")

    if len(sub) == 0:
        print("[pear4] empty subset; aborting.")
        return 2

    # Rebuild records (image bytes, etc.) for the subset.
    print("[pear4] rebuilding probe set to recover images...")
    records = build_probe_set(cfg)
    by_id = {r["id"]: r for r in records}
    missing = [i for i in sub["id"] if i not in by_id]
    if missing:
        print(f"[pear4] WARN: {len(missing)} ids missing from rebuilt probe set")
        sub = sub[sub["id"].isin(by_id)].reset_index(drop=True)
        print(f"[pear4] after dropping missing: {len(sub)} rows")

    print(f"[pear4] loading model {cfg.model_id} ...")
    model, processor = load_model_and_processor(cfg)
    print(f"[pear4] resampling at G={cfg.G} (image-only)")

    new_p1 = []   # pass_at_1 at G
    new_pk = []   # pass_at_k at G
    new_pr = []   # pass_rate at G
    new_ml = []   # mean_logprob at G

    for i, row in tqdm(sub.iterrows(), total=len(sub), desc=f"pear4 G={cfg.G}"):
        rec = by_id[row["id"]]
        try:
            res = image_only_resample(model, processor, rec, cfg,
                                      seed=seed_offset + i)
            new_p1.append(res.pass_at_1)
            new_pk.append(res.pass_at_k)
            new_pr.append(res.pass_rate)
            new_ml.append(res.mean_logprob)
        except Exception as e:                       # noqa: BLE001
            print(f"[pear4] warn id={rec['id']}  {type(e).__name__}: {e}")
            new_p1.append(False); new_pk.append(False)
            new_pr.append(0.0);   new_ml.append(float("nan"))

    sub[f"pass_at_1_g{cfg.G}"] = new_p1
    sub[f"pass_at_k_g{cfg.G}"] = new_pk
    sub[f"pass_rate_g{cfg.G}"]  = new_pr
    sub[f"mean_logprob_g{cfg.G}"] = new_ml

    # Quick summary.
    print(f"\n[pear4] new label rates (G={cfg.G}):")
    print(f"   pass@1   = {sum(new_p1)/len(new_p1):.3f}  (was {sub['pass_at_1'].mean():.3f} at G=16)")
    print(f"   pass@k   = {sum(new_pk)/len(new_pk):.3f}  (was {sub['pass_at_k'].mean():.3f} at G=16)")
    new_reach = sum(1 for p1, pk in zip(new_p1, new_pk) if pk and not p1)
    print(f"   reachable (pass@k & !pass@1)  = {new_reach}/{len(sub)} = {new_reach/len(sub):.3f}")

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(out_parquet)
    print(f"[pear4] wrote {out_parquet}")
    return 0
