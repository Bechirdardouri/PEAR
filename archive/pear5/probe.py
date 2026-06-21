"""PEAR-5 probe: shuffled-patch image + m_shuf extension.

Reads outputs/results_pear4.parquet (358 chartqa-hard rows with m_img,
m_blank already computed), rebuilds the probe set to recover images,
runs ONE additional teacher-forced forward pass per row on a
patch-shuffled image, writes outputs/results_pear5.parquet with two
new columns: ``m_shuf`` and ``delta_spec = m_img - m_shuf``.

Also re-derives ``delta_vis = m_img - m_blank`` (=PEAR-4's ``delta``)
under the PEAR-5 name for clarity downstream.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from pear.config import Config
from pear.data import build_probe_set
from pear.model import load_model_and_processor
from pear.scoring import teacher_forced_logprob


def shuffled_image(img: Image.Image, patch: int = 32, seed: int = 0) -> Image.Image:
    """Return a patch-shuffled copy of ``img``.

    The image is divided into a grid of ``patch x patch`` tiles; the
    tiles are permuted with a deterministic RNG and reassembled. The
    edge strip (right/bottom remainder) is left untouched so output
    size, mode, and pixel statistics match the input exactly.

    This destroys spatial semantics while preserving:
      - image size (vision token count is unchanged)
      - colour histogram
      - local-patch texture statistics

    Used as a counterfactual baseline that isolates the contribution
    of *spatial structure* (i.e. *this* image) from the contribution
    of "vision tower runs at all".
    """
    arr = np.asarray(img.convert("RGB"))
    H, W, _ = arr.shape
    nH = H // patch
    nW = W // patch
    if nH < 2 or nW < 2:
        # too small to meaningfully shuffle; just return a copy
        return img.copy()

    # Crop to the largest patch-aligned rectangle, shuffle, paste back.
    crop = arr[: nH * patch, : nW * patch].copy()
    # (nH, nW, patch, patch, 3)
    tiles = crop.reshape(nH, patch, nW, patch, 3).swapaxes(1, 2).reshape(
        nH * nW, patch, patch, 3
    )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(nH * nW)
    tiles = tiles[perm]
    crop = tiles.reshape(nH, nW, patch, patch, 3).swapaxes(1, 2).reshape(
        nH * patch, nW * patch, 3
    )
    out = arr.copy()
    out[: nH * patch, : nW * patch] = crop
    return Image.fromarray(out, mode="RGB")


def run_probe_full_chartqa(
    cfg: Config,
    *,
    in_pear1: Path,
    in_pear3: Path,
    out_parquet: Path,
    patch: int = 32,
    seed: int = 0,
) -> int:
    """PEAR-5 on the FULL chartqa (800 rows), using G=16 labels from
    PEAR-1 and m_img/m_blank from PEAR-3, then computing m_shuf for
    every row. Restores the proper RLVR-selector evaluation regime
    (no hard / blank-solvable filter)."""
    p1 = pd.read_parquet(in_pear1)
    p3 = pd.read_parquet(in_pear3)
    keep1 = p1[["id", "source", "answer_type", "pass_at_1", "pass_at_k",
                "pass_rate", "mean_logprob", "blank_pass_at_k",
                "blank_pass_rate"]].copy()
    keep3 = p3[["id", "m_inf", "m_blank"]].rename(
        columns={"m_inf": "m_img"}
    ).copy()
    df = keep1.merge(keep3, on="id", how="inner")
    df = df[df["source"] == "chartqa"].reset_index(drop=True)
    print(f"[pear5-full] {len(df)} chartqa rows after join")

    print("[pear5-full] rebuilding probe set to recover images...")
    records = build_probe_set(cfg)
    by_id = {r["id"]: r for r in records}
    df = df[df["id"].isin(by_id)].reset_index(drop=True)
    print(f"[pear5-full] after id match: {len(df)}")

    print(f"[pear5-full] loading model {cfg.model_id} ...")
    model, processor = load_model_and_processor(cfg)

    m_shuf: list[float] = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="pear5-full m_shuf"):
        rec = by_id[row["id"]]
        gold = rec["answer"]
        if isinstance(gold, list):
            gold = gold[0]
        try:
            shuf = shuffled_image(rec["image"], patch=patch, seed=seed + int(i))
            lp = teacher_forced_logprob(
                model, processor, shuf, rec["question"], gold, cfg
            )
            m_shuf.append(float(lp))
        except Exception as e:                       # noqa: BLE001
            print(f"[pear5-full] warn id={rec['id']}  {type(e).__name__}: {e}")
            m_shuf.append(float("nan"))

    df["m_shuf"] = m_shuf
    df["delta_vis"]  = df["m_img"] - df["m_blank"]
    df["delta_spec"] = df["m_img"] - df["m_shuf"]
    # Synthesize G=16 columns under the *_g16 names so the analyzer can
    # read them with the same target-construction code.
    df["pass_at_1_g16"]    = df["pass_at_1"].astype(bool)
    df["pass_at_k_g16"]    = df["pass_at_k"].astype(bool)
    df["pass_rate_g16"]    = df["pass_rate"].astype(float)
    df["mean_logprob_g16"] = df["mean_logprob"].astype(float)

    valid = df.dropna(subset=["m_shuf"])
    print(f"\n[pear5-full] summary (n={len(valid)}/{len(df)}):")
    print(f"   mean m_img      = {valid['m_img'].mean():+.4f}")
    print(f"   mean m_blank    = {valid['m_blank'].mean():+.4f}")
    print(f"   mean m_shuf     = {valid['m_shuf'].mean():+.4f}")
    print(f"   mean delta_vis  = {valid['delta_vis'].mean():+.4f}")
    print(f"   mean delta_spec = {valid['delta_spec'].mean():+.4f}")
    corr = valid[["delta_vis", "delta_spec"]].corr().iloc[0, 1]
    print(f"   corr(delta_vis, delta_spec) = {corr:+.4f}")
    print(f"   pass_rate distribution:")
    pr = valid["pass_rate"].to_numpy()
    bins = [0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
    counts, _ = np.histogram(pr, bins=bins)
    for i in range(len(counts)):
        print(f"     [{bins[i]:.2f},{bins[i+1]:.2f}): {counts[i]:>3}")

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet)
    print(f"[pear5-full] wrote {out_parquet}")
    return 0


def run_probe(
    cfg: Config,
    *,
    in_parquet: Path,
    out_parquet: Path,
    patch: int = 32,
    seed: int = 0,
) -> int:
    df = pd.read_parquet(in_parquet)
    print(f"[pear5] loaded {len(df)} rows from {in_parquet}")
    required = {"id", "source", "m_img", "m_blank"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        print(f"[pear5] ERROR: parquet missing columns {missing_cols}")
        return 2

    # Rebuild records for image recovery (same recipe as PEAR-4).
    print("[pear5] rebuilding probe set to recover images...")
    records = build_probe_set(cfg)
    by_id = {r["id"]: r for r in records}

    missing = [i for i in df["id"] if i not in by_id]
    if missing:
        print(f"[pear5] WARN: {len(missing)} ids missing from rebuilt probe set; dropping.")
        df = df[df["id"].isin(by_id)].reset_index(drop=True)
        print(f"[pear5] after drop: {len(df)} rows")

    print(f"[pear5] loading model {cfg.model_id} ...")
    model, processor = load_model_and_processor(cfg)
    print(f"[pear5] one TF pass per row on patch-shuffled image "
          f"(patch={patch}, seed={seed})")

    m_shuf: list[float] = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="pear5 m_shuf"):
        rec = by_id[row["id"]]
        gold = rec["answer"]
        if isinstance(gold, list):
            gold = gold[0]
        try:
            shuf = shuffled_image(rec["image"], patch=patch, seed=seed + int(i))
            lp = teacher_forced_logprob(
                model, processor, shuf, rec["question"], gold, cfg
            )
            m_shuf.append(float(lp))
        except Exception as e:                       # noqa: BLE001
            print(f"[pear5] warn id={rec['id']}  {type(e).__name__}: {e}")
            m_shuf.append(float("nan"))

    df["m_shuf"] = m_shuf
    df["delta_vis"]  = df["m_img"] - df["m_blank"]
    df["delta_spec"] = df["m_img"] - df["m_shuf"]

    # Quick sanity summary.
    valid = df.dropna(subset=["m_shuf"])
    print(f"\n[pear5] summary (n={len(valid)} / {len(df)} valid):")
    print(f"   mean m_img      = {valid['m_img'].mean():+.4f}")
    print(f"   mean m_blank    = {valid['m_blank'].mean():+.4f}")
    print(f"   mean m_shuf     = {valid['m_shuf'].mean():+.4f}")
    print(f"   mean delta_vis  = {valid['delta_vis'].mean():+.4f}")
    print(f"   mean delta_spec = {valid['delta_spec'].mean():+.4f}")
    corr = valid[["delta_vis", "delta_spec"]].corr().iloc[0, 1]
    print(f"   corr(delta_vis, delta_spec) = {corr:+.4f}")
    print(f"   (expect 0.3..0.7 — related but not identical)")

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet)
    print(f"[pear5] wrote {out_parquet}")
    return 0
