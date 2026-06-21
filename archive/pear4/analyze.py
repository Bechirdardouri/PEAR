"""PEAR-4 analysis.

Per-source partial Spearman test:

    rho_partial(Δ, reachable | m_blank, mean_logprob)

with example-level bootstrap CI (B=2000) and a three-way verdict.

Also prints a 2x2 decision surface (m_blank quartile × Δ quartile →
fraction reachable) — this is the diagnostic the corrected thesis is
built on.

Reuses ``pear2.score.partial_spearman`` and ``bootstrap_partial_spearman``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pear2.score import (
    GO_EFFECT_FLOOR,
    bootstrap_partial_spearman,
    partial_spearman,
    verdict,
)


def _quartile(x: np.ndarray) -> np.ndarray:
    """Return 0..3 quartile label for each element (stable, ties broken to lower bin)."""
    q = np.quantile(x, [0.25, 0.5, 0.75])
    return np.digitize(x, q)


def decision_surface(work: pd.DataFrame) -> None:
    """Print the 2×2 m_blank × Δ cell summary used by the new thesis."""
    mb = work["m_blank"].to_numpy()
    dl = work["delta"].to_numpy()
    y  = (work["reachable"] == True).astype(int).to_numpy()

    mb_med = np.median(mb)
    dl_med = np.median(dl)
    cells = [
        ("LOW prior  & HIGH lift  (visual edge — train here)",   mb <= mb_med, dl <= dl_med),
        ("LOW prior  & LOW  lift  (abstain — vision not helping)", mb <= mb_med, dl >  dl_med),
        ("HIGH prior & HIGH lift  (prior-confirmed by vision)",   mb >  mb_med, dl <= dl_med),
        ("HIGH prior & LOW  lift  (prior-only — vision drag)",    mb >  mb_med, dl >  dl_med),
    ]
    # Note: "HIGH lift" means Δ low/negative (image *raises* gold logprob),
    # because Δ = m_img - m_blank; here `dl <= dl_med` is the high-lift half.
    print(f"\n   median m_blank = {mb_med:+.3f}   median Δ = {dl_med:+.3f}   "
          f"(N = {len(work)})")
    print(f"   {'cell':<55} {'n':>5} {'reach':>7}")
    for label, mb_mask, dl_mask in cells:
        sel = mb_mask & dl_mask
        n = int(sel.sum())
        rate = float(y[sel].mean()) if n > 0 else float("nan")
        print(f"   {label:<55} {n:>5d} {rate:>7.3f}")


def analyze_one(work: pd.DataFrame, label: str, controls: list[str]) -> None:
    n = len(work)
    if n < 50:
        print(f"\n[{label}] n={n} too small, skipping.")
        return
    x = work["delta"].to_numpy(dtype=float)
    y = (work["reachable"] == True).astype(int).to_numpy()
    Z = work[controls].to_numpy(dtype=float)

    ps = bootstrap_partial_spearman(x, y, Z)
    v, direction = verdict(ps)

    print(f"\n[{label}] n={n}  reachable={int(y.sum())} ({y.mean():.1%})")
    print(f"   partial Spearman(Δ, reachable | {', '.join(controls)})")
    print(f"     rho     = {ps.rho:+.4f}")
    print(f"     95% CI  = [{ps.ci_lo:+.4f}, {ps.ci_hi:+.4f}]")
    print(f"     p       = {ps.p_value:.2e}")

    # Marginal references.
    rho_d, p_d = stats.spearmanr(x, y)
    rho_mb, p_mb = stats.spearmanr(work["m_blank"], y)
    rho_mi, p_mi = stats.spearmanr(work["m_img"],   y)
    print(f"   marginal Spearman(Δ,       reachable) = {rho_d:+.4f}  p={p_d:.2e}")
    print(f"   marginal Spearman(m_blank, reachable) = {rho_mb:+.4f}  p={p_mb:.2e}")
    print(f"   marginal Spearman(m_img,   reachable) = {rho_mi:+.4f}  p={p_mi:.2e}")

    decision_surface(work)

    print(f"\n   VERDICT [{label}]: {v}   (direction: rho {direction})  "
          f"GO floor=|rho_lo|>{GO_EFFECT_FLOOR}")


def run(parquet_path: Path, *, g: int, sources: list[str] | None,
        drop_blank_solvable: bool) -> int:
    df = pd.read_parquet(parquet_path)
    print(f"== PEAR-4 analyze: {parquet_path}  n={len(df)} ==")

    # Use the resampled G labels when present; else fall back to PEAR-1 G=16.
    p1_col = f"pass_at_1_g{g}"
    pk_col = f"pass_at_k_g{g}"
    ml_col = f"mean_logprob_g{g}"
    if p1_col in df.columns and pk_col in df.columns:
        df["_pass1"] = df[p1_col].astype(bool)
        df["_passk"] = df[pk_col].astype(bool)
        df["_meanlp"] = df[ml_col].astype(float)
        label_src = f"resampled G={g}"
    else:
        df["_pass1"] = df["pass_at_1"].astype(bool)
        df["_passk"] = df["pass_at_k"].astype(bool)
        df["_meanlp"] = df["mean_logprob"].astype(float)
        label_src = "PEAR-1 G=16"
    print(f"   reachability labels: {label_src}")

    df["reachable"] = df["_passk"] & ~df["_pass1"]
    df["mean_logprob"] = df["_meanlp"]

    src_list = sources or sorted(df["source"].unique())

    for src in src_list:
        sub = df[df["source"] == src].copy()
        if drop_blank_solvable and "blank_pass_at_k" in sub.columns:
            sub = sub[sub["blank_pass_at_k"] == False]
        # Always restrict to hard cases — that's the universe where
        # "reachable" is well-defined (pass@1=0).
        sub = sub[sub["_pass1"] == False].reset_index(drop=True)
        analyze_one(sub, label=src,
                    controls=["m_blank", "mean_logprob"])

    if len(src_list) > 1:
        # Reference: aggregated.
        agg = df.copy()
        if drop_blank_solvable and "blank_pass_at_k" in agg.columns:
            agg = agg[agg["blank_pass_at_k"] == False]
        agg = agg[agg["_pass1"] == False].reset_index(drop=True)
        analyze_one(agg, label="ALL (aggregated, for reference only)",
                    controls=["m_blank", "mean_logprob"])
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="pear4.analyze")
    p.add_argument("--parquet", type=Path, default=Path("outputs/results_pear4.parquet"))
    p.add_argument("--g", type=int, default=64,
                   help="sampling budget used for reachability labels")
    p.add_argument("--source", type=str, action="append",
                   help="restrict to this source (repeatable). default: all sources in parquet")
    p.add_argument("--keep-blank-solvable", action="store_true",
                   help="do NOT drop blank-solvable examples")
    args = p.parse_args()
    raise SystemExit(run(args.parquet,
                         g=args.g,
                         sources=args.source,
                         drop_blank_solvable=not args.keep_blank_solvable))


if __name__ == "__main__":
    main()
