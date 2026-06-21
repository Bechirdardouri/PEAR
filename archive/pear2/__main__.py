"""PEAR-2 pipeline: read PEAR-1 parquet, derive Delta, run the test, print verdict.

Usage:
    python -m pear2 analyze                 # full PEAR-2 from existing data
    python -m pear2 analyze --no-blank-filter  # skip blank-solvable filter
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .calibrate import Calibration, calibrate, m_at_sigma_star
from .score import (
    PartialSpearman,
    bootstrap_partial_spearman,
    partial_spearman,
    verdict,
)


DEFAULT_PARQUET = Path("outputs/results.parquet")


def load(parquet_path: Path) -> pd.DataFrame:
    """Read parquet and add derived columns we need."""
    df = pd.read_parquet(parquet_path)
    if "margins" in df.columns:
        M = np.stack([np.asarray(m, dtype=float) for m in df["margins"]])
        assert M.shape[1] == 7, f"expected 7-element margins, got {M.shape}"
        df["_margins_mat"] = list(M)
    df["hard"] = df["pass_at_1"] == False
    df["reachable"] = df["pass_at_k"] == True
    df["blank_solvable"] = df["blank_pass_at_k"] == True
    return df


def run(parquet_path: Path, *, blank_filter: bool, hard_only: bool,
        print_curve: bool, delta_column: str | None) -> int:
    df = load(parquet_path)
    print(f"== PEAR-2 on {len(df)} rows from {parquet_path} ==")
    print(f"   pass@1 = {df['pass_at_1'].mean():.3f}   "
          f"pass@k = {df['reachable'].mean():.3f}   "
          f"blank_solvable = {df['blank_solvable'].mean():.3f}")

    # ------------------------------------------------------------ Delta
    if delta_column is not None:
        # PEAR-3 mode: use a precomputed Delta column.
        if delta_column not in df.columns:
            print(f"[!] column '{delta_column}' not in parquet; aborting.")
            return 2
        df["Delta"] = df[delta_column].astype(float)
        # Drop NaN rows so the test runs cleanly.
        n_before = len(df)
        df = df.dropna(subset=["Delta", "m_inf"]).reset_index(drop=True)
        print(f"\n-- PEAR-3 Delta from column '{delta_column}' --")
        print(f"   {len(df)}/{n_before} rows have valid Delta")
        print(f"   Delta : mean={df['Delta'].mean():+.3f}  std={df['Delta'].std():.3f}  "
              f"min={df['Delta'].min():+.3f}  max={df['Delta'].max():+.3f}")
    else:
        # PEAR-2 mode: calibrate from margins matrix.
        M_all = np.stack(df["_margins_mat"].tolist())
        cal = calibrate(M_all)
        print(f"\n-- calibration --")
        print(f"   sigma* = {cal.sigma_star:.4f}   "
              f"mean drop at sigma* = {cal.drop_at_sigma_star:.3f} nat   "
              f"(target = 1.0 nat over {cal.n_examples} examples)")
        if print_curve:
            drops = (M_all[:, :1] - M_all).mean(axis=0)
            from .calibrate import SIGMAS
            print("   per-sigma mean drop:")
            for s, d in zip(SIGMAS, drops):
                print(f"     sigma={s:>5.3f}   drop={d:+.3f} nat")
        df["m_at_sigma_star"] = m_at_sigma_star(M_all, cal)
        df["Delta"] = df["m_inf"] - df["m_at_sigma_star"]

    # ------------------------------------------------------------ filter
    work = df.copy()
    if hard_only:
        work = work[work["hard"]]
    if blank_filter:
        work = work[~work["blank_solvable"]]
    work = work.reset_index(drop=True)
    print(f"\n-- analysis sample --")
    print(f"   n = {len(work)}   "
          f"reachable = {int(work['reachable'].sum())} "
          f"({work['reachable'].mean():.1%})")
    if len(work) < 50:
        print("\n[!] sample too small for stable inference; aborting.")
        return 2

    # ------------------------------------------------------------ test
    x = work["Delta"].to_numpy(dtype=float)
    y = work["reachable"].to_numpy(dtype=int)
    if delta_column is not None:
        # PEAR-3: Delta_attr already encodes the blank baseline
        # (m_blank), so the only legitimate control is m_inf.
        Z = work[["m_inf"]].to_numpy(dtype=float)
        controls_label = "m_inf"
    else:
        Z = work[["m_inf", "blank_solvable"]].to_numpy(dtype=float)
        controls_label = "m_inf, blank_solvable"

    # Headline statistic.
    ps = bootstrap_partial_spearman(x, y, Z)
    v, direction = verdict(ps)
    print(f"\n-- partial Spearman: Delta vs reachable | ({controls_label}) --")
    print(f"   rho       = {ps.rho:+.4f}")
    print(f"   95% CI    = [{ps.ci_lo:+.4f}, {ps.ci_hi:+.4f}]")
    print(f"   p-value   = {ps.p_value:.2e}   n = {ps.n}")

    # Reference: marginal Spearman with no controls.
    rho_marg, p_marg = partial_spearman(x, y, np.zeros((len(work), 0)))
    print(f"\n-- reference (no controls) --")
    print(f"   marginal Spearman(Delta, reachable) = {rho_marg:+.4f}   p={p_marg:.2e}")

    # Counter-test: replace Delta with intrinsic-difficulty proxy alone.
    rho_minf, p_minf = partial_spearman(
        work["m_inf"].to_numpy(dtype=float), y,
        np.zeros((len(work), 0)),
    )
    print(f"   marginal Spearman(m_inf, reachable) = {rho_minf:+.4f}   p={p_minf:.2e}")

    # ------------------------------------------------------------ verdict
    print("\n" + "=" * 60)
    print(f"  PEAR-2 VERDICT: {v}   (direction: rho {direction})")
    print("=" * 60)
    if v == "GO":
        sign_explain = (
            "low-Delta examples are preferentially reachable"
            if direction == "-" else
            "high-Delta examples are preferentially reachable"
        )
        print(f"  -> Delta is a usable per-example reachability signal.")
        print(f"     Direction: {sign_explain}.")
        print(f"     Effect size |lower CI| > GO threshold 0.10.")
    elif v == "WEAK":
        print("  -> Real but small effect (CI excludes 0 but magnitude < 0.10).")
        print("     Consider G=64 resampling for sharper labels before deploying.")
    else:
        print("  -> CI includes 0. Perceptual-response idea does not survive on this model.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="pear2")
    sub = p.add_subparsers(dest="cmd", required=True)
    ana = sub.add_parser("analyze", help="run PEAR-2 on existing parquet")
    ana.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ana.add_argument("--no-blank-filter", dest="blank_filter",
                     action="store_false", default=True,
                     help="keep text-solvable examples in the analysis")
    ana.add_argument("--all", dest="hard_only", action="store_false",
                     default=True,
                     help="don't restrict to hard subset (pass@1=False)")
    ana.add_argument("--print-curve", action="store_true",
                     help="dump per-sigma mean drop table")
    ana.add_argument("--delta-column", type=str, default=None,
                     help="use a precomputed Delta column (e.g. delta_attr from PEAR-3)")
    args = p.parse_args()
    if args.cmd == "analyze":
        raise SystemExit(run(args.parquet,
                             blank_filter=args.blank_filter,
                             hard_only=args.hard_only,
                             print_curve=args.print_curve,
                             delta_column=args.delta_column))


if __name__ == "__main__":
    main()
