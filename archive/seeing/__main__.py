"""SEEING CLI.

Subcommands:
    decompose  --parquet outputs/results_pear6_chartqa.parquet
    smoke      hand-built 5-row synthetic check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import decompose as _d


def cmd_decompose(args: argparse.Namespace) -> int:
    df = pd.read_parquet(args.parquet)
    print(f"== SEEING decompose: {args.parquet}   n={len(df)} ==")
    if "source" in df.columns:
        print(f"   sources: {dict(df['source'].value_counts())}")
    results = _d.decompose_grouped(
        df,
        group_col="source",
        bootstrap=args.bootstrap,
        seed=args.seed,
        noise_floor_nat=args.noise_floor,
    )
    for r in results:
        print()
        print(_d.format_result(r))
    return 0


def cmd_smoke(_: argparse.Namespace) -> int:
    for r in _d._synthetic_smoke():
        print(_d.format_result(r))
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="seeing")
    sub = p.add_subparsers(dest="cmd", required=True)

    dc = sub.add_parser("decompose",
                        help="vision-vs-prior decomposition on a parquet")
    dc.add_argument("--parquet", type=Path, required=True)
    dc.add_argument("--bootstrap", type=int, default=2000)
    dc.add_argument("--seed", type=int, default=0)
    dc.add_argument("--noise-floor", type=float,
                    default=_d.DEFAULT_NOISE_FLOOR_NAT,
                    help="drop rows with m_img_sum below this (nat)")
    dc.set_defaults(func=cmd_decompose)

    sm = sub.add_parser("smoke", help="synthetic 5-row sanity check")
    sm.set_defaults(func=cmd_smoke)

    args = p.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
