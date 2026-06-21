"""PEAR-5 CLI.

Subcommands:
    probe     read PEAR-4 parquet, run 1 extra TF pass per row on a
              patch-shuffled image, write outputs/results_pear5.parquet
              with m_shuf + delta_spec (+ delta_vis alias).
    analyze   head-to-head ranking-quality table for VIA vs baselines
              against r_var / reachable / advantage targets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pear.config import Config

from . import analyze as _analyze
from .probe import run_probe, run_probe_full_chartqa


def cmd_probe(args: argparse.Namespace) -> int:
    cfg = Config()
    cfg.ensure_dirs()
    return run_probe(
        cfg,
        in_parquet=args.in_parquet,
        out_parquet=args.out,
        patch=args.patch,
        seed=args.seed,
    )


def cmd_probe_full(args: argparse.Namespace) -> int:
    cfg = Config()
    cfg.ensure_dirs()
    return run_probe_full_chartqa(
        cfg,
        in_pear1=args.in_pear1,
        in_pear3=args.in_pear3,
        out_parquet=args.out,
        patch=args.patch,
        seed=args.seed,
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    return _analyze.run(
        args.parquet,
        B=args.bootstrap,
        seed=args.seed,
        n_cal=args.n_cal,
        g=args.g,
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="pear5")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("probe", help="add m_shuf to PEAR-4 parquet")
    pr.add_argument("--in-parquet", type=Path,
                    default=Path("outputs/results_pear4.parquet"))
    pr.add_argument("--out", type=Path,
                    default=Path("outputs/results_pear5.parquet"))
    pr.add_argument("--patch", type=int, default=32,
                    help="patch size for shuffle (>= Qwen ViT patch 14)")
    pr.add_argument("--seed", type=int, default=0)
    pr.set_defaults(func=cmd_probe)

    pf = sub.add_parser("probe-full",
                        help="compute m_shuf on full 800-row chartqa (no hard filter)")
    pf.add_argument("--in-pear1", type=Path, default=Path("outputs/results.parquet"))
    pf.add_argument("--in-pear3", type=Path, default=Path("outputs/results_pear3.parquet"))
    pf.add_argument("--out", type=Path,
                    default=Path("outputs/results_pear5_full.parquet"))
    pf.add_argument("--patch", type=int, default=32)
    pf.add_argument("--seed", type=int, default=0)
    pf.set_defaults(func=cmd_probe_full)

    an = sub.add_parser("analyze", help="head-to-head VIA vs baselines")
    an.add_argument("--parquet", type=Path,
                    default=Path("outputs/results_pear5.parquet"))
    an.add_argument("--bootstrap", type=int, default=2000)
    an.add_argument("--seed", type=int, default=0)
    an.add_argument("--n-cal", type=int, default=50)
    an.add_argument("--g", type=int, default=64,
                    help="sampling budget used for the label columns (64 or 16)")
    an.set_defaults(func=cmd_analyze)

    args = p.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
