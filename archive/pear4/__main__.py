"""PEAR-4 CLI.

Subcommands:
    probe-resample   re-sample G answers (image-only) on the joined
                     PEAR-1 + PEAR-3 parquet, restricted by --source /
                     hard / blank-solvable filters. Writes
                     outputs/results_pear4.parquet.
    analyze          per-source partial-Spearman test on the parquet.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from pear.config import Config

from . import analyze as _analyze
from .probe import run_resample_chartqa


def cmd_probe_resample(args: argparse.Namespace) -> int:
    cfg = Config()
    # Override G via CLI; keep all other cfg defaults.
    cfg = replace(cfg, G=args.g)
    cfg.ensure_dirs()
    return run_resample_chartqa(
        cfg,
        in_pear1=args.in_pear1,
        in_pear3=args.in_pear3,
        out_parquet=args.out,
        source=args.source,
        hard_only=not args.all,
        drop_blank_solvable=not args.keep_blank_solvable,
        seed_offset=args.seed_offset,
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    return _analyze.run(
        args.parquet, g=args.g,
        sources=args.source,
        drop_blank_solvable=not args.keep_blank_solvable,
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="pear4")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("probe-resample",
                        help="G=64 image-only resample on PEAR-1+3 join")
    pr.add_argument("--in-pear1", type=Path, default=Path("outputs/results.parquet"))
    pr.add_argument("--in-pear3", type=Path, default=Path("outputs/results_pear3.parquet"))
    pr.add_argument("--out",      type=Path, default=Path("outputs/results_pear4.parquet"))
    pr.add_argument("--source",   type=str,  default="chartqa",
                    help="restrict to this source. Use --source '' for all.")
    pr.add_argument("--g",        type=int,  default=64)
    pr.add_argument("--all",      action="store_true",
                    help="do NOT restrict to hard (pass@1=False) examples")
    pr.add_argument("--keep-blank-solvable", action="store_true",
                    help="do NOT drop blank-solvable examples")
    pr.add_argument("--seed-offset", type=int, default=1000,
                    help="seed offset distinct from PEAR-1 to avoid label re-use")
    pr.set_defaults(func=cmd_probe_resample)

    an = sub.add_parser("analyze", help="per-source partial-Spearman analysis")
    an.add_argument("--parquet", type=Path, default=Path("outputs/results_pear4.parquet"))
    an.add_argument("--g", type=int, default=64)
    an.add_argument("--source", type=str, action="append")
    an.add_argument("--keep-blank-solvable", action="store_true")
    an.set_defaults(func=cmd_analyze)

    args = p.parse_args()
    if args.cmd == "probe-resample" and args.source == "":
        args.source = None
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
