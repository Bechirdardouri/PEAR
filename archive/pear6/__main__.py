"""PEAR-6 CLI.

Subcommands:
    probe     2 fwd + G=4 probe-rollout + G=16 eval-rollout per row.
              Default: Qwen3.5-9B, 400 rows × 3 sources (chartqa, ai2d, textvqa).
    analyze   compute-frontier head-to-head + verdict.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from pear.config import Config

from . import analyze as _analyze
from .probe import run_probe


def cmd_probe(args: argparse.Namespace) -> int:
    cfg = Config()
    if args.model_id:
        cfg = replace(cfg, model_id=args.model_id)
    cfg.ensure_dirs()
    sources = args.source or ["chartqa", "ai2d", "textvqa"]
    return run_probe(
        cfg,
        out_parquet=args.out,
        sources=sources,
        n_per_source=args.n_per_source,
        g_eval=args.g_eval,
        g_probe=args.g_probe,
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    return _analyze.run(
        args.parquet,
        B=args.bootstrap,
        seed=args.seed,
        per_source=not args.no_per_source,
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="pear6")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("probe", help="2 fwd + G=4 probe + G=16 eval per row")
    pr.add_argument("--model-id", type=str, default="Qwen/Qwen3.5-9B")
    pr.add_argument("--out", type=Path,
                    default=Path("outputs/results_pear6.parquet"))
    pr.add_argument("--source", action="append",
                    help="restrict to this source (repeatable). default: all 3")
    pr.add_argument("--n-per-source", type=int, default=400)
    pr.add_argument("--g-eval", type=int, default=16)
    pr.add_argument("--g-probe", type=int, default=4)
    pr.set_defaults(func=cmd_probe)

    an = sub.add_parser("analyze",
                        help="compute-frontier head-to-head + verdict")
    an.add_argument("--parquet", type=Path,
                    default=Path("outputs/results_pear6.parquet"))
    an.add_argument("--bootstrap", type=int, default=2000)
    an.add_argument("--seed", type=int, default=0)
    an.add_argument("--no-per-source", action="store_true")
    an.set_defaults(func=cmd_analyze)

    args = p.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
