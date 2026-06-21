"""Single CLI for pear.

Subcommands:

    probe       run the per-row probe (2 fwd + G rollouts) on a model/dataset
                and write a parquet.
    decompose   run VEST on a parquet, optionally grouped by source.
    audit       run VEST across a list of parquets and print the audit table.
    smoke       sanity-check the decomposer on the synthetic dataset.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .audit import AuditEntry, audit_table, print_audit
from .config import Config
from .vest import decompose_grouped, format_result, synthetic_smoke


def _add_probe(sp):
    p = sp.add_parser("probe", help="run per-row probe -> parquet")
    p.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    p.add_argument("--source", action="append", default=None,
                   help="dataset name; repeat for multiple. "
                        "Default: chartqa only.")
    p.add_argument("--n-per-source", type=int, default=800)
    p.add_argument("--g-eval", type=int, default=16)
    p.add_argument("--g-probe", type=int, default=4)
    p.add_argument("--out", type=Path, required=True,
                   help="output parquet path")
    p.add_argument("--seed", type=int, default=0)


def _add_decompose(sp):
    p = sp.add_parser("decompose", help="run VEST on a parquet")
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--group-by-source", action="store_true")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-floor", type=float, default=-20.0,
                   help="drop rows with m_img_sum <= this (nat). "
                        "Pass a very negative value (e.g. -1e9) to disable.")


def _add_audit(sp):
    p = sp.add_parser("audit", help="run VEST across multiple parquets")
    p.add_argument("--entry", action="append", required=True,
                   metavar="LABEL:PATH",
                   help="label:parquet_path; repeat for each checkpoint.")
    p.add_argument("--group-by-source", action="store_true")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None,
                   help="optional csv path for the audit table.")


def _add_smoke(sp):
    sp.add_parser("smoke", help="sanity-check VEST on synthetic data")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pear")
    sp = parser.add_subparsers(dest="cmd", required=True)
    _add_probe(sp)
    _add_decompose(sp)
    _add_audit(sp)
    _add_smoke(sp)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "smoke":
        for r in synthetic_smoke():
            print(format_result(r))
        return 0

    if args.cmd == "decompose":
        df = pd.read_parquet(args.parquet)
        nf = None if args.noise_floor <= -1e8 else args.noise_floor
        if args.group_by_source and "source" in df.columns:
            results = decompose_grouped(
                df, bootstrap=args.bootstrap, seed=args.seed,
                noise_floor_nat=nf,
            )
        else:
            from .vest import decompose
            results = [decompose(
                df, label="ALL", bootstrap=args.bootstrap, seed=args.seed,
                noise_floor_nat=nf,
            )]
        for r in results:
            print(format_result(r))
            print()
        return 0

    if args.cmd == "audit":
        entries: list[AuditEntry] = []
        for raw in args.entry:
            if ":" not in raw:
                raise SystemExit(f"--entry must be LABEL:PATH, got {raw!r}")
            label, _, path = raw.partition(":")
            entries.append(AuditEntry(label=label, parquet_path=Path(path)))
        table = audit_table(
            entries, bootstrap=args.bootstrap, seed=args.seed,
            group_by_source=args.group_by_source,
        )
        print_audit(table)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(args.out, index=False)
            print(f"\n[cli] wrote {args.out}")
        return 0

    if args.cmd == "probe":
        from . import probe as probe_mod
        sources = args.source or ["chartqa"]
        n_per_source = {s: args.n_per_source for s in sources}
        cfg = Config(
            model_id=args.model_id,
            n_per_source=n_per_source,
            seed=args.seed,
        )
        cfg.ensure_dirs()
        return probe_mod.run(
            cfg, out_parquet=args.out, sources=sources,
            g_eval=args.g_eval, g_probe=args.g_probe,
        )

    raise SystemExit(f"unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
