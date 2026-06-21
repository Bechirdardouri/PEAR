"""CLI dispatcher.

Subcommands:
  doctor          — diagnose environment (CUDA, FA, pkg versions, HF cache).
  print-modules   — load model, print `named_modules()` to confirm
                    the vision module name before any heavy compute.
  smoke           — full pipeline on 8 synthetic examples (<2 min).
  build           — assemble + cache the probe set (no model load).
  probe           — main run: probe + sampling, streams to parquet.
  analyze         — load parquet, run H1/H2, write figures + verdict.

CLI overrides (apply to all subcommands that load Config):
  --model-id STR
  --vision-module STR
  --K INT  --G INT
  --n-chartqa INT  --n-ai2d INT  --n-textvqa INT
  --out-dir PATH
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import Config


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    updates: dict = {}
    if args.model_id is not None:
        updates["model_id"] = args.model_id
    if args.vision_module is not None:
        updates["vision_module_name"] = args.vision_module
    if args.K is not None:
        updates["K"] = args.K
    if args.G is not None:
        updates["G"] = args.G
    if args.out_dir is not None:
        out = Path(args.out_dir)
        updates["out_dir"] = out
        updates["parquet_path"] = out / "results.parquet"
        updates["figs_dir"] = out / "figs"
        updates["probe_set_cache"] = out / "probe_set.jsonl"
    n_overrides = {}
    if args.n_chartqa is not None:
        n_overrides["chartqa"] = args.n_chartqa
    if args.n_ai2d is not None:
        n_overrides["ai2d"] = args.n_ai2d
    if args.n_textvqa is not None:
        n_overrides["textvqa"] = args.n_textvqa
    if n_overrides:
        merged = dict(cfg.n_per_source)
        merged.update(n_overrides)
        updates["n_per_source"] = merged
    return replace(cfg, **updates) if updates else cfg


def _add_common_overrides(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--vision-module", type=str, default=None)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--G", type=int, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--n-chartqa", type=int, default=None)
    p.add_argument("--n-ai2d", type=int, default=None)
    p.add_argument("--n-textvqa", type=int, default=None)


def main() -> None:
    parser = argparse.ArgumentParser(prog="pear")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("doctor", "print-modules", "smoke", "build", "probe", "analyze"):
        sp = sub.add_parser(name)
        _add_common_overrides(sp)

    args = parser.parse_args()
    cfg = _apply_overrides(Config(), args)
    cfg.ensure_dirs()

    if args.cmd == "doctor":
        from .doctor import doctor
        raise SystemExit(doctor())
    if args.cmd == "print-modules":
        from .model import load_model_and_processor, print_module_tree
        model, _ = load_model_and_processor(cfg)
        print_module_tree(model)
    elif args.cmd == "smoke":
        from .smoke import run_smoke
        run_smoke(cfg)
    elif args.cmd == "build":
        from .data import build_probe_set, summarize_probe_set
        records = build_probe_set(cfg)
        summarize_probe_set(records)
    elif args.cmd == "probe":
        from .run import run_pipeline
        run_pipeline(cfg)
    elif args.cmd == "analyze":
        from .analysis import run_analysis
        run_analysis(cfg)


if __name__ == "__main__":
    main()
