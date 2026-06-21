"""Add `m_blank` and `delta_attr` columns to PEAR-1's results parquet.

For each example: one extra teacher-forced forward pass with the
image replaced by a same-size uniform-grey baseline (the canonical
Integrated-Gradients reference).

This is the cheapest possible probe: no sampling, no curve fitting,
no calibration. Output is a single number per example.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from pear.config import Config
from pear.data import build_probe_set
from pear.model import load_model_and_processor
from pear.scoring import teacher_forced_logprob


GREY_RGB = (128, 128, 128)


def blank_image(img: Image.Image) -> Image.Image:
    """Same-size, same-mode uniform-grey baseline.

    Same size as the input so the vision tower produces the same
    number of patch tokens — keeps prompt length matched between
    the m_inf and m_blank probes.
    """
    return Image.new("RGB", img.size, GREY_RGB)


def run_probe(cfg: Config, in_parquet: Path, out_parquet: Path) -> int:
    df = pd.read_parquet(in_parquet)
    print(f"[pear3] loaded {len(df)} rows from {in_parquet}")

    records = build_probe_set(cfg)
    by_id = {r["id"]: r for r in records}
    missing = [i for i in df["id"] if i not in by_id]
    if missing:
        print(f"[pear3] WARN: {len(missing)} ids not found in rebuilt probe set")

    print(f"[pear3] loading model {cfg.model_id} ...")
    model, processor = load_model_and_processor(cfg)

    m_blank: list[float] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="pear3"):
        rec = by_id.get(row["id"])
        if rec is None:
            m_blank.append(float("nan"))
            continue
        answer = rec["answer"]
        if isinstance(answer, list):    # textvqa: list of references
            answer = answer[0]
        try:
            mb = teacher_forced_logprob(
                model, processor,
                blank_image(rec["image"]),
                rec["question"],
                answer,
                cfg,
            )
        except Exception as e:           # noqa: BLE001
            print(f"[pear3] warn: id={rec['id']}  {type(e).__name__}: {e}")
            mb = float("nan")
        m_blank.append(float(mb))

    df["m_blank"] = m_blank
    df["delta_attr"] = df["m_inf"] - df["m_blank"]

    n_ok = int(df["m_blank"].notna().sum())
    print(f"[pear3] m_blank computed for {n_ok}/{len(df)} rows")
    print(f"[pear3]   m_inf    : mean={df['m_inf'].mean():+.3f}   "
          f"std={df['m_inf'].std():.3f}")
    print(f"[pear3]   m_blank  : mean={df['m_blank'].mean():+.3f}   "
          f"std={df['m_blank'].std():.3f}")
    print(f"[pear3]   delta    : mean={df['delta_attr'].mean():+.3f}   "
          f"std={df['delta_attr'].std():.3f}")

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet)
    print(f"[pear3] wrote {out_parquet}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="pear3")
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("probe", help="add m_blank / delta_attr to existing parquet")
    pp.add_argument("--in",  dest="in_parquet",  type=Path, default=Path("outputs/results.parquet"))
    pp.add_argument("--out", dest="out_parquet", type=Path, default=Path("outputs/results_pear3.parquet"))
    args = p.parse_args()
    if args.cmd == "probe":
        cfg = Config()
        cfg.ensure_dirs()
        raise SystemExit(run_probe(cfg, args.in_parquet, args.out_parquet))


if __name__ == "__main__":
    main()
