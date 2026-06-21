"""Per (model, dataset) probe driver.

For each example, write one parquet row with:

    id, source, answer_type,
    m_img_sum, m_img_norm, n_img_tokens,
    m_blank_sum, m_blank_norm,
    correct_g16_eval: list[int],
    correct_g4_probe: list[int],
    pass_rate_g16, pass_rate_g4, r_var_g16

Compute per row = 2 forward passes + (G_eval + G_probe) rollouts.
Default budget: 2 fwd + 20 rollouts.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .config import Config
from .data import build_probe_set
from .model import load_model_and_processor
from .score import sample_answers, teacher_forced_logprob
from .verifiers import verify

GREY_RGB = (127, 127, 127)


def _gold_str(ans) -> str:
    if isinstance(ans, list):
        return ans[0] if ans else ""
    return str(ans)


def run(
    cfg: Config,
    *,
    out_parquet: Path,
    sources: list[str] | None = None,
    g_eval: int = 16,
    g_probe: int = 4,
    eval_seed_offset: int = 5000,
    probe_seed_offset: int = 9000,
) -> int:
    """Drive the per-row probe and write a parquet.

    ``sources`` defaults to ``list(cfg.n_per_source.keys())``.
    """
    sources = sources or list(cfg.n_per_source.keys())
    print(f"[probe] model = {cfg.model_id}")
    records = build_probe_set(cfg)
    by_src: dict[str, list[dict]] = {s: [] for s in sources}
    for r in records:
        if r["source"] in by_src:
            by_src[r["source"]].append(r)
    rows_total = sum(len(v) for v in by_src.values())
    print(f"[probe] per-source counts: "
          f"{ {k: len(v) for k, v in by_src.items()} }   total={rows_total}")

    print(f"[probe] loading model {cfg.model_id} ...")
    model, processor = load_model_and_processor(cfg)

    out_rows: list[dict] = []
    pbar = tqdm(total=rows_total, desc="probe")
    for src, recs in by_src.items():
        for i, rec in enumerate(recs):
            gold = _gold_str(rec["answer"])
            ans_type = rec["answer_type"]
            try:
                ti = teacher_forced_logprob(
                    model, processor, rec["image"], rec["question"], gold, cfg
                )
                blank = Image.new("RGB", rec["image"].size, GREY_RGB)
                tb = teacher_forced_logprob(
                    model, processor, blank, rec["question"], gold, cfg
                )

                eval_cfg = replace(cfg, G=g_eval)
                eval_samples = sample_answers(
                    model, processor, rec["image"], rec["question"],
                    eval_cfg, n=g_eval,
                    seed=eval_seed_offset + i + 100000 * sources.index(src),
                )
                eval_correct = [
                    int(verify(s.text, rec["answer"], ans_type))
                    for s in eval_samples
                ]

                probe_cfg = replace(cfg, G=g_probe)
                probe_samples = sample_answers(
                    model, processor, rec["image"], rec["question"],
                    probe_cfg, n=g_probe,
                    seed=probe_seed_offset + i + 100000 * sources.index(src),
                )
                probe_correct = [
                    int(verify(s.text, rec["answer"], ans_type))
                    for s in probe_samples
                ]

                out_rows.append({
                    "id":              rec["id"],
                    "source":          src,
                    "answer_type":     ans_type,
                    "m_img_sum":       ti.sum_logprob,
                    "m_img_norm":      ti.mean_logprob,
                    "n_img_tokens":    ti.n_ans_tokens,
                    "m_blank_sum":     tb.sum_logprob,
                    "m_blank_norm":    tb.mean_logprob,
                    "correct_g16_eval":  eval_correct,
                    "correct_g4_probe":  probe_correct,
                })
            except Exception as e:                                  # noqa: BLE001
                print(f"[probe] warn id={rec['id']}  {type(e).__name__}: {e}")
                out_rows.append({
                    "id":              rec["id"],
                    "source":          src,
                    "answer_type":     ans_type,
                    "m_img_sum":       float("nan"),
                    "m_img_norm":      float("nan"),
                    "n_img_tokens":    0,
                    "m_blank_sum":     float("nan"),
                    "m_blank_norm":    float("nan"),
                    "correct_g16_eval":  [0] * g_eval,
                    "correct_g4_probe":  [0] * g_probe,
                })
            pbar.update(1)
    pbar.close()

    df = pd.DataFrame(out_rows)
    pr_eval  = df["correct_g16_eval"].apply(lambda xs: float(np.mean(xs)))
    pr_probe = df["correct_g4_probe"].apply(lambda xs: float(np.mean(xs)))
    df["pass_rate_g16"]  = pr_eval
    df["pass_rate_g4"]   = pr_probe
    df["r_var_g16"]      = pr_eval * (1 - pr_eval)

    print(f"\n[probe] summary:")
    for src in sources:
        sub = df[df["source"] == src]
        valid = sub.dropna(subset=["m_img_sum"])
        if len(valid) == 0:
            print(f"  [{src}] no valid rows")
            continue
        print(f"  [{src}] n={len(sub)}   pass_rate_g16 = "
              f"{valid['pass_rate_g16'].mean():.3f}   "
              f"r_var_g16 = {valid['r_var_g16'].mean():.4f}")

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet)
    print(f"[probe] wrote {out_parquet}  ({len(df)} rows)")
    return 0
