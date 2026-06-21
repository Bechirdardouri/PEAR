"""PEAR-6 probe: 2 forward passes + G=16 image-only sampling per row.

Outputs one parquet per source with per-row columns:
    id, source, answer_type, question, gold,
    m_img_sum, m_img_norm, n_img_tokens,
    m_blank_sum, m_blank_norm,
    correct_g16_eval: list[int]   length 16, the EVAL labels
    correct_g4_probe: list[int]   length 4,  SEPARATE seeds, the CHEAP-LABEL probe

The probe rollout (G=4) uses a different seed pool from the eval
rollout (G=16) so the cheap-label score isn't biased by being a
subset of the oracle.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from pear.config import Config
from pear.data import build_probe_set
from pear.model import load_model_and_processor
from pear.scoring import sample_answers
from pear.verifiers import verify

from .scoring import teacher_forced_both


GREY_RGB = (127, 127, 127)


def _gold_str(ans) -> str:
    if isinstance(ans, list):
        return ans[0] if ans else ""
    return str(ans)


def run_probe(
    cfg: Config,
    *,
    out_parquet: Path,
    sources: list[str],
    n_per_source: int,
    g_eval: int = 16,
    g_probe: int = 4,
    eval_seed_offset: int = 5000,
    probe_seed_offset: int = 9000,
) -> int:
    print(f"[pear6] cfg.model_id = {cfg.model_id}")
    print(f"[pear6] building probe set...")
    records = build_probe_set(cfg)
    # Filter + truncate per source.
    by_src: dict[str, list[dict]] = {s: [] for s in sources}
    for r in records:
        if r["source"] in by_src and len(by_src[r["source"]]) < n_per_source:
            by_src[r["source"]].append(r)
    rows_total = sum(len(v) for v in by_src.values())
    print(f"[pear6] per-source counts: "
          f"{ {k: len(v) for k, v in by_src.items()} }   total={rows_total}")

    print(f"[pear6] loading model {cfg.model_id} ...")
    model, processor = load_model_and_processor(cfg)

    out_rows: list[dict] = []
    pbar = tqdm(total=rows_total, desc="pear6")
    for src, recs in by_src.items():
        for i, rec in enumerate(recs):
            gold = _gold_str(rec["answer"])
            ans_type = rec["answer_type"]
            try:
                # --- 2 forward passes ---
                ti = teacher_forced_both(
                    model, processor, rec["image"], rec["question"], gold, cfg
                )
                blank = Image.new("RGB", rec["image"].size, GREY_RGB)
                tb = teacher_forced_both(
                    model, processor, blank, rec["question"], gold, cfg
                )

                # --- G=g_eval image-only rollout (eval label) ---
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

                # --- G=g_probe image-only rollout (cheap-label probe) ---
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
                print(f"[pear6] warn id={rec['id']}  {type(e).__name__}: {e}")
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

    print(f"\n[pear6] summary:")
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
    print(f"[pear6] wrote {out_parquet}  ({len(df)} rows)")
    return 0
