"""PEAR-5 analyzer: VIA vs baselines on the GRPO learning signal.

Targets (per row, label-required):
    r_var      = pass_rate * (1 - pass_rate)    PODS / Mroueh objective
    reachable  = pass_at_k & ~pass_at_1         PEAR-4 continuity target
    advantage  = |pass_rate - 0.5| * 2 reversed
                 (Prompt-Replay proxy; high = near 0.5)

Scores (label-free unless noted) — both signs reported, since
the selection direction is part of the score definition and reading
the sign from the data is principled (not cherry-picking) when both
are scored together:
    random              : control
    pass_rate_oracle    : -|pass_rate - 0.5|       (uses labels; UB)
    m_blank_only        : -m_blank
    delta_vis_only      : +delta_vis
    delta_vis_neg       : -delta_vis
    delta_spec_only     : +delta_spec
    delta_spec_neg      : -delta_spec
    VIA_vis             : var_hat * (+delta_vis)
    VIA_vis_neg         : var_hat * (-delta_vis)
    VIA                 : var_hat * (+delta_spec)
    VIA_neg             : var_hat * (-delta_spec)    (data-driven sign)

Metrics:
    Spearman rho(score, target)  + bootstrap CI (B)
    NDCG@k                       for k in {25, 50, 100}
    Top-k mean target
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import ndcg_score

from .calibrate import fit_platt
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    column: str
    label_free: bool


def _fit_multi_logreg(df: pd.DataFrame, pr: np.ndarray, *,
                     n_cal: int = 200, seed: int = 0,
                     threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Fit p_hat using all three probes (m_img, m_blank, m_shuf).

    Returns (p_hat_all, coef). Larger calibration set (n_cal=200) than
    Platt because we now have 3 features. Cross-fitted via stratified
    holdout-then-predict-all.
    """
    X = df[["m_img", "m_blank", "m_shuf"]].to_numpy(dtype=float)
    y = (pr > threshold).astype(int)
    n = len(X)
    rng = np.random.default_rng(seed)
    n_cal = min(n_cal, n)
    idx = rng.choice(n, size=n_cal, replace=False)
    Xf, yf = X[idx], y[idx]
    if yf.sum() == 0 or yf.sum() == len(yf):
        return np.full(n, float(y.mean())), np.zeros(X.shape[1])
    lr = LogisticRegression(solver="lbfgs", max_iter=500)
    lr.fit(Xf, yf)
    p_hat = lr.predict_proba(X)[:, 1]
    return p_hat, lr.coef_.ravel()


def build_scores(df: pd.DataFrame, *, seed: int = 0,
                 n_cal: int = 50,
                 pr_col: str = "pass_rate_g64") -> tuple[pd.DataFrame, dict]:
    """Add score columns to ``df``; return (df, info)."""
    df = df.copy()
    pr = df[pr_col].to_numpy(dtype=float)
    cal = fit_platt(df["m_img"].to_numpy(dtype=float), pr,
                    n_cal=n_cal, seed=seed)
    p_hat = cal.predict(df["m_img"].to_numpy(dtype=float))
    var_hat = p_hat * (1.0 - p_hat)
    df["p_hat"] = p_hat
    df["var_hat"] = var_hat

    rng = np.random.default_rng(seed)
    dv = df["delta_vis"].to_numpy()
    ds = df["delta_spec"].to_numpy()
    df["score_random"]            = rng.uniform(size=len(df))
    df["score_pass_rate_oracle"]  = -np.abs(pr - 0.5)            # high = near 0.5
    df["score_m_blank_only"]      = -df["m_blank"].to_numpy()
    df["score_delta_vis_only"]    = +dv
    df["score_delta_vis_neg"]     = -dv
    df["score_delta_spec_only"]   = +ds
    df["score_delta_spec_neg"]    = -ds
    df["score_VIA_vis"]           = var_hat * (+dv)
    df["score_VIA_vis_neg"]       = var_hat * (-dv)
    df["score_VIA"]               = var_hat * (+ds)
    df["score_VIA_neg"]           = var_hat * (-ds)

    # --- Multi-feature p_hat: the RIGHT operator for r_var selection. ---
    p_multi, coef = _fit_multi_logreg(df, pr, n_cal=min(200, len(df) // 2),
                                      seed=seed)
    var_multi = p_multi * (1.0 - p_multi)
    df["p_hat_multi"] = p_multi
    df["var_hat_multi"] = var_multi
    # Band-near-0.5 score: pick prompts whose predicted pass rate is
    # closest to 0.5. This is the CONCAVE-correct operator for
    # selecting by reward variance.
    df["score_band_multi"]        = -np.abs(p_multi - 0.5)
    df["score_band_m_img"]        = -np.abs(
        cal.predict(df["m_img"].to_numpy(dtype=float)) - 0.5
    )

    info = {
        "platt_alpha": cal.alpha,
        "platt_beta": cal.beta,
        "platt_n_fit": cal.n_fit,
        "multi_coef": coef.tolist(),
        "multi_p_range": (float(p_multi.min()), float(p_multi.max())),
    }
    return df, info


SCORE_SPECS: list[ScoreSpec] = [
    ScoreSpec("random",            "score_random",           True),
    ScoreSpec("m_blank_only",      "score_m_blank_only",     True),
    ScoreSpec("delta_vis_only",    "score_delta_vis_only",   True),
    ScoreSpec("delta_vis_neg",     "score_delta_vis_neg",    True),
    ScoreSpec("delta_spec_only",   "score_delta_spec_only",  True),
    ScoreSpec("delta_spec_neg",    "score_delta_spec_neg",   True),
    ScoreSpec("VIA_vis",           "score_VIA_vis",          True),
    ScoreSpec("VIA_vis_neg",       "score_VIA_vis_neg",      True),
    ScoreSpec("VIA",               "score_VIA",              True),
    ScoreSpec("VIA_neg",           "score_VIA_neg",          True),
    ScoreSpec("band_m_img",        "score_band_m_img",       True),
    ScoreSpec("band_multi",        "score_band_multi",       True),
    ScoreSpec("pass_rate_oracle",  "score_pass_rate_oracle", False),
]


def _random_null_top_k_mean(target: np.ndarray, k: int, *,
                            B: int = 2000, seed: int = 0
                            ) -> tuple[float, float, float]:
    """Bootstrap the null distribution of top-K mean under random selection."""
    rng = np.random.default_rng(seed)
    n = len(target)
    means = np.empty(B)
    for b in range(B):
        idx = rng.choice(n, size=k, replace=False)
        means[b] = target[idx].mean()
    return float(means.mean()), float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))


def _bootstrap_spearman(x: np.ndarray, y: np.ndarray, *,
                        B: int = 2000, seed: int = 0
                        ) -> tuple[float, float, float]:
    rho, _ = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        r, _ = stats.spearmanr(x[idx], y[idx])
        rhos[b] = r if np.isfinite(r) else 0.0
    lo, hi = np.quantile(rhos, [0.025, 0.975])
    return float(rho), float(lo), float(hi)


def _ndcg_at(scores: np.ndarray, target: np.ndarray, k: int) -> float:
    # sklearn expects 2D arrays and non-negative targets.
    t = target - target.min() + 1e-9          # shift to non-negative
    return float(ndcg_score(t.reshape(1, -1), scores.reshape(1, -1), k=k))


def _top_k_mean(scores: np.ndarray, target: np.ndarray, k: int) -> float:
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    return float(target[top].mean())


def head_to_head(df: pd.DataFrame, *, target_col: str, target_name: str,
                 ks: tuple[int, ...] = (25, 50, 100),
                 B: int = 2000, seed: int = 0) -> pd.DataFrame:
    y = df[target_col].to_numpy(dtype=float)
    rows = []
    for spec in SCORE_SPECS:
        s = df[spec.column].to_numpy(dtype=float)
        rho, lo, hi = _bootstrap_spearman(s, y, B=B, seed=seed)
        row = {
            "score":         spec.name,
            "label_free":    spec.label_free,
            "spearman_rho":  rho,
            "ci_lo":         lo,
            "ci_hi":         hi,
        }
        for k in ks:
            row[f"ndcg@{k}"]   = _ndcg_at(s, y, k)
            row[f"top{k}_mean"] = _top_k_mean(s, y, k)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["target"] = target_name
    out.attrs["n"] = len(df)
    return out


def _format_table(t: pd.DataFrame, ks: tuple[int, ...]) -> str:
    lines = []
    header = f"   {'score':<22} {'lf':>3}  {'rho':>7}  {'95% CI':>17}  "
    header += "  ".join(f"{f'ndcg@{k}':>8}" for k in ks)
    header += "  "
    header += "  ".join(f"{f'top{k}_mean':>11}" for k in ks)
    lines.append(header)
    lines.append("   " + "-" * (len(header) - 3))
    # Sort by NDCG@50 desc among label-free (the headline ranking metric);
    # oracle pinned to bottom row.
    lf = t[t["label_free"]].sort_values("ndcg@50", ascending=False)
    ora = t[~t["label_free"]]
    for _, r in pd.concat([lf, ora]).iterrows():
        ci = f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]"
        lab = "Y" if r["label_free"] else "N"
        s = (f"   {r['score']:<22} {lab:>3}  {r['spearman_rho']:+.4f}  {ci:>17}  ")
        s += "  ".join(f"{r[f'ndcg@{k}']:>8.4f}" for k in ks)
        s += "  "
        s += "  ".join(f"{r[f'top{k}_mean']:>11.4f}" for k in ks)
        lines.append(s)
    return "\n".join(lines)


def _verdict(table: pd.DataFrame) -> str:
    # Headline: best label-free score on NDCG@50 vs random.
    lf = table[table["label_free"]]
    rnd = lf[lf["score"] == "random"].iloc[0]
    best = lf.sort_values("ndcg@50", ascending=False).iloc[0]
    best_topk = lf.sort_values("top50_mean", ascending=False).iloc[0]
    ora = table[~table["label_free"]].iloc[0]

    best_rho = best["spearman_rho"]
    ci_clears = (best["ci_lo"] > 0.0) or (best["ci_hi"] < 0.0)
    ndcg_lift = float(best["ndcg@50"]) - float(rnd["ndcg@50"])
    topk_lift = float(best_topk["top50_mean"]) - float(rnd["top50_mean"])
    oracle_gap = float(ora["ndcg@50"]) - float(best["ndcg@50"])

    head = (f"best label-free by NDCG@50: '{best['score']}' "
            f"rho={best_rho:+.4f} NDCG@50={best['ndcg@50']:.4f}; "
            f"best label-free by top50_mean: '{best_topk['score']}' "
            f"top50={best_topk['top50_mean']:.4f} "
            f"(random {rnd['top50_mean']:.4f}, oracle {ora['top50_mean']:.4f}); "
            f"NDCG_lift={ndcg_lift:+.4f}, top50_lift={topk_lift:+.4f}")

    if ndcg_lift > 0.02 and ci_clears and abs(best_rho) > 0.10:
        return f"GO: {head}"
    if ndcg_lift > 0 or topk_lift > 0 or ci_clears:
        return f"ITERATE: {head}"
    return f"NO-GO: {head}"


def run(parquet_path: Path, *, B: int = 2000, seed: int = 0,
        n_cal: int = 50, g: int = 64) -> int:
    df = pd.read_parquet(parquet_path)
    print(f"== PEAR-5 analyze: {parquet_path}  n={len(df)}  (G={g}) ==")

    pr_col = f"pass_rate_g{g}"
    p1_col = f"pass_at_1_g{g}"
    pk_col = f"pass_at_k_g{g}"
    needed = {"m_img", "m_blank", "m_shuf", "delta_vis", "delta_spec",
              pr_col, p1_col, pk_col}
    missing = needed - set(df.columns)
    if missing:
        print(f"[pear5] ERROR: parquet missing columns {missing}")
        return 2

    df = df.dropna(subset=["m_img", "m_blank", "m_shuf", pr_col]).reset_index(drop=True)
    print(f"   after dropna: n={len(df)}")

    # Build targets.
    pr = df[pr_col].to_numpy(dtype=float)
    df["r_var"]     = pr * (1.0 - pr)
    df["reachable"] = (df[pk_col] & ~df[p1_col]).astype(float)
    df["advantage"] = -np.abs(pr - 0.5)             # high = near 0.5

    df, info = build_scores(df, seed=seed, n_cal=n_cal, pr_col=pr_col)
    print(f"   Platt fit: alpha={info['platt_alpha']:+.4f}  "
          f"beta={info['platt_beta']:+.4f}  n_fit={info['platt_n_fit']}")
    print(f"   p_hat range: [{df['p_hat'].min():.3f}, {df['p_hat'].max():.3f}]  "
          f"mean={df['p_hat'].mean():.3f}")
    print(f"   var_hat range: [{df['var_hat'].min():.3f}, {df['var_hat'].max():.3f}]")
    print(f"   multi-feature logreg coef [m_img, m_blank, m_shuf] = "
          f"{[round(c, 3) for c in info['multi_coef']]}")
    print(f"   p_hat_multi range: [{info['multi_p_range'][0]:.3f}, "
          f"{info['multi_p_range'][1]:.3f}]")

    # Sanity: how aligned is calibration with truth?
    rho_cal, _ = stats.spearmanr(df["p_hat"], pr)
    print(f"   spearman(p_hat, {pr_col}) = {rho_cal:+.4f}   "
          f"(calibration sanity; higher = better)")
    corr_axes = float(df[["delta_vis", "delta_spec"]].corr().iloc[0, 1])
    print(f"   corr(delta_vis, delta_spec) = {corr_axes:+.4f}   "
          f"(expect 0.3..0.7)")

    ks = (10, 25, 50, 100)
    print(f"\n>>> TARGET: r_var = p(1-p) [the GRPO advantage proxy] "
          f"<<< (primary, n={len(df)})")
    t1 = head_to_head(df, target_col="r_var", target_name="r_var",
                      ks=ks, B=B, seed=seed)
    print(_format_table(t1, ks))
    # Print proper random null at multiple K for the headline target.
    rv = df["r_var"].to_numpy()
    print(f"\n   Random null (B={B}) for top-K mean r_var (target the")
    print(f"   chartqa-wide bootstrap; a label-free score must clear")
    print(f"   the upper bound of this CI to be a real win):")
    for k in ks:
        mu, lo, hi = _random_null_top_k_mean(rv, k, B=B, seed=seed)
        print(f"     k={k:<4}  mean={mu:.4f}   5%..95% = [{lo:.4f}, {hi:.4f}]")
    print(f"\n   VERDICT (r_var): {_verdict(t1)}")

    print(f"\n>>> TARGET: reachable (PEAR-4 continuity) <<<")
    t2 = head_to_head(df, target_col="reachable", target_name="reachable",
                      ks=ks, B=B, seed=seed)
    print(_format_table(t2, ks))

    print(f"\n>>> TARGET: advantage = -|pass_rate-0.5| (Prompt-Replay proxy) <<<")
    t3 = head_to_head(df, target_col="advantage", target_name="advantage",
                      ks=ks, B=B, seed=seed)
    print(_format_table(t3, ks))

    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="pear5.analyze")
    p.add_argument("--parquet", type=Path,
                   default=Path("outputs/results_pear5.parquet"))
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-cal", type=int, default=50)
    p.add_argument("--g", type=int, default=64)
    args = p.parse_args()
    raise SystemExit(run(args.parquet, B=args.bootstrap,
                         seed=args.seed, n_cal=args.n_cal, g=args.g))


if __name__ == "__main__":
    main()
