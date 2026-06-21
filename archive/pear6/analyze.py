"""PEAR-6 analyzer: compute frontier of label-free vs cheap-rollout selectors.

Compares predictors on a single metric:
    top-K mean r_var_g16   (vs random null bootstrap, B=2000)

Predictors:
    random
    probe_2fwd      band(p̂_2fwd)             # logreg on 4 probe features
    rollout_K (K=1,2,4)                       # band(p̂_K) from K rollouts
                                              # (K=1: p̂_1 = first sample;
                                              #  K=2,4: ordered prefixes of probe)
    probe + rollout band(p̂_combined)         # logreg on probes + p̂_4
    oracle          band(|pass_rate_g16 - 0.5|)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class Predictor:
    name: str
    label_free: bool
    budget_units: float           # rough cost in "1 forward-equivalent" units


def _fit_logreg(X: np.ndarray, y: np.ndarray, *,
                cal_idx: np.ndarray) -> np.ndarray:
    Xf, yf = X[cal_idx], y[cal_idx]
    if yf.sum() == 0 or yf.sum() == len(yf):
        return np.full(len(X), float(y.mean()))
    lr = LogisticRegression(solver="lbfgs", max_iter=500)
    lr.fit(Xf, yf)
    return lr.predict_proba(X)[:, 1]


def _random_null_top_k_mean(target: np.ndarray, k: int, *,
                            B: int = 2000, seed: int = 0
                            ) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(target)
    means = np.empty(B)
    for b in range(B):
        idx = rng.choice(n, size=k, replace=False)
        means[b] = target[idx].mean()
    return (float(means.mean()),
            float(np.quantile(means, 0.05)),
            float(np.quantile(means, 0.95)),
            float(np.quantile(means, 0.99)))


def build_predictors(df: pd.DataFrame, *,
                     seed: int = 0,
                     n_cal_frac: float = 0.25,
                     y_threshold: float = 0.5
                     ) -> tuple[pd.DataFrame, list[Predictor]]:
    """Compute every predictor's score column on ``df``."""
    df = df.copy()
    n = len(df)
    rng = np.random.default_rng(seed)

    pr16 = df["pass_rate_g16"].to_numpy(dtype=float)
    pr4  = df["pass_rate_g4"].to_numpy(dtype=float)

    # Common calibration index (used by every logreg).
    n_cal = max(50, int(n * n_cal_frac))
    cal_idx = rng.choice(n, size=min(n_cal, n), replace=False)
    y = (pr16 > y_threshold).astype(int)

    df["score_random"] = rng.uniform(size=n)

    # ---- Probe (2 fwd) ----
    Xp = df[["m_img_sum", "m_blank_sum", "m_img_norm", "m_blank_norm"]].to_numpy(dtype=float)
    p_probe = _fit_logreg(Xp, y, cal_idx=cal_idx)
    df["p_probe"] = p_probe
    df["score_probe_2fwd"] = -np.abs(p_probe - 0.5)

    # ---- Pure-rollout predictors ----
    # band(|p-0.5|) on K rollouts. K=1 is structurally tied (|0-.5|=|1-.5|=.5)
    # so we use random-tie-break: it is forced to behave like 'random' and we
    # keep it in the table only to make the no-tie regime (K>=2) explicit.
    corr_probe = np.array(df["correct_g4_probe"].tolist(), dtype=int)  # (n, 4)
    rng_tie = np.random.default_rng(seed + 7)
    for k in (1, 2, 4):
        p_k = corr_probe[:, :k].mean(axis=1)
        df[f"p_rollout_{k}"] = p_k
        score = -np.abs(p_k - 0.5)
        # Tiny random jitter breaks ties consistently per seed.
        score = score + 1e-6 * rng_tie.standard_normal(size=len(score))
        df[f"score_rollout_{k}"] = score

    # ---- Combined (probe + p_4) ----
    Xc = np.concatenate([Xp, corr_probe[:, :4].mean(axis=1, keepdims=True)],
                        axis=1)
    p_comb = _fit_logreg(Xc, y, cal_idx=cal_idx)
    df["p_combined"] = p_comb
    df["score_probe_plus_rollout_4"] = -np.abs(p_comb - 0.5)

    # ---- Oracle ----
    df["score_oracle"] = -np.abs(pr16 - 0.5)

    preds = [
        Predictor("random",                 True,  0.0),
        Predictor("probe_2fwd",             True,  2.0),
        Predictor("rollout_1",              True,  1.0),
        Predictor("rollout_2",              True,  2.0),
        Predictor("rollout_4",              True,  4.0),
        Predictor("probe_2fwd_plus_roll_4", True,  6.0),
        Predictor("oracle_g16",             False, 16.0),
    ]
    return df, preds


def _top_k_mean(scores: np.ndarray, target: np.ndarray, k: int) -> float:
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    return float(target[top].mean())


def head_to_head(df: pd.DataFrame, preds: list[Predictor], *,
                 ks: tuple[int, ...] = (10, 25, 50, 100),
                 B: int = 2000, seed: int = 0,
                 target_col: str = "r_var_g16") -> tuple[pd.DataFrame, dict]:
    y = df[target_col].to_numpy(dtype=float)
    rows = []
    for p in preds:
        s = df[f"score_{_score_key(p.name)}"].to_numpy(dtype=float)
        rho, p_val = stats.spearmanr(s, y)
        # Bootstrap CI on rho.
        rng = np.random.default_rng(seed)
        n = len(y)
        rhos = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            r, _ = stats.spearmanr(s[idx], y[idx])
            rhos[b] = r if np.isfinite(r) else 0.0
        ci = (float(np.quantile(rhos, 0.025)),
              float(np.quantile(rhos, 0.975)))
        row = {"name": p.name, "label_free": p.label_free,
               "budget": p.budget_units,
               "rho": float(rho), "ci_lo": ci[0], "ci_hi": ci[1]}
        for k in ks:
            row[f"top{k}"] = _top_k_mean(s, y, k)
        rows.append(row)
    out = pd.DataFrame(rows)
    nulls: dict[int, tuple[float, float, float, float]] = {}
    for k in ks:
        nulls[k] = _random_null_top_k_mean(y, k, B=B, seed=seed)
    return out, nulls


def _score_key(name: str) -> str:
    mapping = {
        "random":                 "random",
        "probe_2fwd":             "probe_2fwd",
        "rollout_1":              "rollout_1",
        "rollout_2":              "rollout_2",
        "rollout_4":              "rollout_4",
        "probe_2fwd_plus_roll_4": "probe_plus_rollout_4",
        "oracle_g16":             "oracle",
    }
    return mapping[name]


def _format_table(t: pd.DataFrame, nulls: dict, ks: tuple[int, ...]) -> str:
    lines = []
    header = f"   {'name':<26} {'lf':>3} {'B':>5}  {'rho':>7}  {'95% CI':>17}"
    header += "  " + "  ".join(f"{f'top{k}':>9}" for k in ks)
    lines.append(header)
    lines.append("   " + "-" * (len(header) - 3))
    t = t.sort_values("budget", ascending=True)
    for _, r in t.iterrows():
        ci = f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]"
        lab = "Y" if r["label_free"] else "N"
        s = (f"   {r['name']:<26} {lab:>3} {r['budget']:>5.1f}  "
             f"{r['rho']:+.4f}  {ci:>17}")
        for k in ks:
            s += f"  {r[f'top{k}']:>9.4f}"
        lines.append(s)
    # Null line.
    lines.append("   " + "-" * (len(header) - 3))
    s = f"   {'RANDOM_NULL_95%_UB':<26} {'N':>3} {'0.0':>5}  "
    s += f"{'.':>7}  {'.':>17}"
    for k in ks:
        s += f"  {nulls[k][2]:>9.4f}"
    lines.append(s)
    s = f"   {'RANDOM_NULL_99%_UB':<26} {'N':>3} {'0.0':>5}  "
    s += f"{'.':>7}  {'.':>17}"
    for k in ks:
        s += f"  {nulls[k][3]:>9.4f}"
    lines.append(s)
    return "\n".join(lines)


def _verdict(t: pd.DataFrame, nulls: dict, k_focus: int = 50) -> str:
    """Judge at matched compute.

    The single question PEAR-6 asks is: at the same per-example budget,
    does the label-free probe beat the cheap-rollout baseline? If not,
    even a probe that beats random null is dominated.
    """
    null_95 = nulls[k_focus][2]
    probe = t[t["name"] == "probe_2fwd"].iloc[0]
    roll2 = t[t["name"] == "rollout_2"].iloc[0]
    roll4 = t[t["name"] == "rollout_4"].iloc[0]
    combined = t[t["name"] == "probe_2fwd_plus_roll_4"].iloc[0]
    oracle = t[t["name"] == "oracle_g16"].iloc[0]

    pp   = float(probe[f"top{k_focus}"])
    r2   = float(roll2[f"top{k_focus}"])
    r4   = float(roll4[f"top{k_focus}"])
    pc   = float(combined[f"top{k_focus}"])
    orcl = float(oracle[f"top{k_focus}"])

    probe_beats_null = pp > null_95
    probe_beats_matched_roll = pp > r2          # budget = 2 for both
    combined_beats_roll = pc > r4               # budget = 6 vs 4; only worth it if pc > r4

    parts = [
        f"random_null_95={null_95:.4f}  oracle_top{k_focus}={orcl:.4f}",
        f"probe_2fwd ({pp:.4f}, B=2)  vs  rollout_2 ({r2:.4f}, B=2)",
        f"probe+roll4 ({pc:.4f}, B=6)  vs  rollout_4 ({r4:.4f}, B=4)",
    ]
    head = "\n     ".join(parts)

    if probe_beats_null and probe_beats_matched_roll and combined_beats_roll:
        flag = "GO  (probe beats rollouts at matched budget and adds info on top)"
    elif probe_beats_null and probe_beats_matched_roll:
        flag = "PARTIAL GO  (probe beats matched rollout but adds no info on top)"
    elif probe_beats_null:
        flag = ("NO-GO  (probe beats null but loses to cheap rollouts at "
                "matched budget; spend FLOPs on rollouts instead)")
    else:
        flag = "NO-GO  (probe does not beat random null)"
    return f"{flag}\n     {head}"


def run(parquet_path: Path, *, B: int = 2000, seed: int = 0,
        per_source: bool = True) -> int:
    df = pd.read_parquet(parquet_path)
    print(f"== PEAR-6 analyze: {parquet_path}  n={len(df)} ==")

    df = df.dropna(subset=["m_img_sum"]).reset_index(drop=True)
    print(f"   after dropna: n={len(df)}")
    print(f"   sources: {dict(df['source'].value_counts())}")

    ks = (10, 25, 50, 100)

    def _one_block(sub: pd.DataFrame, label: str) -> None:
        if len(sub) < 100:
            print(f"\n[{label}] n={len(sub)} too small, skipping.")
            return
        sub = sub.reset_index(drop=True)
        sub, preds = build_predictors(sub, seed=seed)
        pr16 = sub["pass_rate_g16"].to_numpy()
        print(f"\n>>> [{label}]  n={len(sub)}   mean pass_rate_g16={pr16.mean():.3f}  "
              f"mean r_var={(pr16*(1-pr16)).mean():.4f} <<<")
        # Quick sanity: correlation of probe-p with truth.
        rho_probe, _ = stats.spearmanr(sub["p_probe"], pr16)
        rho_r4, _    = stats.spearmanr(sub["p_rollout_4"], pr16)
        print(f"   sanity: rho(p_probe, pass_rate_g16) = {rho_probe:+.4f}   "
              f"rho(p_rollout_4, pass_rate_g16) = {rho_r4:+.4f}")
        t, nulls = head_to_head(sub, preds, ks=ks, B=B, seed=seed)
        print(_format_table(t, nulls, ks))
        print(f"\n   VERDICT: {_verdict(t, nulls, k_focus=50)}")

    if per_source:
        for src in sorted(df["source"].unique()):
            _one_block(df[df["source"] == src], src)
    _one_block(df, "ALL")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="pear6.analyze")
    p.add_argument("--parquet", type=Path,
                   default=Path("outputs/results_pear6.parquet"))
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-per-source", action="store_true")
    args = p.parse_args()
    raise SystemExit(run(args.parquet, B=args.bootstrap, seed=args.seed,
                         per_source=not args.no_per_source))


if __name__ == "__main__":
    main()
