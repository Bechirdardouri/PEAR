"""H1 + H2 statistics, figures, and GO / ITERATE / NO-GO verdict.

H1 (decisive): on the hard subset (pass@1 == 0), does the perceptual
curve (m0, sigma_star, amplitude) predict reachability (pass@k > 0)
better than difficulty alone (pass_rate, mean_logprob)?

H2 (NEED axis): does the curve's amplitude separate examples the model
can answer from text alone (blank_pass_at_k=True) from those it can't?
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import Config


# --------------------------------------------------------------- IO

def load_results(cfg: Config) -> pd.DataFrame:
    df = pd.read_parquet(cfg.parquet_path)
    # Coerce types and drop pathological rows.
    for c in ("m0", "m_inf", "amplitude", "sigma_star",
              "pass_rate", "mean_logprob", "blank_pass_rate"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["m0", "sigma_star", "amplitude",
                           "pass_rate", "mean_logprob"]).copy()
    df["hard"] = df["pass_at_1"] == False
    df["reachable"] = df["pass_at_k"] == True
    return df


# ----------------------------------------------------- AUROC + bootstrap

def _cv_auroc(X: np.ndarray, y: np.ndarray, seed: int = 0, n_splits: int = 5) -> float:
    """5-fold stratified-CV AUROC for a logistic regression."""
    if len(np.unique(y)) < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        scores[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return float(roc_auc_score(y, scores))


def _bootstrap_delta_auroc(
    X_a: np.ndarray, X_b: np.ndarray, y: np.ndarray,
    iters: int, seed: int,
) -> tuple[float, float, float]:
    """Bootstrap ΔAUROC = AUROC(B) - AUROC(A). Returns (point, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        ya = y[idx]
        if len(np.unique(ya)) < 2:
            deltas[i] = np.nan
            continue
        a = _cv_auroc(X_a[idx], ya, seed=int(i))
        b = _cv_auroc(X_b[idx], ya, seed=int(i))
        deltas[i] = b - a
    deltas = deltas[~np.isnan(deltas)]
    point = float(np.mean(deltas))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return point, float(lo), float(hi)


# --------------------------------------------------------------- H1

@dataclass
class H1Result:
    n_hard: int
    n_reachable: int
    auroc_diff: float       # difficulty-only logistic
    auroc_pear: float       # difficulty + PEAR features
    delta_auroc: float      # mean bootstrap delta
    delta_lo: float
    delta_hi: float
    per_bin_auroc_mean: float
    per_bin_auroc_std: float
    per_bin_auroc: list[float]


def h1(df: pd.DataFrame, cfg: Config) -> H1Result:
    hard = df[df["hard"]].copy()
    n_hard = len(hard)
    n_reachable = int(hard["reachable"].sum())
    if n_hard < 20 or n_reachable < 5 or (n_hard - n_reachable) < 5:
        return H1Result(
            n_hard=n_hard, n_reachable=n_reachable,
            auroc_diff=float("nan"), auroc_pear=float("nan"),
            delta_auroc=float("nan"), delta_lo=float("nan"), delta_hi=float("nan"),
            per_bin_auroc_mean=float("nan"), per_bin_auroc_std=float("nan"),
            per_bin_auroc=[],
        )

    y = hard["reachable"].to_numpy(dtype=int)
    # Difficulty features MUST be independent of pass_at_k to avoid
    # label leakage. `pass_rate` and `pass_at_1` are derived from the
    # same G samples that define `reachable`, so they would give
    # AUROC=1.0 trivially. We use `m_inf` (teacher-forced log-prob of
    # the gold answer with sigma=0 — intrinsic model confidence) as
    # the clean difficulty proxy.
    X_diff = hard[["m_inf"]].to_numpy(dtype=float)
    X_pear = hard[["m_inf",
                   "m0", "sigma_star", "amplitude"]].to_numpy(dtype=float)

    auroc_diff = _cv_auroc(X_diff, y, seed=cfg.seed)
    auroc_pear = _cv_auroc(X_pear, y, seed=cfg.seed)
    delta, lo, hi = _bootstrap_delta_auroc(
        X_diff, X_pear, y, iters=cfg.bootstrap_iters, seed=cfg.seed,
    )

    # Per-decile within-bin AUROC of m0 alone (nonparametric).
    # Bin by m_inf (intrinsic difficulty), NOT pass_rate (leaks label).
    hard["bin"] = pd.qcut(hard["m_inf"], q=cfg.pass_rate_bins,
                          duplicates="drop", labels=False)
    per_bin = []
    for _, g in hard.groupby("bin", observed=True):
        if g["reachable"].nunique() < 2 or len(g) < 10:
            continue
        try:
            per_bin.append(float(roc_auc_score(g["reachable"], g["m0"])))
        except ValueError:
            pass

    return H1Result(
        n_hard=n_hard, n_reachable=n_reachable,
        auroc_diff=auroc_diff, auroc_pear=auroc_pear,
        delta_auroc=delta, delta_lo=lo, delta_hi=hi,
        per_bin_auroc_mean=float(np.mean(per_bin)) if per_bin else float("nan"),
        per_bin_auroc_std=float(np.std(per_bin)) if per_bin else float("nan"),
        per_bin_auroc=per_bin,
    )


# --------------------------------------------------------------- H2

@dataclass
class H2Result:
    auroc_amplitude: float
    n: int
    n_blank_solvable: int


def h2(df: pd.DataFrame) -> H2Result:
    y = df["blank_pass_at_k"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        return H2Result(auroc_amplitude=float("nan"), n=len(df), n_blank_solvable=int(y.sum()))
    # Higher amplitude = more visually-dependent → less blank-solvable.
    auroc = float(roc_auc_score(y, -df["amplitude"].to_numpy(dtype=float)))
    return H2Result(auroc_amplitude=auroc, n=len(df), n_blank_solvable=int(y.sum()))


# --------------------------------------------------------------- figures

def _save_scatter(df: pd.DataFrame, cfg: Config) -> Path:
    import matplotlib.pyplot as plt
    hard = df[df["hard"]]
    path = cfg.figs_dir / "scatter_difficulty_vs_sigma_star.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, color, marker in [
        (True, "tab:green", "o"),
        (False, "tab:red", "x"),
    ]:
        sub = hard[hard["reachable"] == label]
        ax.scatter(sub["pass_rate"], sub["sigma_star"],
                   s=16, alpha=0.6, c=color, marker=marker,
                   label=f"reachable={label} (n={len(sub)})")
    ax.set_xlabel("pass_rate  (difficulty proxy)")
    ax.set_ylabel("sigma*  (perceptual half-amplitude crossing)")
    ax.set_title("PEAR 2-axis dissociation (hard subset)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _save_curves_by_regime(df: pd.DataFrame, cfg: Config) -> Path:
    import matplotlib.pyplot as plt
    path = cfg.figs_dir / "curves_by_regime.png"
    sigmas = cfg.sigmas

    easy = df[~df["hard"]]
    hard = df[df["hard"]]
    reach = hard[hard["reachable"]]
    unread = hard[~hard["reachable"]]

    def _take(d, k=3):
        if len(d) == 0:
            return d
        return d.sample(min(k, len(d)), random_state=cfg.seed)

    panels = [
        ("easy (pass@1=1)", _take(easy), "tab:blue"),
        ("reachable-hard",  _take(reach), "tab:green"),
        ("unreadable-hard", _take(unread), "tab:red"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (title, sub, color) in zip(axes, panels):
        for _, row in sub.iterrows():
            ax.plot(sigmas, row["margins"], "-o", color=color, alpha=0.7, lw=1.2)
        ax.set_title(f"{title} (n shown: {len(sub)})")
        ax.set_xlabel("sigma")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("teacher-forced log-prob (margin)")
    fig.suptitle("Perceptual response curves by regime")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _save_auroc_per_bin(h1res: H1Result, cfg: Config) -> Path:
    import matplotlib.pyplot as plt
    path = cfg.figs_dir / "auroc_per_bin.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(h1res.per_bin_auroc))
    ax.bar(xs, h1res.per_bin_auroc, color="tab:purple", alpha=0.8)
    ax.axhline(0.5, color="k", linestyle="--", lw=1)
    ax.axhline(cfg.auroc_go, color="tab:green", linestyle=":", lw=1,
               label=f"GO ≥ {cfg.auroc_go}")
    ax.set_xlabel("pass_rate decile")
    ax.set_ylabel("AUROC(m0 → reachable)")
    ax.set_ylim(0.35, 1.0)
    ax.set_title("Within-bin AUROC of m0 alone (hard subset)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --------------------------------------------------------------- verdict

def _verdict(h1res: H1Result, cfg: Config) -> str:
    mean_auroc = h1res.per_bin_auroc_mean
    delta = h1res.delta_auroc
    lo = h1res.delta_lo

    if (mean_auroc >= cfg.auroc_go
            and delta >= cfg.delta_auroc_go
            and lo > 0.0):
        return "GO"
    if (mean_auroc >= cfg.auroc_iterate
            or delta >= cfg.delta_auroc_iterate):
        return "ITERATE"
    return "NO-GO"


# --------------------------------------------------------------- entry

def run_analysis(cfg: Config) -> None:
    cfg.ensure_dirs()
    df = load_results(cfg)
    print(f"\n== PEAR analysis on {len(df)} rows ==")
    print(f"   sources: {df['source'].value_counts().to_dict()}")
    print(f"   pass@1 rate: {df['pass_at_1'].mean():.3f}")
    print(f"   pass@k rate: {df['pass_at_k'].mean():.3f}")
    print(f"   blank pass@k: {df['blank_pass_at_k'].mean():.3f}")

    h1res = h1(df, cfg)
    h2res = h2(df)

    print("\n-- H1 (decisive) --")
    print(f"   n_hard={h1res.n_hard}  n_reachable={h1res.n_reachable}")
    print(f"   AUROC(difficulty only) = {h1res.auroc_diff:.3f}")
    print(f"   AUROC(difficulty + PEAR) = {h1res.auroc_pear:.3f}")
    print(f"   ΔAUROC = {h1res.delta_auroc:+.3f}  "
          f"95%% CI [{h1res.delta_lo:+.3f}, {h1res.delta_hi:+.3f}]")
    print(f"   per-decile AUROC(m0) mean ± sd = "
          f"{h1res.per_bin_auroc_mean:.3f} ± {h1res.per_bin_auroc_std:.3f}")

    print("\n-- H2 (NEED axis) --")
    print(f"   AUROC(blank_pass_at_k ← -amplitude) = {h2res.auroc_amplitude:.3f}  "
          f"(n={h2res.n}, blank-solvable n={h2res.n_blank_solvable})")

    p1 = _save_scatter(df, cfg)
    p2 = _save_curves_by_regime(df, cfg)
    p3 = _save_auroc_per_bin(h1res, cfg)
    print("\n-- figures --")
    for p in (p1, p2, p3):
        print(f"   {p}")

    verdict = _verdict(h1res, cfg)
    bar = "=" * 60
    print(f"\n{bar}\n  VERDICT: {verdict}\n{bar}")
    if verdict == "GO":
        print("  → curve carries signal beyond difficulty. Build PEAR full method.")
    elif verdict == "ITERATE":
        print("  → suggestive but not decisive. Re-sample hard set at G=64 or "
              "expand probe set; re-run.")
    else:
        print("  → curve is difficulty in disguise. Pivot away from m0/sigma* "
              "as a learnability signal.")
