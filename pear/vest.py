"""VEST — Vision-vs-prior Equity Score Test.

The headline metric is::

    vision_driven_frac =
        (sum of pass_rate over examples with g > 0) /
        (sum of pass_rate over all examples)

i.e. of all *correct rollouts* the model produced, what fraction came
from examples where seeing the image actually moved belief toward the
gold answer. Operates on a DataFrame already produced by ``probe.run``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

# Default noise floor: drop rows whose teacher-forced gold log-prob
# under the image is below this (very low) value, because g is then a
# difference of two near-(-inf) numbers and uninformative.
DEFAULT_NOISE_FLOOR_NAT: float = -20.0

G_BINS: tuple[float, ...] = (-np.inf, -2.0, -0.5, 0.0, 0.5, 2.0, np.inf)


@dataclass(frozen=True)
class VestResult:
    label: str
    n_examples: int
    n_correct_rollouts: float
    mean_g: float
    median_g: float
    frac_vision_helped: float
    frac_vision_hurt: float
    pass_rate_g_pos: float
    pass_rate_g_nonpos: float
    vision_driven_frac: float
    prior_driven_frac: float
    rho_g_passrate: float
    rho_ci_lo: float
    rho_ci_hi: float
    g_bin_counts: list[int]
    g_bin_correct_mass: list[float]
    notes: list[str] = field(default_factory=list)


def _bootstrap_rho_ci(x: np.ndarray, y: np.ndarray, *,
                      B: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        r, _ = stats.spearmanr(x[idx], y[idx])
        rhos[b] = r if np.isfinite(r) else 0.0
    return float(np.quantile(rhos, 0.025)), float(np.quantile(rhos, 0.975))


def decompose(
    df: pd.DataFrame,
    *,
    label: str = "ALL",
    pass_rate_col: str = "pass_rate_g16",
    img_col: str = "m_img_sum",
    blank_col: str = "m_blank_sum",
    bootstrap: int = 2000,
    seed: int = 0,
    noise_floor_nat: float | None = DEFAULT_NOISE_FLOOR_NAT,
) -> VestResult:
    """Compute the vision-vs-prior decomposition on ``df``.

    Pure function: no I/O, no model loads. Operates on a DataFrame
    produced by :func:`pear.probe.run`.
    """
    sub = df.dropna(subset=[img_col, blank_col, pass_rate_col]).copy()
    notes: list[str] = []
    n_dropped_nan = len(df) - len(sub)
    if n_dropped_nan:
        notes.append(f"dropped {n_dropped_nan} rows with NaN scores")

    if noise_floor_nat is not None:
        mask = sub[img_col] > noise_floor_nat
        n_below = int((~mask).sum())
        if n_below:
            notes.append(
                f"dropped {n_below} rows with {img_col} <= {noise_floor_nat:g}"
                f" (gold-token noise floor)"
            )
        sub = sub[mask].copy()

    if len(sub) == 0:
        raise ValueError("no rows survived filtering")

    g = (sub[img_col] - sub[blank_col]).to_numpy(dtype=float)
    pr = sub[pass_rate_col].to_numpy(dtype=float)

    g_pos = g > 0
    g_neg = g < 0
    g_nonpos = ~g_pos

    total_pr = float(pr.sum())
    if total_pr <= 0:
        notes.append("all rows have pass_rate=0; vision_driven_frac undefined")
        vd_frac = float("nan")
        pd_frac = float("nan")
    else:
        vd_frac = float(pr[g_pos].sum() / total_pr)
        pd_frac = 1.0 - vd_frac

    rho, _ = stats.spearmanr(g, pr)
    ci_lo, ci_hi = _bootstrap_rho_ci(g, pr, B=bootstrap, seed=seed)

    bin_idx = np.digitize(g, G_BINS[1:-1], right=False)
    n_bins = len(G_BINS) - 1
    bin_counts = [int((bin_idx == i).sum()) for i in range(n_bins)]
    bin_correct = [float(pr[bin_idx == i].sum()) for i in range(n_bins)]

    return VestResult(
        label=label,
        n_examples=int(len(sub)),
        n_correct_rollouts=total_pr,
        mean_g=float(g.mean()),
        median_g=float(np.median(g)),
        frac_vision_helped=float(g_pos.mean()),
        frac_vision_hurt=float(g_neg.mean()),
        pass_rate_g_pos=float(pr[g_pos].mean()) if g_pos.any() else float("nan"),
        pass_rate_g_nonpos=float(pr[g_nonpos].mean()) if g_nonpos.any() else float("nan"),
        vision_driven_frac=vd_frac,
        prior_driven_frac=pd_frac,
        rho_g_passrate=float(rho) if np.isfinite(rho) else float("nan"),
        rho_ci_lo=ci_lo,
        rho_ci_hi=ci_hi,
        g_bin_counts=bin_counts,
        g_bin_correct_mass=bin_correct,
        notes=notes,
    )


def format_result(r: VestResult) -> str:
    lines = []
    lines.append(f">>> [{r.label}]  n_examples = {r.n_examples}   "
                 f"n_correct_rollouts (sum pass_rate) = {r.n_correct_rollouts:.2f}")
    for note in r.notes:
        lines.append(f"     note: {note}")
    lines.append(f"   g = log p(gold|image,q) - log p(gold|blank,q)   (units: nat)")
    lines.append(f"   mean g   = {r.mean_g:+.3f}     median g = {r.median_g:+.3f}")
    lines.append(f"   frac(g > 0)  [seeing helps]  = {r.frac_vision_helped:.3f}")
    lines.append(f"   frac(g < 0)  [seeing HURTS]  = {r.frac_vision_hurt:.3f}")
    lines.append(f"   pass_rate | g  > 0    = {r.pass_rate_g_pos:.3f}")
    lines.append(f"   pass_rate | g <= 0    = {r.pass_rate_g_nonpos:.3f}")
    lines.append("")
    lines.append(f"   *** vision_driven_frac = {r.vision_driven_frac:.3f} ***")
    lines.append(f"       prior_driven_frac  = {r.prior_driven_frac:.3f}")
    lines.append("")
    lines.append(f"   rho(g, pass_rate) = {r.rho_g_passrate:+.4f}    "
                 f"95% CI [{r.rho_ci_lo:+.4f}, {r.rho_ci_hi:+.4f}]")
    lines.append("")
    lines.append(f"   g-bin breakdown  ({len(G_BINS)-1} bins):")
    edges = list(G_BINS)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        lo_s = "-inf" if not np.isfinite(lo) else f"{lo:+.1f}"
        hi_s = "+inf" if not np.isfinite(hi) else f"{hi:+.1f}"
        lines.append(f"     g in [{lo_s:>5}, {hi_s:>5}):  "
                     f"n_examples = {r.g_bin_counts[i]:>4}   "
                     f"sum pass_rate = {r.g_bin_correct_mass[i]:>7.2f}")
    return "\n".join(lines)


def decompose_grouped(
    df: pd.DataFrame,
    *,
    group_col: str = "source",
    **kwargs,
) -> list[VestResult]:
    """Run :func:`decompose` per-group plus an ALL block."""
    out: list[VestResult] = []
    if group_col in df.columns and df[group_col].nunique() > 1:
        for name in sorted(df[group_col].dropna().unique()):
            sub = df[df[group_col] == name]
            out.append(decompose(sub, label=str(name), **kwargs))
    out.append(decompose(df, label="ALL", **kwargs))
    return out


def synthetic_smoke() -> Sequence[VestResult]:
    """Hand-built tiny dataset for verification.

    Expectations: vision_driven_frac = 0.75, rho > 0.
    """
    df = pd.DataFrame({
        "m_img_sum":   [-3.0, -2.0, -1.0, -5.0, -4.0],
        "m_blank_sum": [-4.0, -4.0, -4.0, -4.0, -4.0],
        "pass_rate_g16": [0.50, 0.75, 1.00, 0.25, 0.50],
    })
    return [decompose(df, label="SMOKE", bootstrap=200, seed=0,
                      noise_floor_nat=None)]
