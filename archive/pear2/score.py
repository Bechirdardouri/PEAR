"""Partial Spearman with bootstrap CI, and the three-way verdict.

The single statistical test:
    rho_partial(Delta, reachable | m_inf, blank_solvable)

Computed by ranking each variable, regressing both target and Delta on
the ranked controls, and taking the Pearson correlation of residuals.
This is the exact partial-Spearman formulation (Conover 1971).

Bootstrap: 2000 resamples for the 95 % CI on rho_partial.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# Pre-registered decision thresholds. Decided BEFORE seeing PEAR-2 numbers.
GO_EFFECT_FLOOR = 0.10       # lower CI bound must clear this
WEAK_INCLUDES_ZERO_EPS = 0.0  # CI not crossing zero = at least "WEAK"
BOOTSTRAP_ITERS = 2000
RNG_SEED = 0


@dataclass(frozen=True)
class PartialSpearman:
    rho: float
    ci_lo: float          # 95 % bootstrap lower
    ci_hi: float          # 95 % bootstrap upper
    p_value: float        # asymptotic
    n: int                # rows used


def _residualize(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """OLS residuals of y on [1, Z]."""
    n = y.shape[0]
    X = np.column_stack([np.ones(n), Z])
    # Solve normal equations directly (small p, large n, stable enough).
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
) -> tuple[float, float]:
    """Spearman partial correlation of x and y given control matrix Z.

    Returns (rho, p_value). With no controls (Z.shape[1]==0) this
    reduces to plain Spearman. p_value is from the asymptotic
    Pearson test on the residuals after rank transform.
    """
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    if Z.shape[1] == 0:
        rho, p = stats.pearsonr(rx, ry)
        return float(rho), float(p)
    rZ = np.column_stack([stats.rankdata(Z[:, j]) for j in range(Z.shape[1])])
    ex = _residualize(rx, rZ)
    ey = _residualize(ry, rZ)
    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return 0.0, 1.0
    rho, p = stats.pearsonr(ex, ey)
    return float(rho), float(p)


def bootstrap_partial_spearman(
    x: np.ndarray, y: np.ndarray, Z: np.ndarray,
    iters: int = BOOTSTRAP_ITERS, seed: int = RNG_SEED,
) -> PartialSpearman:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    rho, p = partial_spearman(x, y, Z)
    boots = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        boots[i], _ = partial_spearman(x[idx], y[idx], Z[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return PartialSpearman(rho=rho, ci_lo=float(lo), ci_hi=float(hi),
                           p_value=p, n=n)


def verdict(ps: PartialSpearman) -> tuple[str, str]:
    """Pre-registered three-way decision rule on *predictive power*.

    What matters is |rho| (sign just tells us the direction of the
    selector); a negative rho means low-Delta examples are
    preferentially reachable, which is just as usable as the
    positive direction — invert the selector.

    GO   : 95 % CI of |rho| has lower bound > GO_EFFECT_FLOOR
    WEAK : CI of rho excludes 0 in EITHER direction, magnitude < floor
    NO   : CI of rho includes 0

    Returns (verdict, direction) where direction in {"+", "-", "0"}.
    """
    # CI for |rho|: if CI is fully on one side of 0, |lower| of the CI
    # in that direction; if it crosses 0, the |rho| CI includes 0.
    if ps.ci_lo > 0:
        abs_lo = ps.ci_lo
        direction = "+"
    elif ps.ci_hi < 0:
        abs_lo = -ps.ci_hi
        direction = "-"
    else:
        return "NO", "0"
    if abs_lo > GO_EFFECT_FLOOR:
        return "GO", direction
    return "WEAK", direction
