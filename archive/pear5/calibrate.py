"""Platt calibration: map teacher-forced m_img to expected pass rate.

One-shot logistic regression of ``pass_rate_g > 0.5`` on ``m_img``
using a stratified subset of rows. Returns a callable that maps
m_img -> p_hat in [0, 1].

This is a calibration, not a held-out generalization test: alpha/beta
are fit on the same parquet they're applied to. The role is to turn
the unbounded log-prob into a probability so we can compute the
GRPO-objective proxy p_hat * (1 - p_hat).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class PlattCalibration:
    alpha: float
    beta: float
    n_fit: int

    def predict(self, m_img: np.ndarray) -> np.ndarray:
        z = self.alpha * np.asarray(m_img, dtype=float) + self.beta
        return 1.0 / (1.0 + np.exp(-z))


def fit_platt(
    m_img: np.ndarray,
    pass_rate: np.ndarray,
    *,
    n_cal: int = 50,
    seed: int = 0,
    threshold: float = 0.5,
) -> PlattCalibration:
    """Fit p_hat = sigmoid(alpha * m_img + beta).

    Uses a stratified subsample of size ``n_cal`` (by pass-rate
    quintile) for fitting. Falls back to using all rows if n < n_cal.
    """
    m_img = np.asarray(m_img, dtype=float)
    pr = np.asarray(pass_rate, dtype=float)
    y = (pr > threshold).astype(int)

    n = len(m_img)
    if n <= n_cal:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        # Stratify by pass-rate quintile (best effort; small groups
        # contribute everything they have).
        try:
            quintiles = np.quantile(pr, np.linspace(0, 1, 6))
            bins = np.clip(np.digitize(pr, quintiles[1:-1]), 0, 4)
            per_bin = max(1, n_cal // 5)
            chosen: list[int] = []
            for b in range(5):
                pool = np.where(bins == b)[0]
                if len(pool) == 0:
                    continue
                take = min(per_bin, len(pool))
                chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
            idx = np.array(sorted(set(chosen)))
            if len(idx) < n_cal // 2:
                idx = rng.choice(n, size=n_cal, replace=False)
        except Exception:                                  # noqa: BLE001
            idx = rng.choice(n, size=n_cal, replace=False)

    Xf = m_img[idx].reshape(-1, 1)
    yf = y[idx]

    if yf.sum() == 0 or yf.sum() == len(yf):
        # Degenerate: all one class. Return a flat mapping at the empirical rate.
        p = float(y.mean())
        # alpha=0, beta=logit(p)
        eps = 1e-6
        p = min(max(p, eps), 1 - eps)
        beta = float(np.log(p / (1 - p)))
        return PlattCalibration(alpha=0.0, beta=beta, n_fit=int(len(idx)))

    lr = LogisticRegression(solver="lbfgs", max_iter=200)
    lr.fit(Xf, yf)
    return PlattCalibration(
        alpha=float(lr.coef_[0, 0]),
        beta=float(lr.intercept_[0]),
        n_fit=int(len(idx)),
    )
