"""Pick a single global sigma* that drops mean gold-log-prob by 1 nat.

Reuses the 7-element `margins` vector already stored in PEAR-1's
parquet — no new forward passes required.

The contract: sigma* is calibrated *once* and locked. From there on
the perceptual sensitivity of each example is the scalar
    Delta(ex) = m_inf(ex) - m_at_sigma_star(ex)
which is interpretable as "how many nats of confidence this example
loses at the noise level that costs the average example 1 nat."

This subtracts away the model's global noise fragility and isolates
per-example perceptual dependence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# The sigma grid that PEAR-1 used; kept here so calibration is
# reproducible without importing the old Config.
SIGMAS = np.concatenate([[0.0], np.geomspace(0.05, 3.0, 6)])  # len 7
TARGET_DROP_NATS = 1.0


@dataclass(frozen=True)
class Calibration:
    sigma_star: float        # the chosen noise level
    sigma_star_idx: float    # fractional index into SIGMAS for interpolation
    drop_at_sigma_star: float  # achieved mean drop (should ≈ TARGET_DROP_NATS)
    n_examples: int


def calibrate(margins: np.ndarray) -> Calibration:
    """Compute sigma* from a (N, 7) matrix of per-example log-probs.

    margins[i, k] = log P(gold_i | image_i, question_i, sigma=SIGMAS[k]).
    """
    assert margins.ndim == 2 and margins.shape[1] == len(SIGMAS), \
        f"expected (N, {len(SIGMAS)}), got {margins.shape}"
    # Per-example drop relative to sigma=0.
    drops = margins[:, :1] - margins                # (N, 7); drops[:, 0] == 0
    mean_drop = drops.mean(axis=0)                  # (7,)
    # Find the first index where mean drop crosses TARGET_DROP_NATS.
    # mean_drop is monotone-ish increasing in sigma; we linearly
    # interpolate the crossing.
    if mean_drop[-1] < TARGET_DROP_NATS:
        # The biggest sigma in the grid isn't enough; clamp.
        return Calibration(
            sigma_star=float(SIGMAS[-1]),
            sigma_star_idx=float(len(SIGMAS) - 1),
            drop_at_sigma_star=float(mean_drop[-1]),
            n_examples=int(margins.shape[0]),
        )
    above = np.where(mean_drop >= TARGET_DROP_NATS)[0]
    k = int(above[0])
    if k == 0:
        return Calibration(
            sigma_star=float(SIGMAS[0]),
            sigma_star_idx=0.0,
            drop_at_sigma_star=float(mean_drop[0]),
            n_examples=int(margins.shape[0]),
        )
    # Linear interpolation between SIGMAS[k-1] and SIGMAS[k].
    d_lo, d_hi = mean_drop[k - 1], mean_drop[k]
    s_lo, s_hi = SIGMAS[k - 1], SIGMAS[k]
    t = (TARGET_DROP_NATS - d_lo) / (d_hi - d_lo)
    sigma_star = float(s_lo + t * (s_hi - s_lo))
    sigma_star_idx = float((k - 1) + t)
    return Calibration(
        sigma_star=sigma_star,
        sigma_star_idx=sigma_star_idx,
        drop_at_sigma_star=float(d_lo + t * (d_hi - d_lo)),
        n_examples=int(margins.shape[0]),
    )


def m_at_sigma_star(margins: np.ndarray, cal: Calibration) -> np.ndarray:
    """Per-example log-prob at the calibrated sigma*, by linear interp.

    margins: (N, 7). Returns (N,).
    """
    k_floor = int(np.floor(cal.sigma_star_idx))
    k_ceil = min(k_floor + 1, margins.shape[1] - 1)
    t = cal.sigma_star_idx - k_floor
    return (1 - t) * margins[:, k_floor] + t * margins[:, k_ceil]
