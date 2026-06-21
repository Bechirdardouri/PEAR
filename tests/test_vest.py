"""Lightweight smoke tests for the VEST decomposition.

These do not load any model; they exercise the pure functions in
``pear.vest`` against synthetic data so they run in well under
a second on CPU. The aim is to catch regressions in the math (sum vs.
mean log-prob handling, noise-floor masking, bootstrap CI shape) --
not to certify the empirical numbers reported in REPORT.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pear.vest import decompose


def _synthetic_probe(n: int = 80, vision_share: float = 0.75, seed: int = 0) -> pd.DataFrame:
    """A toy probe-output frame with a known vision-driven fraction.

    ``vision_share`` of the rows have g >> 0 and pass_rate_g16 ~ 1; the
    rest have g << 0 and pass_rate_g16 ~ 0.25. After the default noise
    floor (m_img_sum > -20 nat), the decomposition should report a
    ``vision_driven_frac`` close to ``vision_share``.
    """
    rng = np.random.default_rng(seed)
    n_vis = int(round(n * vision_share))
    n_prior = n - n_vis

    rows = []
    for _ in range(n_vis):
        m_img = rng.uniform(-2.0, -0.5)
        m_blk = rng.uniform(-8.0, -5.0)
        rows.append(dict(
            m_img_sum=m_img, m_blank_sum=m_blk,
            pass_rate_g16=rng.uniform(0.75, 1.0),
        ))
    for _ in range(n_prior):
        m_img = rng.uniform(-3.0, -1.0)
        m_blk = rng.uniform(-3.0, -1.0)
        rows.append(dict(
            m_img_sum=m_img, m_blank_sum=m_blk,
            pass_rate_g16=rng.uniform(0.0, 0.5),
        ))
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def test_decompose_recovers_known_split():
    df = _synthetic_probe(n=200, vision_share=0.75, seed=0)
    r = decompose(df, bootstrap=500, seed=0)

    # Synthetic "vision" rows carry ~3x the pass_rate of "prior" rows, so
    # the mass-weighted vision_driven_frac is meaningfully higher than
    # the 0.75 row share. Just require it lands clearly on the high side.
    assert r.vision_driven_frac >= 0.70
    assert abs(r.vision_driven_frac + r.prior_driven_frac - 1.0) < 1e-6
    assert r.rho_g_passrate > 0
    assert r.rho_ci_lo <= r.rho_g_passrate <= r.rho_ci_hi


def test_noise_floor_drops_garbage_rows():
    df = _synthetic_probe(n=100, vision_share=0.5, seed=0)
    garbage = pd.DataFrame([
        dict(m_img_sum=-50.0, m_blank_sum=-50.0, pass_rate_g16=0.0)
        for _ in range(50)
    ])
    df_all = pd.concat([df, garbage], ignore_index=True)
    r = decompose(df_all, bootstrap=200, seed=0)

    assert r.n_examples == 100
    assert any("noise floor" in n for n in r.notes)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_decompose_is_deterministic(seed):
    df = _synthetic_probe(n=120, vision_share=0.6, seed=seed)
    a = decompose(df, bootstrap=300, seed=42)
    b = decompose(df, bootstrap=300, seed=42)
    assert a.vision_driven_frac == b.vision_driven_frac
    assert a.rho_g_passrate == b.rho_g_passrate
    assert a.rho_ci_lo == b.rho_ci_lo and a.rho_ci_hi == b.rho_ci_hi


def test_bin_counts_sum_to_n_examples():
    df = _synthetic_probe(n=150, vision_share=0.6, seed=3)
    r = decompose(df, bootstrap=200, seed=0)
    assert sum(r.g_bin_counts) == r.n_examples
