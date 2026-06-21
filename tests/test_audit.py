"""Tests for ``pear.audit``.

End-to-end test on a synthetic two-checkpoint audit using saved
parquets in a temp directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pear.audit import AuditEntry, audit_table


def _synth_parquet(path: Path, *, vd_target: float, seed: int) -> Path:
    """Write a synthetic probe parquet with a known VDF."""
    rng = np.random.default_rng(seed)
    n = 200
    n_vis = int(round(n * vd_target))
    rows = []
    for _ in range(n_vis):
        rows.append(dict(
            source="synthsrc",
            m_img_sum   = float(rng.uniform(-2.0, -0.5)),
            m_blank_sum = float(rng.uniform(-8.0, -5.0)),
            pass_rate_g16 = float(rng.uniform(0.6, 1.0)),
        ))
    for _ in range(n - n_vis):
        rows.append(dict(
            source="synthsrc",
            m_img_sum   = float(rng.uniform(-3.0, -1.0)),
            m_blank_sum = float(rng.uniform(-3.0, -1.0)),
            pass_rate_g16 = float(rng.uniform(0.0, 0.5)),
        ))
    df = pd.DataFrame(rows)
    df.to_parquet(path)
    return path


def test_audit_table_two_checkpoints(tmp_path: Path) -> None:
    p_base = _synth_parquet(tmp_path / "base.parquet", vd_target=0.50, seed=0)
    p_post = _synth_parquet(tmp_path / "post.parquet", vd_target=0.80, seed=1)

    table = audit_table(
        [AuditEntry("base", p_base), AuditEntry("post", p_post)],
        bootstrap=200, seed=0,
    )
    assert len(table) == 2
    assert set(table["label"]) == {"base", "post"}
    base = table[table["label"] == "base"].iloc[0]
    post = table[table["label"] == "post"].iloc[0]
    # The "post" checkpoint was generated with a much higher vision share;
    # VDF must order them correctly.
    assert post["vision_driven_frac"] > base["vision_driven_frac"]
    # rho is well-defined and inside its CI.
    for row in (base, post):
        assert row["rho_ci_lo"] <= row["rho"] <= row["rho_ci_hi"]


def test_audit_table_grouped_by_source(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for src in ("a", "b"):
        for _ in range(80):
            rows.append(dict(
                source=src,
                m_img_sum=float(rng.uniform(-2, -0.5)),
                m_blank_sum=float(rng.uniform(-6, -3)),
                pass_rate_g16=float(rng.uniform(0.5, 1.0)),
            ))
    df = pd.DataFrame(rows)
    p = tmp_path / "twosrc.parquet"
    df.to_parquet(p)
    table = audit_table(
        [AuditEntry("one", p)],
        bootstrap=100, seed=0, group_by_source=True,
    )
    assert set(table["source"]) == {"a", "b"}
