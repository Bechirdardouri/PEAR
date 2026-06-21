"""Cross-checkpoint VEST runner — the E3 audit.

Given a list of ``(label, parquet_path)`` tuples (one per checkpoint /
method / training-step), assemble the 2xK audit table:

           |  vision-driven   prior-driven    rho(g, pass_rate)
    -------+-----------------------------------------------------
    base   |     0.85            0.15            +0.24 [+0.16, +0.30]
    +GRPO  |     ...             ...             ...
    +PGPO  |     ...             ...             ...
    +VPPO  |     ...             ...             ...

This is the core deliverable of the MIRAGE paper (see README.md).
The runner is intentionally thin -- audit data come from probe runs
that this package already produces, so adding a new method or
checkpoint means: train it, run :mod:`pear.probe`, append the
parquet path to the audit list, re-run :func:`audit_table`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .vest import VestResult, decompose, format_result


@dataclass(frozen=True)
class AuditEntry:
    label: str
    parquet_path: Path


def audit_table(
    entries: list[AuditEntry],
    *,
    pass_rate_col: str = "pass_rate_g16",
    img_col: str = "m_img_sum",
    blank_col: str = "m_blank_sum",
    bootstrap: int = 2000,
    seed: int = 0,
    group_by_source: bool = False,
) -> pd.DataFrame:
    """Run VEST on each entry; return a long-form DataFrame.

    Columns: label, source, n_examples, vision_driven_frac,
    prior_driven_frac, mean_g, rho, rho_ci_lo, rho_ci_hi,
    frac_vision_helped, frac_vision_hurt.
    """
    rows: list[dict] = []
    for entry in entries:
        df = pd.read_parquet(entry.parquet_path)
        if group_by_source and "source" in df.columns:
            srcs = sorted(df["source"].dropna().unique())
        else:
            srcs = [None]
        for src in srcs:
            sub = df if src is None else df[df["source"] == src]
            label = entry.label if src is None else f"{entry.label}/{src}"
            res = decompose(
                sub, label=label, pass_rate_col=pass_rate_col,
                img_col=img_col, blank_col=blank_col,
                bootstrap=bootstrap, seed=seed,
            )
            rows.append(_to_row(res, entry.label, src))
    return pd.DataFrame(rows)


def _to_row(res: VestResult, label: str, source: str | None) -> dict:
    return {
        "label":               label,
        "source":              source or "ALL",
        "n_examples":          res.n_examples,
        "n_correct_rollouts":  res.n_correct_rollouts,
        "vision_driven_frac":  res.vision_driven_frac,
        "prior_driven_frac":   res.prior_driven_frac,
        "mean_g":              res.mean_g,
        "frac_vision_helped":  res.frac_vision_helped,
        "frac_vision_hurt":    res.frac_vision_hurt,
        "pass_rate_g_pos":     res.pass_rate_g_pos,
        "pass_rate_g_nonpos":  res.pass_rate_g_nonpos,
        "rho":                 res.rho_g_passrate,
        "rho_ci_lo":           res.rho_ci_lo,
        "rho_ci_hi":           res.rho_ci_hi,
    }


def print_audit(table: pd.DataFrame) -> None:
    """Pretty-print the audit table."""
    cols = [
        "label", "source", "n_examples",
        "vision_driven_frac", "prior_driven_frac",
        "mean_g", "rho", "rho_ci_lo", "rho_ci_hi",
    ]
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
