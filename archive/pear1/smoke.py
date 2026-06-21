"""Synthetic end-to-end smoke test.

Runs the full pipeline on 8 dummy examples with K=3, G=2 so it
finishes in <2 min on a single GPU. Asserts the parquet has the
expected schema. Does NOT assert anything about model accuracy — the
images are synthetic and the answers arbitrary.
"""

from __future__ import annotations

from dataclasses import replace

import pyarrow.parquet as pq

from .config import Config
from .data import synthetic_smoke_set
from .run import run_pipeline


EXPECTED_COLUMNS = {
    "id", "source", "answer_type",
    "m0", "m_inf", "amplitude", "sigma_star", "margins",
    "pass_at_1", "pass_at_k", "pass_rate", "mean_logprob",
    "blank_pass_at_k", "blank_pass_rate",
}


def run_smoke(cfg: Config) -> None:
    smoke_cfg = replace(
        cfg,
        K=3,
        G=2,
        max_new_tokens=8,
        n_per_source={},  # ignored, we pass records directly
        bootstrap_iters=50,
    )
    smoke_cfg.ensure_dirs()

    records = synthetic_smoke_set(n=8)
    path = run_pipeline(smoke_cfg, records=records)

    table = pq.read_table(path)
    cols = set(table.column_names)
    missing = EXPECTED_COLUMNS - cols
    assert not missing, f"smoke: missing columns {missing}"
    assert len(table) >= 1, "smoke: no rows written"
    print(f"\n[smoke] OK: {len(table)} rows, columns = {sorted(cols)}")
