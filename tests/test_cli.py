"""Tests for ``pear.cli``.

Exercises the CLI parser and the lightweight subcommands (``smoke``,
``decompose``, ``audit``) without touching any model. The heavy
subcommand (``probe``) needs a model on GPU and is not tested here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pear.cli import build_parser, main


def _toy_parquet(path: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    n = 100
    df = pd.DataFrame({
        "source": ["s"] * n,
        "m_img_sum":    rng.uniform(-3, -0.5, size=n),
        "m_blank_sum":  rng.uniform(-6, -2,   size=n),
        "pass_rate_g16": rng.uniform(0.3, 1.0, size=n),
    })
    df.to_parquet(path)
    return path


class TestParser:
    def test_smoke_parses(self):
        ns = build_parser().parse_args(["smoke"])
        assert ns.cmd == "smoke"

    def test_decompose_requires_parquet(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["decompose"])

    def test_audit_entry_format_required(self):
        # The CLI happily accepts malformed entries at parse time;
        # the LABEL:PATH check fires inside main(). Verify both halves:
        ns = build_parser().parse_args(
            ["audit", "--entry", "no-colon-here"]
        )
        assert ns.cmd == "audit"


class TestSmokeCmd:
    def test_returns_zero(self, capsys):
        rc = main(["smoke"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "SMOKE" in out


class TestDecomposeCmd:
    def test_runs_on_toy_parquet(self, tmp_path, capsys):
        p = _toy_parquet(tmp_path / "toy.parquet")
        rc = main([
            "decompose", "--parquet", str(p),
            "--bootstrap", "100", "--noise-floor=-1e9",
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "vision_driven_frac" in out


class TestAuditCmd:
    def test_writes_csv(self, tmp_path, capsys):
        p1 = _toy_parquet(tmp_path / "a.parquet", seed=0)
        p2 = _toy_parquet(tmp_path / "b.parquet", seed=1)
        out_csv = tmp_path / "audit.csv"
        rc = main([
            "audit",
            "--entry", f"base:{p1}",
            "--entry", f"post:{p2}",
            "--bootstrap", "100",
            "--out", str(out_csv),
        ])
        assert rc == 0
        assert out_csv.exists()
        df = pd.read_csv(out_csv)
        assert set(df["label"]) == {"base", "post"}

    def test_malformed_entry_rejected(self, tmp_path):
        with pytest.raises(SystemExit, match="--entry must be"):
            main(["audit", "--entry", "no-colon"])
