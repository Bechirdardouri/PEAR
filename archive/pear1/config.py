"""All tunable knobs for the PEAR go/no-go experiment.

Edit values here, override via CLI flags in `python -m pear ...`, or
construct a `Config(...)` programmatically in notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

# Qwen ViT patch math: 14px patches × 2 spatial merge = 28px per token.
_QWEN_TOKEN_PX = 28 * 28


@dataclass(frozen=True)
class Config:
    # -- Model ---------------------------------------------------------
    # Default: Qwen3.5-2B unified VL (early-fusion, qwen3_5 model_type,
    # dense per user). Requires transformers from main.
    model_id: str = "Qwen/Qwen3.5-2B"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_implementation: str | None = None  # "flash_attention_2" if installed
    # Optional override for the vision-module name returned by
    # `find_vision_module`. When None, auto-detected.
    vision_module_name: str | None = None
    # Disable Qwen3.5 thinking mode: the model card flags the 2B as
    # especially prone to thinking-loops; we want short direct answers
    # for teacher-forced scoring anyway.
    enable_thinking: bool = False

    # -- Image preprocessing -------------------------------------------
    # Fixed, modest resolution band so an "unreadable" regime exists.
    min_pixels: int = 256 * _QWEN_TOKEN_PX
    max_pixels: int = 768 * _QWEN_TOKEN_PX

    # -- Probe (perceptual response curve) -----------------------------
    # K log-spaced sigmas, sigma=0 included so margins[0] is the clean
    # m0. Sigma is a multiplier on per-row embedding std.
    K: int = 7
    sigma_max: float = 3.0

    # -- Sampling (difficulty) -----------------------------------------
    G: int = 16
    # Qwen3.5 VL-non-thinking recommended sampling parameters.
    temperature: float = 0.7
    top_p: float = 0.80
    top_k: int = 20
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    max_new_tokens: int = 64

    # -- Data ----------------------------------------------------------
    n_per_source: Mapping[str, int] = field(
        default_factory=lambda: {
            "chartqa": 800,
            "ai2d": 800,
            "textvqa": 800,
        }
    )

    # -- IO ------------------------------------------------------------
    out_dir: Path = Path("./outputs")
    parquet_path: Path = Path("./outputs/results.parquet")
    figs_dir: Path = Path("./outputs/figs")
    probe_set_cache: Path = Path("./outputs/probe_set.jsonl")

    # -- Reproducibility -----------------------------------------------
    seed: int = 0

    # -- Analysis ------------------------------------------------------
    pass_rate_bins: int = 10
    # `hard` defined as pass@1 == 0 (primary outcome) — we also report
    # a softer `pass_rate < hard_threshold` slice for sensitivity.
    hard_threshold: float = 0.5
    bootstrap_iters: int = 2000
    # Decision thresholds (see README §Decision rule).
    auroc_go: float = 0.60
    auroc_iterate: float = 0.55
    delta_auroc_go: float = 0.03
    delta_auroc_iterate: float = 0.01

    # ------------------------------------------------------------------
    @property
    def sigmas(self) -> np.ndarray:
        """K log-spaced sigmas in [0, sigma_max]. First entry is 0."""
        rest = np.geomspace(0.05, self.sigma_max, self.K - 1)
        return np.concatenate([[0.0], rest])

    def ensure_dirs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.figs_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
