"""All tunable knobs for VEST experiments.

Edit values here or override via CLI flags in ``python -m pear``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

# Qwen ViT patch math: 14px patches x 2 spatial merge = 28px per token.
_QWEN_TOKEN_PX = 28 * 28


@dataclass(frozen=True)
class Config:
    # -- Model ---------------------------------------------------------
    model_id: str = "Qwen/Qwen3.5-9B"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_implementation: str | None = None  # "flash_attention_2" if installed
    # Disable Qwen3.5 thinking mode for teacher-forced scoring.
    enable_thinking: bool = False

    # -- Image preprocessing ------------------------------------------
    min_pixels: int = 256 * _QWEN_TOKEN_PX
    max_pixels: int = 768 * _QWEN_TOKEN_PX

    # -- Rollout sampling ---------------------------------------------
    G: int = 16
    temperature: float = 0.7
    top_p: float = 0.80
    top_k: int = 20
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    max_new_tokens: int = 64

    # -- Data ----------------------------------------------------------
    n_per_source: Mapping[str, int] = field(
        default_factory=lambda: {"chartqa": 800}
    )

    # -- IO ------------------------------------------------------------
    out_dir: Path = Path("./outputs")
    parquet_path: Path = Path("./outputs/results.parquet")

    # -- Reproducibility ----------------------------------------------
    seed: int = 0

    def ensure_dirs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
