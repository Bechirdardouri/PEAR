"""Load any ``AutoModelForImageTextToText`` + matching processor.

The previous codebase (``archive/pear1/model.py``) also exposed a
``NoiseHook`` for vision-tower perturbation. That hook was specific to
the abandoned probe-based selection line; VEST does not need it. If a
future audit pass needs perturbed vision (e.g. patch-shuffle as a
second counterfactual), reintroduce it as ``pear.perturb``.
"""

from __future__ import annotations

import torch

from .config import Config


def load_model_and_processor(cfg: Config):
    """Load model + processor in bf16 with flash-attn-2 when available."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = getattr(torch, cfg.dtype)
    kwargs: dict = {"dtype": dtype, "device_map": cfg.device_map}

    attn_impl = cfg.attn_implementation
    if attn_impl is None:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = None
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    model = AutoModelForImageTextToText.from_pretrained(cfg.model_id, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        cfg.model_id,
        min_pixels=cfg.min_pixels,
        max_pixels=cfg.max_pixels,
    )
    return model, processor
