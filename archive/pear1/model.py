"""Model loading, vision-module detection, and the NoiseHook.

The NoiseHook injects scaled Gaussian noise on the *output of the
vision module* via a forward hook. This is architecture-agnostic: any
VLM whose vision tower is a discoverable submodule works without
touching modelling code.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn

from .config import Config


# Names commonly used by VLM vision towers. Ordered by specificity.
_VISION_NAME_HINTS = ("visual", "vision_tower", "vision_model")


def load_model_and_processor(cfg: Config):
    """Load Qwen3.5-2B (or any AutoModelForImageTextToText) + processor.

    Returns ``(model.eval(), processor)``. The model is loaded in
    `cfg.dtype` (bf16 by default) and dispatched via ``device_map``.
    If ``cfg.attn_implementation`` is None, we auto-prefer
    ``flash_attention_2`` when ``flash_attn`` is importable (big win on
    H100); otherwise we fall back to the model's default ("sdpa").
    """
    # Imported lazily so `python -m pear print-modules --help` works
    # without torch/transformers installed.
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


def find_vision_module(model: nn.Module, override: str | None = None) -> nn.Module:
    """Locate the vision module by name.

    Strategy:
      1. If ``override`` is given, resolve it via ``get_submodule``.
      2. Otherwise scan ``named_modules`` for the shallowest module
         whose dotted path's *last segment* matches a known hint.
      3. Raise with a candidate list if zero or multiple top-level
         hits are ambiguous.
    """
    if override is not None:
        return model.get_submodule(override)

    candidates: list[tuple[int, str, nn.Module]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        last = name.split(".")[-1]
        if last in _VISION_NAME_HINTS:
            depth = name.count(".")
            candidates.append((depth, name, module))

    if not candidates:
        raise RuntimeError(
            "Could not find a vision module. Inspect the model with "
            "`python -m pear print-modules` and pass --vision-module."
        )

    # Pick the shallowest hit (top-level vision tower).
    candidates.sort(key=lambda t: t[0])
    min_depth = candidates[0][0]
    shallow = [c for c in candidates if c[0] == min_depth]
    if len(shallow) > 1:
        names = ", ".join(c[1] for c in shallow)
        raise RuntimeError(
            f"Ambiguous vision-module candidates: {names}. "
            "Disambiguate with --vision-module."
        )
    return shallow[0][2]


def print_module_tree(model: nn.Module, max_depth: int = 2) -> None:
    """Print named modules up to ``max_depth``; flag detected vision module."""
    try:
        vis = find_vision_module(model)
        vis_id = id(vis)
    except RuntimeError:
        vis_id = None

    print(f"== Modules of {model.__class__.__name__} (depth ≤ {max_depth}) ==")
    for name, module in model.named_modules():
        if name and name.count(".") > max_depth:
            continue
        marker = "  <-- vision module" if id(module) == vis_id else ""
        cls = module.__class__.__name__
        print(f"  {name or '<root>':50s}  {cls}{marker}")


class NoiseHook:
    """Forward-hook context manager injecting Gaussian noise on vision output.

    The injected noise is additive Gaussian, scaled per row by the
    embedding's own standard deviation along the last dim:

        h_noisy = h + sigma * h.std(-1, keepdim=True) * randn_like(h)

    This degrades semantics *and* fine-grained detail simultaneously
    (the property we want for the perceptual response curve). When
    ``sigma == 0`` the hook is a no-op and is not actually installed.

    Noise is generator-seeded by ``(example_seed, sigma_index)`` so the
    sweep is fully reproducible.
    """

    def __init__(
        self,
        vision_module: nn.Module,
        sigma: float,
        seed: int,
    ) -> None:
        self.vision_module = vision_module
        self.sigma = float(sigma)
        self.seed = int(seed)
        self._handle = None

    def __enter__(self) -> "NoiseHook":
        if self.sigma == 0.0:
            return self
        self._handle = self.vision_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    # -- hook -----------------------------------------------------------
    def _hook(self, _module, _inputs, output):
        return _apply_noise_to_output(output, self.sigma, self.seed)


def _apply_noise_to_output(output, sigma: float, seed: int):
    """Recursively walk tensors in ``output`` and add scaled Gaussian noise.

    Handles plain Tensor, tuple/list of Tensors, ModelOutput subclasses
    (BaseModelOutput, BaseModelOutputWithPooling, etc.), and plain
    dicts. Non-floating tensors are left untouched.
    """
    if isinstance(output, torch.Tensor):
        return _noise_tensor(output, sigma, seed)
    if isinstance(output, (list, tuple)):
        new = [_apply_noise_to_output(o, sigma, seed) for o in output]
        return type(output)(new) if not isinstance(output, list) else new
    # ModelOutput inherits from OrderedDict but the downstream code may
    # access fields as attributes (e.g. `.pooler_output`). Mutate in
    # place so we preserve the original class and its attribute API.
    from transformers.utils import ModelOutput
    if isinstance(output, ModelOutput):
        for k in list(output.keys()):
            v = output[k]
            if torch.is_tensor(v) and v.is_floating_point():
                output[k] = _noise_tensor(v, sigma, seed)
        return output
    if isinstance(output, dict):
        return {k: _apply_noise_to_output(v, sigma, seed) for k, v in output.items()}
    # Fallback: try attribute walk on objects with last_hidden_state.
    if hasattr(output, "last_hidden_state"):
        output.last_hidden_state = _noise_tensor(
            output.last_hidden_state, sigma, seed
        )
        return output
    return output


def _noise_tensor(h: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    if not h.is_floating_point():
        return h
    gen = torch.Generator(device=h.device).manual_seed(seed)
    # Per-row std along the last (feature) dim, broadcast over the rest.
    std = h.detach().float().std(dim=-1, keepdim=True).clamp_min(1e-6)
    noise = torch.empty_like(h, dtype=torch.float32).normal_(generator=gen)
    return (h.float() + sigma * std * noise).to(h.dtype)


@contextmanager
def noise(vision_module: nn.Module, sigma: float, seed: int) -> Iterator[None]:
    """Convenience wrapper: `with noise(vis, 0.5, seed=42): ...`."""
    with NoiseHook(vision_module, sigma, seed):
        yield
