"""Per-example perceptual response curve and shape-feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .model import NoiseHook
from .scoring import teacher_forced_logprob


@dataclass(frozen=True)
class CurveFit:
    m0: float           # clean-image margin (sigma=0)
    m_inf: float        # asymptotic margin (max sigma)
    amplitude: float    # m0 - m_inf  (≥0 when image helps)
    sigma_star: float   # sigma at which margin = m_inf + 0.5*amplitude


def response_curve(
    model,
    processor,
    vision_module,
    record: dict,
    cfg: Config,
    example_seed: int,
) -> np.ndarray:
    """Teacher-forced gold-answer log-prob across ``cfg.sigmas``.

    Returns array of shape (K,) — the perceptual response curve.
    """
    sigmas = cfg.sigmas
    # For ANLS we pick the modal / first reference as the "gold" string
    # to score. (ANLS is for evaluating generations, not for scoring;
    # any one ref is a reasonable target for a single forward pass.)
    answer = record["answer"]
    if isinstance(answer, list):
        answer = answer[0]

    margins = np.zeros(len(sigmas), dtype=np.float64)
    for k, sigma in enumerate(sigmas):
        seed = example_seed * 1000 + k
        with NoiseHook(vision_module, float(sigma), seed=seed):
            margins[k] = teacher_forced_logprob(
                model, processor, record["image"], record["question"], answer, cfg
            )
    return margins


def fit_curve(margins: np.ndarray, sigmas: np.ndarray) -> CurveFit:
    """Extract the four shape features from a (margins, sigmas) curve.

    ``sigma_star`` is the linear-interpolated crossing of the
    half-amplitude line ``m_inf + 0.5*amplitude``. When the curve is
    flat (amplitude ≤ 1e-6) it is set to the max sigma.
    """
    m0 = float(margins[0])
    m_inf = float(margins[-1])
    amplitude = m0 - m_inf
    sigmas = np.asarray(sigmas, dtype=np.float64)

    if amplitude <= 1e-6:
        return CurveFit(m0=m0, m_inf=m_inf, amplitude=amplitude,
                        sigma_star=float(sigmas[-1]))

    half = m_inf + 0.5 * amplitude
    # Find first k where margin drops below `half`. Curve typically
    # monotonic-decreasing; we don't assume strict monotonicity.
    below = np.where(margins <= half)[0]
    if below.size == 0:
        # Never crosses (shouldn't happen given m_inf <= half), pick last.
        sigma_star = float(sigmas[-1])
    else:
        k = int(below[0])
        if k == 0:
            sigma_star = float(sigmas[0])
        else:
            # Linear interp between k-1 and k.
            m_hi, m_lo = float(margins[k - 1]), float(margins[k])
            s_hi, s_lo = float(sigmas[k - 1]), float(sigmas[k])
            if m_hi == m_lo:
                sigma_star = s_lo
            else:
                t = (m_hi - half) / (m_hi - m_lo)
                sigma_star = s_hi + t * (s_lo - s_hi)
    return CurveFit(m0=m0, m_inf=m_inf, amplitude=amplitude,
                    sigma_star=float(sigma_star))
