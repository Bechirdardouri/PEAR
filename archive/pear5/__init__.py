"""PEAR-5: Visual Information Acquisition (VIA).

A training-free, label-free, multimodal data-selection score for RLVR.

Three teacher-forced forward passes per example, two orthogonal axes
of "what vision contributes", one composed acquisition score:

    VIA(x) = p_hat * (1 - p_hat) * delta_spec
    p_hat  = sigmoid(alpha * m_img + beta)        # one-shot Platt

Probes (length-normalized log-prob of the GOLD answer, teacher-forced):
    m_img    : log p(gold | full image)
    m_blank  : log p(gold | grey image, same size)             -> PRIOR
    m_shuf   : log p(gold | patch-shuffled image, same size)   -> SPECIFICITY
    delta_vis  = m_img - m_blank   ("vision matters at all")
    delta_spec = m_img - m_shuf    ("*this* image matters")

Positioning (Jun 2026):
    Predicts per-example RLVR reward variance (the GRPO advantage
    proxy), not the noisier "reachable" boolean. Closest related work:
    PODS (TMLR 2026), InSight (Mar 2026), SHIFT (ICML 2026),
    Prompt Replay (Mar 2026), DEPO (Sep 2025). All text-only; PEAR-5
    is the multimodal complement.

PEAR-5 builds on outputs/results_pear4.parquet (chartqa-hard, G=64
labels, m_img, m_blank already computed) and adds only m_shuf +
delta_spec via one additional forward pass per example.
"""

__version__ = "0.5.0"
