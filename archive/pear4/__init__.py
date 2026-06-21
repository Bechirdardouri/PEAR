"""PEAR-4: visual_axes — the data-grounded rebuild.

Two forward passes per example, no noise sweep, per-source analysis.

Quantities
----------
  m_img    = teacher-forced log-prob of gold | image
  m_blank  = teacher-forced log-prob of gold | grey image       (PRIOR SUPPORT)
  Δ        = m_img - m_blank                                    (VISUAL LIFT)

Thesis (corrected from PEAR-1..3)
---------------------------------
The trainable visual edge is *low prior support + high visual lift*:
examples the model does not already know, but where seeing the image
moves it toward the truth.

This file is the package marker. See `pear4/probe.py`, `pear4/analyze.py`,
and `pear4/__main__.py`.
"""

__version__ = "0.4.0"
