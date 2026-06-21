"""PEAR-3: integrated-gradients endpoint attribution.

One probe (m_blank), one quantity (Delta_attr = m_inf - m_blank),
one test (partial Spearman | m_inf), three-way verdict.

Replaces PEAR-1's seven-sigma noise sweep with the canonical
information-free baseline: a uniform-grey image of the same size.
"""

__version__ = "0.3.0"
