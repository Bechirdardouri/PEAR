"""SEEING — does this VLM see, or guess?

Decomposes a VLM's correctness into two channels:

    g = log p(gold | image, q) - log p(gold | blank, q)        (per example)

    vision_helped(row)  := g > 0        # image moves belief toward truth
    prior_aided(row)    := g <= 0       # image is neutral or hurts truth
    vision_hurt(row)    := g < 0        # image actively suppresses truth

Each correct rollout is then VISION-DRIVEN if its example has g > 0
and PRIOR-DRIVEN otherwise. The "vision-driven fraction" of total
correctness is the central number this module reports.

This is the M1 instrument from the *Seeing or Guessing?* proposal,
run on a single (model, dataset, checkpoint). On a sequence of
checkpoints across an RLVR run, repeating M1 yields the
decomposition-over-training law the paper targets.

Inputs: a parquet with columns
    m_img_sum, m_blank_sum            (sum log-prob of gold over answer span)
    correct_g16_eval (list[int])      (per-rollout correctness)
    pass_rate_g16                     (= mean of correct_g16_eval)
    source                            (optional; per-source breakdown)

Same schema as PEAR-6 (`outputs/results_pear6_*.parquet`).
"""

__version__ = "0.1.0"
