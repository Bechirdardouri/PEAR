"""PEAR — Perceptual Edge-of-competence And Reachability go/no-go experiment.

This package tests one falsifiable claim: that the per-example
"perceptual response curve" of a VLM (how its confidence in the gold
answer decays as visual tokens are noised) carries information about
RL-learnability beyond what raw difficulty (pass-rate, mean log-prob)
explains, *after* controlling for difficulty.

See README.md for the full motivation, decision rule, and run recipe.
"""

from .config import Config

__all__ = ["Config"]
__version__ = "0.1.0"
