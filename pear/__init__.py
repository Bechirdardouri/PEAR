"""pear — a measurement-first toolkit for RL in vision-language models.

This package replaces the six PEAR iterations and the SEEING module
(archived under ``archive/``). It is intentionally small: one
counterfactual ``g = log p(gold|image,q) - log p(gold|blank,q)``
computed per example, summarized at the population level (VEST), and
used to audit perception-aware RLVR methods across models, datasets,
and checkpoints.

The paper this package supports is sketched in :doc:`README.md`.

Modules
-------
config      Tunable knobs (model id, sampling, paths).
model       Load any ``AutoModelForImageTextToText`` + processor.
data        Multi-dataset loaders (chartqa / ai2d / textvqa today).
verifiers   Pure answer verifiers (mc / numeric / exact / anls).
score       Teacher-forced gold-answer log-prob (sum + mean).
vest        The vision-vs-prior decomposer (the central instrument).
probe       Per (model, dataset) probe driver writing one parquet.
audit       Cross-checkpoint VEST runner (E3 in the proposal).
cli         Single CLI: ``python -m pear {probe,decompose,smoke,audit}``.
"""

__version__ = "0.1.0"
