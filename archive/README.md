# archive/ — the research diary that led to MIRAGE

Eight iterations across two arcs. Read this if you want to understand
*why* `pear/` looks the way it does — most of the decisions are
load-bearing failures from earlier attempts.

## Arc 1 — PEAR (probe-as-selector)

**The question (original):** can a cheap probe — a few-shot
perturbation of the vision tower's input embeddings — *select* the
hard-but-learnable examples for RLVR, so we spend rollout compute
where it matters?

| iter | what changed | outcome |
|---|---|---|
| `pear/`  | Initial design. NoiseHook injects per-row std-scaled Gaussian noise into the vision module's output; K sigmas yield a "perceptual response curve" m_k. Logreg over (m_img_sum, m_img_norm, m_shuf, m_blank, ...) → predict r_var = p(1-p) from G=16 rollouts. Compute frontier vs. K-rollout band-selection. | Smoke worked. Real run never decisively beat 1-2 rollouts. |
| `pear2/` | Calibrate/score reframe — split into calibrate.py (fit logreg on a holdout) and score.py (apply). | Cleaner code but same selection ceiling. |
| `pear3/` | Minimal redo — strip back to the essentials. | Confirmed there's signal but it's small. |
| `pear4/` | chartqa-focused probe. Smaller scope to find a regime where the probe wins. | Marginal lift on chartqa, not generalizing. |
| `pear5/` | **Key finding: SUM log-prob vs MEAN log-prob disagree on small models.** Length-norm bias was distorting probe rankings on 2B. Switched to reporting both; logreg picks. Also: m_shuf coef ≈ 0 (the shuffle probe was redundant). | Methodological fix, but lift still modest. |
| `pear6/` | The matched-compute NO-GO experiment. One plot: random vs. logreg(2 fwd) vs. K rollouts vs. logreg(2 fwd + 4 rollouts) vs. oracle(G=16). Backbone bumped to Qwen3.5-9B to kill small-model length confound. **Result: the 2-fwd probe does NOT beat spending the equivalent compute on G=2-4 rollouts.** The probe-as-selector line ends here. | NO-GO, but the parquet it wrote — per-row m_img_sum, m_blank_sum, G=16 rollout pass-rates — was *exactly* the input the next arc needed. |

## Arc 2 — SEEING (decompose, don't select)

**The question (pivoted):** given the probe quantities are cheap and
the parquet exists, what do they *say* about why the model gets
things right when it does?

| iter | what changed | outcome |
|---|---|---|
| `seeing/` | Drop selection entirely. Define `g = m_img_sum - m_blank_sum` and decompose the population: how much of the model's win mass comes from `g > 0` (vision helped) vs `g <= 0` (prior already preferred gold)? | **`vision_driven_frac = 0.852` on Qwen3.5-9B chartqa**, with positive Spearman rho between g and pass-rate (bootstrap CI excludes zero). The 85% figure contradicts the field's "VLMs lean on language priors" framing for this regime and is the publishable observation. |

## Why this matters for `pear/`

The PEAR work answered "no" for selection, but produced the
counterfactual `g = m_img - m_blank` as a *measurement* primitive. The
SEEING work showed that measurement, aggregated across a population,
re-frames whether the field's perception-aware RL methods are even
attacking the right population. **`pear/` keeps only the
parts that survived:**

- `pear/score.py` ← `archive/pear1/scoring.py` + `archive/pear6/scoring.py` (the SUM-not-MEAN fix from PEAR-5).
- `pear/model.py` ← `archive/pear1/model.py` (minus `NoiseHook` — the probe is the simple two-forward `g`).
- `pear/data.py` ← `archive/pear1/data.py` (multi-source loaders + stubs for the bigger E1 grid).
- `pear/verifiers.py` ← `archive/pear1/verifiers.py` verbatim.
- `pear/vest.py` ← `archive/seeing/decompose.py` verbatim (renamed).
- `pear/probe.py` ← `archive/pear6/probe.py` distilled (G=16 eval + G=4 cheap-label rollouts per row).
- `pear/audit.py` is new — the multi-checkpoint VEST roll-up the proposal needs.

The two arcs took eight iterations because they were the wrong shape
for the question and the right shape for *finding* the question.
