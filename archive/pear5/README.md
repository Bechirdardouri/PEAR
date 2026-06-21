# PEAR-5: Visual Information Acquisition (VIA)

A training-free, label-free, multimodal data-selection score for RLVR.
Three teacher-forced forward passes per example; one composed
acquisition score; head-to-head against random / oracle baselines
under the proper bootstrap null.

## TL;DR

**On Qwen3.5-2B / ChartQA, PEAR-5 produces a clean NO-GO for the VIA
hypothesis.** No label-free score we constructed from `m_img`,
`m_blank`, `m_shuf` (or their composites) selects high-`r_var`
examples better than uniform random under a proper bootstrap null
(B=2000). The `m_shuf` (patch-shuffled) baseline carries `coef ≈
-0.001` in a 3-feature logistic regression — i.e. strictly redundant
given `m_img` + `m_blank`.

This is the falsifiable claim the framework was built to test, and
the data answer is "no".

## Probes

Three teacher-forced gold-answer log-probs (length-normalized):

| probe | image fed to vision tower | meaning |
|---|---|---|
| `m_img`   | real image | full multimodal support |
| `m_blank` | grey RGB(127), same size | prior (text + image-presence) |
| `m_shuf`  | patch-shuffled (32 px tiles), same size | "image-shaped prior" |

Derived axes:

```
delta_vis  = m_img - m_blank      # "vision matters at all"
delta_spec = m_img - m_shuf       # "*this* image matters"
```

## Acquisition scores

The 2026 RLVR-selection literature (PODS, InSight, Prompt Replay,
DEPO) converges on the form `uncertainty × value-of-information`.

```
p_hat   = sigmoid(alpha * m_img + beta)       # one-shot Platt cal
var_hat = p_hat * (1 - p_hat)                 # GRPO advantage proxy
VIA      = var_hat * delta_spec               # ours
VIA_vis  = var_hat * delta_vis                # ablation
band_multi  = -|p_multi - 0.5|                # p_multi from 3-feature logreg
band_m_img  = -|sigmoid(alpha*m_img+beta) - 0.5|
```

Both signs of every Δ-based score are reported (`+`, `_neg`) since
the selection direction is part of the score and reading it from data
is principled when both are scored together.

## Headline result (chartqa, n=800, G=16)

```
random null (B=2000) for top-K mean r_var:
  k=10   mean=0.1042   5%..95% = [0.0519, 0.1571]
  k=25   mean=0.1045   5%..95% = [0.0733, 0.1370]
  k=50   mean=0.1042   5%..95% = [0.0824, 0.1266]
  k=100  mean=0.1043   5%..95% = [0.0889, 0.1198]

best label-free top50_mean across ALL scores: 0.0977 (= random)
→ no label-free score clears the random 95% upper bound at any K.
```

Multi-feature logreg fit (the right operator for `r_var` selection):

```
coef [m_img, m_blank, m_shuf] = [-0.222, +0.100, -0.001]
                                                  ^^^^^^^
                                                  redundant
```

The `m_shuf` probe carries **zero predictive signal** beyond `m_img`
and `m_blank`. The third forward pass adds no information.

## Why this is the right negative result (not a coding mistake)

1. **Probes work**: `corr(delta_vis, delta_spec) = 0.65` — the two
   axes are different but related, as designed.
2. **Spearman signal exists**: `delta_spec_neg` ρ = +0.157 (CI
   [+0.087, +0.224]) against `r_var`. Real, just monotone in
   `pass_rate` rather than concave in `p(1-p)`.
3. **The proper operator is the band**: `band_multi` is the only
   label-free score that beats a single random draw on top-K mean.
   It does not beat the bootstrap null because the m_img→pass_rate
   relationship has the wrong sign at length-normalized log-prob
   (`ρ(m_img, pass_rate) = -0.14`), reflecting a length-bias in
   teacher-forced scoring.

This sign reversal is itself the punchline. Teacher-forced
log-prob of a short numeric answer is dominated by token-length
effects on a 2B-parameter model; the "easy" answers (high
pass_rate) get short, low-entropy completions, but their gold
strings are *not* the highest per-token log-prob in the dataset.
A proper acquisition score on this model needs either (a) total
log-prob (not length-normalized), (b) a free-form-sampling proxy
of pass_rate (which defeats the "training-free" goal), or (c) a
larger backbone where the per-token log-prob landscape is less
length-warped.

## File map

| file | role |
|---|---|
| [pear5/__init__.py](pear5/__init__.py) | version + design docstring |
| [pear5/probe.py](pear5/probe.py) | `shuffled_image`, `run_probe`, `run_probe_full_chartqa` |
| [pear5/calibrate.py](pear5/calibrate.py) | Platt calibration `m_img → p_hat` |
| [pear5/analyze.py](pear5/analyze.py) | scores, Spearman + NDCG@k + bootstrap |
| [pear5/__main__.py](pear5/__main__.py) | CLI: `probe`, `probe-full`, `analyze` |

## Reproducing

```bash
# (PEAR-1/3/4 outputs are prerequisites in outputs/)

# Path A: on the chartqa-hard ∧ ¬blank-solvable subset (n=358, G=64)
python -u -m pear5 probe        # ~45s on H100
python -m pear5 analyze         # ~30s

# Path B: on full chartqa (n=800, G=16) — the headline regime
python -u -m pear5 probe-full   # ~1.5min on H100
python -m pear5 analyze --parquet outputs/results_pear5_full.parquet --g 16
```

## What replaces this

The cleanest next step is not another probe-engineering iteration —
the falsifier has spoken on this model. Two natural paths forward:

1. **Backbone scaling**: rerun PEAR-5 on Qwen3-VL-8B or Gemma-3-12B
   where the length-bias is weaker. If `m_shuf` coef stays ≈0, the
   shuffled-patch baseline is dead; if it picks up, this was a
   small-model artefact.
2. **Drop length normalization**: re-emit `m_img` / `m_blank` /
   `m_shuf` as *total* log-prob and refit `band_multi`. This breaks
   the "comparable across questions" property but may restore the
   right sign on the small model.

Neither path requires running the GRPO loop. Both are clean go/no-go
extensions of the existing parquet pipeline.
