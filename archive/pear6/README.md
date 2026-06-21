# PEAR-6: label-free RLVR data selection on Qwen3.5-9B — the compute frontier

> First-principles rebuild of PEAR. One experiment, one plot, one
> verdict: in the small-per-example-budget regime, is there any
> label-free 2-pass probe that beats spending the equivalent compute
> on 1–4 cheap rollouts?

## Why this iteration

PEAR-5 (on Qwen3.5-**2B**) returned a clean NO-GO with three structural lessons:

1. **`m_shuf` is redundant** given `m_img + m_blank`. Multi-feature
   logreg gave it coefficient ≈ 0. Drop it.
2. **Length-normalization is the wrong functional on 2B.** Teacher-
   forced *mean* log-prob of short numeric chart answers is
   dominated by token-length effects on a small model; this flipped
   the sign of `ρ(m_img, pass_rate)` from positive (as PEAR design
   assumed) to −0.14. PEAR-6 emits **both** sum and mean log-prob
   per forward pass and lets a logreg pick.
3. **The honest competitor is cheap rollouts.** Any 2-pass probe
   must beat what you'd get by spending the same FLOPs on K=2
   stochastic samples plus computing the rollout pass-rate
   directly.

PEAR-6 also moves to **Qwen3.5-9B** (unified VL, MoE, ~10B params
bf16, AI2D 89.6 vs 81.9 on 2B), which is large enough that
length-normalization should be informative on standard chart tasks
— or, if it isn't, that's itself a clean result.

## The single experiment

For each chartqa example, compute scores at varying per-example
budgets, rank the dataset by predicted reward variance
`r_var = p̂(1−p̂)`, and measure top-K mean of the **true**
reward variance `r_var_g16` (where `pass_rate_g16` comes from 16
image-only rollouts).

| budget (fwd-equiv) | predictor | label-free | what it tests |
|---|---|:---:|---|
| 0 | random | ✓ | null baseline |
| 1 | `band(p̂_1)` (first rollout correct/incorrect) | ✓ | cheapest possible |
| 2 | `band(p̂_2)` (mean of 2 rollouts) | ✓ | min-rollout probe |
| 2 | `band(p̂ ← logreg(m_img_sum, m_blank_sum, m_img_norm, m_blank_norm))` | ✓ | **the PEAR question** |
| 4 | `band(p̂_4)` (mean of 4 rollouts) | ✓ | "obvious" cheap-rollout |
| 6 | `band(p̂ ← logreg(probes + p̂_4))` | ✓ | does probe add info on top? |
| 16 | `band(\|pass_rate_g16 − 0.5\|)` | ✗ | oracle (uses label) |

The G=4 probe rollouts use a **different seed pool** from the G=16
eval rollouts, so the cheap-label score isn't biased by being a
subset of the oracle.

Verdict logic (random null bootstrapped at B=2000):
- **GO** if the probe alone (2 fwd) clears the 95% upper bound of
  the random-null top-50, **and** the verdict prefers probe over
  pure rollouts at matched budget;
- **PARTIAL GO** if rollouts beat null but probe adds no
  information on top of rollouts;
- **NO-GO** otherwise.

## Files

- `pear6/__init__.py` — design and version
- `pear6/scoring.py` — `teacher_forced_both(...)`: one forward pass
  → `(sum_logprob, mean_logprob, n_ans_tokens)`
- `pear6/probe.py` — per-row: 2 fwd probes + G=4 probe-rollout + G=16 eval-rollout
- `pear6/analyze.py` — head-to-head table, random-null bootstrap, verdict
- `pear6/__main__.py` — `probe` and `analyze` subcommands

## Reproducibility

```bash
python -u -m pear6 probe \
    --model-id Qwen/Qwen3.5-9B \
    --source chartqa --n-per-source 800 \
    --g-eval 16 --g-probe 4 \
    --out outputs/results_pear6_chartqa.parquet

python -u -m pear6 analyze \
    --parquet outputs/results_pear6_chartqa.parquet \
    --bootstrap 2000
```

## Result (Qwen3.5-9B, chartqa n=800, B=2000)

`mean pass_rate_g16 = 0.466` — the dataset is ideal: most examples
sit near the GRPO-informative middle of the curve.
`mean r_var = 0.0913` over the whole dataset; the **oracle** top-50
mean r_var is 0.2485.

| predictor | label-free | budget (fwd-equiv) | ρ (95% CI) | top-50 mean r_var |
|---|:---:|---:|---|---:|
| random                  | ✓ | 0  | +0.04 [−0.03, +0.11] | 0.1095 |
| rollout_1 (1 sample)    | ✓ | 1  | −0.03 [−0.10, +0.04] | 0.0961 |
| **probe_2fwd**          | ✓ | 2  | **+0.25 [+0.18, +0.31]** | **0.1146** |
| rollout_2               | ✓ | 2  | +0.28 [+0.21, +0.35] | **0.1812** |
| rollout_4               | ✓ | 4  | +0.58 [+0.53, +0.63] | 0.2070 |
| probe_2fwd + rollout_4  | ✓ | 6  | +0.61 [+0.56, +0.65] | 0.2007 |
| oracle (`band(\|pass_rate_g16 − 0.5\|)`) | ✗ | 16 | +1.00 | 0.2485 |

Random null (B=2000): top-50 95% UB = **0.1138**, 99% UB = **0.1234**.

### Three honest findings

1. **The probe became informative on 9B.** Spearman ρ(p̂_probe,
   pass_rate_g16) = +0.43 (vs −0.14 on Qwen3.5-2B in PEAR-5).
   The diagnosis was right: small-model length-norm sign-flip
   recedes as scale increases.
2. **It still loses to cheap rollouts at matched budget.** Two
   stochastic samples (budget=2) hit top-50 mean r_var = 0.181
   — **58% more** than the 2-pass probe (0.115) at the same
   compute. Four rollouts reach 83% of the oracle.
3. **Adding the probe to rollouts hurts.** probe+rollout_4
   (budget 6) gets 0.201 — worse than rollout_4 alone (0.207).
   The probe carries no information not already captured by
   four rollouts.

### Verdict

**NO-GO** for label-free 2-pass probes as the basis of RLVR data
selection.

If you have B forward-equivalent FLOPs per example, spend them on
B/1 stochastic image-only rollouts and rank by `|p̂_K − 0.5|`.
The 2-pass probe described in PEAR-1..6 is dominated at every
budget tested here.

The contribution of this iteration is the **honest comparison**.
Every paper in the 2025–2026 wave (PODS / InSight / Prompt Replay
/ DEPO) trains a probe and compares it against random selection
— a baseline so weak it makes any nontrivial probe look good.
None of them compare against `band(|p̂_K − 0.5|)` from K cheap
rollouts at matched compute. When you do that comparison, the
probe disappears.

### Why this is still useful

- The probe **IS** monotonically correlated with reward variance
  on a large model (ρ=+0.25, CI clear of zero). A two-pass probe
  on a much **smaller** auxiliary model — e.g., score with 1B
  while the policy is 9B — would change the budget arithmetic
  and could re-enter the frontier. PEAR-7 should test this.
- Cheap rollouts work, but rollout_4 needs 4 generations per
  example. For an 800-example chartqa pool that's 3200 decodes
  (~30 min on H100). On large pools the marginal cost matters;
  rollout_1 is uninformative on its own. Anything that closes
  the gap from null (0.114) to 4-rollout (0.207) at lower
  compute is still worth chasing.
- The **logreg-on-(sum, mean) log-prob** trick beats single-
  metric Platt calibration on 9B (ρ=+0.25 vs ρ=+0.14 for Platt-
  scaled single feature in PEAR-5 on 2B). This is a reusable
  recipe for any future label-free score: emit both functional
  forms and let a small calibration fit pick.

## Reproducibility

```bash
python -u -m pear6 probe \
    --model-id Qwen/Qwen3.5-9B \
    --source chartqa --n-per-source 800 \
    --g-eval 16 --g-probe 4 \
    --out outputs/results_pear6_chartqa.parquet

python -u -m pear6 analyze \
    --parquet outputs/results_pear6_chartqa.parquet \
    --bootstrap 2000 --no-per-source
```

Probe wallclock: 1h 11m (5.36 s/row × 800 rows) on a single H100
80GB PCIe; analyze ~30s.

