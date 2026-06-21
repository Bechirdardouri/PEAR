"""PEAR-6: the first-principles RLVR-selection compute frontier.

ONE experiment, ONE plot.

For each example, pay a compute budget and rank the dataset by
predicted reward variance r_var = p̂(1-p̂). Compete across budgets:

    0 fwd       random
    2 fwd       band(p̂ ← logreg(m_img_sum, m_blank_sum, m_img_norm, m_blank_norm))
    G=1..8      band(p̂_K) from K image-only rollouts
    2 fwd + G=4 band(p̂ ← logreg(probes + p̂_4))
    G=16        band(|pass_rate - 0.5|)   ← oracle (uses label)

Eval label: r_var_16 = pass_rate_16 × (1 - pass_rate_16) from G=16
image-only sampling.

What's new vs PEAR-1..5:
  1. Backbone: Qwen3.5-9B (vs 2B), kills small-model length-bias confound.
  2. Drop m_shuf (proved redundant in PEAR-5; coef ≈ 0).
  3. Report BOTH per-token AND total log-prob; logreg picks the
     informative one.
  4. Same 16 rollouts serve as K=1,2,4,8 probes AND the G=16 oracle.
     Sub-prefix sampling is biased (probe ⊂ oracle), so we sample
     K=4 with a different seed pool to be independent.
  5. Honest random-null bootstrap (B=2000) for the verdict.

What this answers: in the small-per-example-budget regime, is there
any label-free 2-pass probe that beats spending the equivalent
compute on 1-4 rollouts? This is the question every 2026 paper
(PODS, InSight, Prompt Replay, DEPO) implicitly avoids by either
using lots of rollouts or assuming the proxy is free.
"""

__version__ = "0.6.0"
