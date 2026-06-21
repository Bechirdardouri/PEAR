"""PEAR-6 scoring: teacher-forced log-prob returning BOTH sum and per-token mean.

The single fix PEAR-5 surfaced: length-normalization warps the
predictive landscape on small models, so always emit both forms and
let downstream logreg pick.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pear.config import Config
from pear.scoring import build_scoring_inputs


@dataclass(frozen=True)
class TFResult:
    sum_logprob: float          # sum over answer tokens
    mean_logprob: float         # mean over answer tokens (length-normalized)
    n_ans_tokens: int


@torch.inference_mode()
def teacher_forced_both(
    model,
    processor,
    image,
    question: str,
    answer: str,
    cfg: Config,
) -> TFResult:
    """One forward pass; returns sum + mean log-prob of ``answer`` tokens."""
    inputs, prompt_len = build_scoring_inputs(
        processor, image, question, answer, cfg
    )
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v
              for k, v in inputs.items()}
    out = model(**inputs)
    logits = out.logits                                       # (1, T, V)
    full_ids = inputs["input_ids"]
    target = full_ids[:, prompt_len:]                         # (1, A)
    pred_logits = logits[:, prompt_len - 1 : -1, :]           # (1, A, V)
    A = int(target.shape[1])
    if A == 0:
        return TFResult(sum_logprob=0.0, mean_logprob=0.0, n_ans_tokens=0)
    logp = torch.log_softmax(pred_logits.float(), dim=-1)
    token_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    s = float(token_logp.sum().item())
    return TFResult(sum_logprob=s, mean_logprob=s / A, n_ans_tokens=A)
