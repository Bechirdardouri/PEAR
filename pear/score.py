"""Teacher-forced gold-answer log-prob (sum + mean) and free-form sampling.

The central scoring primitive for VEST is :func:`teacher_forced_logprob`,
which returns ``(sum_logprob, mean_logprob, n_ans_tokens)`` for the gold
answer span under the model. ``sum_logprob`` is the correct quantity to
use in the per-example counterfactual

    g = m_img_sum - m_blank_sum

because length-normalization (mean) systematically distorts comparison
between long and short answer strings and was shown in PEAR-5/6 to
flip the sign of the correlation with correctness on small models.

Lifted with minor cleanup from ``archive/pear1/scoring.py`` and
``archive/pear6/scoring.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import Config


@dataclass(frozen=True)
class TFResult:
    sum_logprob: float
    mean_logprob: float
    n_ans_tokens: int


@dataclass(frozen=True)
class Sample:
    text: str
    mean_logprob: float


def build_chat_inputs(processor, image, question: str, cfg: Config):
    """Tokenize a single (image, question) chat turn."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=cfg.enable_thinking,
    )


def build_scoring_inputs(processor, image, question: str, answer: str, cfg: Config):
    """Build inputs for teacher-forced scoring of ``answer``."""
    prompt_only = build_chat_inputs(processor, image, question, cfg)
    prompt_len = int(prompt_only["input_ids"].shape[1])

    tokenizer = processor.tokenizer
    ans_ids = tokenizer(
        answer, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(prompt_only["input_ids"].device)
    ans_len = int(ans_ids.shape[1])
    full_ids = torch.cat([prompt_only["input_ids"], ans_ids], dim=1)
    full_mask = torch.cat(
        [prompt_only["attention_mask"], torch.ones_like(ans_ids)], dim=1
    )
    inputs = dict(prompt_only)
    inputs["input_ids"] = full_ids
    inputs["attention_mask"] = full_mask
    # Extend per-token tensors (e.g. mm_token_type_ids) with zeros.
    for k, v in prompt_only.items():
        if k in ("input_ids", "attention_mask"):
            continue
        if torch.is_tensor(v) and v.dim() >= 2 and v.shape[:2] == (1, prompt_len):
            pad = torch.zeros(
                (1, ans_len, *v.shape[2:]), dtype=v.dtype, device=v.device
            )
            inputs[k] = torch.cat([v, pad], dim=1)
    return inputs, prompt_len


@torch.inference_mode()
def teacher_forced_logprob(
    model, processor, image, question: str, answer: str, cfg: Config,
) -> TFResult:
    """One forward pass; return (sum, mean, n_tokens) of ``answer`` log-probs."""
    inputs, prompt_len = build_scoring_inputs(
        processor, image, question, answer, cfg
    )
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v
              for k, v in inputs.items()}
    out = model(**inputs)
    logits = out.logits                                   # (1, T, V)
    full_ids = inputs["input_ids"]
    target = full_ids[:, prompt_len:]                     # (1, A)
    pred_logits = logits[:, prompt_len - 1 : -1, :]       # (1, A, V)
    A = int(target.shape[1])
    if A == 0:
        return TFResult(sum_logprob=0.0, mean_logprob=0.0, n_ans_tokens=0)
    logp = torch.log_softmax(pred_logits.float(), dim=-1)
    token_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    s = float(token_logp.sum().item())
    return TFResult(sum_logprob=s, mean_logprob=s / A, n_ans_tokens=A)


@torch.inference_mode()
def sample_answers(
    model, processor, image, question: str, cfg: Config,
    n: int, seed: int,
) -> list[Sample]:
    """Draw ``n`` answers via stochastic decoding."""
    inputs = build_chat_inputs(processor, image, question, cfg)
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v
              for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    gen_kwargs = dict(
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        max_new_tokens=cfg.max_new_tokens,
        num_return_sequences=n,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=processor.tokenizer.pad_token_id
        or processor.tokenizer.eos_token_id,
    )
    out = model.generate(**inputs, **gen_kwargs)
    sequences = out.sequences
    scores = out.scores

    samples: list[Sample] = []
    gen_ids = sequences[:, prompt_len:]
    new_T = gen_ids.shape[1]
    eos_id = processor.tokenizer.eos_token_id
    for i in range(n):
        valid_T = new_T
        if eos_id is not None:
            eos_positions = (gen_ids[i] == eos_id).nonzero(as_tuple=False)
            if eos_positions.numel():
                valid_T = int(eos_positions[0].item())
        if valid_T == 0:
            samples.append(Sample(text="", mean_logprob=0.0))
            continue
        lps = []
        for t in range(valid_T):
            step_logits = scores[t][i]
            step_logp = torch.log_softmax(step_logits.float(), dim=-1)
            lps.append(float(step_logp[gen_ids[i, t]].item()))
        mean_lp = sum(lps) / len(lps)
        text = processor.tokenizer.decode(
            gen_ids[i, :valid_T], skip_special_tokens=True
        ).strip()
        samples.append(Sample(text=text, mean_logprob=mean_lp))
    return samples
