"""Teacher-forced gold-answer log-prob and free-form sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .config import Config


@dataclass(frozen=True)
class Sample:
    text: str
    mean_logprob: float  # length-normalized log-prob of the generated tokens


def build_chat_inputs(processor, image, question: str, cfg: Config):
    """Build model-ready inputs for a single (image, question) pair.

    Uses ``processor.apply_chat_template`` with ``enable_thinking=False``
    (Qwen3.5 default) so the model is asked to answer directly.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=cfg.enable_thinking,
    )
    return inputs


def build_scoring_inputs(processor, image, question: str, answer: str, cfg: Config):
    """Build inputs for teacher-forced scoring of ``answer``.

    Returns ``(inputs, prompt_length)`` where ``prompt_length`` is the
    number of tokens up to and including the assistant's
    generation-prompt header — i.e. the index from which the answer
    tokens start.
    """
    prompt_only = build_chat_inputs(processor, image, question, cfg)
    prompt_len = int(prompt_only["input_ids"].shape[1])

    # Re-build with the assistant turn closed by the answer text. We do
    # this by appending the answer to the prompt's input_ids and
    # extending the attention mask. This is robust across chat
    # templates (no need to re-tokenize the full chat with an assistant
    # turn, which some templates render differently than continuation).
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
    # Extend any other per-token tensor (e.g. mm_token_type_ids in
    # Qwen3.5) so the model's sequence-length checks pass. Answer
    # tokens are plain text, so we extend with zeros.
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
    model,
    processor,
    image,
    question: str,
    answer: str,
    cfg: Config,
) -> float:
    """Length-normalized log-prob of ``answer`` tokens under the model.

    A single forward pass; returns mean log-prob per answer token.
    """
    inputs, prompt_len = build_scoring_inputs(processor, image, question, answer, cfg)
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v
              for k, v in inputs.items()}
    out = model(**inputs)
    logits = out.logits  # (1, T, V)
    # Token at position t is predicted by logits at position t-1.
    # Answer tokens occupy [prompt_len, T); their predictions come from
    # logits[:, prompt_len - 1 : T - 1, :].
    full_ids = inputs["input_ids"]
    target = full_ids[:, prompt_len:]                       # (1, A)
    pred_logits = logits[:, prompt_len - 1 : -1, :]         # (1, A, V)
    if target.shape[1] == 0:
        return 0.0
    logp = torch.log_softmax(pred_logits.float(), dim=-1)
    token_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (1, A)
    return float(token_logp.mean().item())


@torch.inference_mode()
def sample_answers(
    model,
    processor,
    image,
    question: str,
    cfg: Config,
    n: int,
    seed: int,
) -> list[Sample]:
    """Draw ``n`` answers via stochastic decoding.

    Returns a list of (text, mean_logprob) tuples. Mean log-prob is
    computed from the model's per-step scores so we never re-tokenize.
    """
    inputs = build_chat_inputs(processor, image, question, cfg)
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v
              for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])

    # Deterministic across calls with the same seed.
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
    sequences = out.sequences  # (n, T_total)
    scores = out.scores        # tuple of (n, V) length new_T

    # Compute per-sample mean log-prob over the generated tokens.
    samples: list[Sample] = []
    gen_ids = sequences[:, prompt_len:]   # (n, new_T)
    new_T = gen_ids.shape[1]
    eos_id = processor.tokenizer.eos_token_id
    for i in range(n):
        # Stop at eos (exclude eos token itself from mean).
        valid_T = new_T
        if eos_id is not None:
            eos_positions = (gen_ids[i] == eos_id).nonzero(as_tuple=False)
            if eos_positions.numel():
                valid_T = int(eos_positions[0].item())
        if valid_T == 0:
            samples.append(Sample(text="", mean_logprob=0.0))
            continue
        # log-prob of token t came from scores[t]
        lps = []
        for t in range(valid_T):
            step_logits = scores[t][i]                     # (V,)
            step_logp = torch.log_softmax(step_logits.float(), dim=-1)
            lps.append(float(step_logp[gen_ids[i, t]].item()))
        mean_lp = sum(lps) / len(lps)
        text = processor.tokenizer.decode(
            gen_ids[i, :valid_T], skip_special_tokens=True
        ).strip()
        samples.append(Sample(text=text, mean_logprob=mean_lp))
    return samples
