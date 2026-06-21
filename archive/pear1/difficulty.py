"""Per-example difficulty: sample G answers, score pass@1 / pass@k.

Also samples G answers with a blank white image (NEED baseline) so we
can ask, per example, whether the model could have produced the right
answer from the text alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .config import Config
from .scoring import sample_answers
from .verifiers import verify


@dataclass(frozen=True)
class DifficultyResult:
    pass_at_1: bool
    pass_at_k: bool
    pass_rate: float       # fraction of G samples correct
    mean_logprob: float    # mean over G of sample-mean logprob
    blank_pass_at_k: bool
    blank_pass_rate: float


def _blank_image_like(image: Image.Image) -> Image.Image:
    """White image with the same size as ``image`` (preserves layout cost)."""
    return Image.new("RGB", image.size, color=(255, 255, 255))


def evaluate_sampling(
    model,
    processor,
    record: dict,
    cfg: Config,
    example_seed: int,
) -> DifficultyResult:
    image = record["image"]
    question = record["question"]
    gold = record["answer"]
    ans_type = record["answer_type"]

    # G samples with the real image.
    samples = sample_answers(
        model, processor, image, question, cfg,
        n=cfg.G, seed=example_seed,
    )
    correct = [verify(s.text, gold, ans_type) for s in samples]
    pass_at_1 = bool(correct[0]) if correct else False
    pass_at_k = any(correct)
    pass_rate = sum(correct) / len(correct) if correct else 0.0
    mean_logprob = sum(s.mean_logprob for s in samples) / max(len(samples), 1)

    # G samples with a blank image (NEED baseline). Same G, same seed
    # offset so the comparison is fair.
    blank = _blank_image_like(image)
    blank_samples = sample_answers(
        model, processor, blank, question, cfg,
        n=cfg.G, seed=example_seed + 1,
    )
    blank_correct = [verify(s.text, gold, ans_type) for s in blank_samples]
    blank_pass_at_k = any(blank_correct)
    blank_pass_rate = (sum(blank_correct) / len(blank_correct)
                       if blank_correct else 0.0)

    return DifficultyResult(
        pass_at_1=pass_at_1,
        pass_at_k=pass_at_k,
        pass_rate=pass_rate,
        mean_logprob=mean_logprob,
        blank_pass_at_k=blank_pass_at_k,
        blank_pass_rate=blank_pass_rate,
    )
