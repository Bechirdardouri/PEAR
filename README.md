# PEAR

**Perceptual Edge Audit for RL in vision-language models.**

A measurement-first toolkit. The package provides a single, two-forward-pass
diagnostic that asks, of every benchmark example: *did seeing the image
actually move the model toward the correct answer?*

The diagnostic is called **VEST** — Vision-vs-prior Equity Score Test.
Run it on any teacher-forced VLM and any VQA benchmark in a few minutes
on a single GPU.

```
g(x) = log p(gold | image, question) − log p(gold | blank, question)
```

A positive `g` means the image moved belief toward gold (the example is
*vision-driven*); a non-positive `g` means the model already preferred
gold from the question alone (the example is *prior-driven*).
Aggregated across a benchmark, the **vision-driven fraction of correct
rollouts** is the share of model successes that can be attributed to
seeing rather than guessing.

## Why this matters

Through 2025–2026 a new "perception-aware" RL algorithm appeared roughly
every six weeks — PAPO, VPPO, PGPO, PRPO, Vision-SR1, SRPO, PDCR, PEPO,
VGPO, Perceval. They share a common premise: the bottleneck in VLM
reasoning is insufficient grounding in the image, and the way to fix it
is to regularize the policy gradient toward image-conditional behaviour.

That premise is rarely measured. None of these papers report what
fraction of their model's correct answers actually depend on the image.
Without that number, the cost-benefit story for a grounding regularizer
is unfalsifiable.

This repository measures it.

## The headline result

We ran VEST across `{Qwen3.5-2B, Qwen3.5-9B} × {chartqa, ai2d, textvqa,
realworldqa}` — eight cells, 800 examples per cell (600 for realworldqa,
which has 765 total). The vision-driven fraction of correct rollouts:

|                 |  chartqa  |   ai2d    |  textvqa  | realworldqa |
| --------------- | --------: | --------: | --------: | ----------: |
| **Qwen3.5-9B**  | **0.880** | **0.502** | **0.983** |   **0.793** |
| **Qwen3.5-2B**  | **0.783** | **0.114** | **1.000** |   **0.775** |

![grid](results/figures/fig1_grid.png)

The spread is **roughly nine-fold** — from 0.114 to 1.000 — and the
ordering is not what the perception-aware framing predicts. On chartqa,
textvqa, and realworldqa, even the small model gets the bulk of its
correct rollouts from genuinely vision-driven examples. On ai2d, the
small model is essentially answering from language prior alone: only
11.4% of its correct rollouts come from examples where seeing the
diagram moved it toward gold. Scaling four-fold to 9B lifts that number
to 50.2% — still a coin flip.

Plain reading: the perception-bottleneck premise is **benchmark-specific**.
It is empirically backwards on three of the four benchmarks we tested,
and only describes the world on multiple-choice diagram QA at small
scale. A grounding regularizer applied uniformly cannot move the
vision-driven majority that already exists, and can only act on the
prior-driven minority — which on most benchmarks is small.

The full discussion, including a per-cell limitation analysis (textvqa
has near-zero pass rates and so the 0.98+ number is dominated by a tiny
numerator), is in [REPORT.md](REPORT.md).

## Auditing the one public perception-aware checkpoint

Of the ten published perception-aware RL methods reviewed, **one**
released model weights publicly: VGPO. We probed both the base and the
VGPO-tuned checkpoint of Qwen2.5-VL-7B-Instruct on chartqa.

| metric                          | base 7B | VGPO-RL-7B | Δ           |
| ------------------------------- | ------: | ---------: | ----------: |
| vision-driven fraction          |   0.801 |      0.794 | −0.007      |
| Pr[correct \| g > 0]            |   0.752 |      0.772 | +0.020      |
| Pr[correct \| g ≤ 0]            |   0.566 |      0.557 | −0.009      |
| Spearman ρ(g, pass-rate)        |  +0.194 |     +0.194 |  0.000      |
| n (after −20 nat noise floor)   |     161 |        140 | —           |

![e3](results/figures/fig4_e3_contrast.png)

On out-of-distribution chartqa, the VGPO objective did not measurably
shift the vision-driven composition of its successes — the metric the
paper's framing predicts should move the most. Pr[correct | g > 0]
ticked up by 0.02; Pr[correct | g ≤ 0] ticked down by the same
magnitude. We do not call this a counterexample to VGPO — the sample is
small, the benchmark is not the training distribution, and the
direction is consistent with random variation — but the *measurement*
this comparison requires has not previously been reported by VGPO or by
any sister paper.

## What's in this repository

```
PEAR/
  pear/                   the package
    config.py             tunable knobs
    model.py              AutoModelForImageTextToText loader
    data.py               loaders: chartqa, ai2d, textvqa, realworldqa
    verifiers.py          answer verification (mc / numeric / exact / anls)
    score.py              teacher-forced log-prob (sum + mean), sampling
    vest.py               the decomposer; this is the central instrument
    probe.py              per-(model, dataset) driver writing one parquet
    audit.py              cross-checkpoint VEST roll-up
    cli.py                python -m pear {probe,decompose,audit,smoke}

  scripts/
    run_e1_grid.sh        the 8-cell E1 chained probe runner
    run_e3_audit.sh       the base-vs-VGPO probe runner
    make_figures.py       reproduces all figures in REPORT.md

  results/
    probes/               ten .parquet files, one per cell
    vest/                 per-cell VEST text dumps
    audits/               E1 grid + E3 contrast as CSV + text
    figures/              .pdf and .png for every figure in this README

  archive/                the eight-iteration research history that led
                          to VEST: pear1/ .. pear6/ and seeing/, kept
                          verbatim (see archive/README.md)

  REPORT.md               full narrative report with chain of thought
  README.md               this file
```

## Quickstart

```bash
git clone git@github.com:Bechirdardouri/PEAR.git
cd PEAR

# 1. environment (single H100 or any 80 GB device; bf16, flash-attn optional)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. sanity check (synthetic, ~2 seconds)
python -m pear smoke

# 3. reproduce one cell of the grid (~50 minutes on H100)
python -m pear probe \
    --model-id Qwen/Qwen3.5-9B --source chartqa --n-per-source 800 \
    --out results/probes/probe_qwen35_9b_chartqa.parquet

# 4. run VEST on it
python -m pear decompose \
    --parquet results/probes/probe_qwen35_9b_chartqa.parquet \
    --group-by-source

# 5. compare two checkpoints
python -m pear audit \
    --entry base:results/probes/probe_qwen25vl_7b_base_chartqa.parquet \
    --entry vgpo:results/probes/probe_qwen25vl_7b_vgpo_chartqa.parquet \
    --out results/audits/audit_e3_vgpo_chartqa.csv

# 6. regenerate every figure in the report
python scripts/make_figures.py
```

All ten parquet files used in the report are committed under
`results/probes/`, so any figure or table can be regenerated without
re-running a single forward pass:

```bash
python scripts/make_figures.py
python -m pear decompose --parquet results/probes/probe_qwen35_2b_ai2d.parquet
```

## What this work is, and is not

It **is** a measurement instrument and a public audit. It surfaces a
single number, well-defined and cheap to compute, and shows that the
number disagrees substantially with the premises of an entire
sub-literature.

It **is not** a new RL algorithm. We deliberately did not propose
"VEST-GRPO" or a partitioned policy or any other method, because the
measurement is the contribution. The next experiments — E2 (in-training
VEST tracks) and E4 (a method that routes prior-driven examples
separately) — are described in [REPORT.md](REPORT.md) and are the
natural follow-ups once VEST is established as a default diagnostic.

## License

MIT. See [LICENSE](LICENSE).
