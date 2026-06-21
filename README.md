# PEAR

### Perceptual Edge Audit for RL in vision-language models

> *Measurement before optimization. One number, two forward passes,
> a public audit of a fast-moving sub-literature.*

PEAR is a measurement-first toolkit. It provides one diagnostic, called
**VEST**, that asks of every benchmark example a single question:

> *Did seeing the image actually move the model toward the correct answer,
> or could it have produced the same answer from the question alone?*

Two teacher-forced forward passes are enough to answer that question per
example. Aggregated across a benchmark, the result is a single number — the
**vision-driven fraction of correct rollouts**, written **VDF** — that the
perception-aware RL literature has, to our knowledge, never reported.

```
g(x)  =  log p(gold | image, q)  −  log p(gold | blank, q)

VDF   =  (mass of correct rollouts on examples with g > 0)
         ────────────────────────────────────────────────
         (mass of correct rollouts on all examples)
```

A positive `g` means the image moved belief toward gold; that example is
*vision-driven*. A non-positive `g` means the model already preferred gold
from the question alone; that example is *prior-driven*. VDF is the share
of model successes that can be attributed to seeing rather than guessing.

---

## At a glance

| | |
| --- | --- |
| **What it is** | A two-forward-pass diagnostic plus a small Python package. |
| **What it measures** | The fraction of a model's correct answers that depend on the image. |
| **What it produces** | One number per (model, benchmark) cell, plus full per-example parquets. |
| **Cost** | About a second per example on an H100; a full benchmark in 15–60 minutes. |
| **What it does not require** | No auxiliary classifier, no teacher model, no human labels beyond the gold answer, no training. |

---

## The headline result

We ran VEST across a Cartesian grid:

> `{Qwen3.5-2B, Qwen3.5-9B} × {ChartQA, AI2D, TextVQA, RealWorldQA}`
> — eight cells, 800 examples drawn per cell (600 for RealWorldQA, which
> has 765 total). After the conservative noise floor on numerically
> unreliable rows, between 600 and 800 examples per cell are retained.

The vision-driven fraction of correct rollouts:

|                 |  ChartQA  |   AI2D    |  TextVQA  | RealWorldQA |
| :-------------- | --------: | --------: | --------: | ----------: |
| **Qwen3.5-9B**  | **0.880** | **0.502** | **0.983** |   **0.793** |
| **Qwen3.5-2B**  | **0.783** | **0.114** | **1.000** |   **0.775** |

![VEST grid heatmap across eight cells](results/figures/fig1_grid.png)

The spread is **roughly nine-fold** — from 0.114 to 1.000 — and the
ordering is not what the perception-aware framing predicts. On ChartQA,
TextVQA, and RealWorldQA, even the small model gets the bulk of its
correct rollouts from genuinely vision-driven examples. On AI2D, the
small model is essentially answering from language prior alone: only
11.4% of its correct rollouts come from examples where seeing the diagram
moved it toward gold. Scaling four-fold to 9B lifts that number to
50.2% — still a coin flip.

The same picture in per-example detail:

![g distributions across all eight cells](results/figures/fig2_g_distributions.png)

Plain reading: **the perception-bottleneck premise is benchmark-specific.**
It is empirically backwards on three of the four benchmarks we tested,
and only describes the world on multiple-choice diagram QA at small
scale. A grounding regularizer applied uniformly cannot move the
vision-driven majority that already exists; it can only act on the
prior-driven minority, which on most benchmarks is small.

The full per-cell discussion, including the limitation that TextVQA's
0.98+ number is dominated by a tiny absolute numerator, is in
[REPORT.md](REPORT.md).

---

## Auditing the one public perception-aware checkpoint

Through 2025 and 2026 a new perception-aware RL algorithm appeared roughly
every six weeks — PAPO, VPPO, PGPO, PRPO, Vision-SR1, SRPO, PDCR, PEPO,
VGPO, Perceval. They share one premise: that the bottleneck in VLM
reasoning is insufficient grounding in the image, and that the fix is a
training objective which regularizes the policy gradient toward
image-conditional behaviour.

Of those ten methods, **exactly one has released model weights publicly**:
VGPO (`MuMing0102/VGPO-RL-7B`). VGPO is trained from the Qwen2.5-VL-7B
base — *not* from the Qwen3-family backbones we used for the grid — so
the audit experiment must use the Qwen2.5-VL-7B base to be a fair
before/after comparison. We probed both on ChartQA:

| metric                          | base 7B | VGPO-RL-7B |           Δ |
| :------------------------------ | ------: | ---------: | ----------: |
| vision-driven fraction          |   0.801 |      0.794 |      −0.007 |
| Pr[correct \| g > 0]            |   0.752 |      0.772 |      +0.020 |
| Pr[correct \| g ≤ 0]            |   0.566 |      0.557 |      −0.009 |
| Spearman ρ(g, pass-rate)        |  +0.194 |     +0.194 |       0.000 |
| n (after −20 nat noise floor)   |     161 |        140 |           — |

![Base vs VGPO contrast on ChartQA](results/figures/fig4_e3_contrast.png)

On out-of-distribution ChartQA, the VGPO objective did not measurably
shift the vision-driven composition of its successes — the metric the
paper's framing predicts should move the most. Pr[correct | g > 0] ticked
up by 0.02; Pr[correct | g ≤ 0] ticked down by the same magnitude. We do
*not* call this a counterexample to VGPO — the sample is small, the
benchmark is not the training distribution, and the direction is
consistent with random variation — but the *measurement* this comparison
requires has not previously been reported by VGPO or by any sister paper.

---

## Why two model families?

A natural question on first read: *the grid uses Qwen3.5, the audit uses
Qwen2.5-VL. Why?*

|                      | grid (E1)                              | audit (E3)                                  |
| :------------------- | :------------------------------------- | :------------------------------------------ |
| **Question asked**   | Is the perception-aware premise true?  | Did this perception-aware method change it? |
| **Model**            | Qwen3.5-2B and Qwen3.5-9B (open dense) | Qwen2.5-VL-7B-Instruct and VGPO-RL-7B       |
| **Why this model**   | Newest open dense unified-VL family    | Only public perception-aware checkpoint     |
| **What's the pair**  | Two scales of the same family          | A base model and its tuned descendant       |

The audit had to use Qwen2.5-VL-7B because VGPO-RL-7B is trained from
exactly that base; no equivalent perception-aware checkpoint has been
released on a Qwen3 backbone. The grid uses Qwen3.5 because it is the
most current open dense VL family that runs end-to-end on a single
80 GB GPU. We deliberately did *not* mix the two: every number reported
above comes from a strict within-family comparison.

---

## How VEST is computed

![VEST schematic: two forward passes, image and blank, gold scored under each](results/figures/fig5_schematic.png)

One example, two passes. The image pass yields `log p(gold | image, q)`;
the blank pass yields `log p(gold | blank, q)`. Their difference is `g`.
That is the entire signal. The rest of the package is bookkeeping:
loaders, verifiers, sampling for the empirical pass-rate, the
decomposition that turns a per-example `g` into the headline VDF, and
the audit driver that aggregates many (label, parquet) pairs into a
single table.

---

## Repository layout

```
PEAR/
  pear/                       the package
    config.py                   tunable knobs (image sizes, generation params)
    model.py                    AutoModelForImageTextToText loader
    data.py                     loaders: chartqa, ai2d, textvqa, realworldqa
    verifiers.py                answer verification (mc, numeric, exact, anls)
    score.py                    teacher-forced log-prob (sum + mean), sampling
    vest.py                     the decomposer; this is the central instrument
    probe.py                    per-(model, dataset) driver writing one parquet
    audit.py                    cross-checkpoint VEST roll-up
    cli.py                      python -m pear {probe, decompose, audit, smoke}

  scripts/
    run_e1_grid.sh              the 8-cell E1 chained probe runner
    run_e3_audit.sh             the base-vs-VGPO probe runner
    make_figures.py             reproduces every figure in REPORT.md

  results/
    probes/                     ten .parquet files, one per cell, all committed
    vest/                       per-cell VEST text dumps
    audits/                     E1 grid + E3 contrast as .csv and .txt
    figures/                    every figure as .pdf and .png

  archive/                      the eight-iteration research history that led
                                to VEST: pear1/ .. pear6/ and seeing/, kept
                                verbatim (see archive/README.md)

  REPORT.md                     full narrative report with chain of thought
  README.md                     this file
```

---

## Quickstart

```bash
git clone git@github.com:Bechirdardouri/PEAR.git
cd PEAR

# 1. environment (single H100 or any 80 GB device; bf16, flash-attn optional)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. sanity check (synthetic, ~2 seconds, runs on CPU)
python -m pear smoke

# 3. reproduce one cell of the grid (~50 minutes on H100)
python -m pear probe \
    --model-id Qwen/Qwen3.5-9B --source chartqa --n-per-source 800 \
    --out results/probes/probe_qwen35_9b_chartqa.parquet

# 4. run VEST on it
python -m pear decompose \
    --parquet results/probes/probe_qwen35_9b_chartqa.parquet \
    --group-by-source

# 5. compare two checkpoints (the E3 audit)
python -m pear audit \
    --entry base:results/probes/probe_qwen25vl_7b_base_chartqa.parquet \
    --entry vgpo:results/probes/probe_qwen25vl_7b_vgpo_chartqa.parquet \
    --out results/audits/audit_e3_vgpo_chartqa.csv

# 6. regenerate every figure
python scripts/make_figures.py
```

All ten probe parquets used in the report are committed under
[results/probes/](results/probes/), so any number, table, or figure can
be regenerated without re-running a single forward pass:

```bash
python scripts/make_figures.py
python -m pear decompose --parquet results/probes/probe_qwen35_2b_ai2d.parquet
```

End-to-end chained reproduction of the grid and the audit, on one H100:

```bash
bash scripts/run_e1_grid.sh        # ~5 hours wall time
bash scripts/run_e3_audit.sh       # ~2 hours wall time
```

---

## What this work is, and is not

It **is** a measurement instrument and a public audit. It surfaces one
number that is well-defined, cheap to compute, and reproducible from the
files in this repository — and it shows that the number disagrees
substantially with the premise of an entire sub-literature.

It **is not** a new RL algorithm. We deliberately did not propose
*VEST-GRPO* or a partitioned policy or any other method, because the
measurement is the contribution. The natural next experiments — *E2*
(in-training VEST tracks under vanilla GRPO) and *E4* (a policy that
routes prior-driven examples through one objective and vision-driven
through another) — are described in [REPORT.md](REPORT.md) and are
the obvious follow-ups once VEST is established as a default diagnostic.

---

## License

MIT. See [LICENSE](LICENSE).
