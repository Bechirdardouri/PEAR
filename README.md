<h1 align="center">PEAR</h1>

<p align="center"><em>Perceptual Edge Audit for RL in vision-language models.</em></p>

<p align="center">
  <a href="https://github.com/Bechirdardouri/PEAR/actions"><img alt="tests"    src="https://img.shields.io/badge/tests-50%20passing-2e6e8e?style=flat-square&labelColor=1c1c1e"></a>
  <a href="LICENSE">                                       <img alt="license"  src="https://img.shields.io/badge/license-MIT-1c1c1e?style=flat-square"></a>
  <a href="REPORT.md">                                     <img alt="report"   src="https://img.shields.io/badge/report-REPORT.md-b3261e?style=flat-square&labelColor=1c1c1e"></a>
  <a href="proposal/PROPOSAL.md">                          <img alt="proposal" src="https://img.shields.io/badge/proposal-PEAR%20method-2e6e8e?style=flat-square&labelColor=1c1c1e"></a>
</p>

<p align="center"><b>One measurement. Two forward passes. A public audit of a fast-moving sub-literature.</b></p>

---

## The question

A new *perception-aware* RL algorithm for vision-language models has appeared roughly every six weeks through 2025–2026 — PAPO, VPPO, PGPO, PRPO, Vision-SR1, SRPO, PDCR, PEPO, VGPO, Perceval. They all open with the same claim:

> *VLMs underperform on visual reasoning because their policies are not sufficiently grounded in the image.*

That claim has driven a lot of work. Almost no one tested it. The obvious question, before optimising for it, is:

> **Of all the correct answers a modern VLM produces on a standard VQA benchmark, what fraction actually depended on the image?**

This repository is the instrument that answers that question, and a public audit of the only perception-aware method whose weights are public (VGPO).

---

## The measurement

For each example $x = (\text{image},\, q,\, \text{gold})$, score the gold answer **twice** under teacher forcing — once with the real image, once with a uniform-grey blank of the same dimensions. The difference is the contribution of the pixels, isolated from sampling noise:

$$
g(x) \;=\; \log p(\text{gold} \mid \text{image},\, q) \;-\; \log p(\text{gold} \mid \emptyset,\, q)
$$

$g(x) > 0$ → the example is **vision-driven**.  $g(x) \leq 0$ → it is **prior-driven**.

The headline number per (model, benchmark) cell is the **vision-driven fraction of correct rollouts**:

$$
\mathrm{VDF} \;=\; \frac{\sum_{x:\, g(x) > 0} \text{pass-rate}(x)}{\sum_{x} \text{pass-rate}(x)}
$$

That is the entire signal. **~1 second per example on an H100.** No auxiliary classifier, no teacher model, no training.

![The VEST measurement: two teacher-forced passes per example, gold scored under image vs. blank](results/figures/fig5_schematic.png)

---

## Finding 1 — the grid (E1)

We ran VEST across two model scales × four public VQA benchmarks. 800 examples per cell (600 for RealWorldQA).

![Per-cell VDF — the headline 2x4 grid](results/figures/fig1_grid.png)

|                 |  ChartQA  |   AI2D    |  TextVQA  | RealWorldQA |
| :-------------- | --------: | --------: | --------: | ----------: |
| **Qwen3.5-9B**  | **0.880** | **0.502** | **0.983** |   **0.793** |
| **Qwen3.5-2B**  | **0.783** | **0.114** | **1.000** |   **0.775** |

Three observations:

> **The spread is nearly nine-fold.** From 0.114 to 1.000 across eight cells. A single per-cell number conceals an order of magnitude.

> **The ordering is backwards from the perception-aware framing.** AI2D (lowest cell, 0.114) is multiple-choice diagram reasoning — closest to *guess from options*. TextVQA (highest, 1.000) is *read text in the image* — least possible language shortcut. The framing predicts the opposite ordering.

> **Three of four benchmarks already produce most correct rollouts from vision-driven examples**, even at 2B. A grounding regularizer applied uniformly cannot move the vision-driven majority that already exists; it can only act on the prior-driven minority, which on most benchmarks is small.

The per-example distributions tell the same story:

![Per-cell distributions of g(x) — cool: vision-driven, warm: prior-driven](results/figures/fig2_g_distributions.png)

And $g$ does predict downstream correctness on every cell except TextVQA, where the absolute prior-driven denominator is tiny:

![VDF vs Pr[correct|g>0] − Pr[correct|g<=0], one point per cell](results/figures/fig3_leverage.png)

> **Plain reading.** The perception-bottleneck premise is *benchmark-specific*. It is empirically backwards on three of the four benchmarks we tested, and only describes the world on multiple-choice diagram QA at small scale.

---

## Finding 2 — the audit (E3)

Of the ten perception-aware RL methods listed above, **exactly one has released weights**: VGPO (`MuMing0102/VGPO-RL-7B`), trained from `Qwen/Qwen2.5-VL-7B-Instruct`. We probed both on ChartQA.

![Base Qwen2.5-VL-7B vs VGPO-RL-7B on ChartQA: VDF, Pr[correct|g>0], Pr[correct|g<=0]](results/figures/fig4_e3_contrast.png)

| metric                          | base 7B | VGPO-RL-7B |       Δ |
| :------------------------------ | ------: | ---------: | ------: |
| **vision-driven fraction (VDF)**|   0.801 |      0.794 |  −0.007 |
| Pr[correct \| g > 0]            |   0.752 |      0.772 |  +0.020 |
| Pr[correct \| g ≤ 0]            |   0.566 |      0.557 |  −0.009 |
| Spearman ρ(g, pass-rate)        |  +0.194 |     +0.194 |   0.000 |
| n (after −20 nat noise floor)   |     161 |        140 |       — |

On out-of-distribution ChartQA, the VGPO objective **did not measurably shift the metric its framing predicts should move the most**. The Spearman correlation between $g$ and downstream correctness is unchanged to three decimal places.

We are careful not to over-read this. ChartQA is not VGPO's training distribution; the post-floor sample sizes (161, 140) are modest; the Δ values are within bootstrap noise. **But the measurement this comparison requires has not previously been reported by VGPO or by any sister paper.** As more checkpoints become public, the same audit script extends to them.

---

## What this is, and is not

> **Is:** a measurement instrument and a public audit. One number, well-defined, cheap to compute, reproducible from the files in this repository — and that number disagrees substantially with the premise of an entire sub-literature.

> **Is not:** a new RL algorithm. We deliberately did not propose *VEST-GRPO* or a partitioned policy. The *measurement* is the contribution. The natural follow-ups — **E2** (VEST tracks during vanilla GRPO) and **E4** (a policy that routes prior-driven and vision-driven examples through different objectives) — are developed in full in [proposal/PROPOSAL.md](proposal/PROPOSAL.md).

A focused list of seven explicit threats to validity is in [REPORT.md §9](REPORT.md#9-limitations-threats-to-validity-and-what-would-change-the-picture).

---

## How to read the rest of the repo

| if you want to… | go to |
| :--- | :--- |
| Read the full chain of thought, including the eight prior iterations that led to VEST | [REPORT.md](REPORT.md) |
| Read the forward-looking research proposal (perceptual-edge curriculum + trained abstention) | [proposal/PROPOSAL.md](proposal/PROPOSAL.md) |
| Re-run a single cell or the whole grid | [Quickstart](#quickstart) below |
| Inspect any reported number row by row | [results/probes/](results/probes/) — ten committed parquets |
| Regenerate every figure | `make figures` |
| Read the seven explicit limitations | [REPORT.md §9](REPORT.md#9-limitations-threats-to-validity-and-what-would-change-the-picture) |
| Browse the eight-iteration research history | [archive/README.md](archive/README.md) |

---

## Quickstart

```bash
git clone git@github.com:Bechirdardouri/PEAR.git
cd PEAR
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Then:

```bash
make test               # 50 tests, ~2 s, CPU
make figures            # regenerate all 9 figures from committed parquets
python -m pear smoke    # synthetic VEST sanity check
```

Reproduce one cell of the grid (~50 min on a single H100):

```bash
python -m pear probe \
    --model-id Qwen/Qwen3.5-9B --source chartqa --n-per-source 800 \
    --out results/probes/probe_qwen35_9b_chartqa.parquet

python -m pear decompose \
    --parquet results/probes/probe_qwen35_9b_chartqa.parquet \
    --group-by-source
```

Compare two checkpoints (the E3 audit):

```bash
python -m pear audit \
    --entry base:results/probes/probe_qwen25vl_7b_base_chartqa.parquet \
    --entry vgpo:results/probes/probe_qwen25vl_7b_vgpo_chartqa.parquet \
    --out results/audits/audit_e3_vgpo_chartqa.csv
```

All ten probe parquets used in the report are committed, so any number, table, or figure regenerates without re-running a single forward pass. End-to-end chained reproduction:

```bash
bash scripts/run_e1_grid.sh    # ~5 h wall time on one H100
bash scripts/run_e3_audit.sh   # ~2 h wall time on one H100
```

---

## Layout

```
PEAR/
├── pear/                         the package — config, model, data, verifiers,
│                                 score, vest, probe, audit, cli
├── scripts/
│   ├── _style.py                 design system: palette, typography, rcParams
│   ├── make_figures.py           5 audit figures from probes/
│   └── make_proposal_figures.py  4 proposal schematics
├── tests/                        50 tests, ~2 s on CPU
├── results/
│   ├── probes/                   ten parquets, one per cell (committed)
│   ├── vest/                     per-cell VEST text dumps
│   ├── audits/                   E1 grid + E3 contrast as csv and txt
│   └── figures/                  every figure as pdf and png
├── proposal/                     forward-looking method proposal
├── archive/                      eight-iteration research history
├── REPORT.md                     full narrative report
├── Makefile                      make {install, test, figures, clean, all}
├── CITATION.cff
└── pyproject.toml
```

---

## Glossary

| name | role |
| :--- | :--- |
| **PEAR** | the project, the repository, and the Python package — *Perceptual Edge Audit for RL* |
| **VEST** | the diagnostic procedure PEAR implements — *Vision-vs-prior Equity Score Test* |
| **VDF**  | vision-driven fraction; the one headline number VEST produces per cell |
| `g(x)`   | the per-example score — `log p(gold | image, q) − log p(gold | blank, q)` |

---

## Citation

```bibtex
@misc{dardouri2026pear,
  author       = {Dardouri, Bechir},
  title        = {{PEAR}: Perceptual Edge Audit for {RL} in vision-language models},
  year         = {2026},
  howpublished = {\url{https://github.com/Bechirdardouri/PEAR}},
}
```

See [CITATION.cff](CITATION.cff) for a machine-readable version.

## License

MIT. See [LICENSE](LICENSE).
