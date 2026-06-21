# Proposal

This folder holds the forward-looking research proposal that builds on
the audit work in the rest of this repository.

| file | what it is |
| :--- | :--- |
| [PROPOSAL.md](PROPOSAL.md) | the full nine-section proposal |
| [figures/](figures/) | four schematic figures, all programmatically generated |

## Relationship to the rest of the repo

The audit arm ([pear/](../pear/), [results/](../results/),
[REPORT.md](../REPORT.md), [README.md](../README.md)) is the
**diagnostic**: it establishes that a per-cell perceptual edge exists
and is heterogeneous. This proposal is the **prescriptive**
counterpart: it uses a per-example version of the same edge to drive
training allocation and supervise an abstention action.

The two arms share one name (PEAR) and one core construct (the
perceptual edge); they answer different questions, on different units
of analysis. The first paragraph of
[PROPOSAL.md](PROPOSAL.md#relation-to-the-rest-of-this-repository)
spells out the relationship in a short table.

## Regenerating the figures

```bash
make proposal-figures
# or:
python scripts/make_proposal_figures.py
```

All four figures are produced from `scripts/make_proposal_figures.py`
in a single shot, using the design system in `scripts/_style.py`. They
share that design with the five audit figures in `results/figures/`.

| filename | what it shows | referenced from |
| :--- | :--- | :--- |
| `figures/fig1_perceptual_edge_map.png` | Two-axis regime diagram: NEED × resolvability, four regions labelled (text-answerable, readable edge, unreadable, rare-needs-image-not-detail). | "At a glance" |
| `figures/fig2_pear_loop.png` | The PEAR training loop with the dual payoff: training-efficiency on one branch, inference-honesty on the other. | §1 |
| `figures/fig3_margin_probe.png` | The three-condition probe: full image, detail-destroyed image, blank, all scored teacher-forced over the same draft answer. | §3 |
| `figures/fig4_smooth_curriculum.png` | The smooth perceptual-difficulty ramp: the edge moves as the policy improves; the unreadable wall does not. | §3.4 |
