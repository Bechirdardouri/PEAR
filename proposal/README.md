# Proposal

This folder holds the forward-looking research proposal that builds on
the audit work in the rest of this repository.

- [PROPOSAL.md](PROPOSAL.md) — the proposal itself, in nine sections.
- [figures/](figures/) — drop your figures here using the filenames
  below; they are already referenced from the proposal.

## Figures expected

| filename | what it shows | where it is referenced |
| :--- | :--- | :--- |
| `figures/fig1_perceptual_edge_map.png` | The two-axis map: NEED on one axis, resolvability on the other, with the four regimes (text-answerable, readable edge, unreadable, trivially-solved) labelled. | end of "At a glance" |
| `figures/fig2_pear_loop.png` | The PEAR training loop with the dual payoff (training efficiency on one branch, inference honesty on the other). | end of §1 |
| `figures/fig3_margin_probe.png` | The three-condition probe: full image, detail-destroyed image (downsample-then-restore), blank image, all scored teacher-forced over the same draft answer. | start of §3 |
| `figures/fig4_smooth_curriculum.png` | The smooth perceptual-difficulty ramp: the edge moves as the policy improves; the unreadable wall does not. | end of §3.4 |

If your filenames differ, search-and-replace inside `PROPOSAL.md`; the
references are the only thing that needs to match.

## Relationship to the rest of the repo

The audit arm (`pear/`, `results/`, `REPORT.md`, `README.md`) is the
*diagnostic*: it establishes that a per-cell perceptual edge exists
and is heterogeneous. This proposal is the *prescriptive*
counterpart: it uses a per-example version of the same edge to drive
training allocation and supervise an abstention action.

The two arms share one name (PEAR) and one core construct (the
perceptual edge); they answer different questions, on different units
of analysis. The first paragraph of [PROPOSAL.md](PROPOSAL.md#relation-to-the-rest-of-this-repository)
spells out the relationship in a short table.
