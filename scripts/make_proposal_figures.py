"""Schematic figures for proposal/PROPOSAL.md.

Four figures, all drawn programmatically with matplotlib, sharing
the design system in ``scripts/_style.py`` so they live in the same
visual family as the audit figures in ``scripts/make_figures.py``.

    fig1_perceptual_edge_map  -- two-axis regime diagram
    fig2_pear_loop            -- training loop with the two payoffs
    fig3_margin_probe         -- three-condition probe schematic
    fig4_smooth_curriculum    -- the edge moves; the wall does not

Outputs go to ``proposal/figures/``.

Run:
    python scripts/make_proposal_figures.py [--only fig1_perceptual_edge_map]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import _style as S

S.install()

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "proposal" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def _stem(name: str) -> Path:
    return OUT / name


def _rbox(ax, x, y, w, h, text, *, fill="white", edge=S.INK,
          fontsize=9, weight="regular"):
    patch = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.02,rounding_size=0.10",
                            facecolor=fill, edgecolor=edge, linewidth=0.7)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text,
            ha="center", va="center", fontsize=fontsize,
            color=S.INK, fontweight=weight)


def _arrow(ax, a, b, *, color=S.INK, lw=0.7):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>",
                                 mutation_scale=10, color=color, linewidth=lw))


# ================================================================= fig 1
# The perceptual edge of competence: two-axis regime diagram.

def fig1_perceptual_edge_map() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.set_xlim(-0.05, 1.10); ax.set_ylim(-0.05, 1.10)

    # Quadrant shading.
    ax.axvspan(0.5, 1.10, ymin=(0.5 - (-0.05)) / 1.15, ymax=1.0,
               color=S.VISION_SOFT, alpha=0.40, zorder=0)   # readable edge
    ax.axvspan(0.5, 1.10, ymin=0.0, ymax=(0.5 - (-0.05)) / 1.15,
               color=S.PRIOR_SOFT, alpha=0.40, zorder=0)    # unreadable
    ax.axhspan(-0.05, 0.5, xmin=0.0, xmax=(0.5 - (-0.05)) / 1.15,
               color="#f0f0f2", alpha=0.5, zorder=0)        # text-answerable

    # Crosshair.
    ax.axhline(0.5, color=S.MUTED, linewidth=0.5, linestyle=":")
    ax.axvline(0.5, color=S.MUTED, linewidth=0.5, linestyle=":")

    # Quadrant labels.
    ax.text(0.25, 0.25, "text-answerable",
            ha="center", va="center", fontsize=10, color=S.MUTED, style="italic")
    ax.text(0.25, 0.25 - 0.08, "(skip)",
            ha="center", va="center", fontsize=8, color=S.MUTED)

    ax.text(0.25, 0.75, "needs image,\nbut not high-frequency",
            ha="center", va="center", fontsize=10, color=S.FG, style="italic")
    ax.text(0.25, 0.75 - 0.13, "(rare; treat as edge)",
            ha="center", va="center", fontsize=8, color=S.MUTED)

    ax.text(0.79, 0.78, "READABLE EDGE",
            ha="center", va="center", fontsize=11.5, color=S.VISION,
            fontweight="bold")
    ax.text(0.79, 0.71, "concentrate rollouts here",
            ha="center", va="center", fontsize=9, color=S.FG)
    ax.text(0.79, 0.65, "(the only region where RL gains capability)",
            ha="center", va="center", fontsize=8, color=S.MUTED, style="italic")

    ax.text(0.79, 0.27, "UNREADABLE",
            ha="center", va="center", fontsize=11.5, color=S.PRIOR,
            fontweight="bold")
    ax.text(0.79, 0.20, "withhold rollouts;",
            ha="center", va="center", fontsize=9, color=S.FG)
    ax.text(0.79, 0.14, "supervise abstention here",
            ha="center", va="center", fontsize=9, color=S.FG)

    # Axes.
    ax.set_xlabel("NEED  $= \\log p(y\\mid I,q) - \\log p(y\\mid \\emptyset,q)$",
                  fontsize=10)
    ax.set_ylabel("resolvability  $= \\log p(y\\mid I,q) - \\log p(y\\mid \\tilde I,q)$",
                  fontsize=10)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "threshold", "high"])
    ax.set_yticklabels(["0", "threshold", "high"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    fig.text(0.5, -0.02,
             "$y$: gold answer.    "
             "$\\tilde I$: image with fine detail destroyed.    "
             "$\\emptyset$: blank.    "
             "anchored during training by verifiable correctness.",
             ha="center", va="top", fontsize=8, color=S.MUTED, style="italic")

    S.save_both(fig, _stem("fig1_perceptual_edge_map"))
    plt.close(fig)
    print("[proposal-fig] fig1_perceptual_edge_map")


# ================================================================= fig 2
# The PEAR training loop with the dual payoff.

def fig2_pear_loop() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.set_xlim(0, 14.0); ax.set_ylim(0, 6.0); ax.axis("off")

    # Top row: the 5-step loop.
    boxes = [
        (0.4,  4.4, 1.9, 1.1, "1. PROBE\n3 fwd passes",                  S.VISION_SOFT),
        (2.7,  4.4, 1.9, 1.1, "2. ROUTE\nNEED $\\times$ resolvability",  "white"),
        (5.0,  4.4, 1.9, 1.1, "3. SAMPLE\n$G$ rollouts at edge",         S.VISION_SOFT),
        (7.3,  4.4, 1.9, 1.1, "4. UPDATE\nGRPO with abstain reward",     "white"),
        (9.6,  4.4, 1.9, 1.1, "5. RE-PROBE\nevery $K$ steps",            S.VISION_SOFT),
    ]
    for x, y, w, h, txt, fill in boxes:
        _rbox(ax, x, y, w, h, txt, fill=fill, fontsize=9.5)

    # Forward arrows.
    for i in range(len(boxes) - 1):
        x = boxes[i][0] + boxes[i][2]
        y = boxes[i][1] + boxes[i][3] / 2
        _arrow(ax, (x, y), (boxes[i + 1][0], y))

    # Re-probe loop-back arrow.
    _arrow(ax, (boxes[-1][0] + boxes[-1][2] / 2, boxes[-1][1]),
              (boxes[1][0] + boxes[1][2] / 2,  boxes[1][1]),
              color=S.MUTED)
    ax.text(6.5, 3.95, "re-route every $K$ steps as the edge moves",
            ha="center", va="top", fontsize=8.5, color=S.MUTED, style="italic")

    # Branches off step 2: 3 routes.
    _arrow(ax, (boxes[1][0] + boxes[1][2] / 2, boxes[1][1]),
              (1.7, 2.2), color=S.MUTED)
    _arrow(ax, (boxes[1][0] + boxes[1][2] / 2, boxes[1][1]),
              (boxes[2][0] + boxes[2][2] / 2, boxes[2][1]), color=S.VISION)
    _arrow(ax, (boxes[1][0] + boxes[1][2] / 2, boxes[1][1]),
              (11.5, 2.2), color=S.PRIOR)

    _rbox(ax, 0.7,  1.0, 2.5, 1.0, "text-answerable\n(skip)",
          fill="#f0f0f2", edge=S.MUTED)
    _rbox(ax, 5.4,  1.0, 2.0, 1.0, "readable edge\n($G$ rollouts)",
          fill=S.VISION_SOFT, edge=S.VISION, weight="bold")
    _rbox(ax, 10.3, 1.0, 2.5, 1.0, "unreadable\nabstain supervision",
          fill=S.PRIOR_SOFT, edge=S.PRIOR, weight="bold")

    # Payoff annotations.
    ax.text(6.4, 0.55, "training-efficiency payoff",
            ha="center", va="center", fontsize=8.5, color=S.VISION,
            fontweight="bold")
    ax.text(11.55, 0.55, "inference-honesty payoff",
            ha="center", va="center", fontsize=8.5, color=S.PRIOR,
            fontweight="bold")

    S.save_both(fig, _stem("fig2_pear_loop"))
    plt.close(fig)
    print("[proposal-fig] fig2_pear_loop")


# ================================================================= fig 3
# Three-condition probe schematic.

def fig3_margin_probe() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    ax.set_xlim(0, 12.0); ax.set_ylim(0, 5.2); ax.axis("off")

    # Three image conditions on the left.
    _rbox(ax, 0.4, 3.7, 1.9, 0.9, "image  $I$",            fill=S.VISION_SOFT)
    _rbox(ax, 0.4, 2.4, 1.9, 0.9, "degraded  $\\tilde I$\n(downsample by $\\rho$)",
          fill="white", fontsize=8.5)
    _rbox(ax, 0.4, 1.1, 1.9, 0.9, "blank  $\\emptyset$",   fill=S.PRIOR_SOFT)

    # Question.
    _rbox(ax, 2.7, 2.45, 1.9, 0.8, "question  $q$\n+ draft  $\\hat{a}$",
          fontsize=8.5)

    # Forward passes.
    _rbox(ax, 5.0, 3.7, 2.4, 0.9, "VLM   teacher-force  $\\hat{a}$",
          fill=S.VISION_SOFT, fontsize=8.5)
    _rbox(ax, 5.0, 2.4, 2.4, 0.9, "VLM   teacher-force  $\\hat{a}$",
          fontsize=8.5)
    _rbox(ax, 5.0, 1.1, 2.4, 0.9, "VLM   teacher-force  $\\hat{a}$",
          fill=S.PRIOR_SOFT, fontsize=8.5)

    # Outputs.
    _rbox(ax, 7.8, 3.7, 1.7, 0.9, "$m_{\\text{full}}$",   fill=S.VISION_SOFT)
    _rbox(ax, 7.8, 2.4, 1.7, 0.9, "$m_{\\text{deg}}$")
    _rbox(ax, 7.8, 1.1, 1.7, 0.9, "$m_{\\text{blank}}$",  fill=S.PRIOR_SOFT)

    # The two derived axes on the right.
    _rbox(ax, 10.0, 3.05, 1.8, 0.9,
          "NEED\n$m_{\\text{full}} - m_{\\text{blank}}$",
          fill="white", edge=S.ACCENT, fontsize=8.5, weight="bold")
    _rbox(ax, 10.0, 1.75, 1.8, 0.9,
          "resolvability\n$m_{\\text{full}} - m_{\\text{deg}}$",
          fill="white", edge=S.ACCENT, fontsize=8.5, weight="bold")

    # Arrows.
    for y_in, y_mid in [(4.15, 4.15), (2.85, 2.85), (1.55, 1.55)]:
        _arrow(ax, (2.3, y_in), (5.0, y_mid))
    for y in (4.15, 2.85, 1.55):
        _arrow(ax, (4.6, 2.85), (5.0, y))
        _arrow(ax, (7.4, y), (7.8, y))

    _arrow(ax, (9.5, 4.15), (10.0, 3.55), color=S.ACCENT)
    _arrow(ax, (9.5, 1.55), (10.0, 2.85), color=S.ACCENT)
    _arrow(ax, (9.5, 4.15), (10.0, 2.25), color=S.ACCENT)
    _arrow(ax, (9.5, 2.85), (10.0, 2.25), color=S.ACCENT)

    ax.text(6.0, 0.40,
            "Three teacher-forced forward passes per example yield two "
            "cheap, readable axes — the per-example perceptual margin.",
            ha="center", va="center", fontsize=8.5, color=S.MUTED, style="italic")

    S.save_both(fig, _stem("fig3_margin_probe"))
    plt.close(fig)
    print("[proposal-fig] fig3_margin_probe")


# ================================================================= fig 4
# The edge moves; the wall does not.

def fig4_smooth_curriculum() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))

    x = np.linspace(0.0, 10.0, 500)

    # Three snapshots of "reachable" as the model improves: a sigmoidal frontier
    # that slides right, but is hard-capped at a fixed perceptual wall.
    wall = 7.5

    def reach(x, x0):
        return 1.0 / (1.0 + np.exp(2.2 * (x - x0)))

    step_colors = ["#cfd9e1", S.VISION_SOFT, S.VISION]
    step_labels = ["step 0", "step 1k", "step 5k"]
    edges = [2.5, 4.0, 5.8]
    for x0, c, lbl in zip(edges, step_colors, step_labels):
        y = reach(x, x0) * (x < wall)
        ax.plot(x, y, color=c, linewidth=2.0, label=lbl,
                solid_capstyle="round")

    # The unreadable wall.
    ax.axvspan(wall, 10.0, color=S.PRIOR_SOFT, alpha=0.45, zorder=0)
    ax.axvline(wall, color=S.PRIOR, linewidth=1.2)
    ax.text(wall + 0.07, 0.55, "the unreadable wall\n(does not move)",
            ha="left", va="center", fontsize=9, color=S.PRIOR,
            fontweight="bold", style="italic")

    # Annotations on the edges.
    for x0, c, lbl in zip(edges, step_colors, step_labels):
        ax.scatter([x0], [reach(np.array([x0]), x0)[0]], s=42, color=c,
                   edgecolor=S.BG, linewidth=1.2, zorder=4)
        ax.annotate(lbl, (x0, reach(np.array([x0]), x0)[0]),
                    xytext=(x0 - 0.05, reach(np.array([x0]), x0)[0] + 0.06),
                    fontsize=8.5, color=S.FG)

    # Arrow indicating motion of the edge.
    _arrow(ax, (edges[0] + 0.15, 0.92), (edges[-1] - 0.15, 0.92), color=S.MUTED)
    ax.text((edges[0] + edges[-1]) / 2, 0.97,
            "the edge moves as the model improves",
            ha="center", va="bottom", fontsize=8.5, color=S.MUTED, style="italic")

    ax.set_xlabel("perceptual difficulty  $\\to$  (resolvability $\\downarrow$)")
    ax.set_ylabel("$\\Pr[\\text{reachable by current model}]$")
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.05)
    ax.set_xticks([])
    ax.set_yticks([0, 0.5, 1])
    ax.legend(loc="upper right")

    S.save_both(fig, _stem("fig4_smooth_curriculum"))
    plt.close(fig)
    print("[proposal-fig] fig4_smooth_curriculum")


# ----------------------------------------------------------------- main

ALL_FIGS = {
    "fig1_perceptual_edge_map": fig1_perceptual_edge_map,
    "fig2_pear_loop":           fig2_pear_loop,
    "fig3_margin_probe":        fig3_margin_probe,
    "fig4_smooth_curriculum":   fig4_smooth_curriculum,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", choices=list(ALL_FIGS), default=None)
    args = p.parse_args()
    to_run = [args.only] if args.only else list(ALL_FIGS)
    for name in to_run:
        ALL_FIGS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
