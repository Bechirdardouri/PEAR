"""Generate the figures referenced by the README and REPORT.

Outputs go to ``results/figures/``. Both .pdf (vector) and .png
(raster, 300 dpi) versions are produced. The figures are designed
for print: serif font, restrained palette, no chart-junk.

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

# --------------------------------------------------------------- style

mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif"],
    "font.size":        10,
    "axes.titlesize":   10,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth":   1.4,
    "savefig.bbox":      "tight",
    "savefig.dpi":       300,
    "figure.dpi":        110,
})

INK     = "#1a1a1a"
ACCENT  = "#b3261e"   # restrained red, for the line we want the eye to follow
MUTED   = "#7a7a7a"
RULE    = "#cfcfcf"
FILL_HI = "#e8eef5"   # cool grey-blue for "vision-driven"
FILL_LO = "#f5e8e8"   # warm grey-pink for "prior-driven"

# --------------------------------------------------------------- paths

ROOT   = Path(__file__).resolve().parents[1]
PROBES = ROOT / "results" / "probes"
FIGS   = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# (model, dataset, file_stem, n_per_source)
CELLS: list[tuple[str, str, str, int]] = [
    ("Qwen3.5-9B", "chartqa",     "probe_qwen35_9b_chartqa",     800),
    ("Qwen3.5-9B", "ai2d",        "probe_qwen35_9b_ai2d",        800),
    ("Qwen3.5-9B", "textvqa",     "probe_qwen35_9b_textvqa",     800),
    ("Qwen3.5-9B", "realworldqa", "probe_qwen35_9b_realworldqa", 600),
    ("Qwen3.5-2B", "chartqa",     "probe_qwen35_2b_chartqa",     800),
    ("Qwen3.5-2B", "ai2d",        "probe_qwen35_2b_ai2d",        800),
    ("Qwen3.5-2B", "textvqa",     "probe_qwen35_2b_textvqa",     800),
    ("Qwen3.5-2B", "realworldqa", "probe_qwen35_2b_realworldqa", 600),
]
MODELS = ["Qwen3.5-9B", "Qwen3.5-2B"]
DATASETS = ["chartqa", "ai2d", "textvqa", "realworldqa"]
NOISE_FLOOR = -20.0

# --------------------------------------------------------------- helpers

def load_cell(stem: str) -> pd.DataFrame:
    path = PROBES / f"{stem}.parquet"
    df = pd.read_parquet(path).dropna(subset=["m_img_sum", "m_blank_sum"]).copy()
    df["g"] = df["m_img_sum"] - df["m_blank_sum"]
    return df


def vest_stats(df: pd.DataFrame, *, apply_floor: bool = True) -> dict:
    sub = df[df["m_img_sum"] > NOISE_FLOOR] if apply_floor else df.copy()
    g  = sub["g"].to_numpy(dtype=float)
    pr = sub["pass_rate_g16"].to_numpy(dtype=float)
    pos = g > 0
    return {
        "n":              int(len(sub)),
        "vision_driven":  float(pr[pos].sum() / pr.sum()) if pr.sum() > 0 else math.nan,
        "frac_g_pos":     float(pos.mean()),
        "mean_g":         float(g.mean()),
        "pass_rate_pos":  float(pr[pos].mean()) if pos.any() else math.nan,
        "pass_rate_nonpos": float(pr[~pos].mean()) if (~pos).any() else math.nan,
        "pass_rate":      float(pr.mean()),
    }


# --------------------------------------------------------------- Figure 1
# The headline 2x4 heatmap of vision_driven_frac.

def figure_grid() -> None:
    M = np.full((len(MODELS), len(DATASETS)), np.nan)
    PASS = np.full_like(M, np.nan)
    for model, dataset, stem, _ in CELLS:
        df = load_cell(stem)
        s = vest_stats(df)
        i = MODELS.index(model)
        j = DATASETS.index(dataset)
        M[i, j]    = s["vision_driven"]
        PASS[i, j] = s["pass_rate"]

    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="Greys", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            txt_color = "white" if val > 0.55 else INK
            ax.text(
                j, i - 0.10, f"{val:.3f}",
                ha="center", va="center", fontsize=12,
                color=txt_color, fontweight="bold",
            )
            ax.text(
                j, i + 0.22, f"pass-rate {PASS[i, j]:.2f}",
                ha="center", va="center", fontsize=7.5,
                color=txt_color, alpha=0.85,
            )
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_xlabel("benchmark")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("vision-driven fraction of correct rollouts", fontsize=9)
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(length=0, labelsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig1_grid.{ext}")
    plt.close(fig)
    print(f"[fig] wrote fig1_grid.{{pdf,png}}")


# --------------------------------------------------------------- Figure 2
# g-distribution per cell, with the prior-driven mass shaded.

def figure_g_distributions() -> None:
    fig, axes = plt.subplots(
        len(MODELS), len(DATASETS),
        figsize=(8.5, 3.6), sharex="col",
    )
    for (model, dataset, stem, _) in CELLS:
        i = MODELS.index(model)
        j = DATASETS.index(dataset)
        ax = axes[i, j]
        df = load_cell(stem)
        sub = df[df["m_img_sum"] > NOISE_FLOOR]
        g = sub["g"].to_numpy(dtype=float)
        # Choose bin range per column so columns share an x-axis.
        lo, hi = np.percentile(g, [1, 99])
        pad = max(1.0, 0.05 * (hi - lo))
        ax.hist(
            g, bins=30, range=(lo - pad, hi + pad),
            color="#bcbcbc", edgecolor=INK, linewidth=0.4,
        )
        ax.axvline(0.0, color=ACCENT, linewidth=1.0)
        # Shade prior-driven mass.
        s = vest_stats(df)
        ax.text(
            0.04, 0.92, f"$g{{>}}0$: {s['frac_g_pos']:.2f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=INK,
        )
        ax.text(
            0.04, 0.78, f"vis-driven {s['vision_driven']:.2f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=INK, fontweight="bold",
        )
        if i == 0:
            ax.set_title(dataset, fontsize=10, pad=4)
        if j == 0:
            ax.set_ylabel(model, fontsize=10)
        ax.tick_params(left=False, labelleft=False)
        ax.set_yticks([])
        for spine in ("left",):
            ax.spines[spine].set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("$g(x) = \\log p(\\mathrm{gold}\\mid I,q) - \\log p(\\mathrm{gold}\\mid \\emptyset,q)$",
                      fontsize=9)
    fig.subplots_adjust(wspace=0.18, hspace=0.32)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig2_g_distributions.{ext}")
    plt.close(fig)
    print(f"[fig] wrote fig2_g_distributions.{{pdf,png}}")


# --------------------------------------------------------------- Figure 3
# Leverage scatter: vision-driven fraction vs the pass-rate gap.

def figure_leverage() -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for model, dataset, stem, _ in CELLS:
        df = load_cell(stem)
        s = vest_stats(df)
        x = s["vision_driven"]
        # "Leverage" = how much pass-rate the vision-driven slice gives
        # over the prior-driven slice. A positive value means VEST g is
        # genuinely predictive of correctness.
        y = s["pass_rate_pos"] - s["pass_rate_nonpos"]
        marker = "o" if model.endswith("9B") else "s"
        color  = INK if model.endswith("9B") else MUTED
        ax.scatter([x], [y], s=90, marker=marker, color=color,
                   edgecolor="white", linewidth=1.0, zorder=3)
        # Offset labels to avoid overlap.
        dx, dy = {
            ("Qwen3.5-9B", "chartqa"):     (+0.012, +0.010),
            ("Qwen3.5-9B", "ai2d"):        (+0.012, +0.010),
            ("Qwen3.5-9B", "textvqa"):     (-0.012, +0.018),
            ("Qwen3.5-9B", "realworldqa"): (+0.012, +0.012),
            ("Qwen3.5-2B", "chartqa"):     (+0.012, -0.020),
            ("Qwen3.5-2B", "ai2d"):        (+0.020, +0.000),
            ("Qwen3.5-2B", "textvqa"):     (-0.012, -0.020),
            ("Qwen3.5-2B", "realworldqa"): (+0.012, +0.012),
        }[(model, dataset)]
        ha = "left" if dx > 0 else "right"
        ax.annotate(dataset, (x + dx, y + dy), fontsize=8, color=color, ha=ha)
    ax.axhline(0.0, color=RULE, linewidth=0.6)
    ax.axvline(0.5, color=RULE, linewidth=0.6, linestyle=":")
    ax.set_xlabel("vision-driven fraction of correct rollouts")
    ax.set_ylabel("$\\Pr[\\mathrm{correct} \\mid g{>}0] - \\Pr[\\mathrm{correct} \\mid g{\\leq}0]$")
    ax.set_xlim(-0.02, 1.05)
    # Legend.
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=INK,
                   markeredgecolor="white", markersize=8, label="Qwen3.5-9B"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=MUTED,
                   markeredgecolor="white", markersize=8, label="Qwen3.5-2B"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig3_leverage.{ext}")
    plt.close(fig)
    print(f"[fig] wrote fig3_leverage.{{pdf,png}}")


# --------------------------------------------------------------- Figure 4
# E3 contrast: base Qwen2.5-VL-7B vs VGPO-RL-7B on chartqa.

def figure_e3_contrast() -> None:
    base = load_cell("probe_qwen25vl_7b_base_chartqa")
    vgpo = load_cell("probe_qwen25vl_7b_vgpo_chartqa")
    sb = vest_stats(base)
    sv = vest_stats(vgpo)

    metrics = [
        ("vision-driven\nfraction",          "vision_driven"),
        ("$\\Pr[\\text{correct}\\mid g{>}0]$",       "pass_rate_pos"),
        ("$\\Pr[\\text{correct}\\mid g{\\leq}0]$",   "pass_rate_nonpos"),
    ]
    labels    = [m[0] for m in metrics]
    base_vals = [sb[m[1]] for m in metrics]
    vgpo_vals = [sv[m[1]] for m in metrics]

    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    x = np.arange(len(metrics))
    w = 0.36
    ax.bar(x - w/2, base_vals, w, color=MUTED, edgecolor=INK, linewidth=0.6,
           label="Qwen2.5-VL-7B-Instruct (base)")
    ax.bar(x + w/2, vgpo_vals, w, color=ACCENT, edgecolor=INK, linewidth=0.6,
           label="MuMing0102/VGPO-RL-7B")
    for xi, b, v in zip(x, base_vals, vgpo_vals):
        ax.text(xi - w/2, b + 0.01, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    ax.set_ylabel("value (after noise floor)")
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig4_e3_contrast.{ext}")
    plt.close(fig)
    print(f"[fig] wrote fig4_e3_contrast.{{pdf,png}}")


# --------------------------------------------------------------- Figure 5
# Schematic of the VEST measurement procedure.

def figure_schematic() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, text, fill="white"):
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor=fill,
                                    edgecolor=INK, linewidth=0.8))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9.5)

    # Inputs.
    box(0.1, 2.5, 1.6, 0.9, "image $I$",   fill=FILL_HI)
    box(0.1, 1.1, 1.6, 0.9, "blank",       fill=FILL_LO)
    box(2.0, 1.8, 1.6, 0.9, "question $q$\n+ gold $y$")

    # Forward passes.
    box(4.2, 2.5, 2.2, 0.9, "VLM forward\nteacher-forced on $y$", fill=FILL_HI)
    box(4.2, 1.1, 2.2, 0.9, "VLM forward\nteacher-forced on $y$", fill=FILL_LO)

    # Outputs.
    box(6.9, 2.5, 1.7, 0.9, "$\\log p(y\\mid I,q)$", fill=FILL_HI)
    box(6.9, 1.1, 1.7, 0.9, "$\\log p(y\\mid \\emptyset,q)$", fill=FILL_LO)

    # Subtractor.
    box(9.1, 1.8, 0.8, 0.9, "$g$", fill="white")
    # Arrows.
    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="->",
                                     mutation_scale=10, color=INK, linewidth=0.8))
    arrow((1.7, 2.95), (4.2, 2.95))
    arrow((1.7, 1.55), (4.2, 1.55))
    arrow((3.6, 2.25), (4.2, 2.65))
    arrow((3.6, 2.25), (4.2, 1.85))
    arrow((6.4, 2.95), (6.9, 2.95))
    arrow((6.4, 1.55), (6.9, 1.55))
    arrow((8.6, 2.95), (9.1, 2.45))
    arrow((8.6, 1.55), (9.1, 2.05))
    ax.text(9.5, 1.45, "(per example)", ha="center", va="top",
            fontsize=8, color=MUTED, style="italic")

    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig5_schematic.{ext}")
    plt.close(fig)
    print(f"[fig] wrote fig5_schematic.{{pdf,png}}")


# --------------------------------------------------------------- main

def main() -> int:
    figure_schematic()
    figure_grid()
    figure_g_distributions()
    figure_leverage()
    figure_e3_contrast()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
