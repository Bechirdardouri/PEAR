"""Audit figures referenced by README.md and REPORT.md.

Five figures, generated from the ten probe parquets in
``results/probes/``. Every figure obeys the design system in
``scripts/_style.py``.

    fig1_grid              -- 2x4 vision-driven-fraction heatmap (headline)
    fig2_g_distributions   -- per-cell histograms of g(x)
    fig3_leverage          -- leverage scatter (VDF vs pass-rate gap)
    fig4_e3_contrast       -- base vs VGPO on chartqa (audit panel)
    fig5_schematic         -- the VEST measurement diagram

Run:
    python scripts/make_figures.py [--only fig1_grid]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import _style as S

S.install()

# ----------------------------------------------------------------- paths

ROOT   = Path(__file__).resolve().parents[1]
PROBES = ROOT / "results" / "probes"
FIGS   = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------- cells

@dataclass(frozen=True)
class Cell:
    model: str
    dataset: str
    stem: str
    n: int


CELLS: tuple[Cell, ...] = (
    Cell("Qwen3.5-9B", "chartqa",     "probe_qwen35_9b_chartqa",     800),
    Cell("Qwen3.5-9B", "ai2d",        "probe_qwen35_9b_ai2d",        800),
    Cell("Qwen3.5-9B", "textvqa",     "probe_qwen35_9b_textvqa",     800),
    Cell("Qwen3.5-9B", "realworldqa", "probe_qwen35_9b_realworldqa", 600),
    Cell("Qwen3.5-2B", "chartqa",     "probe_qwen35_2b_chartqa",     800),
    Cell("Qwen3.5-2B", "ai2d",        "probe_qwen35_2b_ai2d",        800),
    Cell("Qwen3.5-2B", "textvqa",     "probe_qwen35_2b_textvqa",     800),
    Cell("Qwen3.5-2B", "realworldqa", "probe_qwen35_2b_realworldqa", 600),
)
MODELS:   tuple[str, ...] = ("Qwen3.5-9B", "Qwen3.5-2B")
DATASETS: tuple[str, ...] = ("chartqa", "ai2d", "textvqa", "realworldqa")
DATASETS_PRETTY = {
    "chartqa":     "ChartQA",
    "ai2d":        "AI2D",
    "textvqa":     "TextVQA",
    "realworldqa": "RealWorldQA",
}
NOISE_FLOOR = -20.0


# ---------------------------------------------------------------- helpers

def _load(stem: str) -> pd.DataFrame:
    df = pd.read_parquet(PROBES / f"{stem}.parquet")
    df = df.dropna(subset=["m_img_sum", "m_blank_sum"]).copy()
    df["g"] = df["m_img_sum"] - df["m_blank_sum"]
    return df


def _vest(df: pd.DataFrame, *, floor: bool = True) -> dict:
    sub = df[df["m_img_sum"] > NOISE_FLOOR] if floor else df.copy()
    g  = sub["g"].to_numpy(dtype=float)
    pr = sub["pass_rate_g16"].to_numpy(dtype=float)
    pos = g > 0
    vd = float(pr[pos].sum() / pr.sum()) if pr.sum() > 0 else math.nan
    return dict(
        n                = int(len(sub)),
        vision_driven    = vd,
        frac_g_pos       = float(pos.mean()),
        mean_g           = float(g.mean()),
        median_g         = float(np.median(g)),
        pass_rate_pos    = float(pr[pos].mean()) if pos.any() else math.nan,
        pass_rate_nonpos = float(pr[~pos].mean()) if (~pos).any() else math.nan,
        pass_rate        = float(pr.mean()),
    )


def _stem(name: str) -> Path:
    return FIGS / name


# ================================================================= fig 1

def fig1_grid() -> None:
    M    = np.full((len(MODELS), len(DATASETS)), np.nan)
    PASS = np.full_like(M, np.nan)
    N    = np.full((len(MODELS), len(DATASETS)), 0, dtype=int)
    for c in CELLS:
        s = _vest(_load(c.stem))
        i, j = MODELS.index(c.model), DATASETS.index(c.dataset)
        M[i, j]    = s["vision_driven"]
        PASS[i, j] = s["pass_rate"]
        N[i, j]    = s["n"]

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    cmap = S.vision_prior_cmap()
    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            txt = S.BG if val >= 0.85 else S.INK
            ax.text(j, i - 0.18, f"{val:.3f}",
                    ha="center", va="center", fontsize=14,
                    color=txt, fontweight="bold")
            ax.text(j, i + 0.22, f"pass-rate {PASS[i, j]:.2f}    n = {N[i, j]}",
                    ha="center", va="center", fontsize=7.5,
                    color=txt, alpha=0.85)

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DATASETS_PRETTY[d] for d in DATASETS])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
    cbar.set_label("vision-driven fraction of correct rollouts",
                   fontsize=9, color=S.FG)
    cbar.outline.set_linewidth(0.0)
    cbar.ax.tick_params(length=0, labelsize=8, colors=S.FG)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    fig.text(0.5, -0.02,
             "cool = vision-driven majority    "
             "warm = prior-driven majority    "
             "white = even split",
             ha="center", va="top", fontsize=8, color=S.MUTED, style="italic")
    S.save_both(fig, _stem("fig1_grid"))
    plt.close(fig)
    print("[fig] fig1_grid")


# ================================================================= fig 2

def fig2_g_distributions() -> None:
    fig, axes = plt.subplots(len(MODELS), len(DATASETS),
                             figsize=(9.0, 3.8), sharex="col")
    for c in CELLS:
        i, j = MODELS.index(c.model), DATASETS.index(c.dataset)
        ax = axes[i, j]
        df = _load(c.stem)
        sub = df[df["m_img_sum"] > NOISE_FLOOR]
        g = sub["g"].to_numpy(dtype=float)
        lo, hi = np.percentile(g, [1, 99])
        pad = max(1.0, 0.05 * (hi - lo))
        bins = np.linspace(lo - pad, hi + pad, 32)

        ax.hist(g[g <= 0], bins=bins, color=S.PRIOR_SOFT,
                edgecolor=S.PRIOR, linewidth=0.4)
        ax.hist(g[g > 0],  bins=bins, color=S.VISION_SOFT,
                edgecolor=S.VISION, linewidth=0.4)
        ax.axvline(0.0, color=S.INK, linewidth=0.6)

        s = _vest(df)
        ax.text(0.04, 0.93, f"VDF {s['vision_driven']:.2f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8.5, color=S.INK, fontweight="bold")
        ax.text(0.04, 0.78, f"$\\Pr[g{{>}}0]$ = {s['frac_g_pos']:.2f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, color=S.FG)
        ax.text(0.04, 0.65, f"mean $g$ = {s['mean_g']:+.2f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, color=S.MUTED)

        if i == 0:
            ax.set_title(DATASETS_PRETTY[c.dataset], fontsize=10, pad=6)
        if j == 0:
            ax.set_ylabel(c.model, fontsize=10, labelpad=8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)

    for ax in axes[-1, :]:
        ax.set_xlabel(
            "$g(x) \\;=\\; \\log p(y\\mid I,q) - \\log p(y\\mid \\emptyset,q)$",
            fontsize=9,
        )
    fig.subplots_adjust(wspace=0.18, hspace=0.35)
    S.save_both(fig, _stem("fig2_g_distributions"))
    plt.close(fig)
    print("[fig] fig2_g_distributions")


# ================================================================= fig 3

def fig3_leverage() -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    offsets = {
        ("Qwen3.5-9B", "chartqa"):     (+0.014, +0.014, "left"),
        ("Qwen3.5-9B", "ai2d"):        (+0.014, +0.012, "left"),
        ("Qwen3.5-9B", "textvqa"):     (-0.014, +0.020, "right"),
        ("Qwen3.5-9B", "realworldqa"): (+0.014, +0.014, "left"),
        ("Qwen3.5-2B", "chartqa"):     (+0.014, -0.024, "left"),
        ("Qwen3.5-2B", "ai2d"):        (+0.022, +0.000, "left"),
        ("Qwen3.5-2B", "textvqa"):     (-0.014, -0.024, "right"),
        ("Qwen3.5-2B", "realworldqa"): (+0.014, +0.014, "left"),
    }
    for c in CELLS:
        s = _vest(_load(c.stem))
        x = s["vision_driven"]
        y = s["pass_rate_pos"] - s["pass_rate_nonpos"]
        marker = "o" if c.model.endswith("9B") else "s"
        color  = S.INK if c.model.endswith("9B") else S.MUTED
        ax.scatter([x], [y], s=130, marker=marker, color=color,
                   edgecolor=S.BG, linewidth=1.2, zorder=3)
        dx, dy, ha = offsets[(c.model, c.dataset)]
        ax.annotate(DATASETS_PRETTY[c.dataset], (x + dx, y + dy),
                    fontsize=8, color=color, ha=ha)

    ax.axhline(0.0, color=S.RULE, linewidth=0.5)
    ax.axvline(0.5, color=S.RULE, linewidth=0.5, linestyle=":")
    ax.set_xlabel("vision-driven fraction (VDF)")
    ax.set_ylabel("$\\Pr[\\mathrm{correct}\\mid g{>}0] \\;-\\; "
                  "\\Pr[\\mathrm{correct}\\mid g{\\leq}0]$")
    ax.set_xlim(-0.02, 1.05)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=S.INK,
                   markeredgecolor=S.BG, markersize=10, label="Qwen3.5-9B"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=S.MUTED,
                   markeredgecolor=S.BG, markersize=9, label="Qwen3.5-2B"),
    ]
    ax.legend(handles=handles, loc="lower right")
    S.save_both(fig, _stem("fig3_leverage"))
    plt.close(fig)
    print("[fig] fig3_leverage")


# ================================================================= fig 4

def fig4_e3_contrast() -> None:
    sb = _vest(_load("probe_qwen25vl_7b_base_chartqa"))
    sv = _vest(_load("probe_qwen25vl_7b_vgpo_chartqa"))

    metrics = [
        ("VDF\n(vision-driven fraction)", "vision_driven"),
        ("$\\Pr[\\text{correct}\\mid g{>}0]$",     "pass_rate_pos"),
        ("$\\Pr[\\text{correct}\\mid g{\\leq}0]$", "pass_rate_nonpos"),
    ]
    labels    = [m[0] for m in metrics]
    base_vals = [sb[m[1]] for m in metrics]
    vgpo_vals = [sv[m[1]] for m in metrics]
    deltas    = [v - b for b, v in zip(base_vals, vgpo_vals)]

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(len(metrics))
    w = 0.36
    ax.bar(x - w/2, base_vals, w, color=S.MUTED, edgecolor=S.INK,
           linewidth=0.5, label="base (Qwen2.5-VL-7B-Instruct)")
    ax.bar(x + w/2, vgpo_vals, w, color=S.ACCENT, edgecolor=S.INK,
           linewidth=0.5, label="VGPO-RL-7B")
    for xi, b, v, d in zip(x, base_vals, vgpo_vals, deltas):
        ax.text(xi - w/2, b + 0.012, f"{b:.3f}",
                ha="center", va="bottom", fontsize=8, color=S.INK)
        ax.text(xi + w/2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, color=S.INK)
        sign = "+" if d >= 0 else "−"
        ax.text(xi, max(b, v) + 0.09,
                f"$\\Delta$ = {sign}{abs(d):.3f}",
                ha="center", va="bottom", fontsize=8.5,
                color=S.ACCENT if abs(d) > 0.01 else S.MUTED,
                fontweight="bold" if abs(d) > 0.01 else "regular")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right")
    ax.set_ylabel("value (after −20 nat noise floor)")
    fig.text(0.5, -0.03,
             "OOD evaluation on ChartQA.    "
             "n=161 (base), n=140 (VGPO).    "
             "all three $\\Delta$ within bootstrap noise",
             ha="center", va="top", fontsize=8, color=S.MUTED, style="italic")
    S.save_both(fig, _stem("fig4_e3_contrast"))
    plt.close(fig)
    print("[fig] fig4_e3_contrast")


# ================================================================= fig 5

def fig5_schematic() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 4.0); ax.axis("off")

    def rbox(x, y, w, h, text, *, fill="white", edge=S.INK, fontsize=9):
        patch = FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                facecolor=fill, edgecolor=edge, linewidth=0.7)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, text,
                ha="center", va="center", fontsize=fontsize, color=S.INK)

    def arrow(a, b, *, color=S.INK):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>",
                                     mutation_scale=10, color=color, linewidth=0.7))

    # Inputs (left).
    rbox(0.2, 2.55, 1.8, 0.85, "image  $I$",            fill=S.VISION_SOFT)
    rbox(0.2, 1.10, 1.8, 0.85, "blank  $\\emptyset$",   fill=S.PRIOR_SOFT)
    rbox(2.4, 1.85, 2.0, 0.85, "question  $q$\n+ gold  $y$")

    # Forward passes (middle).
    rbox(5.0, 2.55, 2.6, 0.85, "VLM   teacher-force  $y$", fill=S.VISION_SOFT)
    rbox(5.0, 1.10, 2.6, 0.85, "VLM   teacher-force  $y$", fill=S.PRIOR_SOFT)

    # Log-probs (right).
    rbox(8.1, 2.55, 2.5, 0.85, "$\\log p(y \\mid I, q)$",          fill=S.VISION_SOFT)
    rbox(8.1, 1.10, 2.5, 0.85, "$\\log p(y \\mid \\emptyset, q)$", fill=S.PRIOR_SOFT)

    # Subtractor pill.
    rbox(10.85, 1.85, 0.55, 0.85, "$g$", fill="white", edge=S.ACCENT)

    # Arrows.
    arrow((2.0, 2.97), (5.0, 2.97))
    arrow((2.0, 1.53), (5.0, 1.53))
    arrow((4.4, 2.27), (5.0, 2.67))
    arrow((4.4, 2.27), (5.0, 1.83))
    arrow((7.6, 2.97), (8.1, 2.97))
    arrow((7.6, 1.53), (8.1, 1.53))
    arrow((10.6, 2.97), (10.85, 2.45), color=S.ACCENT)
    arrow((10.6, 1.53), (10.85, 2.05), color=S.ACCENT)

    ax.text(5.75, 0.30,
            "Two teacher-forced forward passes per example.   "
            "Their difference is the per-example vision contribution  "
            "$g(x) = \\log p(y\\mid I,q) - \\log p(y\\mid \\emptyset,q)$.",
            ha="center", va="center", fontsize=8.5, color=S.MUTED, style="italic")

    S.save_both(fig, _stem("fig5_schematic"))
    plt.close(fig)
    print("[fig] fig5_schematic")


# ----------------------------------------------------------------- main

ALL_FIGS = {
    "fig1_grid":            fig1_grid,
    "fig2_g_distributions": fig2_g_distributions,
    "fig3_leverage":        fig3_leverage,
    "fig4_e3_contrast":     fig4_e3_contrast,
    "fig5_schematic":       fig5_schematic,
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
