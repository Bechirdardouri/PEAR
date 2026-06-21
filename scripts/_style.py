"""Unified visual design system for every figure in this repository.

A single source of truth for palette, typography, and rcParams. Importing
this module configures matplotlib globally; the named constants are also
exported so figure code can opt into specific tones.

Design principles
-----------------
1. One accent colour, used sparingly. Everything else is greyscale.
2. Semantic colours: cool ``VISION`` for "image-driven" data; warm ``PRIOR``
   for "prior-driven" data. The two are deliberately desaturated so the
   accent always wins the eye.
3. Generous whitespace. Spines off by default. Ticks short.
4. One font family. Two weights (regular, bold). No italics outside math.
5. Annotations live inside the axes, not in a separate legend, whenever
   the figure has fewer than four series.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------- palette

# Greys and ink.
INK         = "#1c1c1e"
FG          = "#2a2a2e"
MUTED       = "#6e6e73"
RULE        = "#d2d2d7"
BG          = "#ffffff"
CANVAS      = "#fafafa"

# Single accent for the line the eye should follow.
ACCENT      = "#b3261e"
ACCENT_SOFT = "#e57373"

# Semantic two-tone for vision-driven vs prior-driven.
VISION      = "#2e6e8e"   # cool teal-blue
VISION_SOFT = "#cfe1ec"
PRIOR       = "#a3603a"   # warm sienna
PRIOR_SOFT  = "#ecd6c2"

# A small categorical palette for benchmark-coloured plots.
CATEGORICAL = ("#2e6e8e", "#a3603a", "#5a7d4f", "#7b5d8c")


# ------------------------------------------------------------- colormaps

def vision_prior_cmap() -> LinearSegmentedColormap:
    """Diverging map: PRIOR_SOFT at 0, white at 0.5, VISION at 1."""
    return LinearSegmentedColormap.from_list(
        "vision_prior",
        [PRIOR_SOFT, "#ffffff", VISION_SOFT, VISION],
        N=256,
    )


# ------------------------------------------------------------ typography

FONT_FAMILY = "serif"
FONT_SERIF  = ["DejaVu Serif", "Source Serif Pro", "Liberation Serif"]


def install(*, base_size: float = 10.0) -> None:
    """Apply the design system to matplotlib globally."""
    mpl.rcParams.update({
        # Fonts.
        "font.family":             FONT_FAMILY,
        "font.serif":              FONT_SERIF,
        "font.size":               base_size,
        "axes.titlesize":          base_size,
        "axes.labelsize":          base_size,
        "axes.titleweight":        "regular",
        "axes.labelcolor":         FG,
        "axes.titlecolor":         INK,
        "xtick.labelsize":         base_size - 1.0,
        "ytick.labelsize":         base_size - 1.0,
        "xtick.color":             FG,
        "ytick.color":             FG,
        "legend.fontsize":         base_size - 1.0,
        "legend.frameon":          False,
        "mathtext.fontset":        "dejavuserif",

        # Spines.
        "axes.spines.top":         False,
        "axes.spines.right":       False,
        "axes.spines.left":        True,
        "axes.spines.bottom":      True,
        "axes.linewidth":          0.6,
        "axes.edgecolor":          RULE,
        "axes.labelpad":           6.0,

        # Ticks.
        "xtick.major.width":       0.5,
        "ytick.major.width":       0.5,
        "xtick.major.size":        3.0,
        "ytick.major.size":        3.0,
        "xtick.direction":         "out",
        "ytick.direction":         "out",

        # Grid.
        "axes.grid":               False,
        "grid.color":              RULE,
        "grid.linewidth":          0.4,
        "grid.alpha":              0.6,

        # Lines & markers.
        "lines.linewidth":         1.4,
        "lines.markersize":        5.0,
        "patch.linewidth":         0.5,

        # Saving.
        "savefig.bbox":            "tight",
        "savefig.dpi":             300,
        "savefig.facecolor":       BG,
        "figure.facecolor":        BG,
        "figure.dpi":              110,
    })


# ------------------------------------------------------------- utilities

def hairline(ax, *, axis: str = "both") -> None:
    """Replace spines with a hairline only on the axes that are kept."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if axis in ("y", "neither"):
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False)
    if axis in ("x", "neither"):
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)


def save_both(fig, stem, *, dpi: int = 300) -> None:
    """Save ``fig`` as ``stem.pdf`` (vector) and ``stem.png`` (raster)."""
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=dpi)


def figure(width: float, height: float, *, dpi: int = 110):
    """Create a new figure with our design defaults applied."""
    return plt.subplots(figsize=(width, height), dpi=dpi)


def annotate(ax, text, xy, *, color=FG, fontsize=8, **kwargs):
    """Default annotation style: small, dark-grey, no arrow unless asked."""
    return ax.annotate(text, xy, fontsize=fontsize, color=color, **kwargs)
