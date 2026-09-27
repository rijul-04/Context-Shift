"""Shared chart style: one visual system across every figure.

Palette validated with the dataviz validator (light surface #fcfcfb):
lightness band, chroma floor, CVD separation and normal-vision floor all pass.
Aqua sits below 3:1 contrast, so every series carries a direct label or a
legend entry - colour never carries identity alone.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"

SERIES = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "violet": "#4a3aa7",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "red": "#e34948",
}
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
       "#0d366b"]

POLICY_COLOR = {
    "SEL-META-both": SERIES["blue"],
    "SEL-META-t1": SERIES["aqua"],
    "SEL-META-t2": SERIES["violet"],
    "SEL-RULE-both": SERIES["magenta"],
    "TCV": SERIES["orange"],
    "CTP": SERIES["yellow"],
    "best-fixed": INK_MUTED,
    "always-S0": INK_MUTED,
    "oracle": INK_2,
}


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "font.family": "DejaVu Sans",
        "figure.dpi": 150,
    })


def label_end(ax, x, y, text, color, dx=0.15, fontsize=9):
    """Direct label at the end of a line, so identity is never colour alone."""
    ax.annotate(text, xy=(x, y), xytext=(dx, 0), textcoords="offset fontsize",
                color=color, fontsize=fontsize, va="center", fontweight="bold")


def save(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
