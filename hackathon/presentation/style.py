"""One look for every slide chart: 16:9, Avenir Next, big titles with a grey subtitle,
quiet axes, one palette (validated default palette, dataviz skill).

    from style import apply, new_slide, titled, save
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parent.parent / "results" / "figures" / "slides"

# palette
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT, BLUE_DARK = "#86b6ef", "#104281"
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, TINT = "#ebeae4", "#c3c2b7", "#f0efec"

SLIDE = (12.8, 7.2)  # inches; 16:9


def apply() -> None:
    plt.rcParams.update({
        "font.family": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"],
        "font.size": 14,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.labelsize": 15,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1.0,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 13, "ytick.labelsize": 13,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 8, "ytick.major.pad": 8,
        "legend.frameon": False, "legend.fontsize": 14, "legend.labelcolor": INK_2,
        "lines.solid_capstyle": "round",
    })


def new_slide(**kwargs):
    apply()
    return plt.subplots(figsize=SLIDE, **kwargs)


def titled(fig, title: str, subtitle: str = "", footnote: str = "") -> None:
    """Big left-aligned title, grey subtitle under it, small footnote at the bottom."""
    fig.text(0.045, 0.93, title, fontsize=28, fontweight="bold", color=INK, ha="left", va="top")
    if subtitle:
        fig.text(0.045, 0.855, subtitle, fontsize=16, color=INK_2, ha="left", va="top")
    if footnote:
        fig.text(0.045, 0.03, footnote, fontsize=10, color=MUTED, ha="left", va="bottom")


def save(fig, name: str, top: float = 0.78, bottom: float = 0.12, left: float = 0.1,
         right: float = 0.95) -> Path:
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right)
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
