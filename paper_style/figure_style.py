"""Shared vector-figure system for the Kirin paper pair."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

INK = "#172027"
MUTED = "#52606A"
RULE = "#D6DDE1"
BASELINE = "#7D858C"
P1 = "#3A5683"
P1_DARK = "#263C63"
P1_PALE = "#EDF1F7"
P2 = "#AC5F31"
P2_DARK = "#7E4423"
P2_PALE = "#F8EFE7"
POSITIVE = "#16756A"
UNRESOLVED = "#8F6400"
NEGATIVE = "#A94442"
WHITE = "#FFFFFF"

# Compatibility names used by the pre-redesign scripts.
BLUE = P1
GREEN = POSITIVE
ORANGE = P2
RED = NEGATIVE
GRAY = BASELINE
GRID = RULE
TEXT = INK
TEXT2 = MUTED
SURFACE = WHITE
YELLOW = UNRESOLVED
VIOLET = "#62558A"
MAGENTA = "#9C5470"
SEQ_BLUES = [
    "#EDF1F7", "#D9E0EA", "#C5CFDD", "#B1BDD0", "#9DACC3", "#8A9BB7",
    "#768AAA", "#62789D", "#4E6790", P1, "#30476F", "#25395B", "#1B2A47",
]


def apply(paper: int = 1) -> None:
    """Apply common chart typography and the selected paper accent."""
    accent = P1 if paper == 1 else P2
    mpl.rcParams.update({
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "savefig.transparent": False,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Source Sans Pro", "Noto Sans", "DejaVu Sans"],
        "font.size": 9.2,
        "text.color": INK,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.65,
        "axes.titlesize": 10.2,
        "axes.titleweight": "semibold",
        "axes.titlepad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": RULE,
        "grid.linewidth": 0.55,
        "grid.alpha": 0.85,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.4,
        "lines.linewidth": 1.65,
        "lines.markersize": 5.5,
        "patch.edgecolor": WHITE,
        "axes.prop_cycle": mpl.cycler(color=[accent, POSITIVE, P2, BASELINE]),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def bar_label(ax, bars, fmt="{:.3f}", dy=0.004, fontsize=8.7, color=INK) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            (bar.get_x() + bar.get_width() / 2, height + dy),
            ha="center", va="bottom", fontsize=fontsize, color=color,
        )


def panel_label(ax, label: str, *, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10.5, fontweight="bold", color=INK)


def save(fig, path: str | Path) -> None:
    """Write vector PDF with live/selectable text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.03)

