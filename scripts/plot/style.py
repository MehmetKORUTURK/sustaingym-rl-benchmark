"""
Shared publication-quality plotting style for all SustainGym plot scripts.
Import this module at the top of any plot script to apply consistent styling.

Usage:
    from style import apply_style, ALGO_COLORS, ALGO_MARKERS, ALGO_LINESTYLES
"""

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# Publication-quality rcParams (applied on import via apply_style())
# ============================================================================

RC_PARAMS = {
    # Figure
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    # Font — serif for paper, LaTeX-compatible math
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 11,
    "mathtext.fontset": "cm",
    # Axes
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    # Ticks
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # Legend
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7",
    "legend.fancybox": True,
    # PDF / PS — TrueType for editable text
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style():
    """Apply publication-quality rcParams."""
    plt.rcParams.update(RC_PARAMS)


# ============================================================================
# Colorblind-safe algorithm colors (Okabe-Ito inspired)
# ============================================================================

ALGO_COLORS = {
    "PPO":    "#0072B2",  # blue
    "SAC":    "#E69F00",  # amber
    "TD3":    "#009E73",  # teal
    "APPO":   "#CC79A7",  # pink
    "IMPALA": "#56B4E9",  # sky blue
    "PPOLag": "#0072B2",  # blue (same family as PPO)
    "CPO":    "#D55E00",  # vermilion
    "OnCRPO": "#009E73",  # teal
    "FOCOPS": "#F0E442",  # yellow
    # Non-RL baselines
    "Random":         "#999999",  # grey
    "Greedy":         "#8B4513",  # saddle brown
    "MPC":            "#800080",  # purple
    "OfflineOptimal": "#2F4F4F",  # dark slate
    "DoNothing":      "#BC8F8F",  # rosy brown
}

ALGO_MARKERS = {
    "PPO": "o",  "SAC": "s",  "TD3": "^",
    "APPO": "D", "IMPALA": "v",
    "PPOLag": "o", "CPO": "^", "OnCRPO": "s", "FOCOPS": "D",
    # Non-RL baselines
    "Random": "x", "Greedy": "P", "MPC": "*", "OfflineOptimal": "H", "DoNothing": "X",
}

ALGO_LINESTYLES = {
    "PPO": "-",  "SAC": "--",  "TD3": "-.",
    "APPO": ":", "IMPALA": "-",
    "PPOLag": "-", "CPO": "--", "OnCRPO": "-.", "FOCOPS": ":",
    # Non-RL baselines
    "Random": ":", "Greedy": "-.", "MPC": "--", "OfflineOptimal": "-", "DoNothing": ":",
}


def get_color(algo: str) -> str:
    return ALGO_COLORS.get(algo, "#555555")

def get_marker(algo: str) -> str:
    return ALGO_MARKERS.get(algo, "o")

def get_linestyle(algo: str) -> str:
    return ALGO_LINESTYLES.get(algo, "-")


# ============================================================================
# Fallback indexed style (for scripts that cycle by index, not algo name)
# ============================================================================

COLORS_INDEXED = sns.color_palette("tab10", n_colors=10)
# Custom dash patterns: (on, off, ...) in points — wider gaps for visibility
LINE_STYLES = [
    "-",                    # solid
    (0, (5, 3)),            # dashed (wide)
    (0, (5, 2, 1, 2)),     # dash-dot
    (0, (1, 2)),            # dotted (wide gaps)
    (0, (8, 3)),            # long dash
    (0, (5, 2, 1, 2, 1, 2)),  # dash-dot-dot
    (0, (3, 2)),            # short dash
    (0, (8, 3, 1, 3)),     # long dash-dot
    (0, (1, 3)),            # sparse dots
    (0, (5, 1)),            # tight dash
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]
MARK_EVERY_FRAC = 0.08
FILL_ALPHA = 0.20


# ============================================================================
# Environment labels
# ============================================================================

ENV_TITLES = {
    "evcharging": "EV Charging",
    "building": "Building",
    "cogen": "Cogeneration",
}

ENV_SHORT = {
    "evcharging": "ev",
    "building": "bu",
    "cogen": "co",
}


# ============================================================================
# Common helpers
# ============================================================================

def style_axis(ax, grid_y_only=False):
    """Apply consistent axis styling."""
    if grid_y_only:
        ax.yaxis.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    else:
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)


# Auto-apply on import
apply_style()
