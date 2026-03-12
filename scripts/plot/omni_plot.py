"""
OmniSafe (Safe RL) Training Curve Plotter
------------------------------------------
Publication-quality dual-panel plots: Reward (left) + Cost (right).

Usage:
    # Dual-panel (reward + cost) — default
    python ttp/omni_plot.py --env building --t_steps 20000 --w_size 500

    # Single metric
    python ttp/omni_plot.py --env building --state reward --t_steps 20000

    # With cost limit line
    python ttp/omni_plot.py --env cogen --climit 25 --t_steps 20000

    # Save PDF too
    python ttp/omni_plot.py --env evcharging --pdf
"""

import os
import sys
import argparse
import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Publication-quality matplotlib settings
# =============================================================================
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["pdf.fonttype"] = 42   # TrueType fonts in PDF
plt.rcParams["ps.fonttype"] = 42

# =============================================================================
# CONFIGURATION — Add your CSV paths here
# =============================================================================
# Format: (path_to_progress.csv, label_for_legend)

EV_PATHS: List[Tuple[str, str]] = [
    # (r"/path/to/runs/OnCRPO-{SustaingymEVCharging-v0}/seed-.../progress.csv", "OnCRPO"),
    # (r"/path/to/runs/PPOLag-{SustaingymEVCharging-v0}/seed-.../progress.csv", "PPOLag"),
]

BUILDING_PATHS: List[Tuple[str, str]] = [
    (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/runs/OnCRPO-{SustaingymBuilding-v0}/seed-000-2025-04-30-09-59-48/progress.csv", "OnCRPO"),
]

COGEN_PATHS: List[Tuple[str, str]] = [
    # (r"/path/to/runs/OnCRPO-{SustaingymCogen-v0}/seed-.../progress.csv", "OnCRPO"),
]

ENV_PATHS = {
    "evcharging": EV_PATHS,
    "building": BUILDING_PATHS,
    "cogen": COGEN_PATHS,
}

ENV_TITLES = {
    "evcharging": "EVCharging-v0",
    "building": "Building-v0",
    "cogen": "Cogen-v0",
}

OUTPUT_DIR = "./graphs/C_SRL/"

# =============================================================================
# Styling
# =============================================================================
COLORS = sns.color_palette("tab10", n_colors=10)
LINE_STYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]
MARK_EVERY_FRAC = 0.08
FILL_FACTOR = 0.5      # std band width: mean ± FILL_FACTOR * std
FILL_ALPHA = 0.12


# =============================================================================
# Smoothing
# =============================================================================
def ema_smooth(data: np.ndarray, span: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute EMA-smoothed mean and std."""
    series = pd.Series(data)
    mean = series.ewm(span=span, adjust=True).mean().values
    std = series.ewm(span=span, adjust=True).std().fillna(0).values
    return mean, std


# =============================================================================
# Core plotting
# =============================================================================
def plot_single_panel(
    ax: plt.Axes,
    experiments: List[Tuple[str, str]],
    metric: str,
    ylabel: str,
    t_steps: int,
    w_size: int,
    skip: int,
    climit: Optional[float] = None,
):
    """Plot a single metric (reward or cost) on the given axes."""
    col_map = {
        "reward": "Metrics/EpRet",
        "cost": "Metrics/EpCost",
    }
    col_name = col_map.get(metric, f"Value/{metric}")

    for i, (path, label) in enumerate(experiments):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] Cannot read {path}: {e}")
            continue

        if col_name not in df.columns:
            print(f"[WARN] Column '{col_name}' not found in {path}. Skipping.")
            continue

        slice_end = min(t_steps, len(df))
        values = df[col_name].astype(float).iloc[:slice_end].values

        mean, std = ema_smooth(values, w_size)

        # Skip initial episodes
        x = np.arange(len(mean))
        if skip > 0 and len(x) > skip:
            x = x[skip:]
            mean = mean[skip:]
            std = std[skip:]

        color = COLORS[i % len(COLORS)]
        ls = LINE_STYLES[i % len(LINE_STYLES)]
        marker = MARKERS[i % len(MARKERS)]
        me = max(1, int(len(x) * MARK_EVERY_FRAC))

        ax.plot(
            x, mean,
            label=label,
            linewidth=1.8,
            color=color,
            linestyle=ls,
            marker=marker,
            markersize=4,
            markevery=me,
            markeredgewidth=0.5,
        )
        ax.fill_between(
            x,
            mean - FILL_FACTOR * std,
            mean + FILL_FACTOR * std,
            alpha=FILL_ALPHA,
            color=color,
            linewidth=0,
        )

    # Cost limit line
    if climit is not None and metric == "cost":
        ax.axhline(y=climit, color="red", linestyle="--", linewidth=1.5,
                    label=f"Cost Limit ({climit})", zorder=5)

    ax.set_xlabel("Training Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)


def plot_dual_panel(
    experiments: List[Tuple[str, str]],
    env: str,
    t_steps: int,
    w_size: int,
    skip: int,
    climit: Optional[float] = None,
    save_pdf: bool = False,
):
    """Create side-by-side Reward + Cost figure."""
    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{ENV_TITLES.get(env, env)} — Safe RL Training", fontsize=15)

    plot_single_panel(ax_r, experiments, "reward", "Average Episode Reward",
                      t_steps, w_size, skip)
    plot_single_panel(ax_c, experiments, "cost", "Average Episode Cost",
                      t_steps, w_size, skip, climit)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    algo_names = "_".join(sorted(set(label.split("_")[0] for _, label in experiments)))
    _save_figure(fig, env, "dual", t_steps, w_size, save_pdf, algo_names)


def plot_single_metric(
    experiments: List[Tuple[str, str]],
    env: str,
    state: str,
    t_steps: int,
    w_size: int,
    skip: int,
    climit: Optional[float] = None,
    save_pdf: bool = False,
):
    """Create a single-metric figure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ylabel = f"Average Episode {state.capitalize()}"
    ax.set_title(f"{ENV_TITLES.get(env, env)} — Safe RL")

    plot_single_panel(ax, experiments, state, ylabel,
                      t_steps, w_size, skip, climit)

    fig.tight_layout()
    algo_names = "_".join(sorted(set(label.split("_")[0] for _, label in experiments)))
    _save_figure(fig, env, state, t_steps, w_size, save_pdf, algo_names)


# =============================================================================
# Save
# =============================================================================
def _save_figure(fig, env: str, mode: str, t_steps: int, w_size: int,
                 save_pdf: bool, algo_names: str = ""):
    """Save figure to PNG (and optionally PDF)."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    subdir = os.path.join(OUTPUT_DIR, env)
    os.makedirs(subdir, exist_ok=True)

    algo_tag = f"_{algo_names}" if algo_names else ""
    base = f"{timestamp}_{mode}{algo_tag}_{t_steps}tsteps_{w_size}wsize"
    png_path = os.path.join(subdir, f"{base}.png")
    fig.savefig(png_path, bbox_inches="tight")
    print(f"Saved: {png_path}")

    if save_pdf:
        pdf_path = os.path.join(subdir, f"{base}.pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {pdf_path}")

    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="OmniSafe Safe RL Training Plotter")
    parser.add_argument("--env", type=str, default="building",
                        choices=["evcharging", "building", "cogen"],
                        help="Environment to plot")
    parser.add_argument("--state", type=str, default="dual",
                        choices=["dual", "reward", "cost"],
                        help="'dual' for side-by-side reward+cost, or single metric")
    parser.add_argument("--t_steps", type=int, default=20000,
                        help="Max training epochs to plot")
    parser.add_argument("--w_size", type=int, default=500,
                        help="EMA smoothing span")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip initial epochs")
    parser.add_argument("--climit", type=float, default=None,
                        help="Cost constraint limit (horizontal line on cost plot)")
    parser.add_argument("--pdf", action="store_true",
                        help="Also save PDF (vector graphics)")
    args = parser.parse_args()

    experiments = ENV_PATHS.get(args.env, [])
    if not experiments:
        print(f"[ERROR] No CSV paths defined for env='{args.env}'. "
              f"Add paths to the {args.env.upper()}_PATHS list in the script.")
        return

    print(f"Plotting {args.env} | state={args.state} | "
          f"t_steps={args.t_steps} | w_size={args.w_size} | "
          f"experiments={len(experiments)}")

    if args.state == "dual":
        plot_dual_panel(experiments, args.env, args.t_steps, args.w_size,
                        args.skip, args.climit, args.pdf)
    else:
        plot_single_metric(experiments, args.env, args.state, args.t_steps,
                           args.w_size, args.skip, args.climit, args.pdf)


if __name__ == "__main__":
    main()
