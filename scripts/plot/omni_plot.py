"""
OmniSafe (Safe RL) Training Curve Plotter — Auto-discovery version
-------------------------------------------------------------------
Publication-quality dual-panel plots: Reward (left) + Cost (right).

Generates ONE figure per algorithm per environment, with all cost limits
as separate lines on the same plot. This is the paper-ready format.

Usage:
    # Per-algorithm plots for an environment (auto-discovers all data)
    python scripts/plot/omni_plot.py --env cogen
    python scripts/plot/omni_plot.py --env building --t_steps 3000 --w_size 200

    # Single algorithm only
    python scripts/plot/omni_plot.py --env cogen --algo PPOLag

    # With specific cost limit reference line
    python scripts/plot/omni_plot.py --env cogen --climit 25

    # Save PDF too (vector graphics for LaTeX)
    python scripts/plot/omni_plot.py --env cogen --pdf

    # Auto y-axis limits
    python scripts/plot/omni_plot.py --env building --auto_ylim
"""

import os
import sys
import re
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from style import (apply_style, FILL_ALPHA, ENV_TITLES, style_axis,
                   MARK_EVERY_FRAC)

# Re-apply style (already auto-applied on import, but be explicit)
apply_style()

# =============================================================================
# Cost-limit color scheme (cool→warm for tight→loose)
# =============================================================================
# Up to 6 cost limits; colorblind-friendly
CL_COLORS = [
    "#0072B2",  # blue   (tightest)
    "#E69F00",  # amber
    "#009E73",  # teal
    "#D55E00",  # vermilion (loosest)
    "#CC79A7",  # pink   (extra)
    "#56B4E9",  # sky    (extra)
]

CL_LINESTYLES = ["-", "--", "-.", ":", (0, (8, 3)), (0, (5, 2, 1, 2))]
CL_MARKERS = ["o", "s", "^", "D", "v", "P"]

FILL_FACTOR = 0.5
OUTPUT_DIR = "./graphs/C_SRL/"
LOGS_BASE = "./logs_saferl_train"


# =============================================================================
# Auto-discovery
# =============================================================================
def discover_experiments(env: str) -> Dict[str, List[Tuple[str, float]]]:
    """
    Scan logs_saferl_train/omnisafe_{env}/ and return:
        {algo_name: [(progress_csv_path, cost_limit), ...]}
    sorted by cost_limit ascending.
    """
    env_dir = Path(LOGS_BASE) / f"omnisafe_{env}"
    if not env_dir.exists():
        print(f"[ERROR] Directory not found: {env_dir}")
        return {}

    results: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    for run_dir in sorted(env_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name

        # Skip old version directories (v1, v2, v3)
        if re.match(r'^v\d+$', name):
            continue

        # Parse algo and climit from folder name
        # New format: 20260310_144235_PPOLag_CL_10.0
        # Old format: PPOLag_CL_25.0_DS_None_DA_None_DE_None_20260306_163213
        match_new = re.match(r'^\d{8}_\d{6}_(\w+)_CL_([\d.]+)', name)
        match_old = re.match(r'^(\w+)_CL_([\d.]+)_', name)

        if match_new:
            algo, climit = match_new.group(1), float(match_new.group(2))
        elif match_old:
            algo, climit = match_old.group(1), float(match_old.group(2))
        else:
            continue

        # Skip SACLag (removed from benchmark)
        if algo == "SACLag":
            continue

        # Find progress.csv (nested inside OmniSafe's directory structure)
        csv_files = list(run_dir.rglob("progress.csv"))
        if not csv_files:
            print(f"[WARN] No progress.csv in {run_dir}")
            continue

        results[algo].append((str(csv_files[0]), climit))

    # Sort each algo's experiments by cost limit
    for algo in results:
        results[algo].sort(key=lambda x: x[1])

    return dict(results)


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
def plot_algo_dual_panel(
    algo: str,
    experiments: List[Tuple[str, float]],
    env: str,
    t_steps: int,
    w_size: int,
    skip: int,
    climit_ref: Optional[float],
    auto_ylim: bool,
    save_pdf: bool,
):
    """
    Create a dual-panel (Reward + Cost) figure for a single algorithm,
    with each cost limit as a separate line.
    """
    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(14, 5))
    env_title = ENV_TITLES.get(env, env)
    fig.suptitle(f"{env_title} — Safe RL Training", fontsize=15)

    has_data = False

    for i, (csv_path, cl) in enumerate(experiments):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [WARN] Cannot read {csv_path}: {e}")
            continue

        if "Metrics/EpRet" not in df.columns or "Metrics/EpCost" not in df.columns:
            print(f"  [WARN] Missing columns in {csv_path}")
            continue

        slice_end = min(t_steps, len(df))
        rewards = df["Metrics/EpRet"].astype(float).iloc[:slice_end].values
        costs = df["Metrics/EpCost"].astype(float).iloc[:slice_end].values

        r_mean, r_std = ema_smooth(rewards, w_size)
        c_mean, c_std = ema_smooth(costs, w_size)

        x = np.arange(len(r_mean))
        if skip > 0 and len(x) > skip:
            x = x[skip:]
            r_mean, r_std = r_mean[skip:], r_std[skip:]
            c_mean, c_std = c_mean[skip:], c_std[skip:]

        color = CL_COLORS[i % len(CL_COLORS)]
        ls = CL_LINESTYLES[i % len(CL_LINESTYLES)]
        marker = CL_MARKERS[i % len(CL_MARKERS)]
        me = max(1, int(len(x) * MARK_EVERY_FRAC))

        # Format label: integer if whole number, else one decimal
        cl_label = f"{algo}_{int(cl)}" if cl == int(cl) else f"{algo}_{cl}"

        plot_kwargs = dict(
            linewidth=1.8, color=color, linestyle=ls,
            marker=marker, markersize=4, markevery=me,
            markeredgewidth=0.5,
        )

        # Reward panel
        ax_r.plot(x, r_mean, label=cl_label, **plot_kwargs)
        ax_r.fill_between(x, r_mean - FILL_FACTOR * r_std,
                          r_mean + FILL_FACTOR * r_std,
                          alpha=FILL_ALPHA, color=color, linewidth=0)

        # Cost panel
        ax_c.plot(x, c_mean, label=cl_label, **plot_kwargs)
        ax_c.fill_between(x, c_mean - FILL_FACTOR * c_std,
                          c_mean + FILL_FACTOR * c_std,
                          alpha=FILL_ALPHA, color=color, linewidth=0)

        has_data = True

    if not has_data:
        print(f"  [SKIP] No valid data for {algo}")
        plt.close(fig)
        return

    # Cost limit reference line
    if climit_ref is not None:
        ax_c.axhline(y=climit_ref, color="red", linestyle="--", linewidth=1.5,
                     label=f"Cost Limit ({climit_ref})", zorder=5)

    # Styling
    ax_r.set_xlabel("Training Epoch")
    ax_r.set_ylabel("Average Episode Reward")
    ax_c.set_xlabel("Training Epoch")
    ax_c.set_ylabel("Average Episode Cost")

    ax_r.legend(framealpha=0.9, loc="best")
    ax_c.legend(framealpha=0.9, loc="best")

    style_axis(ax_r)
    style_axis(ax_c)

    if not auto_ylim:
        # Keep cost y-axis starting at 0 for clarity
        ax_c.set_ylim(bottom=0)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    _save_figure(fig, env, algo, t_steps, w_size, save_pdf)


def plot_all_algos_single(
    all_experiments: Dict[str, List[Tuple[str, float]]],
    env: str,
    t_steps: int,
    w_size: int,
    skip: int,
    cl_value: float,
    auto_ylim: bool,
    save_pdf: bool,
):
    """
    Create a single dual-panel figure comparing ALL algorithms at a FIXED
    cost limit. Each algorithm is one line. Best for cross-algorithm comparison.
    """
    from style import get_color, get_marker, get_linestyle

    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(14, 5))
    env_title = ENV_TITLES.get(env, env)
    fig.suptitle(f"{env_title} — Safe RL (Cost Limit = {int(cl_value)})", fontsize=15)

    has_data = False

    for algo in ["PPOLag", "CPO", "OnCRPO", "FOCOPS"]:
        if algo not in all_experiments:
            continue

        # Find the experiment matching cl_value
        match = None
        for csv_path, cl in all_experiments[algo]:
            if abs(cl - cl_value) < 0.01:
                match = (csv_path, cl)
                break
        if match is None:
            continue

        csv_path, cl = match
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [WARN] Cannot read {csv_path}: {e}")
            continue

        slice_end = min(t_steps, len(df))
        rewards = df["Metrics/EpRet"].astype(float).iloc[:slice_end].values
        costs = df["Metrics/EpCost"].astype(float).iloc[:slice_end].values

        r_mean, r_std = ema_smooth(rewards, w_size)
        c_mean, c_std = ema_smooth(costs, w_size)

        x = np.arange(len(r_mean))
        if skip > 0 and len(x) > skip:
            x = x[skip:]
            r_mean, r_std = r_mean[skip:], r_std[skip:]
            c_mean, c_std = c_mean[skip:], c_std[skip:]

        color = get_color(algo)
        ls = get_linestyle(algo)
        marker = get_marker(algo)
        me = max(1, int(len(x) * MARK_EVERY_FRAC))

        plot_kwargs = dict(
            linewidth=1.8, color=color, linestyle=ls,
            marker=marker, markersize=4, markevery=me,
            markeredgewidth=0.5,
        )

        ax_r.plot(x, r_mean, label=algo, **plot_kwargs)
        ax_r.fill_between(x, r_mean - FILL_FACTOR * r_std,
                          r_mean + FILL_FACTOR * r_std,
                          alpha=FILL_ALPHA, color=color, linewidth=0)

        ax_c.plot(x, c_mean, label=algo, **plot_kwargs)
        ax_c.fill_between(x, c_mean - FILL_FACTOR * c_std,
                          c_mean + FILL_FACTOR * c_std,
                          alpha=FILL_ALPHA, color=color, linewidth=0)

        has_data = True

    if not has_data:
        print(f"  [SKIP] No data for CL={cl_value}")
        plt.close(fig)
        return

    # Cost limit reference line
    ax_c.axhline(y=cl_value, color="red", linestyle="--", linewidth=1.5,
                 label=f"Cost Limit ({int(cl_value)})", zorder=5)

    ax_r.set_xlabel("Training Epoch")
    ax_r.set_ylabel("Average Episode Reward")
    ax_c.set_xlabel("Training Epoch")
    ax_c.set_ylabel("Average Episode Cost")

    ax_r.legend(framealpha=0.9, loc="best")
    ax_c.legend(framealpha=0.9, loc="best")
    style_axis(ax_r)
    style_axis(ax_c)

    if not auto_ylim:
        ax_c.set_ylim(bottom=0)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure(fig, env, f"compare_CL{int(cl_value)}", t_steps, w_size, save_pdf)


# =============================================================================
# Save
# =============================================================================
def _save_figure(fig, env: str, tag: str, t_steps: int, w_size: int,
                 save_pdf: bool):
    """Save figure to PNG (and optionally PDF)."""
    subdir = os.path.join(OUTPUT_DIR, env)
    os.makedirs(subdir, exist_ok=True)

    base = f"saferl_{env}_{tag}_{t_steps}ep_{w_size}ema"
    png_path = os.path.join(subdir, f"{base}.png")
    fig.savefig(png_path, bbox_inches="tight")
    print(f"  Saved: {png_path}")

    if save_pdf:
        pdf_path = os.path.join(subdir, f"{base}.pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"  Saved: {pdf_path}")

    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="OmniSafe Safe RL Training Plotter (auto-discovery)")
    parser.add_argument("--env", type=str, required=True,
                        choices=["evcharging", "building", "cogen"],
                        help="Environment to plot")
    parser.add_argument("--algo", type=str, default=None,
                        help="Plot only this algorithm (default: all)")
    parser.add_argument("--t_steps", type=int, default=3000,
                        help="Max training epochs to plot")
    parser.add_argument("--w_size", type=int, default=200,
                        help="EMA smoothing span")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip initial epochs")
    parser.add_argument("--climit", type=float, default=None,
                        help="Cost limit reference line (default: middle limit)")
    parser.add_argument("--compare", type=float, default=None,
                        help="Generate cross-algorithm comparison at this cost limit")
    parser.add_argument("--auto_ylim", action="store_true",
                        help="Let matplotlib auto-determine y-axis limits")
    parser.add_argument("--pdf", action="store_true",
                        help="Also save PDF (vector graphics)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Safe RL Plotter — {args.env}")
    print(f"{'='*60}")

    # Discover all experiments
    all_experiments = discover_experiments(args.env)

    if not all_experiments:
        print(f"[ERROR] No experiments found in {LOGS_BASE}/omnisafe_{args.env}/")
        return

    # Report what was found
    for algo, exps in sorted(all_experiments.items()):
        limits = [cl for _, cl in exps]
        print(f"  {algo}: CL = {limits} ({len(exps)} runs)")

    # Determine default climit reference line (use middle value)
    if args.climit is None:
        # Pick the second cost limit from the first algo as reference
        first_algo_exps = list(all_experiments.values())[0]
        if len(first_algo_exps) >= 2:
            args.climit = first_algo_exps[1][1]  # second limit
        else:
            args.climit = first_algo_exps[0][1]

    print(f"\n  Reference cost limit line: {args.climit}")
    print(f"  Epochs: {args.t_steps}, EMA: {args.w_size}")
    print()

    # Mode 1: Cross-algorithm comparison at fixed cost limit
    if args.compare is not None:
        print(f"[MODE] Cross-algorithm comparison at CL={args.compare}")
        plot_all_algos_single(
            all_experiments, args.env, args.t_steps, args.w_size,
            args.skip, args.compare, args.auto_ylim, args.pdf)
        return

    # Mode 2: Per-algorithm plots (default — one figure per algo)
    algos_to_plot = [args.algo] if args.algo else sorted(all_experiments.keys())

    for algo in algos_to_plot:
        if algo not in all_experiments:
            print(f"[WARN] No data for {algo}, skipping")
            continue

        exps = all_experiments[algo]
        print(f"[PLOT] {algo} ({len(exps)} cost limits)")
        plot_algo_dual_panel(
            algo, exps, args.env, args.t_steps, args.w_size,
            args.skip, args.climit, args.auto_ylim, args.pdf)

    # Also generate cross-algorithm comparison for each available cost limit
    # Find common cost limits across all algos
    all_limits = set()
    for exps in all_experiments.values():
        for _, cl in exps:
            all_limits.add(cl)

    # Only generate comparison for limits that have >= 3 algos
    for cl in sorted(all_limits):
        algos_with_cl = sum(
            1 for exps in all_experiments.values()
            if any(abs(c - cl) < 0.01 for _, c in exps)
        )
        if algos_with_cl >= 3:
            print(f"[PLOT] Cross-algorithm comparison at CL={int(cl)}")
            plot_all_algos_single(
                all_experiments, args.env, args.t_steps, args.w_size,
                args.skip, cl, args.auto_ylim, args.pdf)

    print(f"\nDone! Plots saved to {OUTPUT_DIR}{args.env}/")


if __name__ == "__main__":
    main()
