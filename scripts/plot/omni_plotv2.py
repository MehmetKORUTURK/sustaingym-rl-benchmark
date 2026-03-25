"""
OmniSafe (Safe RL) Advanced Plotter — ICLR-quality visualizations
------------------------------------------------------------------
Extends omni_plot.py with five additional plot types beyond standard
training curves:

  1. csr        — Constraint Satisfaction Rate bar chart
  2. pareto     — Reward–Cost Pareto frontier scatter
  3. grid       — Multi-environment 3×2 summary grid (reward + cost)
  4. heatmap    — Cost tracking ratio heatmap (algo × cost limit)
  5. convergence — Epochs to first constraint satisfaction bar chart
  6. curves     — Original dual-panel training curves (from omni_plot.py)
  7. safety     — "Price of Safety" dual heatmap (reward retention + cost tracking)
  8. all        — Generate all plot types

Usage:
    python scripts/plot/omni_plotv2.py --env cogen --plot all
    python scripts/plot/omni_plotv2.py --env cogen --plot pareto
    python scripts/plot/omni_plotv2.py --plot grid              # all 3 envs
    python scripts/plot/omni_plotv2.py --env building --plot csr,heatmap
    python scripts/plot/omni_plotv2.py --env cogen --plot curves --t_steps 3000
    python scripts/plot/omni_plotv2.py --env cogen --plot convergence --tail 500
    python scripts/plot/omni_plotv2.py --env cogen --plot safety --tail 500
    python scripts/plot/omni_plotv2.py --plot safety            # all 3 envs

    # Save PDF (vector graphics for LaTeX)
    python scripts/plot/omni_plotv2.py --env cogen --plot all --pdf
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

from style import (apply_style, FILL_ALPHA, ENV_TITLES, style_axis,
                   MARK_EVERY_FRAC, get_color, get_marker, get_linestyle)

apply_style()

# =============================================================================
# Constants
# =============================================================================
OUTPUT_DIR = "./graphs/C_SRL/"
LOGS_BASE = "./logs_saferl_train"

ALGO_ORDER = ["PPOLag", "CPO", "OnCRPO", "FOCOPS"]
ENV_ORDER = ["evcharging", "building", "cogen"]
ENV_LABELS = {"evcharging": "EV Charging", "building": "Building", "cogen": "Cogeneration"}

# Cost limits per environment — only these are plotted (filters out extras)
ENV_COST_LIMITS = {
    "evcharging": [3, 5, 25, 1000],
    "building":   [25, 50, 100, 200],
    "cogen":      [10, 25, 50, 200],
}

FILL_FACTOR = 0.5

# Algo display colors — consistent with style.py
ALGO_COLORS_ORDERED = [get_color(a) for a in ALGO_ORDER]


# =============================================================================
# Auto-discovery (reused from omni_plot.py)
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

        if re.match(r'^v\d+$', name):
            continue

        match_new = re.match(r'^\d{8}_\d{6}_(\w+)_CL_([\d.]+)', name)
        match_old = re.match(r'^(\w+)_CL_([\d.]+)_', name)

        if match_new:
            algo, climit = match_new.group(1), float(match_new.group(2))
        elif match_old:
            algo, climit = match_old.group(1), float(match_old.group(2))
        else:
            continue

        if algo == "SACLag":
            continue

        csv_files = list(run_dir.rglob("progress.csv"))
        if not csv_files:
            continue

        results[algo].append((str(csv_files[0]), climit))

    # Filter to only official cost limits (if defined for this env)
    allowed_cls = ENV_COST_LIMITS.get(env)
    if allowed_cls is not None:
        for algo in results:
            results[algo] = [(p, cl) for p, cl in results[algo]
                             if any(abs(cl - a) < 0.01 for a in allowed_cls)]

    for algo in results:
        results[algo].sort(key=lambda x: x[1])

    return dict(results)


def load_experiment(csv_path: str) -> Optional[pd.DataFrame]:
    """Load a progress.csv, return None on failure."""
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [WARN] Cannot read {csv_path}: {e}")
        return None


def get_tail_stats(df: pd.DataFrame, tail: int) -> dict:
    """Compute mean/std of reward and cost over the last `tail` epochs."""
    tail_df = df.tail(tail)
    return {
        "reward_mean": tail_df["Metrics/EpRet"].mean(),
        "reward_std":  tail_df["Metrics/EpRet"].std(),
        "cost_mean":   tail_df["Metrics/EpCost"].mean(),
        "cost_std":    tail_df["Metrics/EpCost"].std(),
    }


def ema_smooth(data: np.ndarray, span: int) -> Tuple[np.ndarray, np.ndarray]:
    """EMA-smoothed mean and std."""
    series = pd.Series(data)
    mean = series.ewm(span=span, adjust=True).mean().values
    std = series.ewm(span=span, adjust=True).std().fillna(0).values
    return mean, std


def _cl_label(cl: float) -> str:
    return str(int(cl)) if cl == int(cl) else f"{cl:.1f}"


def _save(fig, name: str, env: str, save_pdf: bool):
    """Save figure to OUTPUT_DIR/env/."""
    subdir = os.path.join(OUTPUT_DIR, env) if env else OUTPUT_DIR
    os.makedirs(subdir, exist_ok=True)
    png = os.path.join(subdir, f"{name}.png")
    fig.savefig(png, bbox_inches="tight")
    print(f"  Saved: {png}")
    if save_pdf:
        pdf = os.path.join(subdir, f"{name}.pdf")
        fig.savefig(pdf, bbox_inches="tight")
        print(f"  Saved: {pdf}")
    plt.close(fig)


# =============================================================================
# 1. Constraint Satisfaction Rate (CSR) Bar Chart
# =============================================================================
def plot_csr(env: str, experiments: Dict, tail: int, save_pdf: bool):
    """
    Bar chart: for each (algo, cost_limit), what fraction of the last `tail`
    epochs have EpCost <= cost_limit?
    """
    print(f"\n[PLOT] CSR bar chart — {env}")

    cost_limits = sorted({cl for exps in experiments.values() for _, cl in exps})
    n_limits = len(cost_limits)
    algos = [a for a in ALGO_ORDER if a in experiments]
    n_algos = len(algos)

    if n_algos == 0:
        print("  [SKIP] No algorithms found")
        return

    fig, ax = plt.subplots(figsize=(max(8, n_limits * 2.5), 5))

    bar_width = 0.8 / n_algos
    x = np.arange(n_limits)

    for i, algo in enumerate(algos):
        csr_values = []
        for cl in cost_limits:
            match = next(((p, c) for p, c in experiments[algo] if abs(c - cl) < 0.01), None)
            if match is None:
                csr_values.append(0)
                continue
            df = load_experiment(match[0])
            if df is None:
                csr_values.append(0)
                continue
            tail_costs = df["Metrics/EpCost"].tail(tail).values
            csr = np.mean(tail_costs <= cl) * 100
            csr_values.append(csr)

        offset = (i - n_algos / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, csr_values, bar_width * 0.9,
                       label=algo, color=get_color(algo), edgecolor="white",
                       linewidth=0.5, zorder=3)

        # Value labels on bars
        for bar, val in zip(bars, csr_values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=8,
                        fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"CL = {_cl_label(cl)}" for cl in cost_limits])
    ax.set_ylabel("Constraint Satisfaction Rate (%)")
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title(f"{ENV_LABELS.get(env, env)} — Constraint Satisfaction Rate (last {tail} epochs)")
    style_axis(ax, grid_y_only=True)

    _save(fig, f"saferl_{env}_csr_tail{tail}", env, save_pdf)


# =============================================================================
# 2. Reward–Cost Pareto Frontier
# =============================================================================
def plot_pareto(env: str, experiments: Dict, tail: int, save_pdf: bool):
    """
    Scatter plot: each (algo, cost_limit) is a point.
    X = final mean cost, Y = final mean reward.
    Vertical dashed lines at each cost limit.
    """
    print(f"\n[PLOT] Pareto frontier — {env}")

    algos = [a for a in ALGO_ORDER if a in experiments]
    cost_limits = sorted({cl for exps in experiments.values() for _, cl in exps})

    fig, ax = plt.subplots(figsize=(9, 6))

    for algo in algos:
        xs, ys, labels = [], [], []
        for csv_path, cl in experiments[algo]:
            df = load_experiment(csv_path)
            if df is None:
                continue
            stats = get_tail_stats(df, tail)
            xs.append(stats["cost_mean"])
            ys.append(stats["reward_mean"])
            labels.append(f"CL={_cl_label(cl)}")

        ax.scatter(xs, ys, color=get_color(algo), marker=get_marker(algo),
                   s=100, label=algo, zorder=5, edgecolors="white", linewidth=0.5)

        # Connect points with a thin line to show the tradeoff path
        if len(xs) > 1:
            sort_idx = np.argsort(xs)
            ax.plot(np.array(xs)[sort_idx], np.array(ys)[sort_idx],
                    color=get_color(algo), linewidth=1.0, alpha=0.4, linestyle="--")

        # Annotate each point with its cost limit
        for xi, yi, lab in zip(xs, ys, labels):
            ax.annotate(lab, (xi, yi), textcoords="offset points",
                        xytext=(6, 6), fontsize=7, alpha=0.7)

    # Vertical reference lines at cost limits
    for cl in cost_limits:
        ax.axvline(x=cl, color="grey", linestyle=":", linewidth=0.7, alpha=0.4)
        ax.text(cl, ax.get_ylim()[1], f" {_cl_label(cl)}", fontsize=7,
                va="top", ha="left", color="grey", alpha=0.6)

    ax.set_xlabel("Average Episode Cost (last {} epochs)".format(tail))
    ax.set_ylabel("Average Episode Reward (last {} epochs)".format(tail))
    ax.set_title(f"{ENV_LABELS.get(env, env)} — Reward–Cost Pareto Frontier")
    ax.legend(loc="best", framealpha=0.9)
    style_axis(ax)

    _save(fig, f"saferl_{env}_pareto_tail{tail}", env, save_pdf)


# =============================================================================
# 3. Multi-Environment Summary Grid (3×2)
# =============================================================================
def plot_grid(all_env_experiments: Dict[str, Dict], t_steps: int, w_size: int,
              save_pdf: bool):
    """
    3-row × 2-col grid: one row per env, columns = reward / cost.
    Each panel shows all 4 algos at a representative cost limit (middle value).
    """
    print(f"\n[PLOT] Multi-environment 3×2 summary grid")

    envs = [e for e in ENV_ORDER if e in all_env_experiments and all_env_experiments[e]]
    if len(envs) == 0:
        print("  [SKIP] No data for any environment")
        return

    fig, axes = plt.subplots(len(envs), 2, figsize=(16, 4.5 * len(envs)),
                              squeeze=False)

    for row, env in enumerate(envs):
        experiments = all_env_experiments[env]
        # Pick the second cost limit as representative
        all_cls = sorted({cl for exps in experiments.values() for _, cl in exps})
        rep_cl = all_cls[len(all_cls) // 2] if len(all_cls) > 1 else all_cls[0]

        ax_r, ax_c = axes[row]

        for algo in ALGO_ORDER:
            if algo not in experiments:
                continue

            match = next(((p, c) for p, c in experiments[algo]
                          if abs(c - rep_cl) < 0.01), None)
            if match is None:
                continue

            df = load_experiment(match[0])
            if df is None:
                continue

            slice_end = min(t_steps, len(df))
            rewards = df["Metrics/EpRet"].astype(float).iloc[:slice_end].values
            costs = df["Metrics/EpCost"].astype(float).iloc[:slice_end].values

            r_mean, r_std = ema_smooth(rewards, w_size)
            c_mean, c_std = ema_smooth(costs, w_size)
            x = np.arange(len(r_mean))
            me = max(1, int(len(x) * MARK_EVERY_FRAC))

            kwargs = dict(linewidth=1.5, color=get_color(algo),
                          linestyle=get_linestyle(algo), marker=get_marker(algo),
                          markersize=3, markevery=me, markeredgewidth=0.4)

            ax_r.plot(x, r_mean, label=algo, **kwargs)
            ax_r.fill_between(x, r_mean - FILL_FACTOR * r_std,
                              r_mean + FILL_FACTOR * r_std,
                              alpha=FILL_ALPHA, color=get_color(algo), linewidth=0)

            ax_c.plot(x, c_mean, label=algo, **kwargs)
            ax_c.fill_between(x, c_mean - FILL_FACTOR * c_std,
                              c_mean + FILL_FACTOR * c_std,
                              alpha=FILL_ALPHA, color=get_color(algo), linewidth=0)

        # Cost limit reference line
        ax_c.axhline(y=rep_cl, color="red", linestyle="--", linewidth=1.2,
                     label=f"CL = {_cl_label(rep_cl)}", zorder=5)

        # Labels
        env_label = ENV_LABELS.get(env, env)
        ax_r.set_ylabel(f"{env_label}\nReward", fontsize=11)
        ax_c.set_ylabel("Cost", fontsize=11)

        if row == len(envs) - 1:
            ax_r.set_xlabel("Training Epoch")
            ax_c.set_xlabel("Training Epoch")
        if row == 0:
            ax_r.set_title("Average Episode Reward", fontsize=12)
            ax_c.set_title("Average Episode Cost", fontsize=12)

        ax_r.legend(fontsize=8, loc="best", framealpha=0.9)
        ax_c.legend(fontsize=8, loc="best", framealpha=0.9)
        style_axis(ax_r)
        style_axis(ax_c)
        ax_c.set_ylim(bottom=0)

    fig.suptitle("Safe RL Training — All Environments", fontsize=14, y=1.01)
    fig.tight_layout()

    _save(fig, "saferl_grid_3x2", "", save_pdf)


# =============================================================================
# 4. Cost Tracking Heatmap
# =============================================================================
def plot_heatmap(env: str, experiments: Dict, tail: int, save_pdf: bool):
    """
    Heatmap: rows = algorithms, columns = cost limits.
    Cell value = actual_cost / cost_limit (tracking ratio).
    Color: green (<= 1.0, constraint satisfied) → red (> 1.0, violation).
    """
    print(f"\n[PLOT] Cost tracking heatmap — {env}")

    algos = [a for a in ALGO_ORDER if a in experiments]
    cost_limits = sorted({cl for exps in experiments.values() for _, cl in exps})

    if not algos or not cost_limits:
        print("  [SKIP] Insufficient data")
        return

    matrix = np.full((len(algos), len(cost_limits)), np.nan)

    for i, algo in enumerate(algos):
        for j, cl in enumerate(cost_limits):
            match = next(((p, c) for p, c in experiments[algo]
                          if abs(c - cl) < 0.01), None)
            if match is None:
                continue
            df = load_experiment(match[0])
            if df is None:
                continue
            mean_cost = df["Metrics/EpCost"].tail(tail).mean()
            matrix[i, j] = mean_cost / cl if cl > 0 else 0

    fig, ax = plt.subplots(figsize=(max(6, len(cost_limits) * 1.8),
                                     max(3, len(algos) * 1.0 + 1.5)))

    # Diverging colormap: green (satisfied) → white (boundary) → red (violation)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "csr_cmap",
        [(0.0, "#2d8a4e"),    # deep green — well under limit
         (0.4, "#8fcb9e"),    # light green
         (0.5, "#ffffff"),    # white — exactly at limit
         (0.65, "#f4a582"),   # light red
         (1.0, "#d73027")],  # deep red — severe violation
    )

    # Determine vmin/vmax — center at 1.0
    valid = matrix[~np.isnan(matrix)]
    if len(valid) == 0:
        print("  [SKIP] All NaN")
        plt.close(fig)
        return

    max_val = max(np.max(valid), 1.5)
    vmin = 0
    vmax = max_val

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # Annotate cells
    for i in range(len(algos)):
        for j in range(len(cost_limits)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=10,
                        color="grey")
            else:
                # Show both ratio and actual cost
                actual = val * cost_limits[j]
                text_color = "white" if val > 1.3 or val < 0.2 else "black"
                ax.text(j, i, f"{val:.2f}\n({actual:.1f})",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color=text_color)

    ax.set_xticks(range(len(cost_limits)))
    ax.set_xticklabels([f"CL = {_cl_label(cl)}" for cl in cost_limits])
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos)
    ax.set_xlabel("Cost Limit")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Cost / Cost Limit (ratio)", fontsize=10)
    # Mark the 1.0 threshold
    cbar.ax.axhline(y=1.0, color="black", linewidth=1.5, linestyle="-")

    ax.set_title(f"{ENV_LABELS.get(env, env)} — Cost Tracking Ratio "
                 f"(last {tail} epochs)", fontsize=13)

    fig.tight_layout()
    _save(fig, f"saferl_{env}_heatmap_tail{tail}", env, save_pdf)


# =============================================================================
# 5. Convergence Speed (Epochs to Constraint Satisfaction)
# =============================================================================
def plot_convergence(env: str, experiments: Dict, w_size: int, save_pdf: bool):
    """
    Bar chart: for each (algo, cost_limit), the first epoch where the
    smoothed cost drops below the cost limit and stays below.
    """
    print(f"\n[PLOT] Convergence speed — {env}")

    algos = [a for a in ALGO_ORDER if a in experiments]
    cost_limits = sorted({cl for exps in experiments.values() for _, cl in exps})
    n_limits = len(cost_limits)
    n_algos = len(algos)

    if n_algos == 0:
        print("  [SKIP] No data")
        return

    fig, ax = plt.subplots(figsize=(max(8, n_limits * 2.5), 5))

    bar_width = 0.8 / n_algos
    x = np.arange(n_limits)

    max_epochs = 0

    for i, algo in enumerate(algos):
        conv_epochs = []
        for cl in cost_limits:
            match = next(((p, c) for p, c in experiments[algo]
                          if abs(c - cl) < 0.01), None)
            if match is None:
                conv_epochs.append(np.nan)
                continue
            df = load_experiment(match[0])
            if df is None:
                conv_epochs.append(np.nan)
                continue

            costs = df["Metrics/EpCost"].astype(float).values
            smoothed, _ = ema_smooth(costs, w_size)

            # Find first epoch where smoothed cost <= cl and stays below
            # (allowing brief excursions: check a rolling window of 50 stays below)
            window = min(50, len(smoothed) // 10)
            found = np.nan
            for ep in range(len(smoothed) - window):
                if np.all(smoothed[ep:ep + window] <= cl):
                    found = ep
                    break

            conv_epochs.append(found)
            if not np.isnan(found):
                max_epochs = max(max_epochs, found)

        offset = (i - n_algos / 2 + 0.5) * bar_width
        vals = [v if not np.isnan(v) else 0 for v in conv_epochs]
        colors = [get_color(algo) if not np.isnan(v) else "#e0e0e0"
                  for v in conv_epochs]

        bars = ax.bar(x + offset, vals, bar_width * 0.9,
                       color=colors, edgecolor="white", linewidth=0.5,
                       zorder=3, label=algo if i == 0 or True else "")

        # Labels
        for bar, val, raw in zip(bars, vals, conv_epochs):
            if np.isnan(raw):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        max_epochs * 0.02 + 5,
                        "N/A", ha="center", va="bottom", fontsize=7,
                        color="grey", fontstyle="italic")
            elif val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max_epochs * 0.01,
                        f"{int(val)}", ha="center", va="bottom", fontsize=7,
                        fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"CL = {_cl_label(cl)}" for cl in cost_limits])
    ax.set_ylabel("Epochs to Constraint Satisfaction")
    ax.set_title(f"{ENV_LABELS.get(env, env)} — Convergence Speed "
                 f"(EMA span={w_size})")
    ax.legend(loc="upper right", framealpha=0.9)
    style_axis(ax, grid_y_only=True)

    _save(fig, f"saferl_{env}_convergence_ema{w_size}", env, save_pdf)


# =============================================================================
# 7. Price of Safety — Dual Heatmap (Reward Retention + Cost Tracking)
# =============================================================================
def _collect_tail_stats(experiments: Dict, tail: int) -> Dict[str, Dict[float, dict]]:
    """Return {algo: {cl: {reward_mean, cost_mean, ...}}} for all experiments."""
    stats = {}
    for algo in ALGO_ORDER:
        if algo not in experiments:
            continue
        stats[algo] = {}
        for csv_path, cl in experiments[algo]:
            df = load_experiment(csv_path)
            if df is None:
                continue
            stats[algo][cl] = get_tail_stats(df, tail)
    return stats


def plot_price_of_safety_single(env: str, experiments: Dict, tail: int,
                                 save_pdf: bool):
    """
    Cost Tracking Ratio heatmap for a SINGLE environment.
    Rows = algorithms, columns = cost limits.
    Cell value = actual_cost / cost_limit.
    Color: green (well under limit) → yellow (at limit) → red (violation).
    Each cell annotated with the ratio + absolute cost in parentheses.
    """
    print(f"\n[PLOT] Cost Tracking Heatmap — {env}")

    algos = [a for a in ALGO_ORDER if a in experiments]
    cost_limits = sorted({cl for exps in experiments.values() for _, cl in exps})

    if not algos or not cost_limits:
        print("  [SKIP] Insufficient data")
        return

    stats = _collect_tail_stats(experiments, tail)

    n_algos = len(algos)
    n_limits = len(cost_limits)

    cost_ratio = np.full((n_algos, n_limits), np.nan)
    cost_abs = np.full((n_algos, n_limits), np.nan)

    for i, algo in enumerate(algos):
        if algo not in stats:
            continue
        for j, cl in enumerate(cost_limits):
            if cl not in stats[algo]:
                continue
            s = stats[algo][cl]
            cost_abs[i, j] = s["cost_mean"]
            cost_ratio[i, j] = s["cost_mean"] / cl if cl > 0 else 0

    fig, ax = plt.subplots(figsize=(max(8, n_limits * 2.2), 3.2))

    env_label = ENV_LABELS.get(env, env)

    # Threshold-centered colormap: green below 1.0, red above 1.0
    # 0 → 1.0: deep green → light green → white (all satisfied)
    # 1.0 → max: white → yellow → orange → red (violation)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cost_track",
        [(0.0, "#006837"),   # deep green — far under limit
         (0.25, "#1a9850"),  # green
         (0.42, "#a6d96a"),  # light green
         (0.5, "#d9f0a3"),   # very light green — just under threshold
         (0.5001, "#ffffbf"),# very light yellow — just over threshold
         (0.6, "#fee08b"),   # yellow — mild violation
         (0.75, "#fc8d59"),  # orange — moderate violation
         (1.0, "#d73027")],  # deep red — severe violation
    )

    valid = cost_ratio[~np.isnan(cost_ratio)]
    if len(valid) == 0:
        print("  [SKIP] All NaN")
        plt.close(fig)
        return

    vmax = max(np.nanmax(cost_ratio) * 1.1, 1.5)

    # TwoSlopeNorm centers the colormap at 1.0 (the constraint threshold)
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=vmax)

    im = ax.imshow(cost_ratio, cmap=cmap, aspect="auto", norm=norm)

    for i in range(n_algos):
        for j in range(n_limits):
            val = cost_ratio[i, j]
            abs_c = cost_abs[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=10, color="grey")
            else:
                text_color = "black"
                ax.text(j, i, f"{val:.2f}\n({abs_c:.1f})",
                        ha="center", va="center", fontsize=10,
                        fontweight="bold", color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Cost / Cost Limit", fontsize=11)
    cbar.ax.axhline(y=1.0, color="black", linewidth=1.5, linestyle="-")

    ax.set_xticks(range(n_limits))
    ax.set_xticklabels([f"CL = {_cl_label(cl)}" for cl in cost_limits])
    ax.set_yticks(range(n_algos))
    ax.set_yticklabels(algos)
    ax.set_xlabel("Cost Limit")
    ax.set_title(f"{env_label}: Constraint Satisfaction Across Cost Limits",
                 fontsize=13)

    fig.tight_layout()
    _save(fig, f"saferl_{env}_cost_tracking_tail{tail}", env, save_pdf)


def plot_price_of_safety_all(all_env_experiments: Dict[str, Dict], tail: int,
                              save_pdf: bool):
    """
    Generate separate Cost Tracking Ratio heatmap per environment.
    Called when --env is not specified: iterates over all envs.
    """
    print(f"\n[PLOT] Cost Tracking — generating per-environment heatmaps")

    envs = [e for e in ENV_ORDER if e in all_env_experiments
            and all_env_experiments[e]]
    if not envs:
        print("  [SKIP] No data")
        return

    for env in envs:
        plot_price_of_safety_single(env, all_env_experiments[env], tail,
                                     save_pdf)


# =============================================================================
# 6. Training Curves (dual panel, from omni_plot.py — per-algo and comparison)
# =============================================================================
def plot_curves(env: str, experiments: Dict, t_steps: int, w_size: int,
                skip: int, auto_ylim: bool, save_pdf: bool):
    """
    Original dual-panel training curves: one figure per algo (cost limits as
    lines) + cross-algorithm comparison figures.
    """
    from omni_plot import (plot_algo_dual_panel, plot_all_algos_single)

    # Per-algo plots
    for algo in ALGO_ORDER:
        if algo not in experiments:
            continue
        exps = experiments[algo]
        print(f"  [CURVES] {algo} ({len(exps)} cost limits)")

        # Default climit ref = second limit
        climit_ref = exps[1][1] if len(exps) >= 2 else exps[0][1]

        plot_algo_dual_panel(algo, exps, env, t_steps, w_size, skip,
                             climit_ref, auto_ylim, save_pdf)

    # Cross-algorithm comparison for each cost limit with >= 3 algos
    all_cls = sorted({cl for exps in experiments.values() for _, cl in exps})
    for cl in all_cls:
        algos_with = sum(1 for exps in experiments.values()
                         if any(abs(c - cl) < 0.01 for _, c in exps))
        if algos_with >= 3:
            print(f"  [CURVES] Cross-algorithm at CL={_cl_label(cl)}")
            plot_all_algos_single(experiments, env, t_steps, w_size, skip,
                                  cl, auto_ylim, save_pdf)


# =============================================================================
# CLI
# =============================================================================
PLOT_CHOICES = ["csr", "pareto", "grid", "heatmap", "convergence", "curves",
                "safety", "all"]


def main():
    parser = argparse.ArgumentParser(
        description="OmniSafe Safe RL — ICLR-quality advanced plots")
    parser.add_argument("--env", type=str, default=None,
                        choices=["evcharging", "building", "cogen"],
                        help="Environment (required for all except --plot grid)")
    parser.add_argument("--plot", type=str, default="all",
                        help=f"Plot type(s), comma-separated: {PLOT_CHOICES}")
    parser.add_argument("--t_steps", type=int, default=3000,
                        help="Max training epochs for curve plots")
    parser.add_argument("--w_size", type=int, default=200,
                        help="EMA smoothing span")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip initial epochs in curves")
    parser.add_argument("--tail", type=int, default=500,
                        help="Number of final epochs for CSR/Pareto/heatmap stats")
    parser.add_argument("--auto_ylim", action="store_true",
                        help="Auto y-axis limits for curves")
    parser.add_argument("--pdf", action="store_true",
                        help="Also save PDF (vector graphics)")
    args = parser.parse_args()

    plots = [p.strip() for p in args.plot.split(",")]
    if "all" in plots:
        plots = ["csr", "pareto", "heatmap", "convergence", "curves", "safety"]

    # Grid and safety-all require all envs
    if "grid" in plots or "grid" in args.plot:
        all_env_data = {}
        for env in ENV_ORDER:
            data = discover_experiments(env)
            if data:
                all_env_data[env] = data
        if all_env_data:
            plot_grid(all_env_data, args.t_steps, args.w_size, args.pdf)
        else:
            print("[ERROR] No data found for grid plot")

        # Remove grid from list; remaining plots need --env
        plots = [p for p in plots if p != "grid"]

    # Safety can run per-env or all-envs (when --env not given)
    if "safety" in plots:
        if args.env is None:
            # All-envs combined figure
            all_env_data = {}
            for env in ENV_ORDER:
                data = discover_experiments(env)
                if data:
                    all_env_data[env] = data
            if all_env_data:
                plot_price_of_safety_all(all_env_data, args.tail, args.pdf)
            plots = [p for p in plots if p != "safety"]
        # else: will be handled per-env below

    if not plots:
        return

    # All remaining plots need --env
    if args.env is None:
        parser.error("--env is required for plot types: " + ", ".join(plots))

    print(f"\n{'='*60}")
    print(f"Safe RL Advanced Plotter — {args.env}")
    print(f"Plots: {plots}")
    print(f"{'='*60}")

    experiments = discover_experiments(args.env)
    if not experiments:
        print(f"[ERROR] No experiments found for {args.env}")
        return

    for algo, exps in sorted(experiments.items()):
        limits = [cl for _, cl in exps]
        print(f"  {algo}: CL = {limits}")

    if "csr" in plots:
        plot_csr(args.env, experiments, args.tail, args.pdf)

    if "pareto" in plots:
        plot_pareto(args.env, experiments, args.tail, args.pdf)

    if "heatmap" in plots:
        plot_heatmap(args.env, experiments, args.tail, args.pdf)

    if "convergence" in plots:
        plot_convergence(args.env, experiments, args.w_size, args.pdf)

    if "curves" in plots:
        plot_curves(args.env, experiments, args.t_steps, args.w_size,
                    args.skip, args.auto_ylim, args.pdf)

    if "safety" in plots:
        plot_price_of_safety_single(args.env, experiments, args.tail, args.pdf)

    print(f"\nDone! Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
