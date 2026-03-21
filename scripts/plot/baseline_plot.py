"""
Baseline Evaluation Plotter
-----------------------------
Auto-discovers baseline_testing.py outputs from logs_baseline_test/ and
optionally stdrl_testing.py outputs from logs_std_test/ for comparison.
Generates publication-quality figures for thesis Chapter 4.

Plot types:
  1. bar        — Grouped bar chart: baselines (+ optional RL) per environment
  2. breakdown  — Env-specific metric subplots (profit/comfort/fuel decomposition)
  3. violin     — Reward distribution per baseline algorithm
  4. compare    — Baselines vs RL side-by-side comparison
  5. all        — All of the above

Usage:
    python scripts/plot/baseline_plot.py --env evcharging --plot bar
    python scripts/plot/baseline_plot.py --env building --plot breakdown
    python scripts/plot/baseline_plot.py --env cogen --plot violin
    python scripts/plot/baseline_plot.py --env evcharging --plot compare
    python scripts/plot/baseline_plot.py --env evcharging --plot all
    python scripts/plot/baseline_plot.py --env evcharging --plot all --with-rl
"""

import os
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from style import (ALGO_COLORS, ALGO_MARKERS, ALGO_LINESTYLES,
                   get_color, get_marker, get_linestyle,
                   ENV_TITLES, style_axis)
# style.py auto-applies rcParams on import


# ============================================================================
# Constants
# ============================================================================
OUTPUT_DIR = "./graphs/C_BASEL/"
BASELINE_LOG_ROOT = "./logs_baseline_test"
RL_LOG_ROOT = "./logs_std_test"

# Canonical ordering for baselines (worst → best expected)
BASELINE_ORDER = ["DoNothing", "Random", "Greedy", "MPC", "OfflineOptimal"]
RL_ORDER = ["PPO", "SAC", "TD3"]

# Environment-specific metric configs for breakdown plots
ENV_METRICS = {
    "evcharging": {
        "columns": ["total_profit", "total_carbon_cost", "total_excess_charge"],
        "labels":  ["Profit ($)", "Carbon Cost ($)", "Excess Charge ($)"],
        "colors":  ["#2ecc71", "#e74c3c", "#3498db"],
    },
    "building": {
        "columns": ["total_comfort_cost", "total_power_cost", "total_cost_usd"],
        "labels":  ["Comfort Cost", "Power Cost", "Total Cost (USD)"],
        "colors":  ["#e74c3c", "#3498db", "#f39c12"],
    },
    "cogen": {
        "columns": ["total_fuel_costs", "total_ramp_costs", "total_non_delivery_cost"],
        "labels":  ["Fuel Cost", "Ramp Cost", "Non-Delivery Cost"],
        "colors":  ["#e67e22", "#9b59b6", "#e74c3c"],
    },
}

# Global save formats
SAVE_FORMATS = ["png"]


# ============================================================================
# Data Loading
# ============================================================================

def load_test_run(run_dir: str) -> Optional[Dict]:
    """Load a single test run: config + episode CSV."""
    config_path = os.path.join(run_dir, "test_config.json")
    csv_path = os.path.join(run_dir, "episode_results.csv")

    if not os.path.exists(config_path) or not os.path.exists(csv_path):
        return None

    with open(config_path) as f:
        config = json.load(f)

    df = pd.read_csv(csv_path)
    if "total_reward" not in df.columns:
        return None

    rewards = df["total_reward"].values
    n = len(rewards)

    return {
        "env": config.get("env", ""),
        "algo": config.get("algo", ""),
        "baseline": config.get("baseline", False),
        "noise_obs": config.get("noise", 0.0),
        "noise_action": config.get("noise_action", 0.0),
        "noise_env": config.get("noise_env", 0.0),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "ci95": float(1.96 * np.std(rewards) / np.sqrt(n)),
        "median_reward": float(np.median(rewards)),
        "n_episodes": n,
        "run_dir": run_dir,
        "df": df,
    }


def discover_runs(env: str = None, algo: str = None,
                  root: str = BASELINE_LOG_ROOT) -> List[Dict]:
    """Auto-discover test runs from a log directory."""
    runs = []
    if not os.path.exists(root):
        print(f"  Warning: {root} not found")
        return runs

    for subdir in sorted(Path(root).iterdir()):
        if not subdir.is_dir():
            continue
        for run_dir in sorted(subdir.iterdir()):
            if not run_dir.is_dir():
                continue
            result = load_test_run(str(run_dir))
            if result is None:
                continue
            if env and result["env"] != env:
                continue
            if algo and result["algo"] != algo:
                continue
            runs.append(result)

    return runs


def discover_baselines(env: str, root: str = BASELINE_LOG_ROOT) -> List[Dict]:
    """Discover baseline-only runs for a given env."""
    runs = discover_runs(env=env, root=root)
    runs = [r for r in runs
            if r["noise_obs"] == 0 and r["noise_action"] == 0 and r["noise_env"] == 0]
    print(f"  Discovered {len(runs)} baseline runs for {env}")
    return runs


def discover_rl(env: str, root: str = RL_LOG_ROOT) -> List[Dict]:
    """Discover RL runs at noise=0 for comparison."""
    runs = discover_runs(env=env, root=root)
    runs = [r for r in runs
            if r["noise_obs"] == 0 and r["noise_action"] == 0 and r["noise_env"] == 0]
    print(f"  Discovered {len(runs)} RL runs for {env}")
    return runs


def _sort_algos(algos: list, include_rl: bool = False) -> list:
    """Sort algorithms: baselines first (in canonical order), then RL."""
    order = BASELINE_ORDER + (RL_ORDER if include_rl else [])
    return sorted(algos, key=lambda a: order.index(a) if a in order else 999)


def _get_algo_data(runs, algos):
    """Compute mean/ci95 per algorithm from runs."""
    means, cis = [], []
    for alg in algos:
        alg_runs = [r for r in runs if r["algo"] == alg]
        if alg_runs:
            all_rewards = np.concatenate([r["df"]["total_reward"].values for r in alg_runs])
            means.append(np.mean(all_rewards))
            cis.append(1.96 * np.std(all_rewards) / np.sqrt(len(all_rewards)))
        else:
            means.append(np.nan)
            cis.append(0)
    return means, cis


# ============================================================================
# Save helper
# ============================================================================

def _save(plot_type: str, env: str):
    """Save figure to graphs/C_BASEL/{env}/{plot_type}.{ext}."""
    subdir = os.path.join(OUTPUT_DIR, env)
    os.makedirs(subdir, exist_ok=True)

    for ext in SAVE_FORMATS:
        path = os.path.join(subdir, f"{plot_type}.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")


# ============================================================================
# Plot 1: Grouped Bar Chart
# ============================================================================

def plot_bar(runs: List[Dict], env: str, save: bool = True):
    """Bar chart comparing all baselines (and optionally RL) for one env."""
    if not runs:
        print("  No data for bar chart")
        return

    algos = _sort_algos(list(set(r["algo"] for r in runs)), include_rl=True)
    n_algos = len(algos)
    means, cis = _get_algo_data(runs, algos)
    colors = [get_color(alg) for alg in algos]

    x = np.arange(n_algos)
    fig, ax = plt.subplots(figsize=(max(7, n_algos * 1.1 + 2), 4.5))

    ax.bar(x, means, yerr=cis, capsize=2.5,
           color=colors, alpha=0.85,
           edgecolor="white", linewidth=0.6, width=0.65,
           error_kw={"linewidth": 1.0, "capthick": 0.8, "color": "0.3"})

    # Y-axis auto-range
    valid = [(m, c) for m, c in zip(means, cis) if not np.isnan(m)]
    if valid:
        ymin = min(m - c for m, c in valid)
        ymax = max(m + c for m, c in valid)
        margin = (ymax - ymin) * 0.12 if ymax != ymin else abs(ymax) * 0.1
        ax.set_ylim(ymin - margin, ymax + margin)

    ax.set_xticks(x)
    ax.set_xticklabels(algos)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title(f"{ENV_TITLES.get(env, env)} \u2014 Algorithm Comparison")
    style_axis(ax, grid_y_only=True)
    fig.tight_layout()

    if save:
        _save("bar", env)


# ============================================================================
# Plot 2: Metric Breakdown
# ============================================================================

def plot_breakdown(runs: List[Dict], env: str, save: bool = True):
    """Subplot breakdown of env-specific metrics per algorithm."""
    metric_cfg = ENV_METRICS.get(env)
    if not metric_cfg:
        print(f"  No metric config for env={env}")
        return

    if not runs:
        print("  No data for breakdown")
        return

    columns = metric_cfg["columns"]
    labels = metric_cfg["labels"]
    colors = metric_cfg["colors"]

    sample_df = runs[0]["df"]
    avail = [(c, l, clr) for c, l, clr in zip(columns, labels, colors)
             if c in sample_df.columns]
    if not avail:
        print(f"  No matching metric columns for env={env}")
        return

    algos = _sort_algos(list(set(r["algo"] for r in runs)), include_rl=True)
    n_metrics = len(avail)
    n_algos = len(algos)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4.2 * n_metrics, 4.5),
                              sharey=False)
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_algos)

    for ax, (col, lbl, clr) in zip(axes, avail):
        means, cis = [], []
        bar_colors = []
        for alg in algos:
            alg_runs = [r for r in runs if r["algo"] == alg]
            if alg_runs and col in alg_runs[0]["df"].columns:
                all_vals = np.concatenate([r["df"][col].dropna().values for r in alg_runs])
                if len(all_vals) > 0:
                    means.append(np.mean(all_vals))
                    cis.append(1.96 * np.std(all_vals) / np.sqrt(len(all_vals)))
                else:
                    means.append(np.nan)
                    cis.append(0)
            else:
                means.append(np.nan)
                cis.append(0)
            bar_colors.append(get_color(alg))

        ax.bar(x, means, yerr=cis, capsize=2.5,
               color=bar_colors, alpha=0.85,
               edgecolor="white", linewidth=0.6, width=0.65,
               error_kw={"linewidth": 0.8, "capthick": 0.7, "color": "0.3"})

        # Y-axis auto-range (don't force zero)
        valid = [(m, c) for m, c in zip(means, cis) if not np.isnan(m)]
        if valid:
            ymin = min(m - c for m, c in valid)
            ymax = max(m + c for m, c in valid)
            margin = (ymax - ymin) * 0.12 if ymax != ymin else abs(ymax) * 0.1
            # Only pull up from zero if data is far from it
            if ymin > 0 and ymin > ymax * 0.3:
                ax.set_ylim(ymin - margin, ymax + margin)

        ax.set_xticks(x)
        ax.set_xticklabels(algos, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel(lbl)
        ax.set_title(lbl)
        style_axis(ax, grid_y_only=True)

    fig.suptitle(f"{ENV_TITLES.get(env, env)} \u2014 Metric Breakdown",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save:
        _save("breakdown", env)


# ============================================================================
# Plot 3: Violin
# ============================================================================

def plot_violin(runs: List[Dict], env: str, save: bool = True):
    """Raincloud plot: half-violin + box + jittered points."""
    if not runs:
        print("  No data for violin")
        return

    all_rows = []
    for r in runs:
        for _, row in r["df"].iterrows():
            all_rows.append({
                "algo": r["algo"],
                "reward": row["total_reward"],
            })

    vdf = pd.DataFrame(all_rows)
    algos = _sort_algos(list(vdf["algo"].unique()), include_rl=True)
    n_algos = len(algos)

    fig, ax = plt.subplots(figsize=(max(7, n_algos * 1.5 + 1), 5))

    rng = np.random.default_rng(42)
    offset_v = 0.15   # half-violin shift
    offset_s = -0.15  # strip shift
    bw = 0.12         # box width

    for i, alg in enumerate(algos):
        data = vdf[vdf["algo"] == alg]["reward"].values
        color = get_color(alg)

        if len(data) < 2:
            continue

        # --- Half-violin (right side) ---
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data, bw_method=0.3)
        y_grid = np.linspace(data.min(), data.max(), 200)
        density = kde(y_grid)
        # Scale density to fit width ~0.3
        density = density / density.max() * 0.3
        ax.fill_betweenx(y_grid, i + offset_v, i + offset_v + density,
                         color=color, alpha=0.45, linewidth=0)
        ax.plot(i + offset_v + density, y_grid, color=color,
                linewidth=0.8, alpha=0.7)

        # --- Box plot (center) ---
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1
        whisker_lo = max(data.min(), q1 - 1.5 * iqr)
        whisker_hi = min(data.max(), q3 + 1.5 * iqr)

        # Box
        box = plt.Rectangle((i - bw / 2, q1), bw, iqr,
                             facecolor=color, alpha=0.6,
                             edgecolor="0.3", linewidth=0.8)
        ax.add_patch(box)
        # Median line
        ax.plot([i - bw / 2, i + bw / 2], [med, med],
                color="white", linewidth=1.8, solid_capstyle="round")
        ax.plot([i - bw / 2, i + bw / 2], [med, med],
                color="0.2", linewidth=1.0, solid_capstyle="round")
        # Whiskers
        ax.plot([i, i], [whisker_lo, q1], color="0.3", linewidth=0.8)
        ax.plot([i, i], [q3, whisker_hi], color="0.3", linewidth=0.8)
        ax.plot([i - bw / 4, i + bw / 4], [whisker_lo, whisker_lo],
                color="0.3", linewidth=0.8)
        ax.plot([i - bw / 4, i + bw / 4], [whisker_hi, whisker_hi],
                color="0.3", linewidth=0.8)

        # --- Jittered strip (left side) ---
        jitter = rng.uniform(-0.08, 0.02, len(data))
        ax.scatter(i + offset_s + jitter, data,
                   s=6, alpha=0.35, color=color,
                   edgecolors="none", zorder=2)

    ax.set_xticks(range(n_algos))
    ax.set_xticklabels(algos)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Episode Reward")
    ax.set_title(f"{ENV_TITLES.get(env, env)} \u2014 Reward Distribution")
    style_axis(ax, grid_y_only=True)
    fig.tight_layout()

    if save:
        _save("violin", env)


# ============================================================================
# Plot 4: Baselines vs RL Comparison
# ============================================================================

def plot_compare(baseline_runs: List[Dict], rl_runs: List[Dict],
                 env: str, save: bool = True):
    """Side-by-side comparison: baselines vs RL with hatch distinction."""
    all_runs = baseline_runs + rl_runs
    if not all_runs:
        print("  No data for comparison")
        return

    baseline_algos = _sort_algos(list(set(r["algo"] for r in baseline_runs)))
    rl_algos = _sort_algos(list(set(r["algo"] for r in rl_runs)), include_rl=True)
    rl_algos = [a for a in rl_algos if a in RL_ORDER]
    all_algos = baseline_algos + rl_algos
    n_total = len(all_algos)

    if n_total == 0:
        print("  No algorithms found for comparison")
        return

    means, cis = _get_algo_data(all_runs, all_algos)
    colors = [get_color(alg) for alg in all_algos]
    is_rl = [alg in RL_ORDER for alg in all_algos]

    x = np.arange(n_total)
    fig, ax = plt.subplots(figsize=(max(7, n_total * 1.1 + 2), 4.5))

    bars = ax.bar(x, means, yerr=cis, capsize=2.5,
                  color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.6, width=0.65,
                  error_kw={"linewidth": 1.0, "capthick": 0.8, "color": "0.3"})

    # Hatch RL bars to visually distinguish from baselines
    for bar, rl in zip(bars, is_rl):
        if rl:
            bar.set_hatch("///")
            bar.set_edgecolor("0.4")

    # Divider line between baselines and RL
    if baseline_algos and rl_algos:
        div_x = len(baseline_algos) - 0.5
        ax.axvline(div_x, color="0.6", linestyle="--", linewidth=1.0, alpha=0.5)

    # Y-axis auto-range
    valid = [(m, c) for m, c in zip(means, cis) if not np.isnan(m)]
    if valid:
        ymin = min(m - c for m, c in valid)
        ymax = max(m + c for m, c in valid)
        margin = (ymax - ymin) * 0.12 if ymax != ymin else abs(ymax) * 0.1
        ax.set_ylim(ymin - margin, ymax + margin)

    ax.set_xticks(x)
    ax.set_xticklabels(all_algos)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title(f"{ENV_TITLES.get(env, env)} \u2014 Baselines vs RL")

    # Legend: solid = baseline, hatched = RL
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor="0.7", edgecolor="white", label="Baseline"),
                      Patch(facecolor="0.7", edgecolor="0.4", hatch="///", label="RL")]
    ax.legend(handles=legend_handles, loc="best", frameon=True)
    style_axis(ax, grid_y_only=True)
    fig.tight_layout()

    if save:
        _save("compare", env)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Baseline Evaluation Plotter (paper-quality)")
    parser.add_argument("--env", type=str, required=True,
                        choices=["evcharging", "building", "cogen"],
                        help="Environment to plot")
    parser.add_argument("--plot", type=str, default="all",
                        choices=["bar", "breakdown", "violin", "compare", "all"],
                        help="Plot type to generate")
    parser.add_argument("--baseline-root", type=str, default=BASELINE_LOG_ROOT,
                        help="Root directory for baseline logs")
    parser.add_argument("--rl-root", type=str, default=RL_LOG_ROOT,
                        help="Root directory for RL test logs (for compare plot)")
    parser.add_argument("--with-rl", action="store_true",
                        help="Include RL results in bar/violin/breakdown plots")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not call plt.show()")
    parser.add_argument("--formats", nargs="+", default=["png"],
                        choices=["png", "pdf", "svg"],
                        help="Output formats (default: png)")

    args = parser.parse_args()

    global SAVE_FORMATS
    SAVE_FORMATS = args.formats

    env = args.env

    # Load baseline runs
    print(f"\n{'='*60}")
    print(f"Loading baseline results for {env}...")
    baseline_runs = discover_baselines(env, root=args.baseline_root)

    # Load RL runs (for compare, or if --with-rl)
    rl_runs = []
    if args.plot in ("compare", "all") or args.with_rl:
        print(f"Loading RL results for {env}...")
        rl_runs = discover_rl(env, root=args.rl_root)

    if not baseline_runs and not rl_runs:
        print("No runs found. Run baseline_testing.py first.")
        return

    # Combined runs (for bar/violin/breakdown when --with-rl)
    combined = baseline_runs + (rl_runs if args.with_rl else [])

    # Summary table
    print(f"\n{'='*60}")
    print("Available runs:")
    for r in baseline_runs + rl_runs:
        tag = "BL" if r.get("baseline") or r["algo"] in BASELINE_ORDER else "RL"
        print(f"  [{tag}] {r['algo']:18s}  reward={r['mean_reward']:.4f} \u00b1 {r['std_reward']:.4f}  "
              f"({r['n_episodes']} eps)")
    print(f"{'='*60}\n")

    # Generate plots
    if args.plot in ("bar", "all"):
        print("[Plot] Bar chart...")
        plot_bar(combined, env)

    if args.plot in ("breakdown", "all"):
        print("[Plot] Metric breakdown...")
        plot_breakdown(combined, env)

    if args.plot in ("violin", "all"):
        print("[Plot] Violin...")
        plot_violin(combined, env)

    if args.plot in ("compare", "all"):
        print("[Plot] Baselines vs RL comparison...")
        plot_compare(baseline_runs, rl_runs, env)

    if not args.no_show:
        plt.show()

    print(f"\nOutput directory: {os.path.join(OUTPUT_DIR, env)}")


if __name__ == "__main__":
    main()
