"""
Post-Training Evaluation Plotter
---------------------------------
Auto-discovers stdrl_testing.py outputs from logs_std_test/ and generates
publication-quality figures for noise robustness analysis.

Produces six plot types:
  1. Grouped bar chart with error bars (main result figure)
  2. Degradation line plot (noise sensitivity)
  3. Heatmap (train-noise x test-noise robustness matrix)
  4. Violin plot (reward distribution per noise level)
  5. Metric breakdown subplot (env-specific component analysis)
  6. Noise panel (obs/act/env noise comparison in 3 subplots)

Usage:
    python scripts/plot/stdrl_post_plot.py --env evcharging --algo SAC --noise-type obs --plot bar
    python scripts/plot/stdrl_post_plot.py --env building --noise-type obs --plot line
    python scripts/plot/stdrl_post_plot.py --env cogen --algo PPO --noise-type obs --plot heatmap
    python scripts/plot/stdrl_post_plot.py --env evcharging --algo SAC --noise-type obs --plot violin
    python scripts/plot/stdrl_post_plot.py --env evcharging --algo SAC --noise-type obs --plot breakdown
    python scripts/plot/stdrl_post_plot.py --env evcharging --algo SAC --plot panel
    python scripts/plot/stdrl_post_plot.py --env evcharging --algo SAC --noise-type obs --plot all
"""

import os
import sys
import json
import re
import argparse
import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from style import (ALGO_COLORS, ALGO_MARKERS, ALGO_LINESTYLES,
                   get_color, get_marker, get_linestyle,
                   ENV_TITLES as _ENV_TITLES, style_axis)
# style.py auto-applies rcParams on import


# ============================================================================
# Constants
# ============================================================================
OUTPUT_DIR = "./graphs/C_POST/"
TEST_LOG_ROOT = "./logs_std_test"

# Post-plot uses longer display names for titles
ENV_TITLES = {
    "evcharging": "EV Charging",
    "building": "Building",
    "cogen": "Cogeneration",
}

NOISE_LABELS = {
    "obs": "Observation Noise ($\\sigma$)",
    "action": "Action Noise ($\\sigma$)",
    "env": "Environment Noise ($\\sigma$)",
}

NOISE_SHORT = {
    "obs": "Obs",
    "action": "Act",
    "env": "Env",
}

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


# ============================================================================
# Helpers (delegating to style.py)
# ============================================================================

_get_color = get_color
_get_marker = get_marker
_get_ls = get_linestyle


# Global: set by main() from --formats arg
SAVE_FORMATS = ["png"]


def _save(plot_type: str, env: str, noise_type: str, algo: str = None):
    """Save figure into organized folder structure:

    graphs/C_POST/
      {env}/
        {algo}/                     # algo-specific plots (bar, heatmap, violin, breakdown)
          {noise_type}_{plot_type}.{ext}
        comparison/                 # multi-algo plots (line, norm — no algo filter)
          {noise_type}_{plot_type}.{ext}
        panel.{ext}                 # noise panel (all noise types, top level)

    Latest run overwrites previous files (no timestamp clutter).
    Formats controlled by --formats flag (default: png only).
    """
    # Determine subfolder
    if plot_type == "panel":
        subdir = os.path.join(OUTPUT_DIR, env)
        if algo:
            subdir = os.path.join(subdir, algo)
        filename = "panel"
    elif algo:
        subdir = os.path.join(OUTPUT_DIR, env, algo)
        filename = f"{noise_type}_{plot_type}"
    else:
        subdir = os.path.join(OUTPUT_DIR, env, "comparison")
        filename = f"{noise_type}_{plot_type}"

    os.makedirs(subdir, exist_ok=True)

    for ext in SAVE_FORMATS:
        path = os.path.join(subdir, f"{filename}.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")


def _nice_noise_label(val: float) -> str:
    """Format noise value for tick labels."""
    if val == 0:
        return "0"
    if val == int(val):
        return str(int(val))
    return f"{val:g}"


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
                  root: str = TEST_LOG_ROOT) -> List[Dict]:
    """Auto-discover all test runs from logs_std_test/."""
    runs = []
    if not os.path.exists(root):
        print(f"Warning: {root} not found")
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

    print(f"Discovered {len(runs)} test runs" +
          (f" for env={env}" if env else "") +
          (f", algo={algo}" if algo else ""))
    return runs


def load_manual_dirs(csv_dirs: List[str]) -> List[Dict]:
    """Load runs from explicitly provided directories."""
    runs = []
    for d in csv_dirs:
        result = load_test_run(d)
        if result:
            runs.append(result)
        else:
            print(f"Warning: could not load {d}")
    return runs


def get_noise_value(run: Dict, noise_type: str) -> float:
    key = {"obs": "noise_obs", "action": "noise_action", "env": "noise_env"}
    return run[key[noise_type]]


def build_summary_table(runs: List[Dict], noise_type: str) -> pd.DataFrame:
    rows = []
    for r in runs:
        rows.append({
            "algo": r["algo"],
            "noise": get_noise_value(r, noise_type),
            "mean_reward": r["mean_reward"],
            "std_reward": r["std_reward"],
            "ci95": r["ci95"],
            "n_episodes": r["n_episodes"],
        })
    return pd.DataFrame(rows).sort_values(["algo", "noise"]).reset_index(drop=True)


def _parse_train_noise(model_path: str, noise_type: str) -> float:
    """Extract train noise from model path like .../NOISE_0.1_ACT_0.0_ENV_0.0/..."""
    key_map = {
        "obs": r"NOISE_([\d.]+)",
        "action": r"ACT_([\d.]+)",
        "env": r"ENV_([\d.]+)",
    }
    pattern = key_map.get(noise_type, r"NOISE_([\d.]+)")
    match = re.search(pattern, model_path)
    if match:
        return float(match.group(1))
    # Fallback: DS_ pattern
    if noise_type == "obs":
        match = re.search(r"DS_([\d.]+)", model_path)
        if match:
            return float(match.group(1))
    return 0.0


# ============================================================================
# Plot 1: Grouped Bar Chart
# ============================================================================

def plot_grouped_bar(runs: List[Dict], noise_type: str, env: str,
                     algo: str = None, save: bool = True):
    """Clean grouped bar chart: noise levels on x-axis, algorithms as groups."""
    df = build_summary_table(runs, noise_type)
    if df.empty:
        print("No data for grouped bar chart")
        return

    algos = sorted(df["algo"].unique())
    noise_levels = sorted(df["noise"].unique())
    n_algos = len(algos)
    n_noise = len(noise_levels)

    bar_width = 0.7 / max(n_algos, 1)
    x = np.arange(n_noise)

    fig, ax = plt.subplots(figsize=(max(7, n_noise * 1.1 + 2), 4.5))

    for i, alg in enumerate(algos):
        alg_df = df[df["algo"] == alg]
        means, cis = [], []
        for nl in noise_levels:
            row = alg_df[alg_df["noise"] == nl]
            if len(row) > 0:
                means.append(row["mean_reward"].values[0])
                cis.append(row["ci95"].values[0])
            else:
                means.append(np.nan)
                cis.append(0)

        offset = (i - n_algos / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width * 0.88,
               yerr=cis, capsize=2.5,
               label=alg, color=_get_color(alg), alpha=0.85,
               edgecolor="white", linewidth=0.6,
               error_kw={"linewidth": 1.0, "capthick": 0.8, "color": "0.3"})

    # Y-axis: auto-range with some padding, don't force zero
    all_means = df["mean_reward"].values
    all_cis = df["ci95"].values
    ymin = np.nanmin(all_means - all_cis)
    ymax = np.nanmax(all_means + all_cis)
    margin = (ymax - ymin) * 0.12 if ymax != ymin else abs(ymax) * 0.1
    ax.set_ylim(ymin - margin, ymax + margin)

    ax.set_xticks(x)
    ax.set_xticklabels([_nice_noise_label(nl) for nl in noise_levels])
    ax.set_xlabel(NOISE_LABELS.get(noise_type, "Noise Level"))
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title(f"{ENV_TITLES.get(env, env)} \u2014 {NOISE_SHORT.get(noise_type, '')} Noise Robustness")

    ax.legend(loc="best", frameon=True)
    ax.yaxis.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if save:
        _save("bar", env, noise_type, algo)


# ============================================================================
# Plot 2: Degradation Line Plot
# ============================================================================

def plot_degradation_line(runs: List[Dict], noise_type: str, env: str,
                          normalize: bool = False, save: bool = True):
    """Line plot: x=noise, y=reward. Optional normalization to baseline."""
    df = build_summary_table(runs, noise_type)
    if df.empty:
        print("No data for degradation line plot")
        return

    algos = sorted(df["algo"].unique())

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for alg in algos:
        alg_df = df[df["algo"] == alg].sort_values("noise")
        x = alg_df["noise"].values
        y = alg_df["mean_reward"].values
        ci = alg_df["ci95"].values

        if normalize and len(y) > 0:
            baseline = y[0] if y[0] != 0 else 1.0
            y = y / abs(baseline) * 100
            ci = ci / abs(baseline) * 100

        ax.plot(x, y, label=alg, color=_get_color(alg),
                marker=_get_marker(alg), markersize=7,
                linestyle=_get_ls(alg),
                linewidth=2.2, markeredgecolor="white", markeredgewidth=0.8)
        ax.fill_between(x, y - ci, y + ci, alpha=0.15, color=_get_color(alg))

    # Baseline reference line (noise=0 performance)
    if not normalize:
        baseline_vals = df[df["noise"] == 0]["mean_reward"]
        if len(baseline_vals) > 0:
            bl = baseline_vals.mean()
            ax.axhline(bl, color="0.5", linestyle=":", linewidth=1.0, alpha=0.6,
                        label=f"Baseline ({bl:.2f})")
    else:
        ax.axhline(100, color="0.5", linestyle=":", linewidth=1.0, alpha=0.6)

    ax.set_xlabel(NOISE_LABELS.get(noise_type, "Noise Level"))
    ylabel = "Performance (% of Baseline)" if normalize else "Mean Episode Reward"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ENV_TITLES.get(env, env)} \u2014 {NOISE_SHORT.get(noise_type, '')} Noise Sensitivity")

    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if save:
        suffix = "norm" if normalize else "line"
        _save(suffix, env, noise_type)


# ============================================================================
# Plot 3: Heatmap (Train Noise x Test Noise)
# ============================================================================

def plot_heatmap(runs: List[Dict], noise_type: str, env: str,
                 algo: str = None, save: bool = True):
    """Heatmap with diagonal highlighting for matched train=test noise."""
    records = []
    for r in runs:
        config_path = os.path.join(r["run_dir"], "test_config.json")
        if not os.path.exists(config_path):
            continue
        with open(config_path) as f:
            cfg = json.load(f)

        if algo and cfg.get("algo") != algo:
            continue

        test_noise = get_noise_value(r, noise_type)
        model_path = cfg.get("model_path", "")
        train_noise = _parse_train_noise(model_path, noise_type)

        records.append({
            "train_noise": train_noise,
            "test_noise": test_noise,
            "mean_reward": r["mean_reward"],
        })

    if not records:
        print("No data for heatmap (could not infer train noise from model paths)")
        return

    hdf = pd.DataFrame(records)
    pivot = hdf.pivot_table(index="train_noise", columns="test_noise",
                            values="mean_reward", aggfunc="mean")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    n_rows, n_cols = pivot.shape
    fig, ax = plt.subplots(figsize=(max(5, n_cols * 0.9 + 1.5),
                                     max(4, n_rows * 0.75 + 1)))

    sns.heatmap(pivot, annot=True, fmt=".2f",
                cmap="RdYlGn", linewidths=0.8, linecolor="white",
                ax=ax, annot_kws={"size": 9, "weight": "bold"},
                cbar_kws={"label": "Mean Reward", "shrink": 0.85})

    # Highlight diagonal (train_noise == test_noise)
    for i, tr in enumerate(pivot.index):
        for j, te in enumerate(pivot.columns):
            if abs(tr - te) < 1e-9:
                ax.add_patch(plt.Rectangle((j, i), 1, 1,
                             fill=False, edgecolor="black",
                             linewidth=2.5, clip_on=False))

    ax.set_xlabel(f"Test {NOISE_SHORT.get(noise_type, '')} Noise ($\\sigma$)")
    ax.set_ylabel(f"Train {NOISE_SHORT.get(noise_type, '')} Noise ($\\sigma$)")
    # Format tick labels
    ax.set_xticklabels([_nice_noise_label(v) for v in pivot.columns], rotation=0)
    ax.set_yticklabels([_nice_noise_label(v) for v in pivot.index], rotation=0)

    title_algo = f" ({algo})" if algo else ""
    ax.set_title(f"{ENV_TITLES.get(env, env)}{title_algo} \u2014 Robustness Matrix")
    fig.tight_layout()

    if save:
        _save("heatmap", env, noise_type, algo)


# ============================================================================
# Plot 4: Violin Plot
# ============================================================================

def plot_violin(runs: List[Dict], noise_type: str, env: str,
                algo: str = None, save: bool = True):
    """Violin plot showing full reward distribution per noise level."""
    # Build per-episode DataFrame
    all_rows = []
    for r in runs:
        if algo and r["algo"] != algo:
            continue
        noise_val = get_noise_value(r, noise_type)
        for _, row in r["df"].iterrows():
            all_rows.append({
                "algo": r["algo"],
                "noise": noise_val,
                "reward": row["total_reward"],
            })

    if not all_rows:
        print("No data for violin plot")
        return

    vdf = pd.DataFrame(all_rows)
    algos = sorted(vdf["algo"].unique())
    noise_levels = sorted(vdf["noise"].unique())
    n_noise = len(noise_levels)

    if len(algos) == 1:
        # Single algo: simple violin per noise level
        fig, ax = plt.subplots(figsize=(max(6, n_noise * 0.9 + 1), 4.5))

        parts = ax.violinplot(
            [vdf[(vdf["noise"] == nl)]["reward"].values for nl in noise_levels],
            positions=range(n_noise), showmeans=True, showmedians=True,
            showextrema=False)

        color = _get_color(algos[0])
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.35)
            pc.set_edgecolor(color)
            pc.set_linewidth(0.8)
        parts["cmeans"].set_color(color)
        parts["cmeans"].set_linewidth(1.5)
        parts["cmedians"].set_color("0.3")
        parts["cmedians"].set_linewidth(1.0)
        parts["cmedians"].set_linestyle("--")

        # Overlay strip/jitter
        for i, nl in enumerate(noise_levels):
            data = vdf[vdf["noise"] == nl]["reward"].values
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(data))
            ax.scatter(i + jitter, data, s=8, alpha=0.3, color=color,
                       edgecolors="none", zorder=3)

        ax.set_xticks(range(n_noise))
        ax.set_xticklabels([_nice_noise_label(nl) for nl in noise_levels])
        title_algo = f" ({algos[0]})" if algos[0] else ""
        ax.set_title(f"{ENV_TITLES.get(env, env)}{title_algo} \u2014 Reward Distribution")

    else:
        # Multiple algos: grouped violin via seaborn
        fig, ax = plt.subplots(figsize=(max(8, n_noise * 1.5 + 2), 5))

        palette = {alg: _get_color(alg) for alg in algos}
        vdf["noise_str"] = vdf["noise"].apply(_nice_noise_label)
        # Maintain numeric ordering
        noise_order = [_nice_noise_label(nl) for nl in noise_levels]

        sns.violinplot(data=vdf, x="noise_str", y="reward", hue="algo",
                       order=noise_order, palette=palette, inner="quartile",
                       linewidth=0.8, saturation=0.8, cut=0, ax=ax)

        ax.set_title(f"{ENV_TITLES.get(env, env)} \u2014 Reward Distribution")

    ax.set_xlabel(NOISE_LABELS.get(noise_type, "Noise Level"))
    ax.set_ylabel("Episode Reward")
    ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    if len(algos) > 1:
        ax.legend(loc="best", frameon=True, title="Algorithm")
    fig.tight_layout()

    if save:
        _save("violin", env, noise_type, algo)


# ============================================================================
# Plot 5: Metric Breakdown Subplot
# ============================================================================

def plot_metric_breakdown(runs: List[Dict], noise_type: str, env: str,
                          algo: str = None, save: bool = True):
    """Subplots showing env-specific metrics breakdown across noise levels."""
    metric_cfg = ENV_METRICS.get(env)
    if not metric_cfg:
        print(f"No metric config for env={env}")
        return

    columns = metric_cfg["columns"]
    labels = metric_cfg["labels"]
    colors = metric_cfg["colors"]

    # Filter runs
    filtered = [r for r in runs if (not algo or r["algo"] == algo)]
    if not filtered:
        print("No data for metric breakdown")
        return

    # Check which columns actually exist
    available_cols = []
    available_labels = []
    available_colors = []
    sample_df = filtered[0]["df"]
    for col, lbl, clr in zip(columns, labels, colors):
        if col in sample_df.columns:
            available_cols.append(col)
            available_labels.append(lbl)
            available_colors.append(clr)

    if not available_cols:
        print(f"No matching metric columns in episode CSV for env={env}")
        return

    n_metrics = len(available_cols)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.2 * n_metrics, 4),
                              sharey=False)
    if n_metrics == 1:
        axes = [axes]

    # Gather data per noise level
    noise_levels = sorted(set(get_noise_value(r, noise_type) for r in filtered))

    for ax, col, lbl, clr in zip(axes, available_cols, available_labels, available_colors):
        means, cis = [], []
        for nl in noise_levels:
            nl_runs = [r for r in filtered if abs(get_noise_value(r, noise_type) - nl) < 1e-9]
            all_vals = np.concatenate([r["df"][col].values for r in nl_runs])
            means.append(np.mean(all_vals))
            cis.append(1.96 * np.std(all_vals) / np.sqrt(len(all_vals)))

        x = np.arange(len(noise_levels))
        ax.bar(x, means, yerr=cis, capsize=2.5, color=clr, alpha=0.75,
               edgecolor="white", linewidth=0.5,
               error_kw={"linewidth": 0.8, "capthick": 0.7, "color": "0.3"})

        # Also overlay line for trend clarity
        ax.plot(x, means, color=clr, marker="o", markersize=5,
                linewidth=1.5, markeredgecolor="white", markeredgewidth=0.6,
                zorder=5, alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels([_nice_noise_label(nl) for nl in noise_levels],
                            fontsize=8)
        ax.set_xlabel(NOISE_LABELS.get(noise_type, "Noise"), fontsize=10)
        ax.set_ylabel(lbl, fontsize=10)
        ax.set_title(lbl, fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

    title_algo = f" ({algo})" if algo else ""
    fig.suptitle(f"{ENV_TITLES.get(env, env)}{title_algo} \u2014 Metric Breakdown",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save:
        _save("breakdown", env, noise_type, algo)


# ============================================================================
# Plot 6: 3-Noise-Type Panel
# ============================================================================

def plot_noise_panel(runs_all: List[Dict], env: str,
                     algo: str = None, save: bool = True):
    """Side-by-side subplots for obs/action/env noise in one figure.

    Uses ALL runs for the given env (optionally filtered by algo), grouping
    by which noise axis is non-zero (or all-zero = baseline).
    """
    noise_types = ["obs", "action", "env"]
    noise_keys = ["noise_obs", "noise_action", "noise_env"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax, nt, nk in zip(axes, noise_types, noise_keys):
        # Select runs where only this noise type varies (others are 0)
        other_keys = [k for k in noise_keys if k != nk]
        panel_runs = [r for r in runs_all
                      if all(r[ok] == 0 for ok in other_keys)]
        if algo:
            panel_runs = [r for r in panel_runs if r["algo"] == algo]

        if not panel_runs:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="0.5")
            ax.set_title(f"{NOISE_SHORT[nt]} Noise", fontweight="bold")
            ax.set_xlabel(NOISE_LABELS[nt])
            continue

        df = build_summary_table(panel_runs, nt)
        algos = sorted(df["algo"].unique())

        for alg in algos:
            alg_df = df[df["algo"] == alg].sort_values("noise")
            x = alg_df["noise"].values
            y = alg_df["mean_reward"].values
            ci = alg_df["ci95"].values

            ax.plot(x, y, label=alg, color=_get_color(alg),
                    marker=_get_marker(alg), markersize=6,
                    linestyle=_get_ls(alg), linewidth=2.0,
                    markeredgecolor="white", markeredgewidth=0.6)
            ax.fill_between(x, y - ci, y + ci, alpha=0.12,
                            color=_get_color(alg))

        ax.set_title(f"{NOISE_SHORT[nt]} Noise", fontweight="bold")
        ax.set_xlabel(NOISE_LABELS[nt])
        ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean Episode Reward")
    # Single shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        # Try other panels
        for ax in axes[1:]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=len(labels), frameon=True,
                   bbox_to_anchor=(0.5, 1.06), fontsize=10)

    title_algo = f" ({algo})" if algo else ""
    fig.suptitle(f"{ENV_TITLES.get(env, env)}{title_algo} \u2014 Noise Comparison",
                 fontsize=14, fontweight="bold", y=1.12)
    fig.tight_layout()

    if save:
        _save("panel", env, "all", algo)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-Training Evaluation Plotter (paper-quality)")
    parser.add_argument("--env", type=str, default=None,
                        choices=["evcharging", "building", "cogen"],
                        help="Filter by environment")
    parser.add_argument("--algo", type=str, default=None,
                        choices=["PPO", "SAC", "TD3"],
                        help="Filter by algorithm")
    parser.add_argument("--noise-type", type=str, default="obs",
                        choices=["obs", "action", "env"],
                        help="Which noise axis to plot (ignored for panel)")
    parser.add_argument("--plot", type=str, default="all",
                        choices=["bar", "line", "norm", "heatmap",
                                 "violin", "breakdown", "panel", "all"],
                        help="Plot type to generate")
    parser.add_argument("--csv-dirs", nargs="+", default=None,
                        help="Manual test run directories")
    parser.add_argument("--test-root", type=str, default=TEST_LOG_ROOT,
                        help="Root directory for test logs")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not call plt.show()")
    parser.add_argument("--figsize", type=float, nargs=2, default=None,
                        metavar=("W", "H"),
                        help="Override figure size (width height)")
    parser.add_argument("--formats", nargs="+", default=["png"],
                        choices=["png", "pdf", "svg"],
                        help="Output formats (default: png). E.g. --formats png pdf svg")

    args = parser.parse_args()

    # Set global save formats
    global SAVE_FORMATS
    SAVE_FORMATS = args.formats

    # Load data
    if args.csv_dirs:
        runs = load_manual_dirs(args.csv_dirs)
    else:
        # For panel plot, load all noise types (don't filter by algo in discover
        # since panel needs all runs for the env)
        if args.plot == "panel":
            runs = discover_runs(env=args.env, root=args.test_root)
        else:
            runs = discover_runs(env=args.env, algo=args.algo,
                                 root=args.test_root)

    if not runs:
        print("No test runs found. Run stdrl_testing.py first, or use --csv-dirs.")
        return

    env = args.env or runs[0]["env"]

    # Filter: when plotting a specific noise type, exclude runs where OTHER
    # noise channels are non-zero. E.g., --noise-type action should exclude
    # obs-noise-only runs (noise_obs>0, noise_action=0) that would pollute
    # the noise=0 baseline group.
    if args.plot != "panel":
        noise_keys = {"obs": "noise_obs", "action": "noise_action", "env": "noise_env"}
        other_keys = [v for k, v in noise_keys.items() if k != args.noise_type]
        before = len(runs)
        runs = [r for r in runs if all(r[ok] == 0 for ok in other_keys)]
        if len(runs) < before:
            print(f"Filtered {before - len(runs)} runs with non-zero noise "
                  f"on other channels (kept {len(runs)})")

    # Print summary table
    df = build_summary_table(runs, args.noise_type)
    print(f"\n{'='*60}")
    print(df.to_string(index=False))
    print(f"{'='*60}\n")

    # Generate plots
    if args.plot in ("bar", "all"):
        print("[Plot] Grouped bar chart...")
        plot_grouped_bar(runs, args.noise_type, env, args.algo)

    if args.plot in ("line", "all"):
        print("[Plot] Degradation line...")
        plot_degradation_line(runs, args.noise_type, env, normalize=False)

    if args.plot in ("norm", "all"):
        print("[Plot] Normalized degradation...")
        plot_degradation_line(runs, args.noise_type, env, normalize=True)

    if args.plot in ("heatmap", "all"):
        print("[Plot] Heatmap...")
        plot_heatmap(runs, args.noise_type, env, args.algo)

    if args.plot in ("violin", "all"):
        print("[Plot] Violin...")
        plot_violin(runs, args.noise_type, env, args.algo)

    if args.plot in ("breakdown", "all"):
        print("[Plot] Metric breakdown...")
        plot_metric_breakdown(runs, args.noise_type, env, args.algo)

    if args.plot in ("panel", "all"):
        print("[Plot] Noise panel...")
        # Panel needs all runs for the env, reload if needed
        if args.plot == "all" and args.algo:
            panel_runs = discover_runs(env=env, root=args.test_root)
        else:
            panel_runs = runs
        plot_noise_panel(panel_runs, env, args.algo)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
