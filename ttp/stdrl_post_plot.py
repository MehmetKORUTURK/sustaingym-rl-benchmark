"""
Post-Training Evaluation Plotter
---------------------------------
Auto-discovers stdrl_testing.py outputs from logs_std_test/ and generates
publication-quality figures for noise robustness analysis.

Produces three plot types:
  1. Grouped bar chart with error bars (main result figure)
  2. Degradation line plot (noise sensitivity)
  3. Heatmap (train-noise x test-noise robustness matrix)

Usage:
    # Bar chart: one algo, one env, one noise type
    python ttp/stdrl_post_plot.py --env evcharging --algo SAC --noise-type obs --plot bar

    # Degradation line: all algos compared on one env
    python ttp/stdrl_post_plot.py --env building --noise-type obs --plot line

    # Heatmap: train-noise x test-noise matrix
    python ttp/stdrl_post_plot.py --env cogen --algo PPO --noise-type obs --plot heatmap

    # All plots at once
    python ttp/stdrl_post_plot.py --env evcharging --algo SAC --noise-type obs --plot all

    # Manual CSV paths (override auto-discovery)
    python ttp/stdrl_post_plot.py --csv-dirs path/to/dir1 path/to/dir2 --plot bar
"""

import os
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Publication-quality matplotlib settings
# ============================================================================
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

OUTPUT_DIR = "./graphs/C_POST/"
TEST_LOG_ROOT = "./logs_std_test"

ENV_TITLES = {
    "evcharging": "EVCharging-v0",
    "building": "Building-v0",
    "cogen": "Cogen-v0",
}

NOISE_LABELS = {
    "obs": "Observation Noise (σ)",
    "action": "Action Noise (σ)",
    "env": "Environment Noise (σ)",
}


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
        # subdir = logs_std_test/evcharging_SAC/
        for run_dir in sorted(subdir.iterdir()):
            if not run_dir.is_dir():
                continue
            result = load_test_run(str(run_dir))
            if result is None:
                continue
            # Filter
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
    """Extract the relevant noise value from a run."""
    key = {"obs": "noise_obs", "action": "noise_action", "env": "noise_env"}
    return run[key[noise_type]]


def build_summary_table(runs: List[Dict], noise_type: str) -> pd.DataFrame:
    """Build a summary DataFrame: algo x noise_level -> mean, ci95, etc."""
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


# ============================================================================
# Plot 1: Grouped Bar Chart with Error Bars
# ============================================================================

def plot_grouped_bar(runs: List[Dict], noise_type: str, env: str,
                     algo: str = None, save: bool = True):
    """Grouped bar chart: noise levels on x-axis, algorithms as groups."""
    df = build_summary_table(runs, noise_type)
    if df.empty:
        print("No data for grouped bar chart")
        return

    algos = sorted(df["algo"].unique())
    noise_levels = sorted(df["noise"].unique())

    fig, ax = plt.subplots(figsize=(max(8, len(noise_levels) * 1.2), 5))
    colors = sns.color_palette("tab10", n_colors=len(algos))
    hatches = ['', '//', '..', 'xx', '\\\\', 'oo']

    n_algos = len(algos)
    bar_width = 0.8 / n_algos
    x = np.arange(len(noise_levels))

    for i, alg in enumerate(algos):
        alg_df = df[df["algo"] == alg]
        means = []
        cis = []
        for nl in noise_levels:
            row = alg_df[alg_df["noise"] == nl]
            if len(row) > 0:
                means.append(row["mean_reward"].values[0])
                cis.append(row["ci95"].values[0])
            else:
                means.append(0)
                cis.append(0)

        offset = (i - n_algos / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width * 0.9,
                      yerr=cis, capsize=3,
                      label=alg, color=colors[i],
                      hatch=hatches[i % len(hatches)],
                      edgecolor='black', linewidth=0.5,
                      error_kw={'linewidth': 1, 'capthick': 1})

        # Value labels on bars
        for bar, m, ci in zip(bars, means, cis):
            if m != 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci + 0.02,
                        f'{m:.2f}', ha='center', va='bottom', fontsize=7, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([f'σ={nl}' for nl in noise_levels])
    ax.set_xlabel(NOISE_LABELS.get(noise_type, "Noise Level"), fontweight='bold')
    ax.set_ylabel("Mean Episode Reward", fontweight='bold')
    ax.set_title(f"{ENV_TITLES.get(env, env)} — {NOISE_LABELS.get(noise_type, '').split('(')[0].strip()}",
                 fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save:
        _save("bar", env, noise_type, algo)


# ============================================================================
# Plot 2: Degradation Line Plot
# ============================================================================

def plot_degradation_line(runs: List[Dict], noise_type: str, env: str,
                          normalize: bool = False, save: bool = True):
    """Line plot: x=noise level, y=mean reward (optionally normalized to baseline)."""
    df = build_summary_table(runs, noise_type)
    if df.empty:
        print("No data for degradation line plot")
        return

    algos = sorted(df["algo"].unique())
    colors = sns.color_palette("tab10", n_colors=len(algos))
    markers = ['o', 's', '^', 'D', 'v', 'P']
    line_styles = ['-', '--', '-.', ':']

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, alg in enumerate(algos):
        alg_df = df[df["algo"] == alg].sort_values("noise")
        x = alg_df["noise"].values
        y = alg_df["mean_reward"].values
        ci = alg_df["ci95"].values

        if normalize and len(y) > 0:
            baseline = y[0] if y[0] != 0 else 1.0
            y = y / abs(baseline) * 100
            ci = ci / abs(baseline) * 100

        ax.plot(x, y, label=alg, color=colors[i],
                marker=markers[i % len(markers)], markersize=6,
                linestyle=line_styles[i % len(line_styles)],
                linewidth=2, markeredgewidth=0.8)
        ax.fill_between(x, y - ci, y + ci, alpha=0.12, color=colors[i])

    ax.set_xlabel(NOISE_LABELS.get(noise_type, "Noise Level"), fontweight='bold')
    ylabel = "Performance (% of baseline)" if normalize else "Mean Episode Reward"
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(f"{ENV_TITLES.get(env, env)} — Noise Sensitivity",
                 fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save:
        suffix = "norm" if normalize else "line"
        _save(suffix, env, noise_type)


# ============================================================================
# Plot 3: Heatmap (Train Noise x Test Noise)
# ============================================================================

def plot_heatmap(runs: List[Dict], noise_type: str, env: str,
                 algo: str = None, save: bool = True):
    """Heatmap: rows=train noise, cols=test noise, cell=mean reward.

    Requires runs where train_noise and test_noise differ.
    The train noise is inferred from the model path (NOISE_X in the training dir).
    The test noise is from the test config.
    """
    # Try to infer train noise from model_path in config
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

        # Infer train noise from model_path
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

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 0.8),
                                     max(4, len(pivot.index) * 0.6)))

    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Mean Reward'})
    ax.set_xlabel("Test Noise (σ)", fontweight='bold')
    ax.set_ylabel("Train Noise (σ)", fontweight='bold')
    title_algo = f" ({algo})" if algo else ""
    ax.set_title(f"{ENV_TITLES.get(env, env)}{title_algo} — Robustness Matrix",
                 fontweight='bold')
    plt.tight_layout()

    if save:
        _save("heatmap", env, noise_type, algo)


def _parse_train_noise(model_path: str, noise_type: str) -> float:
    """Extract train noise from model path like .../NOISE_0.1_ACT_0.0_ENV_0.0/..."""
    import re
    key_map = {"obs": r"NOISE_([\d.]+)", "action": r"ACT_([\d.]+)", "env": r"ENV_([\d.]+)"}
    pattern = key_map.get(noise_type, r"NOISE_([\d.]+)")
    match = re.search(pattern, model_path)
    if match:
        return float(match.group(1))
    # Fallback: try DS_ pattern
    if noise_type == "obs":
        match = re.search(r"DS_([\d.]+)", model_path)
        if match:
            return float(match.group(1))
    return 0.0


# ============================================================================
# Save Helper
# ============================================================================

def _save(plot_type: str, env: str, noise_type: str, algo: str = None):
    """Save current figure to OUTPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parts = [timestamp, env, plot_type, noise_type]
    if algo:
        parts.insert(2, algo)
    filename = "_".join(parts)

    png_path = os.path.join(OUTPUT_DIR, filename + ".png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {png_path}")

    pdf_path = os.path.join(OUTPUT_DIR, filename + ".pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved: {pdf_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Post-Training Evaluation Plotter')
    parser.add_argument('--env', type=str, default=None,
                        choices=['evcharging', 'building', 'cogen'],
                        help='Filter by environment')
    parser.add_argument('--algo', type=str, default=None,
                        choices=['PPO', 'SAC', 'TD3'],
                        help='Filter by algorithm (required for heatmap)')
    parser.add_argument('--noise-type', type=str, default='obs',
                        choices=['obs', 'action', 'env'],
                        help='Which noise axis to plot')
    parser.add_argument('--plot', type=str, default='all',
                        choices=['bar', 'line', 'norm', 'heatmap', 'all'],
                        help='Plot type to generate')
    parser.add_argument('--csv-dirs', nargs='+', default=None,
                        help='Manual test run directories (overrides auto-discovery)')
    parser.add_argument('--test-root', type=str, default=TEST_LOG_ROOT,
                        help='Root directory for test logs')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not call plt.show()')

    args = parser.parse_args()

    # Load data
    if args.csv_dirs:
        runs = load_manual_dirs(args.csv_dirs)
    else:
        runs = discover_runs(env=args.env, algo=args.algo, root=args.test_root)

    if not runs:
        print("No test runs found. Run stdrl_testing.py first, or use --csv-dirs.")
        return

    env = args.env or runs[0]["env"]

    # Print summary table
    df = build_summary_table(runs, args.noise_type)
    print(f"\n{'='*60}")
    print(df.to_string(index=False))
    print(f"{'='*60}\n")

    # Generate plots
    if args.plot in ('bar', 'all'):
        plot_grouped_bar(runs, args.noise_type, env, args.algo)

    if args.plot in ('line', 'all'):
        plot_degradation_line(runs, args.noise_type, env, normalize=False)

    if args.plot in ('norm', 'all'):
        plot_degradation_line(runs, args.noise_type, env, normalize=True)

    if args.plot in ('heatmap', 'all'):
        plot_heatmap(runs, args.noise_type, env, args.algo)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
