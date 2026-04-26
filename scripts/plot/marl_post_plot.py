"""
MARL Post-Evaluation Plotter
-----------------------------
Auto-discovers marl_testing.py outputs from logs_marl_test/ and generates
publication-quality figures for multi-agent RL analysis.

Produces five plot types:
  1. bar       — Grouped bar chart: algo × policy mode per environment
  2. violin    — Reward distribution violin per algo/mode
  3. compare   — SA vs MARL grouped bar (using stdrl test logs as SA baseline)
  4. summary   — 1×3 panel: all envs side-by-side (best mode per algo)
  5. table     — Print LaTeX-ready summary table (mean ± std, CI95)

Usage:
    python scripts/plot/marl_post_plot.py --env building --plot bar
    python scripts/plot/marl_post_plot.py --env cogen --plot violin
    python scripts/plot/marl_post_plot.py --plot compare
    python scripts/plot/marl_post_plot.py --plot summary
    python scripts/plot/marl_post_plot.py --plot table
    python scripts/plot/marl_post_plot.py --plot all
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (
    ENV_TITLES,
    apply_style,
    get_color,
    get_marker,
    get_linestyle,
    style_axis,
)

apply_style()

# ============================================================================
# Constants
# ============================================================================
OUTPUT_DIR = "./graphs/C_MARL_POST"
MARL_TEST_ROOT = "./logs_marl_test"
STDRL_TEST_ROOT = "./logs_std_test"

ALGO_ORDER = ["PPO", "SAC", "APPO", "IMPALA"]
ENV_ORDER = ["evcharging", "building", "cogen"]

MODE_LABELS = {"independent": "Independent", "shared": "Shared"}
MODE_COLORS = {"independent": "#4878A8", "shared": "#E07B54"}
MODE_HATCHES = {"independent": "", "shared": "///"}
SA_COLOR = "#7CAE7A"  # sage green for single-agent

SAVE_FORMATS = ["png"]

# ============================================================================
# Helpers
# ============================================================================

def _save(filename: str, subdir: str = ""):
    """Save figure to OUTPUT_DIR/subdir/filename.{ext}."""
    path_dir = os.path.join(OUTPUT_DIR, subdir) if subdir else OUTPUT_DIR
    os.makedirs(path_dir, exist_ok=True)
    for ext in SAVE_FORMATS:
        path = os.path.join(path_dir, f"{filename}.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")


def _nice_label(algo: str, mode: str) -> str:
    short = "Ind" if mode == "independent" else "Sh"
    return f"{algo} ({short})"


# ============================================================================
# Data Loading
# ============================================================================

def load_marl_run(run_dir: str) -> Optional[Dict]:
    """Load a single MARL test run: config + episode CSV."""
    config_path = os.path.join(run_dir, "test_config.json")
    csv_path = os.path.join(run_dir, "episode_results.csv")

    if not os.path.exists(config_path) or not os.path.exists(csv_path):
        return None

    with open(config_path) as f:
        config = json.load(f)

    df = pd.read_csv(csv_path)
    if "total_reward" not in df.columns or df.empty:
        return None

    rewards = df["total_reward"].values
    n = len(rewards)

    return {
        "env": config.get("env", ""),
        "algo": config.get("algo", ""),
        "shared": config.get("shared_policy", False),
        "mode": "shared" if config.get("shared_policy", False) else "independent",
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "ci95": float(1.96 * np.std(rewards) / np.sqrt(n)),
        "median_reward": float(np.median(rewards)),
        "q25": float(np.percentile(rewards, 25)),
        "q75": float(np.percentile(rewards, 75)),
        "n_episodes": n,
        "run_dir": run_dir,
        "df": df,
    }


def discover_marl_runs(
    env: str = None, root: str = MARL_TEST_ROOT
) -> List[Dict]:
    """Auto-discover all MARL test runs from logs_marl_test/."""
    runs = []
    base = Path(root)
    if not base.exists():
        print(f"Warning: {root} not found")
        return runs

    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        for run_dir in sorted(subdir.iterdir()):
            if not run_dir.is_dir():
                continue
            result = load_marl_run(str(run_dir))
            if result is None:
                continue
            if env and result["env"] != env:
                continue
            runs.append(result)

    print(f"Discovered {len(runs)} MARL test runs" +
          (f" for env={env}" if env else ""))
    return runs


def discover_stdrl_baselines(root: str = STDRL_TEST_ROOT) -> Dict[str, Dict[str, Dict]]:
    """Load single-agent stdrl test results (noise=0 only) for SA vs MARL comparison.

    Returns: {env: {algo: {mean_reward, std_reward, ci95, n_episodes}}}
    """
    results: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    base = Path(root)
    if not base.exists():
        return dict(results)

    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        for run_dir in sorted(subdir.iterdir()):
            if not run_dir.is_dir():
                continue
            config_path = run_dir / "test_config.json"
            csv_path = run_dir / "episode_results.csv"
            if not config_path.exists() or not csv_path.exists():
                continue

            with open(config_path) as f:
                cfg = json.load(f)

            # Only noise=0 runs
            if cfg.get("noise", 0) != 0 or cfg.get("noise_action", 0) != 0 or cfg.get("noise_env", 0) != 0:
                continue

            env = cfg.get("env", "")
            algo = cfg.get("algo", "")
            if not env or not algo:
                continue

            df = pd.read_csv(csv_path)
            if "total_reward" not in df.columns or df.empty:
                continue

            rewards = df["total_reward"].values
            n = len(rewards)
            entry = {
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "ci95": float(1.96 * np.std(rewards) / np.sqrt(n)),
                "n_episodes": n,
            }

            # Keep best result if multiple noise=0 runs
            if algo not in results[env] or entry["mean_reward"] > results[env][algo]["mean_reward"]:
                results[env][algo] = entry

    return dict(results)


def group_runs(runs: List[Dict]) -> Dict[str, Dict[str, Dict[str, Dict]]]:
    """Group runs into {env: {algo: {mode: run_dict}}}."""
    grouped: Dict[str, Dict[str, Dict[str, Dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in runs:
        grouped[r["env"]][r["algo"]][r["mode"]] = r
    return dict(grouped)


# ============================================================================
# Plot 1: Grouped Bar Chart (per env)
# ============================================================================

def plot_bar(runs: List[Dict], env: str, save: bool = True):
    """Grouped bar: each algo has up to 2 bars (independent, shared)."""
    env_runs = [r for r in runs if r["env"] == env]
    if not env_runs:
        print(f"No data for {env} bar chart")
        return

    grouped = group_runs(env_runs)
    if env not in grouped:
        return

    algos = [a for a in ALGO_ORDER if a in grouped[env]]
    modes = ["independent", "shared"]
    # Filter to modes that exist
    avail_modes = set()
    for a in algos:
        avail_modes.update(grouped[env][a].keys())
    modes = [m for m in modes if m in avail_modes]

    n_algos = len(algos)
    n_modes = len(modes)
    bar_width = 0.7 / max(n_modes, 1)
    x = np.arange(n_algos)

    fig, ax = plt.subplots(figsize=(max(6, n_algos * 1.8 + 1), 5))

    for j, mode in enumerate(modes):
        means, cis = [], []
        for algo in algos:
            r = grouped[env].get(algo, {}).get(mode)
            if r:
                means.append(r["mean_reward"])
                cis.append(r["ci95"])
            else:
                means.append(np.nan)
                cis.append(0)

        offset = (j - n_modes / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, means, bar_width * 0.88,
            yerr=cis, capsize=3,
            label=MODE_LABELS[mode], color=MODE_COLORS[mode],
            hatch=MODE_HATCHES[mode], alpha=0.85,
            edgecolor="white", linewidth=0.6,
            error_kw={"linewidth": 1.0, "capthick": 0.8, "color": "0.3"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(algos)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title(f"{ENV_TITLES.get(env, env)} — MARL Evaluation (100 episodes)")
    ax.legend(loc="best", frameon=True)
    style_axis(ax, grid_y_only=True)
    fig.tight_layout()

    if save:
        _save(f"marl_bar_{env}", env)


# ============================================================================
# Plot 2: Violin Plot (per env)
# ============================================================================

def plot_violin(runs: List[Dict], env: str, save: bool = True):
    """Violin plot: reward distribution per algo × mode."""
    env_runs = [r for r in runs if r["env"] == env]
    if not env_runs:
        print(f"No data for {env} violin")
        return

    # Build per-episode DataFrame
    rows = []
    for r in env_runs:
        label = _nice_label(r["algo"], r["mode"])
        for _, row in r["df"].iterrows():
            rows.append({
                "algo": r["algo"],
                "mode": r["mode"],
                "label": label,
                "reward": row["total_reward"],
            })

    vdf = pd.DataFrame(rows)

    # Order: algo order × mode order
    label_order = []
    for algo in ALGO_ORDER:
        for mode in ["independent", "shared"]:
            lbl = _nice_label(algo, mode)
            if lbl in vdf["label"].values:
                label_order.append(lbl)

    palette = {}
    for lbl in label_order:
        mode = "shared" if "(Sh)" in lbl else "independent"
        palette[lbl] = MODE_COLORS[mode]

    fig, ax = plt.subplots(figsize=(max(8, len(label_order) * 1.0 + 1), 5))

    sns.violinplot(
        data=vdf, x="label", y="reward", order=label_order,
        palette=palette, inner="quartile", linewidth=0.8,
        saturation=0.8, cut=0, ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Episode Reward")
    ax.set_title(f"{ENV_TITLES.get(env, env)} — MARL Reward Distribution")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    style_axis(ax, grid_y_only=True)
    fig.tight_layout()

    if save:
        _save(f"marl_violin_{env}", env)


# ============================================================================
# Plot 3: SA vs MARL Comparison (1×3 panel)
# ============================================================================

def plot_sa_vs_marl(
    runs: List[Dict],
    stdrl: Dict[str, Dict[str, Dict]],
    save: bool = True,
):
    """1×3 panel: SA vs MARL-best per algorithm, across all envs."""
    from matplotlib.patches import Patch

    grouped = group_runs(runs)
    envs = [e for e in ENV_ORDER if e in grouped]
    n_envs = len(envs)
    if n_envs == 0:
        print("No MARL data for SA vs MARL comparison")
        return

    fig, axes = plt.subplots(1, n_envs, figsize=(5.5 * n_envs, 5))
    if n_envs == 1:
        axes = [axes]

    BAR_WIDTH = 0.28

    for ax, env in zip(axes, envs):
        env_data = grouped[env]
        # All algos present (union of SA and MARL)
        all_algos = sorted(
            set(list(env_data.keys()) + list(stdrl.get(env, {}).keys())),
            key=lambda a: ALGO_ORDER.index(a) if a in ALGO_ORDER else 99,
        )

        x = np.arange(len(all_algos))
        sa_means, sa_cis = [], []
        ma_ind_means, ma_ind_cis = [], []
        ma_sh_means, ma_sh_cis = [], []

        for algo in all_algos:
            # SA baseline
            sa = stdrl.get(env, {}).get(algo)
            if sa:
                sa_means.append(sa["mean_reward"])
                sa_cis.append(sa["ci95"])
            else:
                sa_means.append(np.nan)
                sa_cis.append(0)

            # MARL independent
            ma_i = env_data.get(algo, {}).get("independent")
            if ma_i:
                ma_ind_means.append(ma_i["mean_reward"])
                ma_ind_cis.append(ma_i["ci95"])
            else:
                ma_ind_means.append(np.nan)
                ma_ind_cis.append(0)

            # MARL shared
            ma_s = env_data.get(algo, {}).get("shared")
            if ma_s:
                ma_sh_means.append(ma_s["mean_reward"])
                ma_sh_cis.append(ma_s["ci95"])
            else:
                ma_sh_means.append(np.nan)
                ma_sh_cis.append(0)

        bar_kw = dict(
            width=BAR_WIDTH * 0.88, capsize=2.5, alpha=0.85,
            edgecolor="white", linewidth=0.6,
            error_kw={"linewidth": 0.8, "capthick": 0.7, "color": "0.3"},
        )

        ax.bar(x - BAR_WIDTH, sa_means, yerr=sa_cis,
               color=SA_COLOR, label="Single-Agent", **bar_kw)
        ax.bar(x, ma_ind_means, yerr=ma_ind_cis,
               color=MODE_COLORS["independent"], label="MARL Indep.", **bar_kw)
        ax.bar(x + BAR_WIDTH, ma_sh_means, yerr=ma_sh_cis,
               color=MODE_COLORS["shared"], hatch="///",
               label="MARL Shared", **bar_kw)

        ax.set_xticks(x)
        ax.set_xticklabels(all_algos)
        ax.set_title(ENV_TITLES.get(env, env), fontweight="bold")
        ax.set_ylabel("Mean Episode Reward" if ax == axes[0] else "")
        style_axis(ax, grid_y_only=True)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               frameon=True, bbox_to_anchor=(0.5, 1.05), fontsize=10)

    fig.suptitle("Single-Agent vs Multi-Agent — 100-Episode Evaluation",
                 fontsize=14, fontweight="bold", y=1.10)
    fig.tight_layout()

    if save:
        _save("marl_sa_vs_marl_eval")


# ============================================================================
# Plot 4: Summary Panel (1×3, best mode per algo)
# ============================================================================

def plot_summary(runs: List[Dict], save: bool = True):
    """1×3 summary: bar chart per env, best policy mode per algo."""
    grouped = group_runs(runs)
    envs = [e for e in ENV_ORDER if e in grouped]
    n_envs = len(envs)
    if n_envs == 0:
        return

    fig, axes = plt.subplots(1, n_envs, figsize=(5 * n_envs, 4.5))
    if n_envs == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        env_data = grouped[env]
        algos = [a for a in ALGO_ORDER if a in env_data]

        means, cis, colors = [], [], []
        labels = []

        for algo in algos:
            modes = env_data[algo]
            # Pick best mode
            best_mode = max(modes, key=lambda m: modes[m]["mean_reward"])
            r = modes[best_mode]
            means.append(r["mean_reward"])
            cis.append(r["ci95"])
            colors.append(get_color(algo))
            mode_tag = "Sh" if best_mode == "shared" else "Ind"
            labels.append(f"{algo}\n({mode_tag})")

        x = np.arange(len(algos))
        ax.bar(x, means, yerr=cis, capsize=3, color=colors, alpha=0.85,
               edgecolor="white", linewidth=0.6,
               error_kw={"linewidth": 0.8, "capthick": 0.7, "color": "0.3"})

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(ENV_TITLES.get(env, env), fontweight="bold")
        ax.set_ylabel("Mean Episode Reward" if ax == axes[0] else "")
        style_axis(ax, grid_y_only=True)

    fig.suptitle("MARL Evaluation Summary (100 episodes, best policy mode)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()

    if save:
        _save("marl_summary_panel")


# ============================================================================
# Table: LaTeX-ready summary
# ============================================================================

def print_latex_table(runs: List[Dict]):
    """Print a LaTeX-ready summary table to stdout."""
    grouped = group_runs(runs)

    print("\n% MARL Evaluation Summary Table (auto-generated by marl_post_plot.py)")
    print("% Copy-paste into chapter5.tex")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{MARL evaluation results (100 episodes). Bold = best per environment.}")
    print(r"\label{tab:marl_eval}")
    print(r"\small")
    print(r"\begin{tabular}{llccc}")
    print(r"\toprule")
    print(r"\textbf{Environment} & \textbf{Algorithm} & \textbf{Mode} & \textbf{Mean $\pm$ Std} & \textbf{95\% CI} \\")
    print(r"\midrule")

    for env in ENV_ORDER:
        if env not in grouped:
            continue

        env_data = grouped[env]
        best_reward = -np.inf
        # Find best
        for algo in env_data:
            for mode in env_data[algo]:
                r = env_data[algo][mode]
                if r["mean_reward"] > best_reward:
                    best_reward = r["mean_reward"]

        first = True
        for algo in ALGO_ORDER:
            if algo not in env_data:
                continue
            for mode in ["independent", "shared"]:
                if mode not in env_data[algo]:
                    continue
                r = env_data[algo][mode]
                env_col = ENV_TITLES.get(env, env) if first else ""
                if first:
                    env_col = r"\multirow{" + str(sum(len(env_data[a]) for a in env_data)) + "}{*}{" + ENV_TITLES.get(env, env) + "}"
                    first = False
                else:
                    env_col = ""

                mode_short = "Ind" if mode == "independent" else "Sh"
                mean_str = f"{r['mean_reward']:.2f} $\\pm$ {r['std_reward']:.2f}"
                ci_lo = r['mean_reward'] - r['ci95']
                ci_hi = r['mean_reward'] + r['ci95']
                ci_str = f"[{ci_lo:.2f}, {ci_hi:.2f}]"

                if abs(r["mean_reward"] - best_reward) < 0.01:
                    mean_str = r"\textbf{" + mean_str + "}"

                print(f"    {env_col} & {algo} & {mode_short} & {mean_str} & {ci_str} \\\\")

        print(r"\midrule")

    # Remove last midrule, add bottomrule
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MARL Post-Evaluation Plotter (paper-quality)")
    parser.add_argument("--env", type=str, default=None,
                        choices=["evcharging", "building", "cogen"],
                        help="Filter by environment (required for bar/violin)")
    parser.add_argument("--plot", type=str, default="all",
                        choices=["bar", "violin", "compare", "summary", "table", "all"],
                        help="Plot type to generate")
    parser.add_argument("--test-root", type=str, default=MARL_TEST_ROOT,
                        help="Root directory for MARL test logs")
    parser.add_argument("--stdrl-root", type=str, default=STDRL_TEST_ROOT,
                        help="Root directory for stdrl test logs (SA baseline)")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not call plt.show()")
    parser.add_argument("--formats", nargs="+", default=["png"],
                        choices=["png", "pdf", "svg"],
                        help="Output formats (default: png)")

    args = parser.parse_args()

    global SAVE_FORMATS
    SAVE_FORMATS = args.formats

    # Load data
    runs = discover_marl_runs(env=args.env, root=args.test_root)
    if not runs:
        print("No MARL test runs found. Run marl_testing.py first.")
        return

    # Print quick summary
    print(f"\n{'='*60}")
    print(f"{'Env':<14} {'Algo':<8} {'Mode':<12} {'Mean':>10} {'Std':>8} {'CI95':>8} {'N':>4}")
    print(f"{'-'*60}")
    for r in sorted(runs, key=lambda x: (x["env"], x["algo"], x["mode"])):
        print(f"{r['env']:<14} {r['algo']:<8} {r['mode']:<12} "
              f"{r['mean_reward']:>10.2f} {r['std_reward']:>8.2f} "
              f"{r['ci95']:>8.2f} {r['n_episodes']:>4}")
    print(f"{'='*60}\n")

    # Generate plots
    if args.plot in ("bar", "all"):
        envs = [args.env] if args.env else ENV_ORDER
        for env in envs:
            env_runs = [r for r in runs if r["env"] == env]
            if env_runs:
                print(f"[Plot] Bar chart — {env}...")
                plot_bar(runs, env)

    if args.plot in ("violin", "all"):
        envs = [args.env] if args.env else ENV_ORDER
        for env in envs:
            env_runs = [r for r in runs if r["env"] == env]
            if env_runs:
                print(f"[Plot] Violin — {env}...")
                plot_violin(runs, env)

    if args.plot in ("compare", "all"):
        print("[Plot] SA vs MARL comparison...")
        stdrl = discover_stdrl_baselines(args.stdrl_root)
        if stdrl:
            plot_sa_vs_marl(runs, stdrl)
        else:
            print("  No stdrl baselines found — skipping SA vs MARL plot")

    if args.plot in ("summary", "all"):
        print("[Plot] Summary panel...")
        plot_summary(runs)

    if args.plot in ("table", "all"):
        print_latex_table(runs)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
