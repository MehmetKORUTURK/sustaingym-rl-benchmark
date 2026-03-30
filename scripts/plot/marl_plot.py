"""
MARL Training Curve Plotter — Auto-discovery version
-----------------------------------------------------
Publication-quality plots for multi-agent RL experiments on SustainGym.

All algorithms on the same figure are truncated to the MINIMUM common
iteration count for fair comparison.

Plot types:
  1. training   — Per-env 1×2 dual panel: Independent (left) vs Shared (right)
  2. compare    — Per-env single figure: best policy mode per algo overlaid
  3. bar        — 1×3 grouped bar chart: final reward across all environments
  4. wallclock  — Per-env reward vs wall-clock hours

Usage:
    python scripts/plot/marl_plot.py --env evcharging --plot training
    python scripts/plot/marl_plot.py --env building   --plot compare --auto_ylim
    python scripts/plot/marl_plot.py --plot bar
    python scripts/plot/marl_plot.py --plot all
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (
    ENV_TITLES,
    FILL_ALPHA,
    MARK_EVERY_FRAC,
    apply_style,
    get_color,
    get_linestyle,
    get_marker,
    style_axis,
)

apply_style()

# ============================================================================
# Constants
# ============================================================================
FILL_FACTOR = 0.5
OUTPUT_DIR = "./graphs/C_MARL"
LOGS_BASE = "./logs_marl_train"

ALGO_ORDER = ["PPO", "SAC", "APPO", "IMPALA"]

POLICY_LABELS = {"independent": "Independent Policy", "shared": "Shared Policy"}

# Bar chart colors
BAR_COLORS = {"independent": "#4C72B0", "shared": "#DD8452"}
BAR_HATCH = {"independent": "", "shared": "///"}

# Per-env EMA defaults (higher = smoother)
EMA_DEFAULTS = {
    "evcharging": 200,
    "building": 300,
    "cogen": 300,
}


# ============================================================================
# Auto-discovery
# ============================================================================
def discover_experiments(
    env_filter: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, dict]]]:
    """
    Scan logs_marl_train/ and return nested dict:
        {env: {algo: {policy_mode: {csv, config, label, n_iters, ...}}}}

    For each (env, algo, policy_mode) combo, selects the run with the
    highest tail reward among runs with ≥50% of max iterations.
    """
    base = Path(LOGS_BASE)
    if not base.exists():
        print(f"[ERROR] Logs directory not found: {base}")
        return {}

    raw: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for algo_dir in sorted(base.iterdir()):
        if not algo_dir.is_dir():
            continue
        dirname = algo_dir.name

        match = re.match(r"^(evcharging|building|cogen)_(\w+?)(_SHARED)?$", dirname)
        if not match:
            continue

        env, algo, shared_flag = match.group(1), match.group(2), match.group(3)
        if env_filter and env != env_filter:
            continue

        policy_mode = "shared" if shared_flag else "independent"

        for run_dir in sorted(algo_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            csv_path = run_dir / "metrics.csv"
            cfg_path = run_dir / "config.json"
            if not csv_path.exists():
                continue

            try:
                n_lines = sum(1 for _ in open(csv_path)) - 1
            except Exception:
                continue

            try:
                df_tmp = pd.read_csv(csv_path)
                tail_reward = df_tmp["mean_reward"].iloc[-min(50, len(df_tmp)):].mean()
            except Exception:
                tail_reward = -np.inf

            cfg = {}
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text())
                except Exception:
                    pass

            raw[env][algo][policy_mode].append({
                "csv": str(csv_path),
                "config": cfg,
                "n_iters": n_lines,
                "run_dir": str(run_dir),
                "tail_reward": tail_reward,
            })

    # Pick best run per combo
    results: Dict[str, Dict[str, Dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for env in raw:
        for algo in raw[env]:
            for pm in raw[env][algo]:
                runs = raw[env][algo][pm]
                max_iters = max(r["n_iters"] for r in runs)
                viable = [r for r in runs if r["n_iters"] >= max_iters * 0.5]
                if not viable:
                    viable = runs
                best = max(viable, key=lambda r: r["tail_reward"])
                label = algo if pm == "independent" else f"{algo} (sh)"
                best["label"] = label
                best["policy_mode"] = pm
                results[env][algo][pm] = best

    return dict(results)


def load_csv(path: str) -> Optional[pd.DataFrame]:
    """Load metrics CSV and return DataFrame or None."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  [WARN] Cannot read {path}: {e}")
        return None
    if "iteration" not in df.columns or "mean_reward" not in df.columns:
        print(f"  [WARN] Missing columns in {path}")
        return None
    if df.empty:
        return None
    return df


def ema_smooth(data: np.ndarray, span: int) -> Tuple[np.ndarray, np.ndarray]:
    """EMA-smoothed mean and std."""
    s = pd.Series(data)
    mean = s.ewm(span=span, adjust=True).mean().values
    std = s.ewm(span=span, adjust=True).std().fillna(0).values
    return mean, std


def compute_fair_cutoff(
    data: Dict[str, Dict[str, dict]], policy_mode: str,
    outlier_ratio: float = 0.3,
) -> Tuple[int, List[str]]:
    """
    Find fair iteration cutoff across all algos for a given policy mode.

    Runs shorter than outlier_ratio * median are excluded as outliers.
    Returns (cutoff_iters, list_of_excluded_algos).
    """
    counts = {}
    for algo in data:
        if policy_mode in data[algo]:
            counts[algo] = data[algo][policy_mode]["n_iters"]
    if not counts:
        return 0, []

    median_iters = sorted(counts.values())[len(counts) // 2]
    threshold = median_iters * outlier_ratio

    excluded = [a for a, n in counts.items() if n < threshold]
    kept = {a: n for a, n in counts.items() if n >= threshold}

    if not kept:
        # All excluded — fall back to global min
        return min(counts.values()), []

    return min(kept.values()), excluded


# ============================================================================
# Plot helpers
# ============================================================================
def _plot_lines(
    ax, data: Dict[str, Dict[str, dict]], policy_mode: str,
    w_size: int, skip: int,
):
    """Plot smoothed reward curves for all algos with given policy mode.
    Each algo is plotted to its full length — no truncation."""
    plotted = 0
    for algo in ALGO_ORDER:
        if algo not in data or policy_mode not in data[algo]:
            continue
        info = data[algo][policy_mode]
        df = load_csv(info["csv"])
        if df is None:
            continue

        if skip > 0:
            df = df.iloc[skip:].reset_index(drop=True)
        if df.empty:
            continue

        mean, std = ema_smooth(df["mean_reward"].values, w_size)
        x = df["iteration"].values
        me = max(1, int(len(x) * MARK_EVERY_FRAC))

        color = get_color(algo)
        marker = get_marker(algo)
        ls = get_linestyle(algo)

        ax.plot(
            x, mean, label=algo, color=color, linestyle=ls,
            linewidth=1.8, marker=marker, markersize=4,
            markevery=me, markeredgewidth=0.5, markeredgecolor="white",
        )
        ax.fill_between(
            x, mean - FILL_FACTOR * std, mean + FILL_FACTOR * std,
            alpha=FILL_ALPHA, color=color, linewidth=0,
        )
        plotted += 1
    return plotted


# ============================================================================
# Plot 1: Training Curves — Independent vs Shared (1×2 per env)
# ============================================================================
def plot_training_curves(
    env: str, data: Dict[str, Dict[str, dict]],
    w_size: int, skip: int, auto_ylim: bool,
):
    has_shared = any("shared" in data[a] for a in data)

    if has_shared:
        fig, (ax_ind, ax_sh) = plt.subplots(1, 2, figsize=(14, 5.5))
        panels = [("independent", ax_ind), ("shared", ax_sh)]
    else:
        fig, ax_ind = plt.subplots(figsize=(8, 5.5))
        panels = [("independent", ax_ind)]

    env_title = ENV_TITLES.get(env, env)
    fig.suptitle(f"{env_title} — MARL Training Curves", fontsize=15)

    for pm, ax in panels:
        n = _plot_lines(ax, data, pm, w_size, skip)

        ax.set_xlabel("Training Episode")
        ax.set_ylabel("Mean Episode Reward")
        ax.set_title(POLICY_LABELS[pm])
        if n > 0:
            ax.legend(loc="best", frameon=True)
        style_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, env, "training")


# ============================================================================
# Plot 2: Algorithm Comparison — best policy mode per algo, single panel
# ============================================================================
def plot_algo_compare(
    env: str, data: Dict[str, Dict[str, dict]],
    w_size: int, skip: int, auto_ylim: bool,
    zoom: Optional[Tuple[float, float, float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
):
    """Single figure: pick best policy mode per algo, plot each to full length."""
    best_per_algo: Dict[str, dict] = {}
    for algo in ALGO_ORDER:
        if algo not in data:
            continue
        candidates = []
        for pm in ["independent", "shared"]:
            if pm in data[algo]:
                candidates.append(data[algo][pm])
        if candidates:
            best = max(candidates, key=lambda r: r["tail_reward"])
            best_per_algo[algo] = best

    if not best_per_algo:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    env_title = ENV_TITLES.get(env, env)
    ax.set_title(f"{env_title} — MARL Algorithm Comparison")

    for algo in ALGO_ORDER:
        if algo not in best_per_algo:
            continue
        info = best_per_algo[algo]
        df = load_csv(info["csv"])
        if df is None:
            continue

        if skip > 0:
            df = df.iloc[skip:].reset_index(drop=True)
        if df.empty:
            continue

        mean, std = ema_smooth(df["mean_reward"].values, w_size)
        x = df["iteration"].values

        # Pad short series to xlim with last value (building only)
        if xlim and env == "building" and len(x) > 0 and x[-1] < xlim[1]:
            pad_x = np.arange(int(x[-1]) + 1, int(xlim[1]) + 1)
            x = np.concatenate([x, pad_x])
            mean = np.concatenate([mean, np.full(len(pad_x), mean[-1])])
            std = np.concatenate([std, np.full(len(pad_x), std[-1])])

        me = max(1, int(len(x) * MARK_EVERY_FRAC))

        pm_tag = " (sh)" if info["policy_mode"] == "shared" else ""
        color = get_color(algo)

        ax.plot(
            x, mean, label=f"{algo}{pm_tag}", color=color,
            linestyle=get_linestyle(algo), linewidth=1.8,
            marker=get_marker(algo), markersize=4,
            markevery=me, markeredgewidth=0.5, markeredgecolor="white",
        )
        ax.fill_between(
            x, mean - FILL_FACTOR * std, mean + FILL_FACTOR * std,
            alpha=FILL_ALPHA, color=color, linewidth=0,
        )

    if xlim:
        ax.set_xlim(xlim)

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Mean Episode Reward")
    if env == "building" and zoom:
        ax.legend(loc="center right", frameon=True,
                  bbox_to_anchor=(1.0, 0.28))
    else:
        ax.legend(loc="lower right", frameon=True)
    style_axis(ax)

    # Inset zoom (same style as stdrl_plot.py)
    if zoom:
        zx1, zx2, zy1, zy2 = zoom
        # Per-env inset position
        if env == "building":
            axins = ax.inset_axes([0.84, 0.42, 0.15, 0.20])
        else:
            axins = ax.inset_axes([0.82, 0.22, 0.15, 0.20])
        for line in ax.get_lines():
            axins.plot(line.get_xdata(), line.get_ydata(),
                       color=line.get_color(), linestyle=line.get_linestyle(),
                       linewidth=1.5)
        axins.set_xlim(zx1, zx2)
        axins.set_ylim(zy1, zy2)
        x_start_k = int(np.ceil(zx1 / 1000))
        x_end_k = int(np.floor(zx2 / 1000))
        if x_end_k >= x_start_k:
            x_range = np.arange(x_start_k, x_end_k + 1) * 1000
            axins.set_xticks(x_range)
            axins.set_xticklabels([f"{int(v)}" for v in x_range])
        axins.set_xlim(zx1, zx2)
        axins.tick_params(labelsize=7)
        axins.grid(True, alpha=0.3)
        for spine in axins.spines.values():
            spine.set_edgecolor('0.4')
            spine.set_linewidth(1.0)
        indicator = ax.indicate_inset_zoom(
            axins, edgecolor='#aa2222', linewidth=2.0, alpha=0.9)
        # Matplotlib 3.10+: returns InsetIndicator with .rectangle/.connectors
        # Matplotlib <3.10: returns (rect, connectors) tuple
        if hasattr(indicator, 'rectangle'):
            rect = indicator.rectangle
            connectors = indicator.connectors
        else:
            rect, connectors = indicator
        rect.set_facecolor('#ee9999')
        rect.set_alpha(0.35)
        for conn in connectors:
            conn.set_edgecolor('#aaaaaa')
            conn.set_linewidth(1.0)

    fig.tight_layout()
    _save(fig, env, "compare")


# ============================================================================
# Plot 3: Final Reward Bar Chart (1×3 across environments)
# ============================================================================
def plot_bar_final(
    all_data: Dict[str, Dict[str, Dict[str, dict]]], tail_n: int,
):
    envs = [e for e in ["evcharging", "building", "cogen"] if e in all_data]
    n_envs = len(envs)
    if n_envs == 0:
        return

    fig, axes = plt.subplots(1, n_envs, figsize=(5.5 * n_envs, 5))
    if n_envs == 1:
        axes = [axes]

    fig.suptitle("MARL — Final Converged Reward by Algorithm", fontsize=15)

    for ax, env in zip(axes, envs):
        data = all_data[env]
        env_title = ENV_TITLES.get(env, env)

        algos_present = [a for a in ALGO_ORDER if a in data]
        has_shared = any("shared" in data[a] for a in algos_present)
        policy_modes = ["independent", "shared"] if has_shared else ["independent"]

        x_pos = np.arange(len(algos_present))
        n_bars = len(policy_modes)
        bar_width = 0.35 if n_bars == 2 else 0.5
        offsets = [-bar_width / 2, bar_width / 2] if n_bars == 2 else [0]

        for j, pm in enumerate(policy_modes):
            means, stds = [], []
            for algo in algos_present:
                if pm in data[algo]:
                    df = load_csv(data[algo][pm]["csv"])
                    if df is not None:
                        effective_tail = min(tail_n, max(1, len(df) // 5))
                        tail = df["mean_reward"].iloc[-effective_tail:]
                        means.append(tail.mean())
                        stds.append(tail.std())
                    else:
                        means.append(np.nan)
                        stds.append(0)
                else:
                    means.append(np.nan)
                    stds.append(0)

            ax.bar(
                x_pos + offsets[j], means, bar_width,
                yerr=stds, capsize=3,
                color=BAR_COLORS[pm], hatch=BAR_HATCH[pm],
                edgecolor="black", linewidth=0.5,
                label=POLICY_LABELS[pm].replace(" Policy", ""),
                error_kw={"linewidth": 0.8},
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(algos_present)
        ax.set_ylabel("Mean Episode Reward")
        ax.set_title(env_title)
        ax.legend(loc="best", frameon=True)
        style_axis(ax, grid_y_only=True)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "all", "bar_final")


# ============================================================================
# Plot 4: Wall-Clock Efficiency
# ============================================================================
def plot_wallclock(
    env: str, data: Dict[str, Dict[str, dict]],
    w_size: int, skip: int, auto_ylim: bool,
):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    env_title = ENV_TITLES.get(env, env)
    ax.set_title(f"{env_title} — MARL Wall-Clock Efficiency")

    plot_data = []

    for algo in ALGO_ORDER:
        if algo not in data:
            continue
        # Pick best policy mode
        best_pm = None
        best_reward = -np.inf
        for pm in ["independent", "shared"]:
            if pm in data[algo] and data[algo][pm]["tail_reward"] > best_reward:
                best_reward = data[algo][pm]["tail_reward"]
                best_pm = pm
        if best_pm is None:
            continue

        info = data[algo][best_pm]
        df = load_csv(info["csv"])
        if df is None or "elapsed_seconds" not in df.columns:
            continue

        pm_tag = " (sh)" if best_pm == "shared" else ""
        plot_data.append((algo, pm_tag, df))

    if not plot_data:
        plt.close(fig)
        return

    for algo, pm_tag, df in plot_data:
        hours = df["elapsed_seconds"].values / 3600.0
        rewards = df["mean_reward"].values

        if len(rewards) == 0:
            continue

        mean, std = ema_smooth(rewards, w_size)
        me = max(1, int(len(hours) * MARK_EVERY_FRAC))
        color = get_color(algo)

        ax.plot(
            hours, mean, label=f"{algo}{pm_tag}", color=color,
            linestyle=get_linestyle(algo), linewidth=1.8,
            marker=get_marker(algo), markersize=4,
            markevery=me, markeredgewidth=0.5, markeredgecolor="white",
        )
        ax.fill_between(
            hours, mean - FILL_FACTOR * std, mean + FILL_FACTOR * std,
            alpha=FILL_ALPHA, color=color, linewidth=0,
        )

    ax.set_xlabel("Wall-Clock Time (hours)")
    ax.set_ylabel("Mean Episode Reward")
    ax.legend(loc="best", frameon=True)
    style_axis(ax)

    fig.tight_layout()
    _save(fig, env, "wallclock")


# ============================================================================
# Save helper
# ============================================================================
def _save(fig, env: str, tag: str):
    subdir = os.path.join(OUTPUT_DIR, env) if env != "all" else OUTPUT_DIR
    os.makedirs(subdir, exist_ok=True)

    png_path = os.path.join(subdir, f"marl_{env}_{tag}.png")
    fig.savefig(png_path, bbox_inches="tight")
    print(f"  [SAVED] {png_path}")
    plt.close(fig)


# ============================================================================
# Single-agent stdrl baseline discovery (noise=0 only)
# ============================================================================
STDRL_LOGS = "./logs_std_train"
STDRL_ALGOS = ["PPO", "SAC", "TD3"]

# Cogen MARL rewards are NOT /1e7 scaled in metrics.csv, while single-agent
# MyCogenEnv divides by 1e7.  Multiply SA rewards by 1e7 for comparison.
COGEN_SA_SCALE = 1e7


def discover_stdrl_baselines() -> Dict[str, Dict[str, dict]]:
    """
    Scan logs_std_train/ for noise=0 runs and return:
        {env: {algo: {reward_mean, reward_std}}}
    Picks the best noise-0 run per (env, algo).
    """
    base = Path(STDRL_LOGS)
    if not base.exists():
        return {}

    raw: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for algo_dir in sorted(base.iterdir()):
        if not algo_dir.is_dir():
            continue
        match = re.match(r"^(evcharging|building|cogen)_(PPO|SAC|TD3)$", algo_dir.name)
        if not match:
            continue
        env, algo = match.group(1), match.group(2)

        for run_dir in sorted(algo_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            name = run_dir.name
            # Accept only noise=0 runs
            is_zero = (
                ("DS_0.0_DA_0.0" in name and "ENV" not in name)
                or (re.search(r"NOISE_0(?:\.0)?_ACT_0(?:\.0)?(?:$|[^_])", name)
                    and "ENV" not in name)
            )
            if not is_zero:
                continue

            mon = run_dir / "monitor.csv"
            if not mon.exists():
                continue
            try:
                df = pd.read_csv(mon, skiprows=1)
                if "r" not in df.columns or df.empty:
                    continue
                n = len(df)
                tail_n = min(100, max(1, n // 5))
                tail = df["r"].iloc[-tail_n:]
                raw[env][algo].append({
                    "reward_mean": tail.mean(),
                    "reward_std": tail.std(),
                    "episodes": n,
                })
            except Exception:
                continue

    # Pick best run per (env, algo)
    results: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for env in raw:
        for algo in raw[env]:
            best = max(raw[env][algo], key=lambda r: r["reward_mean"])
            results[env][algo] = best
    return dict(results)


# ============================================================================
# Plot 5: MARL vs Single-Agent Bar Chart (1×3)
# ============================================================================
def plot_bar_vs_stdrl(
    all_data: Dict[str, Dict[str, Dict[str, dict]]],
    stdrl: Dict[str, Dict[str, dict]],
    tail_n: int,
):
    """Publication-quality grouped bar chart: SA vs MARL (ind & shared) per env.

    Each algorithm family is a group with up to 3 bars:
      SA (solid blue) | MARL-ind (terracotta solid) | MARL-sh (terracotta hatched)
    """
    from matplotlib.patches import Patch

    envs = [e for e in ["evcharging", "building", "cogen"] if e in all_data]
    n_envs = len(envs)
    if n_envs == 0:
        return

    fig, axes = plt.subplots(1, n_envs, figsize=(6.0 * n_envs, 5.0))
    if n_envs == 1:
        axes = [axes]

    # --- Professional muted palette (colorblind-safe) ---
    SA_COLOR    = "#4878A8"   # muted steel blue
    MARL_IND_COLOR = "#E07B54"   # warm terracotta (independent)
    MARL_SH_COLOR  = "#F2C078"   # warm gold (shared)

    BAR_WIDTH = 0.28
    BAR_GAP   = 0.04          # gap between bars within a group
    GROUP_GAP = 0.50          # gap between algorithm groups

    for ax_idx, (ax, env) in enumerate(zip(axes, envs)):
        data = all_data[env]
        env_title = ENV_TITLES.get(env, env)
        is_cogen = env == "cogen"

        # --- Collect single-agent entries ---
        sa_map: Dict[str, Tuple[float, float]] = {}
        if env in stdrl:
            for algo in STDRL_ALGOS:
                if algo not in stdrl[env]:
                    continue
                info = stdrl[env][algo]
                m, s = info["reward_mean"], info["reward_std"]
                if is_cogen:
                    m *= COGEN_SA_SCALE
                    s *= COGEN_SA_SCALE
                sa_map[algo] = (m, s)

        # --- Collect MARL entries: ALL policy modes ---
        # marl_all[algo][pm] = (mean, std)
        marl_all: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for algo in ALGO_ORDER:
            if algo not in data:
                continue
            for pm in ["independent", "shared"]:
                if pm not in data[algo]:
                    continue
                info = data[algo][pm]
                df = load_csv(info["csv"])
                if df is None:
                    continue
                effective_tail = min(tail_n, max(1, len(df) // 5))
                tail = df["mean_reward"].iloc[-effective_tail:]
                if algo not in marl_all:
                    marl_all[algo] = {}
                marl_all[algo][pm] = (tail.mean(), tail.std())

        # --- Determine all unique algos in display order ---
        all_algos_ordered = []
        for algo in STDRL_ALGOS:
            if algo in sa_map or algo in marl_all:
                all_algos_ordered.append(algo)
        for algo in ALGO_ORDER:
            if algo in marl_all and algo not in all_algos_ordered:
                all_algos_ordered.append(algo)

        # --- Place bars per group ---
        tick_positions, tick_labels = [], []
        bar_specs = []   # list of (x, mean, std, color, hatch)
        x_cursor = 0.0

        for algo in all_algos_ordered:
            has_sa   = algo in sa_map
            has_ind  = algo in marl_all and "independent" in marl_all[algo]
            has_sh   = algo in marl_all and "shared" in marl_all[algo]
            n_bars   = int(has_sa) + int(has_ind) + int(has_sh)
            if n_bars == 0:
                continue

            group_start = x_cursor

            if has_sa:
                bar_specs.append((x_cursor, sa_map[algo][0], sa_map[algo][1],
                                  SA_COLOR, ""))
                x_cursor += BAR_WIDTH + BAR_GAP
            if has_ind:
                bar_specs.append((x_cursor, marl_all[algo]["independent"][0],
                                  marl_all[algo]["independent"][1],
                                  MARL_IND_COLOR, ""))
                x_cursor += BAR_WIDTH + BAR_GAP
            if has_sh:
                bar_specs.append((x_cursor, marl_all[algo]["shared"][0],
                                  marl_all[algo]["shared"][1],
                                  MARL_SH_COLOR, "///"))
                x_cursor += BAR_WIDTH + BAR_GAP

            group_end = x_cursor - BAR_GAP
            center = (group_start + group_end) / 2
            tick_positions.append(center)
            tick_labels.append(algo)
            x_cursor += GROUP_GAP

        # --- Draw bars ---
        err_kw = dict(capsize=3, capthick=0.8, ecolor="#333333")
        for (bx, bm, bs, bc, bh) in bar_specs:
            ax.bar(bx, bm, BAR_WIDTH, yerr=bs, color=bc,
                   edgecolor="white", linewidth=0.6, hatch=bh,
                   error_kw=err_kw, zorder=3)

        # --- Axes styling ---
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=25, ha="right", fontsize=9.5)
        if ax_idx == 0:
            ax.set_ylabel("Mean Episode Reward", fontsize=11)
        ax.set_title(env_title, fontsize=13, fontweight="bold", pad=10)

        # Minimal spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

        # Subtle gridlines
        ax.yaxis.grid(True, alpha=0.20, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

    # Single shared legend below all subplots
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=SA_COLOR, edgecolor="white",
              label="Single-Agent"),
        Patch(facecolor=MARL_IND_COLOR, edgecolor="white",
              label="MARL Independent"),
        Patch(facecolor=MARL_SH_COLOR, edgecolor="white", hatch="///",
              label="MARL Shared"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, frameon=True, framealpha=0.9,
               edgecolor="#cccccc", fontsize=10,
               handlelength=1.8, handleheight=1.2,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Single-Agent vs Multi-Agent",
                  fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97], w_pad=3.0)
    _save(fig, "all", "bar_sa_vs_marl")


# ============================================================================
# Summary table
# ============================================================================
def print_summary(all_data: Dict[str, Dict[str, Dict[str, dict]]], tail_n: int):
    print(f"\n{'='*80}")
    print(f"  MARL Training Summary (last {tail_n} iters averaged, at fair cutoff)")
    print(f"{'='*80}")

    for env in ["evcharging", "building", "cogen"]:
        if env not in all_data:
            continue
        data = all_data[env]
        env_title = ENV_TITLES.get(env, env)
        print(f"\n  {env_title}:")

        for pm in ["independent", "shared"]:
            has_any = any(pm in data[a] for a in data)
            if not has_any:
                continue
            print(f"    {POLICY_LABELS[pm]}:")
            print(f"    {'Algorithm':<12} {'Iters':>7} {'Final Reward':>14}")
            print(f"    {'-'*35}")

            for algo in ALGO_ORDER:
                if algo not in data or pm not in data[algo]:
                    continue
                df = load_csv(data[algo][pm]["csv"])
                if df is None:
                    continue
                n = len(df)
                effective_tail = min(tail_n, max(1, n // 5))
                tail = df["mean_reward"].iloc[-effective_tail:]
                print(f"    {algo:<12} {n:>7} {tail.mean():>14.2f}")
            print()

    print(f"{'='*80}\n")


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="MARL Training Plotter (auto-discovery)")
    parser.add_argument("--env", type=str, default=None,
                        choices=["evcharging", "building", "cogen"],
                        help="Environment (default: all)")
    parser.add_argument("--plot", type=str, default="all",
                        choices=["training", "compare", "bar", "bar_vs",
                                 "wallclock", "all"],
                        help="Plot type to generate")
    parser.add_argument("--w_size", type=int, default=0,
                        help="EMA smoothing span (0 = auto per env)")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip first N rows from each CSV")
    parser.add_argument("--tail", type=int, default=100,
                        help="Rows to average for final reward (bar chart)")
    parser.add_argument("--auto_ylim", action="store_true",
                        help="Let matplotlib auto-scale Y axis")
    parser.add_argument("--xlim", nargs=2, type=float,
                        metavar=("X1", "X2"),
                        help="X-axis limits for compare plot")
    parser.add_argument("--zoom", nargs=4, type=float,
                        metavar=("X1", "X2", "Y1", "Y2"),
                        help="Inset zoom region for compare plot")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  MARL Plotter — Auto-discovery")
    print(f"{'='*60}")

    all_data = discover_experiments(args.env)

    if not all_data:
        print("[ERROR] No MARL experiments found.")
        return

    # Report discoveries
    for env in sorted(all_data.keys()):
        print(f"\n  [{env}]")
        for pm in ["independent", "shared"]:
            algos = [a for a in ALGO_ORDER if a in all_data[env] and pm in all_data[env][a]]
            if not algos:
                continue
            iters_str = ", ".join(f"{a}={all_data[env][a][pm]['n_iters']}" for a in algos)
            print(f"    {pm:>12}: {iters_str}")

    print_summary(all_data, args.tail)

    # Discover single-agent baselines for bar_vs plot
    stdrl = discover_stdrl_baselines()
    if stdrl:
        print("\n  [stdrl baselines]")
        for env in sorted(stdrl.keys()):
            algos_str = ", ".join(
                f"{a}={stdrl[env][a]['reward_mean']:.2f}"
                for a in STDRL_ALGOS if a in stdrl[env]
            )
            print(f"    {env}: {algos_str}")

    # Generate plots
    plots = (["training", "compare", "bar", "bar_vs", "wallclock"]
             if args.plot == "all" else [args.plot])
    envs = [args.env] if args.env else ["evcharging", "building", "cogen"]

    for plot_type in plots:
        if plot_type == "bar":
            print(f"\n[PLOT] Final reward bar chart")
            plot_bar_final(all_data, args.tail)
            continue

        if plot_type == "bar_vs":
            print(f"\n[PLOT] MARL vs Single-Agent bar chart")
            plot_bar_vs_stdrl(all_data, stdrl, args.tail)
            continue

        for env in envs:
            if env not in all_data:
                continue
            data = all_data[env]
            w = args.w_size if args.w_size > 0 else EMA_DEFAULTS.get(env, 200)
            print(f"\n[PLOT] {plot_type} — {env} (w_size={w})")

            if plot_type == "training":
                plot_training_curves(env, data, w, args.skip, args.auto_ylim)
            elif plot_type == "compare":
                plot_algo_compare(env, data, w, args.skip, args.auto_ylim,
                                  zoom=tuple(args.zoom) if args.zoom else None,
                                  xlim=tuple(args.xlim) if args.xlim else None)
            elif plot_type == "wallclock":
                plot_wallclock(env, data, w, args.skip, args.auto_ylim)

    print(f"\nDone! Plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
