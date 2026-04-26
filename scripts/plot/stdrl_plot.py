"""
Unified Standard RL Training Curve Plotter
-------------------------------------------
Merged from stdrl_plot_ev.py, stdrl_plot_bu.py, stdrl_plot_co.py

Usage:
    python scripts/plot/stdrl_plot.py --env evcharging --algo PPO --dt DS
    python scripts/plot/stdrl_plot.py --env building --algo SAC --dt DA --auto_ylim
    python scripts/plot/stdrl_plot.py --env cogen --algo PPO --dt DE
    python scripts/plot/stdrl_plot.py --env cogen --algo PPO --dt all   # 1x3 subplot
    python scripts/plot/stdrl_plot.py --env evcharging --compare --zoom 25000 29000 6 7

    # Baseline algorithm comparison (PPO vs SAC vs TD3, noise=0):
    python scripts/plot/stdrl_plot.py --env evcharging --compare
    python scripts/plot/stdrl_plot.py --env building --compare --auto_ylim
    python scripts/plot/stdrl_plot.py --env cogen --compare

    # Multi-seed mode (cross-seed mean +/- std):
    python scripts/plot/stdrl_plot.py --env evcharging --algo PPO --dt all --multiseed
    python scripts/plot/stdrl_plot.py --env building --algo SAC --dt DS --multiseed

    # Generate all 7 multiseed panel figures at once:
    python scripts/plot/stdrl_plot.py --multiseed-all
"""

import os
import re
import sys
import argparse
import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import List, Tuple, Optional, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from style import (COLORS_INDEXED, LINE_STYLES, MARKERS,
                   MARK_EVERY_FRAC, FILL_ALPHA, ENV_SHORT, style_axis,
                   ALGO_COLORS, ALGO_MARKERS, ALGO_LINESTYLES,
                   get_color, get_marker, get_linestyle)
# style.py auto-applies rcParams on import


# ============================================================================
# MULTI-SEED CONSTANTS
# ============================================================================

MULTISEED_LOGS_BASE = Path(__file__).resolve().parents[2] / "docs" / "thesis" / "logs" / "logs_std_train"
MULTISEED_FIGS_BASE = Path(__file__).resolve().parents[2] / "docs" / "thesis" / "figures" / "C_STDRL"

MULTISEED_LEVELS: Dict[str, Dict[str, List[float]]] = {
    "evcharging": {"PS": [0.0, 0.15, 0.30], "PA": [0.0, 0.15, 0.30], "PD": [0.0, 0.15, 0.30]},
    "building":   {"PS": [0.0, 0.05, 0.15], "PA": [0.0, 0.05, 0.15], "PD": [0.0, 0.05, 0.15]},
    "cogen":      {"PS": [0.0, 10.0, 20.0], "PA": [0.0, 0.10, 0.20], "PD": [0.0, 0.30, 0.50]},
}

# Which (env, algo) combos to generate in --multiseed-all
MULTISEED_ALL_COMBOS = [
    ("evcharging", "PPO"),
    ("evcharging", "SAC"),
    ("building", "PPO"),
    ("building", "SAC"),
    ("cogen", "PPO"),
    ("cogen", "SAC"),
    ("cogen", "TD3"),
]

# DT code -> (PS index, PA index, PD index) in the noise tuple
_DT_TO_NOISE = {
    "DS": lambda level: (level, 0.0, 0.0),
    "DA": lambda level: (0.0, level, 0.0),
    "DE": lambda level: (0.0, 0.0, level),
}

# Old format regex: DATE_DS_X_DA_Y
_RE_OLD = re.compile(
    r'^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_DS_([0-9.]+)_DA_([0-9.]+)$'
)
# New format regex: DATE_NOISE_X_ACT_Y[_ENV_Z]
_RE_NEW = re.compile(
    r'^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_NOISE_([0-9.]+)_ACT_([0-9.]+)(?:_ENV_([0-9.]+))?$'
)


def _parse_noise_from_dirname(dirname: str) -> Optional[Tuple[float, float, float]]:
    """Parse (PS, PA, PD) from a run directory name. Returns None if unparseable."""
    m = _RE_OLD.match(dirname)
    if m:
        return (float(m.group(1)), float(m.group(2)), 0.0)
    m = _RE_NEW.match(dirname)
    if m:
        ps = float(m.group(1))
        pa = float(m.group(2))
        pd = float(m.group(3)) if m.group(3) is not None else 0.0
        return (ps, pa, pd)
    return None


def match_run(dirname: str, ps: float, pa: float, pd: float) -> bool:
    """Check if a directory name matches the given (PS, PA, PD) noise config."""
    parsed = _parse_noise_from_dirname(dirname)
    if parsed is None:
        return False
    return (abs(parsed[0] - ps) < 1e-6 and
            abs(parsed[1] - pa) < 1e-6 and
            abs(parsed[2] - pd) < 1e-6)


def find_seed_runs(env: str, algo: str, ps: float, pa: float, pd: float) -> List[Path]:
    """Find all matching seed run monitor.csv paths for a given noise config."""
    base_dir = MULTISEED_LOGS_BASE / f"{env}_{algo}"
    if not base_dir.exists():
        print(f"  WARNING: Directory not found: {base_dir}")
        return []
    results = []
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        if match_run(entry.name, ps, pa, pd):
            csv_path = entry / "monitor.csv"
            if csv_path.exists():
                results.append(csv_path)
    return results


def compute_multiseed_curve(
    seed_paths: List[Path],
    window: int,
    max_steps: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load all seeds, EMA-smooth each, compute cross-seed mean and std.

    Returns (episodes, mean, std) or None if no valid seeds.
    """
    smoothed_list = []
    for p in seed_paths:
        rewards = load_rewards(str(p), max_steps)
        if rewards is None:
            continue
        mean_i, _ = calculate_running_stats(rewards, window)
        smoothed_list.append(mean_i)

    if not smoothed_list:
        return None

    # Align to shortest length
    min_len = min(len(s) for s in smoothed_list)
    aligned = np.array([s[:min_len] for s in smoothed_list])  # (n_seeds, min_len)

    episodes = np.arange(min_len)
    cross_mean = aligned.mean(axis=0)
    cross_std = aligned.std(axis=0)
    return episodes, cross_mean, cross_std


def _plot_multiseed_on_ax(
    ax,
    env: str,
    algo: str,
    dt: str,
    levels: List[float],
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    title: str = "",
    window: int = 4000,
    max_steps: int = 32000,
    skip_initial: int = 250,
    auto_xlim: bool = False,
    auto_ylim: bool = False,
    std_band: float = 0.5,
    total_steps: int = 9_000_000,
    show_ylabel: bool = True,
    legend_loc='lower right',
    zoom: Optional[Tuple[float, float, float, float]] = None,
):
    """Plot multi-seed learning curves (mean +/- std) on a given axes."""
    noise_fn = _DT_TO_NOISE[dt]

    for idx, level in enumerate(levels):
        ps, pa, pd = noise_fn(level)
        seed_paths = find_seed_runs(env, algo, ps, pa, pd)
        n_seeds = len(seed_paths)
        print(f"  {algo} {DT_SHORT[dt]}={level:.2f}: found {n_seeds} seed runs")

        if n_seeds == 0:
            continue

        result = compute_multiseed_curve(seed_paths, window, max_steps)
        if result is None:
            continue

        episodes, cross_mean, cross_std = result

        # Thin data for line style visibility (~1500 points max)
        step = max(1, len(episodes) // 1500)
        x = episodes[::step].copy()
        mean_t = cross_mean[::step]
        std_t = cross_std[::step]

        # Shift x so that skip_initial maps to 0
        x = x - skip_initial

        label = f"{algo}_{DT_SHORT[dt]}_{level:.2f} (n={n_seeds})"

        color = COLORS_INDEXED[idx % len(COLORS_INDEXED)]
        ls = LINE_STYLES[idx % len(LINE_STYLES)]
        marker = MARKERS[idx % len(MARKERS)]
        me = max(1, int(len(x) * MARK_EVERY_FRAC))

        ax.plot(x, mean_t, label=label, linewidth=1.8, color=color,
                linestyle=ls, marker=marker, markersize=5,
                markevery=me, markeredgewidth=0.6,
                markeredgecolor=color)
        ax.fill_between(x, mean_t - std_band * std_t, mean_t + std_band * std_t,
                        alpha=FILL_ALPHA, color=color, linewidth=0)

    # X-axis limits (shifted so skip_initial -> 0)
    if auto_xlim:
        pass
    elif xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0, max_steps - skip_initial)

    # Y-axis limits
    if auto_ylim:
        pass
    elif ylim:
        ax.set_ylim(ylim)

    # Ensure the last x value appears as a tick, then re-apply xlim
    x_end = max_steps - skip_initial
    xticks = [t for t in ax.get_xticks() if 0 <= t <= x_end]
    if x_end not in xticks:
        xticks.append(x_end)
    ax.set_xticks(xticks)
    ax.set_xlim(0, x_end)

    # Show total training steps (top-left corner)
    exponent = int(np.floor(np.log10(total_steps)))
    mantissa = total_steps / 10**exponent
    step_str = f"Total: ~{int(round(mantissa))}e{exponent} steps"
    ax.text(0.02, 0.97, step_str, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_title(title)
    ax.set_xlabel("Training Episode")
    if show_ylabel:
        ax.set_ylabel("Average Episode Reward")

    if isinstance(legend_loc, dict):
        ax.legend(frameon=True, **legend_loc)
    else:
        ax.legend(loc=legend_loc, frameon=True)
    style_axis(ax)

    # Inset zoom for perturbation plots
    if zoom:
        zx1, zx2, zy1, zy2 = zoom
        axins = ax.inset_axes([0.84, 0.22, 0.15, 0.20])
        for line in ax.get_lines():
            axins.plot(line.get_xdata(), line.get_ydata(),
                       color=line.get_color(), linestyle=line.get_linestyle(),
                       linewidth=1.5)
        axins.set_xlim(zx1, zx2)
        axins.set_ylim(zy1, zy2)
        x_start_k = int(np.ceil(zx1 / 1000))
        x_end_k = int(np.floor(zx2 / 1000))
        x_range = np.arange(x_start_k, x_end_k + 1) * 1000
        axins.set_xticks(x_range)
        axins.set_xticklabels([f"{int(v)}" for v in x_range])
        axins.set_xlim(zx1, zx2)
        axins.tick_params(labelsize=7)
        axins.grid(True, alpha=0.3)
        for spine in axins.spines.values():
            spine.set_edgecolor('0.4')
            spine.set_linewidth(1.0)
        rect, connectors = ax.indicate_inset_zoom(axins, edgecolor='#aa2222', linewidth=2.0, alpha=0.9)
        rect.set_facecolor('#ee9999')
        rect.set_alpha(0.35)
        for conn in connectors:
            conn.set_edgecolor('#aaaaaa')
            conn.set_linewidth(1.0)


def plot_multiseed_single(
    env: str,
    algo: str,
    dt: str,
    window: int = 4000,
    max_steps: int = 32000,
    skip_initial: int = 250,
    auto_xlim: bool = False,
    auto_ylim: bool = False,
    std_band: float = 0.5,
    total_steps: int = 9_000_000,
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    zoom: Optional[Tuple[float, float, float, float]] = None,
    legend_loc='lower right',
):
    """Plot multi-seed learning curves (single panel, single DT)."""
    dt_key = {"DS": "PS", "DA": "PA", "DE": "PD"}[dt]
    levels = MULTISEED_LEVELS.get(env, {}).get(dt_key, [])
    if not levels:
        print(f"  No multiseed levels for {env} / {dt_key}")
        return

    env_title = ENV_DEFAULTS[env]["title"]
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_multiseed_on_ax(
        ax, env, algo, dt, levels, ylim, xlim,
        title=f"{env_title} — {algo}: {DT_FULL[dt]}",
        window=window, max_steps=max_steps, skip_initial=skip_initial,
        auto_xlim=auto_xlim, auto_ylim=auto_ylim, std_band=std_band,
        total_steps=total_steps, legend_loc=legend_loc, zoom=zoom,
    )
    plt.tight_layout()
    return fig


def plot_multiseed_all_dt(
    env: str,
    algo: str,
    window: int = 4000,
    max_steps: int = 32000,
    skip_initial: int = 250,
    auto_xlim: bool = False,
    auto_ylim: bool = False,
    std_band: float = 0.5,
    total_steps: int = 9_000_000,
    ylim_override: Optional[Tuple[float, float]] = None,
    xlim_override: Optional[Tuple[float, float]] = None,
    zoom: Optional[Tuple[float, float, float, float]] = None,
):
    """Plot multi-seed DS, DA, DE side by side in 1x3 subplots."""
    env_title = ENV_DEFAULTS[env]["title"]
    env_config = ALL_CONFIGS.get(env, {})
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)

    for i, dt in enumerate(["DS", "DA", "DE"]):
        dt_key = {"DS": "PS", "DA": "PA", "DE": "PD"}[dt]
        levels = MULTISEED_LEVELS.get(env, {}).get(dt_key, [])

        # Use config ylim as default if available
        ylim = ylim_override
        if ylim is None and algo in env_config and dt in env_config[algo]:
            _, default_ylim = env_config[algo][dt]
            ylim = default_ylim

        loc = LEGEND_LOC.get((env, algo, dt), 'lower right')

        _plot_multiseed_on_ax(
            axes[i], env, algo, dt, levels, ylim, xlim_override,
            title=DT_SUBFIG[dt], window=window, max_steps=max_steps,
            skip_initial=skip_initial, auto_xlim=auto_xlim,
            auto_ylim=auto_ylim, std_band=std_band, total_steps=total_steps,
            show_ylabel=(i == 0), legend_loc=loc, zoom=zoom,
        )

    fig.suptitle(f"{env_title} — {algo} (Multi-Seed)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def save_multiseed_plot(env: str, algo: str, dt: str, save_pdf: bool = False):
    """Save multiseed plot to docs/thesis/figures/C_STDRL/{env_short}/{algo}/"""
    subdir = MULTISEED_FIGS_BASE / ENV_SHORT.get(env, env) / algo
    subdir.mkdir(parents=True, exist_ok=True)

    png_path = subdir / f"{dt}_multiseed.png"
    plt.savefig(str(png_path), dpi=300, bbox_inches='tight')
    print(f"\nPNG saved: {png_path}")

    if save_pdf:
        pdf_path = subdir / f"{dt}_multiseed.pdf"
        plt.savefig(str(pdf_path), bbox_inches='tight')
        print(f"PDF saved: {pdf_path}")


# ============================================================================
# PER-ENV DEFAULTS
# ============================================================================

ENV_DEFAULTS = {
    "evcharging": {"title": "EV Charging", "t_steps": 30250, "std_band": 0.03, "ylim": (5, 8), "total_steps": 9_000_000},
    "building":   {"title": "Building",   "t_steps": 20250, "std_band": 0.03, "ylim": None, "total_steps": 9_000_000},
    "cogen":      {"title": "Cogeneration",      "t_steps": 10250, "std_band": 0.03, "ylim": None, "total_steps": 3_000_000},
}

OUTPUT_DIR = "./graphs/C_STDRL/"

# Per (env, algo, dt) legend location overrides  — default is 'lower right'
LEGEND_LOC = {
    ("evcharging", "SAC", "DS"): {"loc": "center right", "bbox_to_anchor": (1.0, 0.3)},
}


# ============================================================================
# EVCHARGING CONFIG
# ============================================================================

EV_CONFIG = {
    "PPO": {
        "DS": ([
            (r"logs_std_train/evcharging_PPO/2026-02-05-20-06-35_DS_0.0_DA_0.0/monitor.csv", "PPO_0.0"),
            (r"logs_std_train/evcharging_PPO/2026-02-09-14-41-09_DS_0.15_DA_0.0/monitor.csv", "PPO_0.15"),
            # (r"logs_std_train/evcharging_PPO/2026-02-09-14-38-32_DS_0.1_DA_0.0/monitor.csv", "PPO_0.1"),
            (r"logs_std_train/evcharging_PPO/2026-02-09-14-49-37_DS_0.3_DA_0.0/monitor.csv", "PPO_0.3"),
            # (r"logs_std_train/evcharging_PPO/2026-02-10-13-00-16_DS_0.6_DA_0.0/monitor.csv", "PPO_0.6"),
            # (r"logs_std_train/evcharging_PPO/2026-02-09-13-32-12_DS_0.01_DA_0.0/monitor.csv", "PPO_0.01"),
            # (r"logs_std_train/evcharging_PPO/2026-02-09-14-31-24_DS_0.05_DA_0.0/monitor.csv", "PPO_0.05"),
            # (r"logs_std_train/evcharging_PPO/2026-02-09-14-49-08_DS_0.2_DA_0.0/monitor.csv", "PPO_0.2"),
            # (r"logs_std_train/evcharging_PPO/2026-02-10-13-00-36_DS_0.4_DA_0.0/monitor.csv", "PPO_0.4"),
            # (r"logs_std_train/evcharging_PPO/2026-02-10-13-00-36_DS_0.5_DA_0.0/monitor.csv", "PPO_0.5"),
        ], (4.5, 8)),
        "DA": ([
            (r"logs_std_train/evcharging_PPO/2026-02-05-20-06-35_DS_0.0_DA_0.0/monitor.csv", "PPO_0.0"),
            (r"logs_std_train/evcharging_PPO/2026-02-11-03-22-23_DS_0.0_DA_0.15/monitor.csv", "PPO_0.15"),
            # (r"logs_std_train/evcharging_PPO/2026-02-11-03-20-57_DS_0.0_DA_0.1/monitor.csv", "PPO_0.1"),
            (r"logs_std_train/evcharging_PPO/2026-02-11-08-02-01_DS_0.0_DA_0.3/monitor.csv", "PPO_0.3"),
            # (r"logs_std_train/evcharging_PPO/2026-02-16-10-40-02_NOISE_0.0_ACT_0.6/monitor.csv", "PPO_0.6"),
            # (r"logs_std_train/evcharging_PPO/2026-02-10-13-36-28_DS_0.0_DA_0.01/monitor.csv", "PPO_0.01"),
            # (r"logs_std_train/evcharging_PPO/2026-02-10-22-42-09_DS_0.0_DA_0.05/monitor.csv", "PPO_0.05"),
            # (r"logs_std_train/evcharging_PPO/2026-02-11-05-41-17_DS_0.0_DA_0.2/monitor.csv", "PPO_0.2"),
            # (r"logs_std_train/evcharging_PPO/2026-02-11-10-13-34_DS_0.0_DA_0.4/monitor.csv", "PPO_0.4"),
        ], (4.5, 8)),
        "DE": ([
            (r"logs_std_train/evcharging_PPO/2026-02-05-20-06-35_DS_0.0_DA_0.0/monitor.csv", "PPO_0.0"),
            # (r"logs_std_train/evcharging_PPO/2026-03-04-22-47-02_NOISE_0.0_ACT_0.0_ENV_0.05/monitor.csv", "PPO_0.05"),
            (r"logs_std_train/evcharging_PPO/2026-03-05-00-29-20_NOISE_0.0_ACT_0.0_ENV_0.15/monitor.csv", "PPO_0.15"),
            (r"logs_std_train/evcharging_PPO/2026-03-05-04-45-55_NOISE_0.0_ACT_0.0_ENV_0.3/monitor.csv", "PPO_0.3"),
            # (r"logs_std_train/evcharging_PPO/2026-03-04-22-32-05_NOISE_0.0_ACT_0.0_ENV_0.01/monitor.csv", "PPO_0.01"),
            # (r"logs_std_train/evcharging_PPO/2026-03-04-22-32-09_NOISE_0.0_ACT_0.0_ENV_0.02/monitor.csv", "PPO_0.02"),
            # (r"logs_std_train/evcharging_PPO/2026-03-04-23-38-39_NOISE_0.0_ACT_0.0_ENV_0.1/monitor.csv", "PPO_0.1"),
            # (r"logs_std_train/evcharging_PPO/2026-03-05-03-39-07_NOISE_0.0_ACT_0.0_ENV_0.2/monitor.csv", "PPO_0.2"),
        ], (4.5, 8)),
    },
    "SAC": {
        "DS": ([
            (r"logs_std_train/evcharging_SAC/2026-02-09-12-45-20_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
            (r"logs_std_train/evcharging_SAC/2026-02-12-15-47-04_NOISE_0.15_ACT_0.0/monitor.csv", "SAC_0.15"),
            # (r"logs_std_train/evcharging_SAC/2026-02-12-15-47-04_NOISE_0.1_ACT_0.0/monitor.csv", "SAC_0.1"),
            (r"logs_std_train/evcharging_SAC/2026-02-13-18-23-14_NOISE_0.3_ACT_0.0/monitor.csv", "SAC_0.3"),
            # (r"logs_std_train/evcharging_SAC/2026-02-15-06-03-15_NOISE_0.6_ACT_0.0/monitor.csv", "SAC_0.6"),
            # (r"logs_std_train/evcharging_SAC/2026-02-12-15-46-22_NOISE_0.01_ACT_0.0/monitor.csv", "SAC_0.01"),
            # (r"logs_std_train/evcharging_SAC/2026-02-12-15-46-22_NOISE_0.05_ACT_0.0/monitor.csv", "SAC_0.05"),
            # (r"logs_std_train/evcharging_SAC/2026-02-13-18-18-09_NOISE_0.2_ACT_0.0/monitor.csv", "SAC_0.2"),
            # (r"logs_std_train/evcharging_SAC/2026-02-13-18-25-51_NOISE_0.4_ACT_0.0/monitor.csv", "SAC_0.4"),
            # (r"logs_std_train/evcharging_SAC/2026-02-13-18-45-20_NOISE_0.5_ACT_0.0/monitor.csv", "SAC_0.5"),
        ], (2.7, 7)),
        "DA": ([
            (r"logs_std_train/evcharging_SAC/2026-02-09-12-45-20_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
            (r"logs_std_train/evcharging_SAC/2026-02-19-22-13-55_NOISE_0.0_ACT_0.15/monitor.csv", "SAC_0.15"),
            # (r"logs_std_train/evcharging_SAC/2026-02-19-22-32-04_NOISE_0.0_ACT_0.1/monitor.csv", "SAC_0.1"),
            (r"logs_std_train/evcharging_SAC/2026-02-18-18-10-06_NOISE_0.0_ACT_0.3/monitor.csv", "SAC_0.3"),
            # (r"logs_std_train/evcharging_SAC/2026-02-18-16-49-39_NOISE_0.0_ACT_0.6/monitor.csv", "SAC_0.6"),
            # (r"logs_std_train/evcharging_SAC/2026-02-15-06-18-46_NOISE_0.0_ACT_0.01/monitor.csv", "SAC_0.01"),
            # (r"logs_std_train/evcharging_SAC/2026-02-20-00-41-29_NOISE_0.0_ACT_0.05/monitor.csv", "SAC_0.05"),
            # (r"logs_std_train/evcharging_SAC/2026-02-19-19-02-28_NOISE_0.0_ACT_0.2/monitor.csv", "SAC_0.2"),
            # (r"logs_std_train/evcharging_SAC/2026-02-18-16-50-26_NOISE_0.0_ACT_0.4/monitor.csv", "SAC_0.4"),
            # (r"logs_std_train/evcharging_SAC/2026-02-18-16-50-26_NOISE_0.0_ACT_0.5/monitor.csv", "SAC_0.5"),
        ], (2.7, 7)),
        "DE": ([
            (r"logs_std_train/evcharging_SAC/2026-02-09-12-45-20_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
            (r"logs_std_train/evcharging_SAC/2026-03-04-17-55-08_NOISE_0.0_ACT_0.0_ENV_0.15/monitor.csv", "SAC_0.15"),
            # (r"logs_std_train/evcharging_SAC/2026-03-04-12-34-10_NOISE_0.0_ACT_0.0_ENV_0.05/monitor.csv", "SAC_0.05"),
            # (r"logs_std_train/evcharging_SAC/2026-03-04-12-35-06_NOISE_0.0_ACT_0.0_ENV_0.1/monitor.csv", "SAC_0.1"),
            (r"logs_std_train/evcharging_SAC/2026-03-04-22-31-19_NOISE_0.0_ACT_0.0_ENV_0.3/monitor.csv", "SAC_0.3"),
            # (r"logs_std_train/evcharging_SAC/2026-03-04-12-02-50_NOISE_0.0_ACT_0.0_ENV_0.01/monitor.csv", "SAC_0.01"),
            # (r"logs_std_train/evcharging_SAC/2026-03-04-12-02-50_NOISE_0.0_ACT_0.0_ENV_0.02/monitor.csv", "SAC_0.02"),
            # (r"logs_std_train/evcharging_SAC/2026-03-04-22-01-04_NOISE_0.0_ACT_0.0_ENV_0.2/monitor.csv", "SAC_0.2"),
        ], (2.7, 7)),
    },
    "TD3": {
        "DS": ([
            (r"logs_std_train/evcharging_TD3/2026-02-07-17-45-25_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),
            # TD3 has no obs noise runs for EVCharging
        ], None),
        "DA": ([
            (r"logs_std_train/evcharging_TD3/2026-02-07-17-45-25_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),
            # TD3 has no action noise runs for EVCharging
        ], None),
        "DE": ([
            (r"logs_std_train/evcharging_TD3/2026-02-07-17-45-25_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),
            # TD3 has no env noise runs for EVCharging
        ], None),
    },
}


# ============================================================================
# BUILDING CONFIG
# ============================================================================

BU_CONFIG = {
    "PPO": {
        "DS": ([
            (r"logs_std_train/building_PPO/2026-02-09-11-18-57_DS_0.0_DA_0.0/monitor.csv", "PPO_0.0"),
            (r"logs_std_train/building_PPO/2026-02-11-16-20-14_NOISE_0.05_ACT_0.0/monitor.csv", "PPO_0.05"),
            # (r"logs_std_train/building_PPO/2026-02-11-16-20-15_NOISE_0.2_ACT_0.0/monitor.csv", "PPO_0.2"),
            # (r"logs_std_train/building_PPO/2026-02-11-16-22-06_NOISE_0.4_ACT_0.0/monitor.csv", "PPO_0.4"),
            # (r"logs_std_train/building_PPO/2026-02-11-15-36-10_NOISE_0.01_ACT_0.0/monitor.csv", "PPO_0.01"),
            # (r"logs_std_train/building_PPO/2026-02-11-16-20-14_NOISE_0.03_ACT_0.0/monitor.csv", "PPO_0.03"),
            # (r"logs_std_train/building_PPO/2026-02-11-16-20-14_NOISE_0.1_ACT_0.0/monitor.csv", "PPO_0.1"),
            (r"logs_std_train/building_PPO/2026-02-11-16-20-15_NOISE_0.15_ACT_0.0/monitor.csv", "PPO_0.15"),
            # (r"logs_std_train/building_PPO/2026-02-11-16-20-15_NOISE_0.3_ACT_0.0/monitor.csv", "PPO_0.3"),
        ], None),
        "DA": ([
            (r"logs_std_train/building_PPO/2026-02-09-11-18-57_DS_0.0_DA_0.0/monitor.csv", "PPO_0.0"),
            (r"logs_std_train/building_PPO/2026-02-25-20-07-54_NOISE_0.0_ACT_0.05/monitor.csv", "PPO_0.05"),
            # (r"logs_std_train/building_PPO/2026-02-25-20-14-27_NOISE_0.0_ACT_0.2/monitor.csv", "PPO_0.2"),
            # (r"logs_std_train/building_PPO/2026-02-25-20-33-49_NOISE_0.0_ACT_0.4/monitor.csv", "PPO_0.4"),
            # (r"logs_std_train/building_PPO/2026-02-25-18-42-09_NOISE_0.0_ACT_0.01/monitor.csv", "PPO_0.01"),
            # (r"logs_std_train/building_PPO/2026-02-25-19-36-47_NOISE_0.0_ACT_0.03/monitor.csv", "PPO_0.03"),
            # (r"logs_std_train/building_PPO/2026-02-25-20-07-54_NOISE_0.0_ACT_0.1/monitor.csv", "PPO_0.1"),
            (r"logs_std_train/building_PPO/2026-02-25-20-14-27_NOISE_0.0_ACT_0.15/monitor.csv", "PPO_0.15"),
            # (r"logs_std_train/building_PPO/2026-02-25-20-31-23_NOISE_0.0_ACT_0.3/monitor.csv", "PPO_0.3"),
        ], None),
        "DE": ([
            (r"logs_std_train/building_PPO/2026-02-09-11-18-57_DS_0.0_DA_0.0/monitor.csv", "PPO_0.0"),
            (r"logs_std_train\building_PPO\2026-03-23-15-29-11_NOISE_0.0_ACT_0.0_ENV_0.05\monitor.csv", "PPO_0.05"),
            (r"logs_std_train\building_PPO\2026-03-23-15-38-44_NOISE_0.0_ACT_0.0_ENV_0.15\monitor.csv", "PPO_0.15"),
            # (r"logs_std_train/building_PPO/2026-03-03-00-53-40_NOISE_0.0_ACT_0.0_ENV_0.5/monitor.csv", "PPO_0.5"),
            # (r"logs_std_train/building_PPO/2026-03-03-00-54-42_NOISE_0.0_ACT_0.0_ENV_1.0/monitor.csv", "PPO_1.0"),
            # (r"logs_std_train/building_PPO/2026-03-03-01-16-28_NOISE_0.0_ACT_0.0_ENV_3.0/monitor.csv", "PPO_3.0"),
            # (r"logs_std_train/building_PPO/2026-03-03-00-53-41_NOISE_0.0_ACT_0.0_ENV_0.3/monitor.csv", "PPO_0.3"),
            # (r"logs_std_train/building_PPO/2026-03-03-01-13-56_NOISE_0.0_ACT_0.0_ENV_1.5/monitor.csv", "PPO_1.5"),
            # (r"logs_std_train/building_PPO/2026-03-03-01-13-56_NOISE_0.0_ACT_0.0_ENV_2.0/monitor.csv", "PPO_2.0"),
        ], None),
    },
    "SAC": {
        "DS": ([
            (r"logs_std_train/building_SAC/2026-02-09-11-18-19_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
            (r"logs_std_train/building_SAC/2026-02-25-08-17-49_NOISE_0.05_ACT_0.0/monitor.csv", "SAC_0.05"),
            # (r"logs_std_train/building_SAC/2026-02-25-08-18-37_NOISE_0.2_ACT_0.0/monitor.csv", "SAC_0.2"),
            # (r"logs_std_train/building_SAC/2026-02-26-15-22-40_NOISE_0.4_ACT_0.0/monitor.csv", "SAC_0.4"),
            # (r"logs_std_train/building_SAC/2026-02-25-08-15-54_NOISE_0.01_ACT_0.0/monitor.csv", "SAC_0.01"),
            # (r"logs_std_train/building_SAC/2026-02-25-08-15-54_NOISE_0.03_ACT_0.0/monitor.csv", "SAC_0.03"),
            # (r"logs_std_train/building_SAC/2026-02-25-08-18-03_NOISE_0.1_ACT_0.0/monitor.csv", "SAC_0.1"),
            (r"logs_std_train/building_SAC/2026-02-25-08-18-37_NOISE_0.15_ACT_0.0/monitor.csv", "SAC_0.15"),
            # (r"logs_std_train/building_SAC/2026-02-25-08-18-55_NOISE_0.3_ACT_0.0/monitor.csv", "SAC_0.3"),
        ], (-125, -25)),
        "DA": ([
            (r"logs_std_train/building_SAC/2026-02-09-11-18-19_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
            (r"logs_std_train/building_SAC/2026-02-26-15-05-50_NOISE_0.0_ACT_0.05/monitor.csv", "SAC_0.05"),
            # (r"logs_std_train/building_SAC/2026-02-26-15-06-10_NOISE_0.0_ACT_0.2/monitor.csv", "SAC_0.2"),
            # (r"logs_std_train/building_SAC/2026-02-26-15-06-56_NOISE_0.0_ACT_0.4/monitor.csv", "SAC_0.4"),
            # (r"logs_std_train/building_SAC/2026-02-26-15-05-43_NOISE_0.0_ACT_0.01/monitor.csv", "SAC_0.01"),
            # (r"logs_std_train/building_SAC/2026-02-26-15-05-50_NOISE_0.0_ACT_0.03/monitor.csv", "SAC_0.03"),
            # (r"logs_std_train/building_SAC/2026-02-26-15-05-50_NOISE_0.0_ACT_0.1/monitor.csv", "SAC_0.1"),
            (r"logs_std_train/building_SAC/2026-02-26-15-05-50_NOISE_0.0_ACT_0.15/monitor.csv", "SAC_0.15"),
            # (r"logs_std_train/building_SAC/2026-02-26-15-06-10_NOISE_0.0_ACT_0.3/monitor.csv", "SAC_0.3"),
        ], (-125, -25)),
        "DE": ([
            (r"logs_std_train/building_SAC/2026-02-09-11-18-19_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
            (r"logs_std_train\building_SAC\2026-03-23-15-51-05_NOISE_0.0_ACT_0.0_ENV_0.05\monitor.csv", "SAC_0.05"),
            (r"logs_std_train\building_SAC\2026-03-23-15-52-45_NOISE_0.0_ACT_0.0_ENV_0.15\monitor.csv", "SAC_0.15"),
            # (r"logs_std_train/building_SAC/2026-03-04-07-56-07_NOISE_0.0_ACT_0.0_ENV_0.5/monitor.csv", "SAC_0.5"),
            # (r"logs_std_train/building_SAC/2026-03-04-08-24-43_NOISE_0.0_ACT_0.0_ENV_1.0/monitor.csv", "SAC_1.0"),
            # (r"logs_std_train/building_SAC/2026-03-04-08-34-02_NOISE_0.0_ACT_0.0_ENV_3.0/monitor.csv", "SAC_3.0"),
            # (r"logs_std_train/building_SAC/2026-03-03-10-50-34_NOISE_0.0_ACT_0.0_ENV_0.1/monitor.csv", "SAC_0.1"),
            # (r"logs_std_train/building_SAC/2026-03-04-07-56-07_NOISE_0.0_ACT_0.0_ENV_0.3/monitor.csv", "SAC_0.3"),
            # (r"logs_std_train/building_SAC/2026-03-04-08-24-43_NOISE_0.0_ACT_0.0_ENV_1.5/monitor.csv", "SAC_1.5"),
            # (r"logs_std_train/building_SAC/2026-03-04-08-34-02_NOISE_0.0_ACT_0.0_ENV_2.0/monitor.csv", "SAC_2.0"),
        ], (-125, -25)),
    },
    "TD3": {
        "DS": ([
            (r"logs_std_train/building_TD3/2026-02-11-08-02-31_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),
            # TD3 has no obs noise runs for Building
        ], None),
        "DA": ([
            (r"logs_std_train/building_TD3/2026-02-11-08-02-31_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),
            # TD3 has no action noise runs for Building
        ], None),
        "DE": ([
            (r"logs_std_train/building_TD3/2026-02-11-08-02-31_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),
            # TD3 has no env noise runs for Building
        ], None),
    },
}


# ============================================================================
# COGEN CONFIG
# ============================================================================

CO_CONFIG = {
    "PPO": {
        "DS": ([
            (r"logs_std_train/cogen_PPO/2026-02-09-18-02-55_DS_0.0_DA_0.0/monitor.csv", "PPO_0.0"),
            (r"logs_std_train/cogen_PPO/2026-02-26-15-06-54_NOISE_10.0_ACT_0.0/monitor.csv", "PPO_10.0"),
            (r"logs_std_train/cogen_PPO/2026-02-26-15-12-21_NOISE_20.0_ACT_0.0/monitor.csv", "PPO_20.0"),
            # (r"logs_std_train/cogen_PPO/2026-02-25-21-00-28_NOISE_1.0_ACT_0.0/monitor.csv", "PPO_1.0"),
            # (r"logs_std_train/cogen_PPO/2026-02-26-00-58-13_NOISE_5.0_ACT_0.0/monitor.csv", "PPO_5.0"),
            # (r"logs_std_train/cogen_PPO/2026-02-25-20-33-57_NOISE_0.5_ACT_0.0/monitor.csv", "PPO_0.5"),
            # (r"logs_std_train/cogen_PPO/2026-02-25-23-21-59_NOISE_2.0_ACT_0.0/monitor.csv", "PPO_2.0"),
            # (r"logs_std_train/cogen_PPO/2026-02-26-00-24-39_NOISE_3.0_ACT_0.0/monitor.csv", "PPO_3.0"),
            # (r"logs_std_train/cogen_PPO/2026-02-26-00-24-39_NOISE_4.0_ACT_0.0/monitor.csv", "PPO_4.0"),
            # (r"logs_std_train/cogen_PPO/2026-02-26-15-12-21_NOISE_12.0_ACT_0.0/monitor.csv", "PPO_12.0"),
            # (r"logs_std_train/cogen_PPO/2026-02-26-15-12-21_NOISE_15.0_ACT_0.0/monitor.csv", "PPO_15.0"),
            # (r"logs_std_train/cogen_PPO/2026-02-26-15-12-21_NOISE_17.0_ACT_0.0/monitor.csv", "PPO_17.0"),
        ], (-2.6, -1.7)),
        "DA": ([
            (r"logs_std_train/cogen_PPO/2026-02-09-18-02-55_DS_0.0_DA_0.0/monitor.csv", "PPO_0.0"),
            (r"logs_std_train/cogen_PPO/2026-03-01-13-02-21_NOISE_0.0_ACT_0.1/monitor.csv", "PPO_0.1"),
            (r"logs_std_train/cogen_PPO/2026-03-01-13-02-50_NOISE_0.0_ACT_0.2/monitor.csv", "PPO_0.2"),
            # (r"logs_std_train/cogen_PPO/2026-03-01-13-05-10_NOISE_0.0_ACT_0.4/monitor.csv", "PPO_0.4"),
            # (r"logs_std_train/cogen_PPO/2026-03-01-13-02-11_NOISE_0.0_ACT_0.05/monitor.csv", "PPO_0.05"),
            # (r"logs_std_train/cogen_PPO/2026-03-01-13-02-21_NOISE_0.0_ACT_0.15/monitor.csv", "PPO_0.15"),
            # (r"logs_std_train/cogen_PPO/2026-03-01-13-03-14_NOISE_0.0_ACT_0.3/monitor.csv", "PPO_0.3"),
        ], (-2.6, -1.7)),
        "DE": ([
            (r"logs_std_train/cogen_PPO/2026-02-09-18-02-55_DS_0.0_DA_0.0/monitor.csv", "PPO_0.0"),
            (r"logs_std_train\cogen_PPO\2026-03-18-04-43-37_NOISE_0.0_ACT_0.0_ENV_0.3\monitor.csv", "PPO_0.3"),
            (r"logs_std_train\cogen_PPO\2026-03-18-07-30-19_NOISE_0.0_ACT_0.0_ENV_0.5\monitor.csv", "PPO_0.5"),
            # (r"logs_std_train\cogen_PPO\2026-03-18-03-50-20_NOISE_0.0_ACT_0.0_ENV_0.1\monitor.csv", "PPO_0.1"),
            # (r"logs_std_train\cogen_PPO\2026-03-18-08-36-57_NOISE_0.0_ACT_0.0_ENV_1.0\monitor.csv", "PPO_1.0"),
            # (r"logs_std_train\cogen_PPO\2026-03-18-09-18-23_NOISE_0.0_ACT_0.0_ENV_1.5\monitor.csv", "PPO_1.5"),
            # (r"logs_std_train\cogen_PPO\2026-03-18-11-27-00_NOISE_0.0_ACT_0.0_ENV_2.0\monitor.csv", "PPO_2.0"),
            # (r"logs_std_train\cogen_PPO\2026-03-18-12-16-04_NOISE_0.0_ACT_0.0_ENV_3.0\monitor.csv", "PPO_3.0"),
        ], (-2.6, -1.7)),
    },
    "SAC": {
        "DS": ([
            (r"logs_std_train/cogen_SAC/2026-03-21-16-19-37_NOISE_0.0_ACT_0.0_ENV_0.0/monitor.csv", "SAC_0.0"),
            (r"logs_std_train/cogen_SAC/2026-03-03-00-52-08_NOISE_10.0_ACT_0.0_ENV_0.0/monitor.csv", "SAC_10.0"),
            (r"logs_std_train/cogen_SAC/2026-03-03-03-07-27_NOISE_20.0_ACT_0.0_ENV_0.0/monitor.csv", "SAC_20.0"),
            # (r"logs_std_train/cogen_SAC/2026-02-09-18-03-16_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
            # (r"logs_std_train/cogen_SAC/2026-03-03-00-52-08_NOISE_17.0_ACT_0.0_ENV_0.0/monitor.csv", "SAC_17.0"),
            # (r"logs_std_train/cogen_SAC/2026-03-03-03-07-27_NOISE_25.0_ACT_0.0_ENV_0.0/monitor.csv", "SAC_25.0"),
            # (r"logs_std_train/cogen_SAC/2026-03-03-00-52-08_NOISE_12.0_ACT_0.0_ENV_0.0/monitor.csv", "SAC_12.0"),
            # NOTE: SAC has no runs at noise 1-5 range; only 10+ available
        ], None),
        "DA": ([
            (r"logs_std_train/cogen_SAC/2026-03-21-16-19-37_NOISE_0.0_ACT_0.0_ENV_0.0/monitor.csv", "SAC_0.0"),
            (r"logs_std_train/cogen_SAC/2026-03-01-13-53-58_NOISE_0.0_ACT_0.1/monitor.csv", "SAC_0.1"),
            (r"logs_std_train/cogen_SAC/2026-03-01-13-54-26_NOISE_0.0_ACT_0.2/monitor.csv", "SAC_0.2"),
            # (r"logs_std_train/cogen_SAC/2026-03-01-13-55-28_NOISE_0.0_ACT_0.4/monitor.csv", "SAC_0.4"),
            # (r"logs_std_train/cogen_SAC/2026-03-01-13-53-35_NOISE_0.0_ACT_0.05/monitor.csv", "SAC_0.05"),
            # (r"logs_std_train/cogen_SAC/2026-03-01-13-53-58_NOISE_0.0_ACT_0.15/monitor.csv", "SAC_0.15"),
            # (r"logs_std_train/cogen_SAC/2026-03-01-13-55-18_NOISE_0.0_ACT_0.3/monitor.csv", "SAC_0.3"),
        ], None),
        "DE": ([
            (r"logs_std_train/cogen_SAC/2026-03-21-16-19-37_NOISE_0.0_ACT_0.0_ENV_0.0/monitor.csv", "SAC_0.0"),
            (r"logs_std_train\cogen_SAC\2026-03-18-14-27-51_NOISE_0.0_ACT_0.0_ENV_0.3\monitor.csv", "SAC_0.3"),
            (r"logs_std_train\cogen_SAC\2026-03-18-21-17-22_NOISE_0.0_ACT_0.0_ENV_0.5\monitor.csv", "SAC_0.5"),
            # (r"logs_std_train\cogen_SAC\2026-03-18-13-32-04_NOISE_0.0_ACT_0.0_ENV_0.1\monitor.csv", "SAC_0.1"),
            # (r"logs_std_train\cogen_SAC\2026-03-18-23-25-23_NOISE_0.0_ACT_0.0_ENV_1.0\monitor.csv", "SAC_1.0"),
            # (r"logs_std_train\cogen_SAC\2026-03-19-00-17-47_NOISE_0.0_ACT_0.0_ENV_1.5\monitor.csv", "SAC_1.5"),
            # (r"logs_std_train\cogen_SAC\2026-03-19-08-57-10_NOISE_0.0_ACT_0.0_ENV_2.0\monitor.csv", "SAC_2.0"),
            # (r"logs_std_train\cogen_SAC\2026-03-19-15-09-38_NOISE_0.0_ACT_0.0_ENV_3.0\monitor.csv", "SAC_3.0"),
        ], None),
    },
    "TD3": {
        "DS": ([
            (r"logs_std_train/cogen_TD3/2026-02-09-18-03-50_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),
            (r"logs_std_train/cogen_TD3/2026-03-01-17-02-51_NOISE_10.0_ACT_0.0/monitor.csv", "TD3_10.0"),
            (r"logs_std_train/cogen_TD3/2026-03-01-17-07-54_NOISE_20.0_ACT_0.0/monitor.csv", "TD3_20.0"),
            # (r"logs_std_train/cogen_TD3/2026-03-01-17-06-27_NOISE_17.0_ACT_0.0/monitor.csv", "TD3_17.0"),
            # (r"logs_std_train/cogen_TD3/2026-03-01-17-11-37_NOISE_25.0_ACT_0.0/monitor.csv", "TD3_25.0"),
            # (r"logs_std_train/cogen_TD3/2026-03-01-17-03-27_NOISE_12.0_ACT_0.0/monitor.csv", "TD3_12.0"),
        ], None),
        "DA": ([
            (r"logs_std_train/cogen_TD3/2026-02-09-18-03-50_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),
            (r"logs_std_train/cogen_TD3/2026-03-01-14-03-10_NOISE_0.0_ACT_0.1/monitor.csv", "TD3_0.1"),
            (r"logs_std_train/cogen_TD3/2026-03-01-14-05-18_NOISE_0.0_ACT_0.2/monitor.csv", "TD3_0.2"),
            # (r"logs_std_train/cogen_TD3/2026-03-01-14-06-06_NOISE_0.0_ACT_0.4/monitor.csv", "TD3_0.4"),
            # (r"logs_std_train/cogen_TD3/2026-03-01-14-03-10_NOISE_0.0_ACT_0.05/monitor.csv", "TD3_0.05"),
            # (r"logs_std_train/cogen_TD3/2026-03-01-14-03-32_NOISE_0.0_ACT_0.15/monitor.csv", "TD3_0.15"),
            # (r"logs_std_train/cogen_TD3/2026-03-01-14-05-35_NOISE_0.0_ACT_0.3/monitor.csv", "TD3_0.3"),
        ], None),
        "DE": ([
            (r"logs_std_train/cogen_TD3/2026-02-09-18-03-50_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),
            (r"logs_std_train\cogen_TD3\2026-03-19-18-44-18_NOISE_0.0_ACT_0.0_ENV_0.3\monitor.csv", "TD3_0.3"),
            (r"logs_std_train\cogen_TD3\2026-03-19-22-27-59_NOISE_0.0_ACT_0.0_ENV_0.5\monitor.csv", "TD3_0.5"),
            # (r"logs_std_train/cogen_TD3/2026-03-08-13-42-13_NOISE_0.0_ACT_0.0_ENV_0.5/monitor.csv", "TD3_0.5"),
            # (r"logs_std_train/cogen_TD3/2026-03-08-13-42-43_NOISE_0.0_ACT_0.0_ENV_1.0/monitor.csv", "TD3_1.0"),
            # (r"logs_std_train/cogen_TD3/2026-03-08-13-43-16_NOISE_0.0_ACT_0.0_ENV_3.0/monitor.csv", "TD3_3.0"),
        ], None),
    },
}


# ============================================================================
# ENV -> CONFIG mapping
# ============================================================================

ALL_CONFIGS = {
    "evcharging": EV_CONFIG,
    "building": BU_CONFIG,
    "cogen": CO_CONFIG,
}


# ============================================================================
# BASELINE PATHS FOR ALGORITHM COMPARISON (noise=0.0 for each algo)
# ============================================================================

BASELINE_PATHS = {
    "evcharging": {
        "PPO": r"logs_std_train/evcharging_PPO/2026-02-05-20-06-35_DS_0.0_DA_0.0/monitor.csv",
        "SAC": r"logs_std_train/evcharging_SAC/2026-02-09-12-45-20_DS_0.0_DA_0.0/monitor.csv",
        "TD3": r"logs_std_train/evcharging_TD3/2026-02-07-17-45-25_DS_0.0_DA_0.0/monitor.csv",
    },
    "building": {
        "PPO": r"logs_std_train/building_PPO/2026-02-09-11-18-57_DS_0.0_DA_0.0/monitor.csv",
        "SAC": r"logs_std_train/building_SAC/2026-02-09-11-18-19_DS_0.0_DA_0.0/monitor.csv",
        "TD3": r"logs_std_train/building_TD3/2026-02-11-08-02-31_DS_0.0_DA_0.0/monitor.csv",
    },
    "cogen": {
        "PPO": r"logs_std_train/cogen_PPO/2026-02-09-18-02-55_DS_0.0_DA_0.0/monitor.csv",
        "SAC": r"logs_std_train/cogen_SAC/2026-03-21-16-19-37_NOISE_0.0_ACT_0.0_ENV_0.0/monitor.csv",
        "TD3": r"logs_std_train/cogen_TD3/2026-02-09-18-03-50_DS_0.0_DA_0.0/monitor.csv",
    },
}


# ============================================================================
# FUNCTIONS
# ============================================================================

def load_rewards(csv_path: str, max_steps: int) -> Optional[np.ndarray]:
    """Load rewards from CSV."""
    if not Path(csv_path).exists():
        print(f"  File not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path, comment="#", header=0)
        if "r" not in df.columns:
            print(f"  No 'r' column: {csv_path}")
            return None

        rewards = df["r"].astype(float).values[:max_steps]
        print(f"  Loaded {len(rewards)} episodes")
        return rewards

    except Exception as e:
        print(f"  Error: {csv_path} - {e}")
        return None


def calculate_running_stats(rewards: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate running mean and std using EMA (no window-edge kink)."""
    series = pd.Series(rewards)
    mean = series.ewm(span=window, adjust=True).mean().values
    std = series.ewm(span=window, adjust=True).std().values
    return mean, std


DT_FULL = {"DS": "Perturbation of State (PS)", "DA": "Perturbation of Action (PA)", "DE": "Perturbation of Dynamics (PD)"}
DT_SHORT = {"DS": "PS", "DA": "PA", "DE": "PD"}
DT_SUBFIG = {"DS": "(a) State", "DA": "(b) Action", "DE": "(c) Dynamics"}


def _plot_on_ax(
    ax,
    experiments: List[Tuple[str, str]],
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    title: str = "",
    window: int = 4000,
    max_steps: int = 32000,
    skip_initial: int = 250,
    auto_xlim: bool = False,
    auto_ylim: bool = False,
    std_band: float = 0.5,
    dt: str = "",
    total_steps: int = 9_000_000,
    show_ylabel: bool = True,
    legend_loc = 'lower right',
    zoom: Optional[Tuple[float, float, float, float]] = None,
):
    """Plot learning curves on a given axes object."""

    for idx, (csv_path, raw_label) in enumerate(experiments):
        # Reformat label: "PPO_0.1" -> "PPO_PS_0.10"
        parts = raw_label.split("_", 1)
        if len(parts) == 2 and dt:
            algo_name, noise_val = parts
            try:
                label = f"{algo_name}_{DT_SHORT.get(dt, dt)}_{float(noise_val):.2f}"
            except ValueError:
                label = raw_label
        else:
            label = raw_label
        rewards = load_rewards(csv_path, max_steps)
        if rewards is None:
            continue

        mean, std = calculate_running_stats(rewards, window)

        # Thin data so dash/dot line styles are visible (~1500 points max)
        x_full = np.arange(len(mean))
        step = max(1, len(x_full) // 1500)
        x = x_full[::step]
        mean_t = mean[::step]
        std_t = std[::step]

        # Shift x so that skip_initial maps to 0
        x = x - skip_initial

        color = COLORS_INDEXED[idx % len(COLORS_INDEXED)]
        ls = LINE_STYLES[idx % len(LINE_STYLES)]
        marker = MARKERS[idx % len(MARKERS)]
        me = max(1, int(len(x) * MARK_EVERY_FRAC))

        ax.plot(x, mean_t, label=label, linewidth=1.8, color=color,
                linestyle=ls, marker=marker, markersize=5,
                markevery=me, markeredgewidth=0.6,
                markeredgecolor=color)
        ax.fill_between(x, mean_t - std_band * std_t, mean_t + std_band * std_t,
                        alpha=FILL_ALPHA, color=color, linewidth=0)

    # X-axis limits (shifted so skip_initial -> 0)
    if auto_xlim:
        pass
    elif xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0, max_steps - skip_initial)

    # Y-axis limits
    if auto_ylim:
        pass
    elif ylim:
        ax.set_ylim(ylim)

    # Ensure the last x value appears as a tick, then re-apply xlim
    x_end = max_steps - skip_initial
    xticks = [t for t in ax.get_xticks() if 0 <= t <= x_end]
    if x_end not in xticks:
        xticks.append(x_end)
    ax.set_xticks(xticks)
    ax.set_xlim(0, x_end)

    # Show total training steps (top-left corner)
    exponent = int(np.floor(np.log10(total_steps)))
    mantissa = total_steps / 10**exponent
    step_str = f"Total: ~{int(round(mantissa))}e{exponent} steps"
    ax.text(0.02, 0.97, step_str, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_title(title)
    ax.set_xlabel("Training Episode")
    if show_ylabel:
        ax.set_ylabel("Average Episode Reward")

    if isinstance(legend_loc, dict):
        ax.legend(frameon=True, **legend_loc)
    else:
        ax.legend(loc=legend_loc, frameon=True)
    style_axis(ax)

    # Inset zoom for perturbation plots
    if zoom:
        zx1, zx2, zy1, zy2 = zoom
        axins = ax.inset_axes([0.84, 0.22, 0.15, 0.20])
        # Re-plot all lines from main axes (without markers)
        for line in ax.get_lines():
            axins.plot(line.get_xdata(), line.get_ydata(),
                       color=line.get_color(), linestyle=line.get_linestyle(),
                       linewidth=1.5)
        axins.set_xlim(zx1, zx2)
        axins.set_ylim(zy1, zy2)
        x_start_k = int(np.ceil(zx1 / 1000))
        x_end_k = int(np.floor(zx2 / 1000))
        x_range = np.arange(x_start_k, x_end_k + 1) * 1000
        axins.set_xticks(x_range)
        axins.set_xticklabels([f"{int(v)}" for v in x_range])
        axins.set_xlim(zx1, zx2)
        axins.tick_params(labelsize=7)
        axins.grid(True, alpha=0.3)
        for spine in axins.spines.values():
            spine.set_edgecolor('0.4')
            spine.set_linewidth(1.0)
        rect, connectors = ax.indicate_inset_zoom(axins, edgecolor='#aa2222', linewidth=2.0, alpha=0.9)
        rect.set_facecolor('#ee9999')
        rect.set_alpha(0.35)
        for conn in connectors:
            conn.set_edgecolor('#aaaaaa')
            conn.set_linewidth(1.0)


def plot_learning_curves(
    experiments: List[Tuple[str, str]],
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    title: str = "",
    window: int = 4000,
    max_steps: int = 32000,
    skip_initial: int = 250,
    auto_xlim: bool = False,
    auto_ylim: bool = False,
    std_band: float = 0.5,
    dt: str = "",
    total_steps: int = 9_000_000,
    zoom: Optional[Tuple[float, float, float, float]] = None,
    legend_loc = 'lower right',
):
    """Plot learning curves (single panel)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_on_ax(ax, experiments, ylim, xlim, title, window, max_steps,
                skip_initial, auto_xlim, auto_ylim, std_band, dt, total_steps,
                zoom=zoom, legend_loc=legend_loc)
    plt.tight_layout()


def plot_learning_curves_all_dt(
    env_config: dict,
    algo: str,
    env_title: str,
    window: int = 4000,
    max_steps: int = 32000,
    skip_initial: int = 250,
    auto_xlim: bool = False,
    auto_ylim: bool = False,
    std_band: float = 0.5,
    total_steps: int = 9_000_000,
    ylim_override: Optional[Tuple[float, float]] = None,
    xlim_override: Optional[Tuple[float, float]] = None,
    zoom: Optional[Tuple[float, float, float, float]] = None,
    env: str = "",
):
    """Plot DS, DA, DE side by side in 1x3 subplots."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)

    for i, dt in enumerate(["DS", "DA", "DE"]):
        if algo not in env_config or dt not in env_config[algo]:
            print(f"  Config not found: {algo} / {dt}, skipping")
            continue

        paths, default_ylim = env_config[algo][dt]
        ylim = ylim_override if ylim_override else default_ylim
        xlim = xlim_override
        loc = LEGEND_LOC.get((env, algo, dt), 'lower right')

        _plot_on_ax(
            axes[i], paths, ylim, xlim, DT_SUBFIG[dt], window, max_steps,
            skip_initial, auto_xlim, auto_ylim, std_band, dt, total_steps,
            show_ylabel=(i == 0), zoom=zoom, legend_loc=loc,
        )

    fig.suptitle(f"{env_title} — {algo}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def save_plot(env: str, algo: str, dt: str, output_dir: str, save_pdf: bool = False):
    """Save plot into organized folder: graphs/C_STDRL/{env}/{algo}/{dt}.png"""
    subdir = os.path.join(output_dir, ENV_SHORT.get(env, env), algo)
    os.makedirs(subdir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_filename = f"{dt}_{timestamp}"

    png_path = os.path.join(subdir, base_filename + ".png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"\nPNG saved: {png_path}")

    if save_pdf:
        pdf_path = os.path.join(subdir, base_filename + ".pdf")
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"PDF saved: {pdf_path}")


def plot_comparison(
    env: str,
    max_steps: int = 32000,
    window: int = 6000,
    skip_initial: int = 250,
    auto_ylim: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    auto_xlim: bool = False,
    std_band: float = 0.03,
    total_steps: int = 9_000_000,
    zoom: Optional[Tuple[float, float, float, float]] = None,
):
    """Plot baseline (noise=0) learning curves for all algorithms on one chart."""
    baselines = BASELINE_PATHS.get(env, {})
    if not baselines:
        print(f"No baseline paths for env: {env}")
        return

    env_def = ENV_DEFAULTS[env]
    _, ax = plt.subplots(figsize=(10, 6))

    for algo, csv_path in baselines.items():
        print(f"Loading {algo} baseline ...")
        rewards = load_rewards(csv_path, max_steps)
        if rewards is None:
            continue

        mean, std = calculate_running_stats(rewards, window)

        x_full = np.arange(len(mean))
        step = max(1, len(x_full) // 1500)
        x = x_full[::step]
        mean_t = mean[::step]
        std_t = std[::step]

        # Shift x so that skip_initial maps to 0
        x = x - skip_initial

        color = get_color(algo)
        ls = get_linestyle(algo)
        marker = get_marker(algo)
        me = max(1, int(len(x) * MARK_EVERY_FRAC))

        ax.plot(x, mean_t, label=algo, linewidth=1.8, color=color,
                linestyle=ls, marker=marker, markersize=5,
                markevery=me, markeredgewidth=0.6,
                markeredgecolor=color)
        ax.fill_between(x, mean_t - std_band * std_t, mean_t + std_band * std_t,
                        alpha=FILL_ALPHA, color=color, linewidth=0)

    # X-axis limits (shifted so skip_initial -> 0)
    if auto_xlim:
        pass
    elif xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0, max_steps - skip_initial)

    # Y-axis limits (CLI override > env default)
    effective_ylim = ylim if ylim else env_def.get("ylim")
    if auto_ylim:
        pass
    elif effective_ylim:
        ax.set_ylim(effective_ylim)

    # Ensure the last x value appears as a tick, then re-apply xlim
    x_end = max_steps - skip_initial
    xticks = [t for t in ax.get_xticks() if 0 <= t <= x_end]
    if x_end not in xticks:
        xticks.append(x_end)
    ax.set_xticks(xticks)
    ax.set_xlim(0, x_end)

    # Show total training steps (top-left corner)
    exponent = int(np.floor(np.log10(total_steps)))
    mantissa = total_steps / 10**exponent
    step_str = f"Total: ~{int(round(mantissa))}e{exponent} steps"
    ax.text(0.02, 0.97, step_str, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_title(f"{env_def['title']} — Algorithm Comparison (Baseline)")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Average Episode Reward")
    ax.legend(loc='lower right', frameon=True)
    style_axis(ax)

    # Inset zoom: zoom = (x1, x2, y1, y2) in shifted episode coordinates
    # Done AFTER style_axis and set_xticks so main axis is finalized
    if zoom:
        zx1, zx2, zy1, zy2 = zoom
        axins = ax.inset_axes([0.84, 0.45, 0.15, 0.20])
        # Re-plot using stored data
        for algo_name, csv_path in baselines.items():
            rewards = load_rewards(csv_path, max_steps)
            if rewards is None:
                continue
            mean_z, _ = calculate_running_stats(rewards, window)
            x_z = np.arange(len(mean_z)) - skip_initial
            axins.plot(x_z, mean_z, color=get_color(algo_name),
                       linestyle=get_linestyle(algo_name), linewidth=1.5)
        axins.set_xlim(zx1, zx2)
        axins.set_ylim(zy1, zy2)
        # Clean ticks: round x ticks to thousands (inclusive of both ends)
        x_start_k = int(np.ceil(zx1 / 1000))
        x_end_k = int(np.floor(zx2 / 1000))
        x_range = np.arange(x_start_k, x_end_k + 1) * 1000
        axins.set_xticks(x_range)
        axins.set_xticklabels([f"{int(v)}" for v in x_range])
        axins.set_xlim(zx1, zx2)
        axins.tick_params(labelsize=7)
        axins.grid(True, alpha=0.3)
        for spine in axins.spines.values():
            spine.set_edgecolor('0.4')
            spine.set_linewidth(1.0)
        # Indicate zoom region on main plot + connector lines (academic style)
        rect, connectors = ax.indicate_inset_zoom(axins, edgecolor='#aa2222', linewidth=2.0, alpha=0.9)
        rect.set_facecolor('#ee9999')
        rect.set_alpha(0.35)
        # Make connector lines pastel gray
        for conn in connectors:
            conn.set_edgecolor('#aaaaaa')
            conn.set_linewidth(1.0)

    plt.tight_layout()


def save_comparison_plot(env: str, output_dir: str, save_pdf: bool = False):
    """Save comparison plot into graphs/C_STDRL/{env_short}/comparison/"""
    subdir = os.path.join(output_dir, ENV_SHORT.get(env, env), "comparison")
    os.makedirs(subdir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_filename = f"baseline_comparison_{timestamp}"

    png_path = os.path.join(subdir, base_filename + ".png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"\nPNG saved: {png_path}")

    if save_pdf:
        pdf_path = os.path.join(subdir, base_filename + ".pdf")
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"PDF saved: {pdf_path}")


# ============================================================================
# MAIN
# ============================================================================

def _run_multiseed_all(args):
    """Generate all 7 multiseed panel figures (1x3 each) in one run."""
    for env, algo in MULTISEED_ALL_COMBOS:
        env_def = ENV_DEFAULTS[env]
        t_steps = args.t_steps if args.t_steps is not None else env_def["t_steps"]

        print(f"\n{'='*70}")
        print(f"Multi-seed 1x3: {env} / {algo}")
        print(f"{'='*70}\n")

        plot_multiseed_all_dt(
            env=env,
            algo=algo,
            window=args.w_size,
            max_steps=t_steps,
            skip_initial=args.skip,
            auto_xlim=False,
            auto_ylim=args.auto_ylim,
            std_band=1.0,  # full ±1σ cross-seed band (not the 0.03 single-seed EMA scaling)
            total_steps=env_def["total_steps"],
        )
        save_multiseed_plot(env, algo, "DS_DA_DE", args.pdf)
        plt.close()

    print(f"\nDone — all {len(MULTISEED_ALL_COMBOS)} figures saved.")


def main():
    parser = argparse.ArgumentParser(description='Unified Standard RL Training Curve Plotter')
    parser.add_argument('--env', type=str, default=None, choices=['evcharging', 'building', 'cogen'],
                        help='Environment (required except with --multiseed-all)')
    parser.add_argument('--algo', type=str, default='PPO', choices=['PPO', 'SAC', 'TD3'],
                        help='Algorithm (ignored with --compare)')
    parser.add_argument('--dt', type=str, default='DS', choices=['DS', 'DA', 'DE', 'all'],
                        help='Noise type: DS/DA/DE or "all" for 1x3 subplot')
    parser.add_argument('--t_steps', type=int, default=None, help='Training steps (default: env-specific)')
    parser.add_argument('--w_size', type=int, default=6000, help='Window size')
    parser.add_argument('--skip', type=int, default=250, help='Skip initial episodes')
    parser.add_argument('--pdf', action='store_true', help='Also save PDF (vector graphics)')
    parser.add_argument('--ylim', nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help='Y-axis limits (e.g., --ylim 5 6.5)')
    parser.add_argument('--xlim', nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help='X-axis limits (e.g., --xlim 250 32000)')
    parser.add_argument('--auto_ylim', action='store_true', help='Force auto y-axis limits')
    parser.add_argument('--auto_xlim', action='store_true', help='Force auto x-axis limits')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all algorithms baseline (noise=0) for the given env')
    parser.add_argument('--zoom', nargs=4, type=float, metavar=('X1', 'X2', 'Y1', 'Y2'),
                        help='Inset zoom region (e.g., --zoom 20000 22000 -35 -25)')
    parser.add_argument('--multiseed', action='store_true',
                        help='Multi-seed mode: auto-discover seed runs, plot cross-seed mean +/- std')
    parser.add_argument('--multiseed-all', action='store_true', dest='multiseed_all',
                        help='Generate all 7 multiseed panel figures (ev PPO/SAC, bu PPO/SAC, co PPO/SAC/TD3)')

    args = parser.parse_args()

    # ── Multiseed-all: no --env required ──
    if args.multiseed_all:
        _run_multiseed_all(args)
        return

    # All other modes require --env
    if args.env is None:
        parser.error("--env is required (unless using --multiseed-all)")

    # Get env defaults
    env_def = ENV_DEFAULTS[args.env]
    t_steps = args.t_steps if args.t_steps is not None else env_def["t_steps"]

    # Determine ylim / xlim
    ylim = tuple(args.ylim) if args.ylim else None
    xlim = tuple(args.xlim) if args.xlim else None
    auto_xlim = args.auto_xlim and not args.xlim

    # ── Comparison mode ──
    if args.compare:
        print(f"\n{'='*70}")
        print(f"Baseline Algorithm Comparison: {args.env}")
        print(f"{'='*70}\n")

        plot_comparison(
            env=args.env,
            max_steps=t_steps,
            window=args.w_size,
            skip_initial=args.skip,
            auto_ylim=args.auto_ylim,
            ylim=ylim,
            xlim=xlim,
            auto_xlim=auto_xlim,
            std_band=env_def["std_band"],
            total_steps=env_def["total_steps"],
            zoom=tuple(args.zoom) if args.zoom else None,
        )
        save_comparison_plot(args.env, OUTPUT_DIR, args.pdf)
        plt.show()
        return

    # ── Multi-seed mode ──
    if args.multiseed:
        if args.dt == "all":
            print(f"\n{'='*70}")
            print(f"Multi-seed 1x3: {args.env} / {args.algo} / DS+DA+DE")
            print(f"{'='*70}\n")

            plot_multiseed_all_dt(
                env=args.env,
                algo=args.algo,
                window=args.w_size,
                max_steps=t_steps,
                skip_initial=args.skip,
                auto_xlim=auto_xlim,
                auto_ylim=args.auto_ylim,
                std_band=1.0,  # full ±1σ cross-seed band
                total_steps=env_def["total_steps"],
                ylim_override=ylim,
                xlim_override=xlim,
                zoom=tuple(args.zoom) if args.zoom else None,
            )
            save_multiseed_plot(args.env, args.algo, "DS_DA_DE", args.pdf)
            plt.show()
        else:
            print(f"\n{'='*70}")
            print(f"Multi-seed: {args.env} / {args.algo} / {args.dt}")
            print(f"{'='*70}\n")

            loc = LEGEND_LOC.get((args.env, args.algo, args.dt), 'lower right')
            plot_multiseed_single(
                env=args.env,
                algo=args.algo,
                dt=args.dt,
                window=args.w_size,
                max_steps=t_steps,
                skip_initial=args.skip,
                auto_xlim=auto_xlim,
                auto_ylim=args.auto_ylim,
                std_band=1.0,  # full ±1σ cross-seed band
                total_steps=env_def["total_steps"],
                ylim=ylim,
                xlim=xlim,
                zoom=tuple(args.zoom) if args.zoom else None,
                legend_loc=loc,
            )
            save_multiseed_plot(args.env, args.algo, args.dt, args.pdf)
            plt.show()
        return

    env_config = ALL_CONFIGS[args.env]

    # ── All noise types side by side (1x3 subplot) ──
    if args.dt == "all":
        print(f"\n{'='*70}")
        print(f"Plotting {args.env} / {args.algo} / DS+DA+DE (1x3)")
        print(f"{'='*70}\n")

        plot_learning_curves_all_dt(
            env_config=env_config,
            algo=args.algo,
            env_title=env_def["title"],
            window=args.w_size,
            max_steps=t_steps,
            skip_initial=args.skip,
            auto_xlim=auto_xlim,
            auto_ylim=args.auto_ylim,
            std_band=env_def["std_band"],
            total_steps=env_def["total_steps"],
            ylim_override=ylim,
            xlim_override=xlim,
            zoom=tuple(args.zoom) if args.zoom else None,
            env=args.env,
        )
        save_plot(args.env, args.algo, "DS_DA_DE", OUTPUT_DIR, args.pdf)
        plt.show()
        return

    # ── Standard single-algo noise plot ──
    print(f"\n{'='*70}")
    print(f"Plotting {args.env} / {args.algo} / {args.dt}")
    print(f"{'='*70}\n")

    if args.algo not in env_config or args.dt not in env_config[args.algo]:
        print(f"Config not found: {args.env} / {args.algo} / {args.dt}")
        return

    paths, default_ylim = env_config[args.algo][args.dt]

    # Override ylim with config default if no explicit flag
    if not args.ylim and not args.auto_ylim:
        ylim = default_ylim

    loc = LEGEND_LOC.get((args.env, args.algo, args.dt), 'lower right')
    plot_learning_curves(
        experiments=paths,
        ylim=ylim,
        xlim=xlim,
        title=f"{env_def['title']} — {args.algo}: {DT_FULL[args.dt]}",
        window=args.w_size,
        max_steps=t_steps,
        skip_initial=args.skip,
        auto_xlim=auto_xlim,
        auto_ylim=args.auto_ylim,
        std_band=env_def["std_band"],
        dt=args.dt,
        total_steps=env_def["total_steps"],
        zoom=tuple(args.zoom) if args.zoom else None,
        legend_loc=loc,
    )

    save_plot(args.env, args.algo, args.dt, OUTPUT_DIR, args.pdf)
    plt.show()


if __name__ == "__main__":
    main()
