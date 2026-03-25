"""
Cross-Noise Robustness Heatmap
------------------------------
Creates a single unified heatmap per environment showing final training
reward across all three noise channels (Observation, Action, Environment)
for PPO and SAC side by side.

Usage:
    python scripts/plot/noise_heatmap.py --env evcharging
    python scripts/plot/noise_heatmap.py --env building
    python scripts/plot/noise_heatmap.py --env cogen
    python scripts/plot/noise_heatmap.py --all              # all three envs
    python scripts/plot/noise_heatmap.py --env evcharging --last-n 200  # avg of last N episodes
"""

import os
import sys
import re
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
import matplotlib.ticker as ticker
from style import apply_style, ENV_TITLES, ALGO_COLORS, style_axis

apply_style()

# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]
TRAIN_DIR = BASE_DIR / "logs_std_train"
OUT_DIR = BASE_DIR / "graphs" / "C_STDRL"

ALGOS = ["PPO", "SAC"]

# Noise channel labels
CHANNEL_LABELS = {
    "obs": "Observation (PS)",
    "action": "Action (PA)",
    "env": "Dynamics (PD)",
}


# ============================================================================
# DATA LOADING
# ============================================================================

def parse_noise(dirname: str) -> dict:
    """Extract noise levels from directory name."""
    result = {"obs": 0.0, "action": 0.0, "env": 0.0}

    # New format: NOISE_X_ACT_Y_ENV_Z
    m_noise = re.search(r"NOISE_([\d.]+)", dirname)
    m_act = re.search(r"ACT_([\d.]+)", dirname)
    m_env = re.search(r"ENV_([\d.]+)", dirname)

    if m_noise:
        result["obs"] = float(m_noise.group(1))
    if m_act:
        result["action"] = float(m_act.group(1))
    if m_env:
        result["env"] = float(m_env.group(1))

    # Legacy format: DS_X_DA_Y
    if not m_noise:
        m_ds = re.search(r"DS_([\d.]+)", dirname)
        if m_ds:
            result["obs"] = float(m_ds.group(1))
    if not m_act:
        m_da = re.search(r"DA_([\d.]+)", dirname)
        if m_da:
            result["action"] = float(m_da.group(1))

    return result


def is_single_channel(noise: dict) -> str | None:
    """Return the active noise channel if exactly one is non-zero, or 'clean' if all zero."""
    active = [ch for ch, v in noise.items() if v > 0]
    if len(active) == 0:
        return "clean"
    if len(active) == 1:
        return active[0]
    return None  # multi-channel noise, skip


def read_final_reward(run_dir: Path, last_n: int = 100) -> float | None:
    """Read mean reward of last N episodes from monitor.csv."""
    monitor = run_dir / "monitor.csv"
    if not monitor.exists():
        return None
    try:
        df = pd.read_csv(monitor, comment="#")
        if len(df) < 10:
            return None
        rewards = df["r"].iloc[-last_n:]
        return float(np.mean(rewards))
    except Exception:
        return None


def collect_data(env: str, algo: str, last_n: int = 100) -> dict:
    """
    Collect final rewards for all runs of env_algo.
    Returns: {channel: {noise_level: reward}}
    """
    env_algo_dir = TRAIN_DIR / f"{env}_{algo}"
    if not env_algo_dir.exists():
        return {}

    data = {"obs": {}, "action": {}, "env": {}}
    clean_rewards = []

    for run_dir in sorted(env_algo_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        noise = parse_noise(run_dir.name)
        channel = is_single_channel(noise)
        if channel is None:
            continue  # skip multi-channel

        reward = read_final_reward(run_dir, last_n)
        if reward is None:
            continue

        if channel == "clean":
            clean_rewards.append(reward)
        else:
            level = noise[channel]
            # If duplicate noise level, keep the best (latest run likely better)
            if level in data[channel]:
                data[channel][level] = max(data[channel][level], reward)
            else:
                data[channel][level] = reward

    # Add clean (noise=0) to each channel — use best run (max reward)
    if clean_rewards:
        clean_best = float(max(clean_rewards))
        for ch in data:
            data[ch][0.0] = clean_best

    return data


# ============================================================================
# PLOTTING
# ============================================================================

def build_heatmap_matrix(data_ppo: dict, data_sac: dict):
    """
    Build a combined matrix for the heatmap.
    Returns: matrix (n_rows x n_cols), row_labels, col_labels
    Each row = (algo, channel), each col = noise_level
    """
    # Collect all noise levels per channel across both algos
    all_levels = {}
    for ch in ["obs", "action", "env"]:
        levels = set()
        for data in [data_ppo, data_sac]:
            if ch in data:
                levels.update(data[ch].keys())
        all_levels[ch] = sorted(levels)

    # Build rows: PPO-obs, PPO-act, PPO-env, SAC-obs, SAC-act, SAC-env
    rows = []
    row_labels = []
    col_data = []  # (channel, level) for each column

    # We'll create a flat column list: all obs levels, then all act levels, then all env levels
    for ch in ["obs", "action", "env"]:
        for level in all_levels[ch]:
            col_data.append((ch, level))

    n_cols = len(col_data)

    for algo, data in [("PPO", data_ppo), ("SAC", data_sac)]:
        row = []
        for ch, level in col_data:
            if ch in data and level in data[ch]:
                row.append(data[ch][level])
            else:
                row.append(np.nan)
        rows.append(row)
        row_labels.append(algo)

    matrix = np.array(rows)
    return matrix, row_labels, col_data, all_levels


def format_level(level: float) -> str:
    """Format noise level for display."""
    if level == 0.0:
        return "0"
    if level == int(level):
        return str(int(level))
    return f"{level:g}"


def plot_env_heatmap(env: str, last_n: int = 100, save: bool = True):
    """Create cross-noise robustness heatmap for one environment."""
    print(f"\n{'='*60}")
    print(f"  {ENV_TITLES.get(env, env)} — Cross-Noise Robustness Heatmap")
    print(f"{'='*60}")

    data_ppo = collect_data(env, "PPO", last_n)
    data_sac = collect_data(env, "SAC", last_n)

    if not data_ppo and not data_sac:
        print(f"  No data found for {env}")
        return

    # Print summary
    for algo, data in [("PPO", data_ppo), ("SAC", data_sac)]:
        print(f"\n  {algo}:")
        for ch in ["obs", "action", "env"]:
            levels = sorted(data.get(ch, {}).keys())
            vals = [f"{data[ch][l]:.2f}" for l in levels]
            print(f"    {ch:8s}: {dict(zip([format_level(l) for l in levels], vals))}")

    # --- Build per-channel sub-heatmaps ---
    channels = ["obs", "action", "env"]

    # Figure layout: 3 sub-heatmaps side by side, each channel
    fig = plt.figure(figsize=(16, 3.8))

    # Compute global vmin/vmax for consistent colorscale
    all_vals = []
    for data in [data_ppo, data_sac]:
        for ch in channels:
            all_vals.extend(data.get(ch, {}).values())

    if not all_vals:
        print("  No values to plot")
        return

    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    # Add small padding
    vrange = vmax - vmin if vmax > vmin else 1.0
    vmin -= vrange * 0.02
    vmax += vrange * 0.02

    # Determine widths proportional to number of noise levels per channel
    widths = []
    channel_levels = {}
    for ch in channels:
        levels = set()
        for data in [data_ppo, data_sac]:
            if ch in data:
                levels.update(data[ch].keys())
        channel_levels[ch] = sorted(levels)
        widths.append(max(len(channel_levels[ch]), 1))

    # Add space for colorbar
    gs = gridspec.GridSpec(1, 4, width_ratios=widths + [0.3],
                           wspace=0.15, left=0.05, right=0.93, top=0.82, bottom=0.18)

    axes = []
    im = None

    for i, ch in enumerate(channels):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)

        levels = channel_levels[ch]
        if not levels:
            ax.set_visible(False)
            continue

        # Build 2-row matrix: [PPO, SAC] x noise_levels
        matrix = np.full((2, len(levels)), np.nan)
        for j, level in enumerate(levels):
            if ch in data_ppo and level in data_ppo[ch]:
                matrix[0, j] = data_ppo[ch][level]
            if ch in data_sac and level in data_sac[ch]:
                matrix[1, j] = data_sac[ch][level]

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                        vmin=vmin, vmax=vmax, interpolation="nearest")

        # Annotate cells with values
        for r in range(2):
            for c in range(len(levels)):
                val = matrix[r, c]
                if np.isnan(val):
                    ax.text(c, r, "—", ha="center", va="center",
                            fontsize=9, color="gray", style="italic")
                else:
                    # Choose text color based on background brightness
                    norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    text_color = "white" if norm_val < 0.3 or norm_val > 0.85 else "black"
                    # Format: short for large numbers, more decimals for small
                    if abs(val) >= 10:
                        txt = f"{val:.1f}"
                    elif abs(val) >= 1:
                        txt = f"{val:.2f}"
                    else:
                        txt = f"{val:.3f}"
                    ax.text(c, r, txt, ha="center", va="center",
                            fontsize=8.5, fontweight="bold", color=text_color)

        # Labels
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([format_level(l) for l in levels], fontsize=9)
        ax.set_xlabel(f"Noise Level", fontsize=10, fontweight="bold")
        ax.set_title(CHANNEL_LABELS[ch], fontsize=12, fontweight="bold", pad=8)

        if i == 0:
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["PPO", "SAC"], fontsize=11, fontweight="bold")
        else:
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["", ""])

        # Remove spines for cleaner look
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("0.7")
            spine.set_linewidth(0.5)

    # Colorbar
    cbar_ax = fig.add_subplot(gs[0, 3])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Mean Episode Reward", fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    # Suptitle
    fig.suptitle(f"{ENV_TITLES.get(env, env)} — Cross-Noise Robustness",
                 fontsize=15, fontweight="bold", y=0.97)

    if save:
        out_path = OUT_DIR / ENV_TITLES.get(env, env).lower().replace(" ", "") / "noise_heatmap.png"
        # Use the short env name directory that already exists
        env_short = {"evcharging": "ev", "building": "bu", "cogen": "co"}.get(env, env)
        out_path = OUT_DIR / env_short / "noise_heatmap.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"\n  Saved: {out_path}")

    plt.show()
    plt.close(fig)


# ============================================================================
# DEGRADATION HEATMAP (% drop from clean baseline)
# ============================================================================

def plot_env_degradation(env: str, last_n: int = 100, save: bool = True):
    """Create % degradation heatmap (relative to noise=0 baseline).

    Uses a single global color scale across all 3 panels so channels are
    directly comparable.  Percentages are clamped to [-100, +20] for
    readability; values beyond the clamp are still shown as text.
    """
    print(f"\n{'='*60}")
    print(f"  {ENV_TITLES.get(env, env)} — Noise Degradation (% from clean)")
    print(f"{'='*60}")

    data_ppo = collect_data(env, "PPO", last_n)
    data_sac = collect_data(env, "SAC", last_n)

    channels = ["obs", "action", "env"]

    # --- Pre-compute all degradation values for global color scale ---
    widths = []
    channel_levels = {}
    matrices = {}          # ch -> 2 x len(levels) array
    raw_matrices = {}      # un-clamped values for text annotations

    for ch in channels:
        levels = set()
        for data in [data_ppo, data_sac]:
            if ch in data:
                levels.update(k for k in data[ch].keys() if k > 0)
        channel_levels[ch] = sorted(levels)
        widths.append(max(len(channel_levels[ch]), 1))

        levels = channel_levels[ch]
        mat = np.full((2, len(levels)), np.nan)
        for algo_idx, data in enumerate([data_ppo, data_sac]):
            if 0.0 not in data.get(ch, {}):
                continue
            baseline = data[ch][0.0]
            for j, level in enumerate(levels):
                if level in data[ch] and abs(baseline) > 1e-9:
                    mat[algo_idx, j] = (data[ch][level] - baseline) / abs(baseline) * 100
        raw_matrices[ch] = mat.copy()
        # Clamp for color mapping (keeps heatmap readable)
        matrices[ch] = np.clip(mat, -100, 20)

    all_clamped = np.concatenate([m.ravel() for m in matrices.values()])
    all_clamped = all_clamped[~np.isnan(all_clamped)]
    if len(all_clamped) == 0:
        print("  No degradation data")
        return

    global_vmin = min(all_clamped.min(), -5)   # at least -5
    global_vmax = max(all_clamped.max(), 5)     # at least +5

    # --- Plot ---
    fig = plt.figure(figsize=(16, 3.8))
    gs = gridspec.GridSpec(1, 4, width_ratios=widths + [0.3],
                           wspace=0.15, left=0.05, right=0.93, top=0.82, bottom=0.18)

    im = None
    for i, ch in enumerate(channels):
        ax = fig.add_subplot(gs[0, i])
        levels = channel_levels[ch]
        if not levels:
            ax.set_visible(False)
            continue

        mat = matrices[ch]
        raw = raw_matrices[ch]

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                        vmin=global_vmin, vmax=global_vmax,
                        interpolation="nearest")

        for r in range(2):
            for c in range(len(levels)):
                val = raw[r, c]
                if np.isnan(val):
                    ax.text(c, r, "—", ha="center", va="center",
                            fontsize=9, color="gray", style="italic")
                else:
                    # Text color: white on dark backgrounds
                    clamped = mat[r, c]
                    norm_val = (clamped - global_vmin) / (global_vmax - global_vmin)
                    color = "white" if norm_val < 0.35 else "black"
                    ax.text(c, r, f"{val:+.1f}%", ha="center", va="center",
                            fontsize=8.5, fontweight="bold", color=color)

        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([format_level(l) for l in levels], fontsize=9)
        ax.set_xlabel("Noise Level", fontsize=10, fontweight="bold")
        ax.set_title(CHANNEL_LABELS[ch], fontsize=12, fontweight="bold", pad=8)

        if i == 0:
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["PPO", "SAC"], fontsize=11, fontweight="bold")
        else:
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["", ""])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("0.7")
            spine.set_linewidth(0.5)

    # Colorbar
    if im is not None:
        cbar_ax = fig.add_subplot(gs[0, 3])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Reward Change (%)", fontsize=10, fontweight="bold")
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle(f"{ENV_TITLES.get(env, env)} — Reward Degradation vs. Clean Baseline",
                 fontsize=15, fontweight="bold", y=0.97)

    if save:
        env_short = {"evcharging": "ev", "building": "bu", "cogen": "co"}.get(env, env)
        out_path = OUT_DIR / env_short / "noise_degradation.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"\n  Saved: {out_path}")

    plt.show()
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-Noise Robustness Heatmap")
    parser.add_argument("--env", choices=["evcharging", "building", "cogen"])
    parser.add_argument("--all", action="store_true", help="Generate for all envs")
    parser.add_argument("--last-n", type=int, default=100,
                        help="Average last N episodes for final reward (default: 100)")
    parser.add_argument("--no-degradation", action="store_true",
                        help="Skip degradation heatmap")
    args = parser.parse_args()

    envs = ["evcharging", "building", "cogen"] if args.all else [args.env]

    if not args.env and not args.all:
        parser.error("Specify --env or --all")

    for env in envs:
        plot_env_heatmap(env, last_n=args.last_n)
        if not args.no_degradation:
            plot_env_degradation(env, last_n=args.last_n)


if __name__ == "__main__":
    main()
