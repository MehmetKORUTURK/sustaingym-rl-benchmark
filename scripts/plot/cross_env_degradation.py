"""
Generate cross-environment degradation figure for IEEE SmartGridComm paper.
3-panel figure: one per perturbation channel (PS, PA, PD).
Each panel shows degradation % vs normalized sigma for best algo per env.
No pandas/numpy — uses stdlib + matplotlib only.

Usage:
    python scripts/plot/cross_env_degradation.py
"""

import csv
import re
import os
import statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEST_DIR = Path(__file__).resolve().parents[2] / "logs_std_test"
OUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "papers" / "paper2_perturbation"

BEST_ALGO = {"evcharging": "PPO", "building": "PPO", "cogen": "PPO"}
SECOND_ALGO = {"evcharging": "SAC", "building": "SAC", "cogen": "SAC"}

ENV_LABELS = {"evcharging": "EV Charging", "building": "Building HVAC", "cogen": "Cogeneration"}
CHANNEL_LABELS = {"obs": "State Perturbation (PS)", "action": "Action Perturbation (PA)", "env": "Dynamics Perturbation (PD)"}
ENV_COLORS = {"evcharging": "#1565C0", "building": "#C62828", "cogen": "#2E7D32"}
ENV_MARKERS = {"evcharging": "o", "building": "s", "cogen": "^"}
MAX_SIGMA = {"evcharging": 0.6, "building": 0.4, "cogen": 5.0}


def parse_noise(dirname):
    result = {"obs": 0.0, "action": 0.0, "env": 0.0}
    m = re.search(r"NOISE_([\d.]+)", dirname)
    if m: result["obs"] = float(m.group(1))
    m = re.search(r"ACT_([\d.]+)", dirname)
    if m: result["action"] = float(m.group(1))
    m = re.search(r"ENV_([\d.]+)", dirname)
    if m: result["env"] = float(m.group(1))
    return result


def is_single_channel(noise):
    active = [ch for ch, v in noise.items() if v > 0]
    if len(active) == 0: return "clean"
    if len(active) == 1: return active[0]
    return None


def read_test_reward(run_dir):
    csv_path = run_dir / "episode_results.csv"
    if not csv_path.exists():
        return None
    try:
        rewards = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rewards.append(float(row["total_reward"]))
        return statistics.mean(rewards) if rewards else None
    except Exception:
        return None


def collect_data(env, algo):
    d = TEST_DIR / f"{env}_{algo}"
    if not d.exists():
        return {}
    data = {"obs": {}, "action": {}, "env": {}}
    cleans = []
    for entry in sorted(os.listdir(d)):
        rd = d / entry
        if not rd.is_dir(): continue
        noise = parse_noise(entry)
        ch = is_single_channel(noise)
        if ch is None: continue
        r = read_test_reward(rd)
        if r is None: continue
        if ch == "clean":
            cleans.append(r)
        else:
            lv = noise[ch]
            if lv not in data[ch] or r > data[ch][lv]:
                data[ch][lv] = r
    if cleans:
        c = statistics.mean(cleans)
        for ch in data:
            data[ch][0.0] = c
    return data


def degradation(data, channel):
    ch_data = data.get(channel, {})
    if not ch_data or 0.0 not in ch_data:
        return [], []
    clean = ch_data[0.0]
    if clean == 0: return [], []
    sigmas = sorted(ch_data.keys())
    degs = [(clean - ch_data[s]) / abs(clean) * 100 for s in sigmas]
    return sigmas, degs


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6))  # IEEE double-column width
    channels = ["obs", "action", "env"]

    for ax_idx, channel in enumerate(channels):
        ax = axes[ax_idx]
        for env in ["evcharging", "building", "cogen"]:
            for algo, ls in [(BEST_ALGO[env], "-"), (SECOND_ALGO[env], "--")]:
                data = collect_data(env, algo)
                sigmas, degs = degradation(data, channel)
                if not sigmas: continue

                # Normalize sigma
                norm = [s / MAX_SIGMA[env] for s in sigmas]
                label = f"{ENV_LABELS[env]} ({algo})"
                ax.plot(norm, degs, color=ENV_COLORS[env], marker=ENV_MARKERS[env],
                        markersize=3.5, linewidth=1.2, linestyle=ls, label=label)

        ax.set_title(CHANNEL_LABELS[channel], fontweight="bold")
        ax.set_xlabel("Normalized $\\sigma$ / $\\sigma_{\\max}$")
        if ax_idx == 0:
            ax.set_ylabel("Degradation (%)")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.set_xlim(-0.02, 1.02)

    # Collect all handles for unified legend
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    for ext in ["pdf", "png"]:
        p = OUT_DIR / f"fig_cross_env_degradation.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
