"""
MARL training curve plotter for SustainGym environments.

Reads metrics.csv files produced by marl_training.py (or marl_tr_*.py)
and plots smoothed reward curves per algorithm, following the same
publication-quality style as stdrl_plot_*.py and omni_plot.py.

Usage:
    python ttp/marl_plot.py --env evcharging --t_steps 32000 --w_size 500
    python ttp/marl_plot.py --env building --w_size 400 --auto_ylim --pdf
    python ttp/marl_plot.py --env cogen --t_steps 100 --w_size 20
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ── Publication-quality Matplotlib settings ──────────────────────────────

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["pdf.fonttype"] = 42  # TrueType in PDFs

# ── Style constants ─────────────────────────────────────────────────────

COLORS = sns.color_palette("tab10", n_colors=10)
LINE_STYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]
MARK_EVERY_FRAC = 0.08  # ~12 markers per curve
FILL_FACTOR = 0.5       # band width: mean +/- 0.5 * std
FILL_ALPHA = 0.12

ENV_TITLES = {
    "evcharging": "EVCharging-v0",
    "building": "Building-v0",
    "cogen": "Cogen-v0",
}

# ── Configuration: (csv_path, label) tuples per environment ─────────────
# Update these paths after each training run on ARC.
# Each entry is (path_to_metrics.csv, display_label).
#
# Example:
#   ("./logs_marl_train/evcharging_APPO/2026-03-08_14-30-00_NOISE_0_ACT_0_ENV_0/metrics.csv", "APPO"),

CONFIG = {
    "evcharging": [
        (r"logs_marl_train/evcharging_PPO/2026-03-09_00-27-27_NOISE_0_ACT_0_ENV_0/metrics.csv", "PPO"),
        (r"logs_marl_train/evcharging_PPO_SHARED/2026-03-09_12-31-13_NOISE_0_ACT_0_ENV_0/metrics.csv", "PPO_sh"),

        # (r"logs_marl_train/evcharging_APPO/2026-03-10_16-08-19_NOISE_0_ACT_0_ENV_0", "APPO"),
        # (r"logs_marl_train/evcharging_APPO/2026-03-10_16-08-19_NOISE_0_ACT_0_ENV_0", "APPO"),
        # (r"logs_marl_train/evcharging_APPO_SHARED/2026-03-09_00-44-07_NOISE_0_ACT_0_ENV_0/metrics.csv", "APPO_sh"),

        # (r"logs_marl_train/evcharging_IMPALA/2026-03-09_12-56-37_NOISE_0_ACT_0_ENV_0/metrics.csv", "IMPALA"),
        # (r"logs_marl_train/evcharging_IMPALA_SHARED/2026-03-09_17-21-09_NOISE_0_ACT_0_ENV_0/metrics.csv", "IMPALA_sh"),

        # (r"logs_marl_train/evcharging_SAC/2026-03-09_12-38-22_NOISE_0_ACT_0_ENV_0/metrics.csv", "SAC"),
        # (r"logs_marl_train/evcharging_SAC_SHARED/2026-03-09_12-31-13_NOISE_0_ACT_0_ENV_0/metrics.csv", "SAC_sh"),
    ],
    "building": [
        (r"logs_marl_train/building_PPO/2026-03-09_17-28-58_NOISE_0_ACT_0_ENV_0/metrics.csv", "PPO"),
        (r"logs_marl_train/building_APPO/2026-03-09_22-02-07_NOISE_0_ACT_0_ENV_0/metrics.csv", "APPO"),
        (r"logs_marl_train/building_IMPALA/2026-03-10_00-03-07_NOISE_0_ACT_0_ENV_0/metrics.csv", "IMPALA"),
        (r"logs_marl_train/building_IMPALA_SHARED/2026-03-10_01-23-22_NOISE_0_ACT_0_ENV_0/metrics.csv", "IMPALA_sh"),
        (r"logs_marl_train/building_SAC/2026-03-09_20-54-35_NOISE_0_ACT_0_ENV_0/metrics.csv", "SAC"),

    ],
    "cogen": [
        # ("./logs_marl_train/cogen_PPO/.../metrics.csv", "PPO"),
        # ("./logs_marl_train/cogen_APPO/.../metrics.csv", "APPO"),
    ],
}

# ── CLI Arguments ────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="MARL training curve plotter for SustainGym environments")
parser.add_argument("--env", type=str, required=True,
                    choices=["evcharging", "building", "cogen"],
                    help="Environment to plot")
parser.add_argument("--t_steps", type=int, default=0,
                    help="Max iterations to plot (0 = all data)")
parser.add_argument("--w_size", type=int, default=500,
                    help="EMA smoothing span (default: 500)")
parser.add_argument("--skip", type=int, default=0,
                    help="Skip first N rows from each CSV")
parser.add_argument("--ylim", type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="Force Y-axis limits [min max]")
parser.add_argument("--auto_ylim", action="store_true",
                    help="Let matplotlib auto-scale Y axis")
parser.add_argument("--xlim", type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="Force X-axis limits [min max]")
parser.add_argument("--auto_xlim", action="store_true",
                    help="Let matplotlib auto-scale X axis")
parser.add_argument("--pdf", action="store_true",
                    help="Also save PDF output")
args = parser.parse_args()

# ── Validate Configuration ──────────────────────────────────────────────

entries = CONFIG[args.env]
if not entries:
    print(f"[ERROR] No CSV paths configured for '{args.env}' in CONFIG dict.")
    print("        Edit ttp/marl_plot.py and add (path, label) entries.")
    raise SystemExit(1)

# ── Plot ────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

for i, (csv_path, label) in enumerate(entries):
    if not os.path.isfile(csv_path):
        print(f"[WARN] File not found, skipping: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    if "iteration" not in df.columns or "mean_reward" not in df.columns:
        print(f"[WARN] Missing 'iteration' or 'mean_reward' column in {csv_path}")
        continue

    # Skip initial rows and truncate
    if args.skip > 0:
        df = df.iloc[args.skip:].reset_index(drop=True)
    if args.t_steps > 0:
        df = df[df["iteration"] <= args.t_steps]

    if df.empty:
        print(f"[WARN] No data after filtering for '{label}'")
        continue

    # EMA smoothing
    mean = df["mean_reward"].ewm(span=args.w_size, adjust=True).mean()
    std = df["mean_reward"].ewm(span=args.w_size, adjust=True).std()

    x = df["iteration"]
    n_rows = len(df)
    mark_every = max(1, int(n_rows * MARK_EVERY_FRAC))

    color = COLORS[i % len(COLORS)]
    marker = MARKERS[i % len(MARKERS)]
    ls = LINE_STYLES[i % len(LINE_STYLES)]

    ax.plot(x, mean, label=label, color=color, linestyle=ls,
            linewidth=1.8, marker=marker, markersize=4,
            markevery=mark_every, markeredgewidth=0.5)
    ax.fill_between(x, mean - FILL_FACTOR * std, mean + FILL_FACTOR * std,
                    alpha=FILL_ALPHA, color=color)

# ── Styling ─────────────────────────────────────────────────────────────

ax.set_title(ENV_TITLES[args.env], fontsize=16, weight="bold")
ax.set_xlabel("Training Iteration")
ax.set_ylabel("Mean Episode Reward")
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(loc="lower right", frameon=True, shadow=True, fancybox=True)

if args.ylim:
    ax.set_ylim(args.ylim)
if args.xlim:
    ax.set_xlim(args.xlim)
elif not args.auto_xlim and args.t_steps > 0:
    ax.set_xlim(0, args.t_steps)

fig.tight_layout()

# ── Save ────────────────────────────────────────────────────────────────

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = "./graphs/C_MARL"
os.makedirs(out_dir, exist_ok=True)

# Extract unique algo names from labels for filename
algo_names = "_".join(dict.fromkeys(lbl for _, lbl in entries))
out_name = f"{timestamp}_{args.env}_MARL_{algo_names}"

png_path = os.path.join(out_dir, f"{out_name}.png")
fig.savefig(png_path)
print(f"[SAVED] {png_path}")

if args.pdf:
    pdf_path = os.path.join(out_dir, f"{out_name}.pdf")
    fig.savefig(pdf_path)
    print(f"[SAVED] {pdf_path}")

plt.show()
