"""
Simple & Clean EV Charging RL Plotter
--------------------------------------
Simplified version maintaining original functionality.

Usage:
    python plot_simple.py --algo PPO --dt DS
"""

import os
import argparse
import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Publication-quality matplotlib settings
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["pdf.fonttype"] = 42  # TrueType fonts in PDF
plt.rcParams["ps.fonttype"] = 42


# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "./graphs/C_STDRL/"

# PPO + DS
PPO_DS_PATHS = [
    (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-05-20-06-35_DS_0.0_DA_0.0/monitor.csv","PPO_0.0",),
    
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-07-14-29-01_DS_0.01_DA_0.0/monitor.csv",   "PPO_0.01",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-07-14-46-43_DS_0.05_DA_0.0/monitor.csv",   "PPO_0.05",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-07-15-58-20_DS_0.1_DA_0.0/monitor.csv",    "PPO_0.1",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-07-16-24-30_DS_0.15_DA_0.0/monitor.csv",   "PPO_0.15",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-07-16-24-30_DS_0.2_DA_0.0/monitor.csv",    "PPO_0.2",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-07-16-54-19_DS_0.3_DA_0.0/monitor.csv",    "PPO_0.3",),
    
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-09-13-32-12_DS_0.01_DA_0.0/monitor.csv",   "PPO_0.01",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-09-14-31-24_DS_0.05_DA_0.0/monitor.csv",   "PPO_0.05",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-09-14-38-32_DS_0.1_DA_0.0/monitor.csv",    "PPO_0.1",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-09-14-41-09_DS_0.15_DA_0.0/monitor.csv",   "PPO_0.15",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-09-14-49-08_DS_0.2_DA_0.0/monitor.csv",    "PPO_0.2",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-09-14-49-37_DS_0.3_DA_0.0/monitor.csv",    "PPO_0.3",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-10-13-00-36_DS_0.4_DA_0.0/monitor.csv",    "PPO_0.4",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-10-13-00-36_DS_0.5_DA_0.0/monitor.csv",    "PPO_0.5",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-10-13-00-16_DS_0.6_DA_0.0/monitor.csv",    "PPO_0.6",),
    
    

    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_SAC/2026-02-05-21-16-55_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_SAC/2026-02-07-17-15-00_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
    (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_SAC/2026-02-09-12-45-20_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_TD3/2026-02-05-21-07-41_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_TD3/2026-02-07-17-45-25_DS_0.0_DA_0.0/monitor.csv", "TD3_0.0"),

]
PPO_DS_YLIM = (5, 8.5)

# PPO + DA
PPO_DA_PATHS = [
    (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-05-20-06-35_DS_0.0_DA_0.0/monitor.csv","PPO_0.0",),
    
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-10-13-36-28_DS_0.0_DA_0.01/monitor.csv","PPO__DA_0.01",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-10-22-42-09_DS_0.0_DA_0.05/monitor.csv","PPO__DA_0.05",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-11-03-20-57_DS_0.0_DA_0.1/monitor.csv","PPO__DA_0.10",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-11-03-22-23_DS_0.0_DA_0.15/monitor.csv","PPO__DA_0.15",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-11-05-41-17_DS_0.0_DA_0.2/monitor.csv","PPO__DA_0.20",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-11-08-02-01_DS_0.0_DA_0.3/monitor.csv","PPO__DA_0.30",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_PPO/2026-02-11-10-13-34_DS_0.0_DA_0.4/monitor.csv","PPO__DA_0.40",),
]
PPO_DA_YLIM = (4.2, 6.25)

# SAC + DS
SAC_DS_PATHS = [
    (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_SAC/2026-02-05-21-16-55_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_SAC/2026-02-12-15-46-22_NOISE_0.01_ACT_0.0/monitor.csv", "SAC_DS_0.01"),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_SAC/2026-02-12-15-46-22_NOISE_0.05_ACT_0.0/monitor.csv", "SAC_DS_0.05"),
    (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_SAC/2026-02-12-15-47-04_NOISE_0.1_ACT_0.0/monitor.csv", "SAC_DS_0.1"),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_SAC/2026-02-12-15-47-04_NOISE_0.15_ACT_0.0/monitor.csv", "SAC_DS_0.15"),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_SAC/2026-02-13-18-18-09_NOISE_0.2_ACT_0.0/monitor.csv", "SAC_DS_0.2"),
    (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_SAC/2026-02-13-18-23-14_NOISE_0.3_ACT_0.0/monitor.csv", "SAC_DS_0.3"),
    (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_SAC/2026-02-13-18-25-51_NOISE_0.4_ACT_0.0/monitor.csv", "SAC_DS_0.4"),
    (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_SAC/2026-02-13-18-45-20_NOISE_0.5_ACT_0.0/monitor.csv", "SAC_DS_0.5"),
    (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_SAC/2026-02-15-06-03-15_NOISE_0.6_ACT_0.0/monitor.csv", "SAC_DS_0.6"),
]
SAC_DS_YLIM = (3, 5)

# SAC + DA
SAC_DA_PATHS = [
    (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_SAC/2026-02-05-21-16-55_DS_0.0_DA_0.0/monitor.csv", "SAC_0.0"),
    # (r"logs_std_train/evcharging_SAC/2026-02-15-06-18-46_NOISE_0.0_ACT_0.01/monitor.csv", "SAC_DA_0.01"),
    # (r"logs_std_train/evcharging_SAC/2026-02-20-00-41-29_NOISE_0.0_ACT_0.05/monitor.csv", "SAC_DA_0.05"),
    (r"logs_std_train/evcharging_SAC/2026-02-19-22-32-04_NOISE_0.0_ACT_0.1/monitor.csv", "SAC_DA_0.1"),
    # (r"logs_std_train/evcharging_SAC/2026-02-19-22-13-55_NOISE_0.0_ACT_0.15/monitor.csv", "SAC_DA_0.15"),
    # (r"logs_std_train/evcharging_SAC/2026-02-19-19-02-28_NOISE_0.0_ACT_0.2/monitor.csv", "SAC_DA_0.2"),
    (r"logs_std_train/evcharging_SAC/2026-02-18-18-10-06_NOISE_0.0_ACT_0.3/monitor.csv", "SAC_DA_0.3"),
    # (r"logs_std_train/evcharging_SAC/2026-02-18-16-50-26_NOISE_0.0_ACT_0.4/monitor.csv", "SAC_DA_0.4"),
    # (r"logs_std_train/evcharging_SAC/2026-02-18-16-50-26_NOISE_0.0_ACT_0.5/monitor.csv", "SAC_DA_0.5"),
    # (r"logs_std_train/evcharging_SAC/2026-02-18-16-49-39_NOISE_0.0_ACT_0.6/monitor.csv", "SAC_DA_0.6"),

]
SAC_DA_YLIM = (3.5, 7.5)

# TD3 + DS
TD3_DS_PATHS = [
    (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/evcharging_TD3/2026-02-05-21-07-41_DS_0.0_DA_0.0/monitor.csv", "TD3_DS_0.2"),
    ("C:/Users/mkoru/OneDrive/Desktop/programs/sustaingym-main/sustaingym-main/sustaingym/logs/evcharging_TD3/2025-03-07-08-56-57_DS_0.4_DA_0.0/monitor.csv", "TD3_DS_0.4"),
    ("C:/Users/mkoru/OneDrive/Desktop/programs/sustaingym-main/sustaingym-main/sustaingym/logs_test/evcharging_TD3/2025-03-09-12-08-09_DS_0.0_DA_0.0/monitor.csv", "TD3_DS_0.0"),
]
TD3_DS_YLIM = (2, 8)

# TD3 + DA
TD3_DA_PATHS = [
    ("C:/Users/mkoru/OneDrive/Desktop/programs/sustaingym-main/sustaingym-main/sustaingym/logs/evcharging_TD3/2025-03-07-08-53-50_DS_0.0_DA_0.0/monitor.csv", "TD3_DA_0.0"),
    ("C:/Users/mkoru/OneDrive/Desktop/programs/sustaingym-main/sustaingym-main/sustaingym/logs/evcharging_TD3/2025-03-07-08-58-15_DS_0.0_DA_0.2/monitor.csv", "TD3_DA_0.2"),
    ("C:/Users/mkoru/OneDrive/Desktop/programs/sustaingym-main/sustaingym-main/sustaingym/logs/evcharging_TD3/2025-03-07-08-59-36_DS_0.0_DA_0.4/monitor.csv", "TD3_DA_0.4"),
]
TD3_DA_YLIM = (3.5, 7.5)

# PPO + DE (Environment Noise)
PPO_DE_PATHS = [
    # (r"logs_std_train/evcharging_PPO/TODO_NOISE_0.0_ACT_0.0_ENV_0.01/monitor.csv", "PPO_ENV_0.01"),
    # (r"logs_std_train/evcharging_PPO/TODO_NOISE_0.0_ACT_0.0_ENV_0.05/monitor.csv", "PPO_ENV_0.05"),
    # (r"logs_std_train/evcharging_PPO/TODO_NOISE_0.0_ACT_0.0_ENV_0.1/monitor.csv", "PPO_ENV_0.1"),
    # (r"logs_std_train/evcharging_PPO/TODO_NOISE_0.0_ACT_0.0_ENV_0.15/monitor.csv", "PPO_ENV_0.15"),
    # (r"logs_std_train/evcharging_PPO/TODO_NOISE_0.0_ACT_0.0_ENV_0.2/monitor.csv", "PPO_ENV_0.2"),
    # (r"logs_std_train/evcharging_PPO/TODO_NOISE_0.0_ACT_0.0_ENV_0.3/monitor.csv", "PPO_ENV_0.3"),
    # (r"logs_std_train/evcharging_PPO/TODO_NOISE_0.0_ACT_0.0_ENV_0.4/monitor.csv", "PPO_ENV_0.4"),
    # (r"logs_std_train/evcharging_PPO/TODO_NOISE_0.0_ACT_0.0_ENV_0.5/monitor.csv", "PPO_ENV_0.5"),
    # (r"logs_std_train/evcharging_PPO/TODO_NOISE_0.0_ACT_0.0_ENV_0.6/monitor.csv", "PPO_ENV_0.6"),
]
PPO_DE_YLIM = (5, 8.5)

# SAC + DE (Environment Noise)
SAC_DE_PATHS = [
    # (r"logs_std_train/evcharging_SAC/TODO_NOISE_0.0_ACT_0.0_ENV_0.01/monitor.csv", "SAC_ENV_0.01"),
    # (r"logs_std_train/evcharging_SAC/TODO_NOISE_0.0_ACT_0.0_ENV_0.05/monitor.csv", "SAC_ENV_0.05"),
    # (r"logs_std_train/evcharging_SAC/TODO_NOISE_0.0_ACT_0.0_ENV_0.1/monitor.csv", "SAC_ENV_0.1"),
    # (r"logs_std_train/evcharging_SAC/TODO_NOISE_0.0_ACT_0.0_ENV_0.15/monitor.csv", "SAC_ENV_0.15"),
    # (r"logs_std_train/evcharging_SAC/TODO_NOISE_0.0_ACT_0.0_ENV_0.2/monitor.csv", "SAC_ENV_0.2"),
    # (r"logs_std_train/evcharging_SAC/TODO_NOISE_0.0_ACT_0.0_ENV_0.3/monitor.csv", "SAC_ENV_0.3"),
    # (r"logs_std_train/evcharging_SAC/TODO_NOISE_0.0_ACT_0.0_ENV_0.4/monitor.csv", "SAC_ENV_0.4"),
    # (r"logs_std_train/evcharging_SAC/TODO_NOISE_0.0_ACT_0.0_ENV_0.5/monitor.csv", "SAC_ENV_0.5"),
    # (r"logs_std_train/evcharging_SAC/TODO_NOISE_0.0_ACT_0.0_ENV_0.6/monitor.csv", "SAC_ENV_0.6"),
]
SAC_DE_YLIM = (3, 5)

# TD3 + DE (Environment Noise)
TD3_DE_PATHS = [
    # (r"logs_std_train/evcharging_TD3/TODO_NOISE_0.0_ACT_0.0_ENV_0.01/monitor.csv", "TD3_ENV_0.01"),
    # (r"logs_std_train/evcharging_TD3/TODO_NOISE_0.0_ACT_0.0_ENV_0.05/monitor.csv", "TD3_ENV_0.05"),
    # (r"logs_std_train/evcharging_TD3/TODO_NOISE_0.0_ACT_0.0_ENV_0.1/monitor.csv", "TD3_ENV_0.1"),
    # (r"logs_std_train/evcharging_TD3/TODO_NOISE_0.0_ACT_0.0_ENV_0.15/monitor.csv", "TD3_ENV_0.15"),
    # (r"logs_std_train/evcharging_TD3/TODO_NOISE_0.0_ACT_0.0_ENV_0.2/monitor.csv", "TD3_ENV_0.2"),
    # (r"logs_std_train/evcharging_TD3/TODO_NOISE_0.0_ACT_0.0_ENV_0.3/monitor.csv", "TD3_ENV_0.3"),
    # (r"logs_std_train/evcharging_TD3/TODO_NOISE_0.0_ACT_0.0_ENV_0.4/monitor.csv", "TD3_ENV_0.4"),
    # (r"logs_std_train/evcharging_TD3/TODO_NOISE_0.0_ACT_0.0_ENV_0.5/monitor.csv", "TD3_ENV_0.5"),
    # (r"logs_std_train/evcharging_TD3/TODO_NOISE_0.0_ACT_0.0_ENV_0.6/monitor.csv", "TD3_ENV_0.6"),
]
TD3_DE_YLIM = (2, 8)

# All configs
CONFIG = {
    "PPO": {"DS": (PPO_DS_PATHS, PPO_DS_YLIM), "DA": (PPO_DA_PATHS, PPO_DA_YLIM), "DE": (PPO_DE_PATHS, PPO_DE_YLIM)},
    "SAC": {"DS": (SAC_DS_PATHS, SAC_DS_YLIM), "DA": (SAC_DA_PATHS, SAC_DA_YLIM), "DE": (SAC_DE_PATHS, SAC_DE_YLIM)},
    "TD3": {"DS": (TD3_DS_PATHS, TD3_DS_YLIM), "DA": (TD3_DA_PATHS, TD3_DA_YLIM), "DE": (TD3_DE_PATHS, TD3_DE_YLIM)},
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_rewards(csv_path: str, max_steps: int) -> Optional[np.ndarray]:
    """Load rewards from CSV."""
    if not Path(csv_path).exists():
        print(f"⚠️  File not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path, comment="#", header=0)
        if "r" not in df.columns:
            print(f"⚠️  No 'r' column: {csv_path}")
            return None

        rewards = df["r"].astype(float).values[:max_steps]
        print(f"✓ Loaded {len(rewards)} episodes")
        return rewards

    except Exception as e:
        print(f"❌ Error: {csv_path} - {e}")
        return None


def calculate_running_stats(rewards: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate running mean and std using EMA (no window-edge kink)."""
    series = pd.Series(rewards)
    mean = series.ewm(span=window, adjust=True).mean().values
    std = series.ewm(span=window, adjust=True).std().values
    return mean, std


def plot_learning_curves(
    experiments: List[Tuple[str, str]],
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    title: str = "",
    window: int = 500,
    max_steps: int = 32000,
    skip_initial: int = 250,
    auto_xlim: bool = False,
    auto_ylim: bool = False
):
    """Plot learning curves with shaded ±1 std band."""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Colorblind-friendly palette with enough distinct colors
    colors = sns.color_palette("tab10", n_colors=max(len(experiments), 10))
    # Line styles and markers cycle for B&W distinguishability
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<']
    mark_every_frac = 0.08  # place ~12 markers per curve

    for idx, (csv_path, label) in enumerate(experiments):
        # Load data
        rewards = load_rewards(csv_path, max_steps)
        if rewards is None:
            continue

        # Calculate running mean and std
        mean, std = calculate_running_stats(rewards, window)

        # Plot with distinct color + linestyle + marker
        x = np.arange(len(mean))
        color = colors[idx % len(colors)]
        ls = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        me = max(1, int(len(x) * mark_every_frac))

        ax.plot(x, mean, label=label, linewidth=1.8, color=color,
                linestyle=ls, marker=marker, markersize=4,
                markevery=me, markeredgewidth=0.5)
        ax.fill_between(x, mean - 0.3 * std, mean + 0.3 * std,
                        alpha=0.12, color=color, linewidth=0)

    # Plot settings
    # X-axis limits
    if auto_xlim:
        pass  # matplotlib auto-scales
    elif xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(skip_initial, max_steps)  # Default behavior

    # Y-axis limits
    if auto_ylim:
        pass  # matplotlib auto-scales
    elif ylim:
        ax.set_ylim(ylim)
    # If neither auto_ylim nor ylim, matplotlib will auto-scale

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Training Episode", fontweight='bold')
    ax.set_ylabel("Average Episode Reward", fontweight='bold')

    # Professional legend
    ax.legend(loc='lower right', frameon=True, shadow=True, fancybox=True)

    # Clean grid and spines
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()


def save_plot(algo: str, dt: str, output_dir: str, train_steps: int, window: int, save_pdf: bool = False):
    """Save plot to file (PNG always, PDF optional)."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_filename = f"{timestamp}_{algo}_Reward_{dt}"

    # Save PNG (always)
    png_path = os.path.join(output_dir, base_filename + ".png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ PNG saved: {png_path}")

    # Save PDF (optional, vector graphics)
    if save_pdf:
        pdf_path = os.path.join(output_dir, base_filename + ".pdf")
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"✅ PDF saved: {pdf_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='EV Charging RL Simple Plotter')
    parser.add_argument('--algo', type=str, default='PPO', choices=['PPO', 'SAC', 'TD3'])
    parser.add_argument('--dt', type=str, default='DS', choices=['DS', 'DA', 'DE'])
    parser.add_argument('--t_steps', type=int, default=32000, help='Training steps')
    parser.add_argument('--w_size', type=int, default=500, help='Window size')
    parser.add_argument('--skip', type=int, default=250, help='Skip initial episodes')
    parser.add_argument('--pdf', action='store_true', help='Also save PDF (vector graphics)')

    # New optional axis limit parameters
    parser.add_argument('--ylim', nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help='Y-axis limits (e.g., --ylim 5 6.5). If not set, uses config defaults.')
    parser.add_argument('--xlim', nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help='X-axis limits (e.g., --xlim 250 32000). If not set, uses skip and t_steps.')
    parser.add_argument('--auto_ylim', action='store_true',
                        help='Force auto y-axis limits (ignore config defaults)')
    parser.add_argument('--auto_xlim', action='store_true',
                        help='Force auto x-axis limits (ignore skip and t_steps)')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"📊 Plotting {args.algo} + {args.dt}...")
    print(f"{'='*70}\n")

    # Get paths from config
    if args.algo not in CONFIG or args.dt not in CONFIG[args.algo]:
        print(f"❌ Config not found: {args.algo} - {args.dt}")
        return

    paths, default_ylim = CONFIG[args.algo][args.dt]

    # Determine ylim
    if args.ylim:
        ylim = tuple(args.ylim)
        print(f"✓ Using custom Y-axis limits: {ylim}")
    elif args.auto_ylim:
        ylim = None
        print("✓ Using auto Y-axis limits")
    else:
        ylim = default_ylim
        print(f"✓ Using config Y-axis limits: {ylim}")

    # Determine xlim
    if args.xlim:
        xlim = tuple(args.xlim)
        auto_xlim = False
        print(f"✓ Using custom X-axis limits: {xlim}")
    elif args.auto_xlim:
        xlim = None
        auto_xlim = True
        print("✓ Using auto X-axis limits")
    else:
        xlim = None
        auto_xlim = False
        print(f"✓ Using default X-axis limits: ({args.skip}, {args.t_steps})")

    # Plot
    title = "EVCharging-v0"
    plot_learning_curves(
        experiments=paths,
        ylim=ylim,
        xlim=xlim,
        title=title,
        window=args.w_size,
        max_steps=args.t_steps,
        skip_initial=args.skip,
        auto_xlim=auto_xlim,
        auto_ylim=args.auto_ylim
    )

    # Save
    save_plot(args.algo, args.dt, OUTPUT_DIR, args.t_steps, args.w_size, args.pdf)

    # Show
    plt.show()


if __name__ == "__main__":
    main()
