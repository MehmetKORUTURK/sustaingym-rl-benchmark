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
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/cogen_PPO/2026-02-05-18-56-37_DS_0.0_DA_0.0/monitor.csv","PPO_0.0",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/cogen_PPO/2026-02-09-11-59-36_DS_0.0_DA_0.0/monitor.csv","PPO_0.0",),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/cogen_PPO/2026-02-09-18-02-55_DS_0.0_DA_0.0/monitor.csv","PPO_0.0",),
    
    # (r"logs_std_train/cogen_PPO/2026-02-25-20-33-57_NOISE_0.5_ACT_0.0/monitor.csv","PPO_0.5",),    
    (r"logs_std_train/cogen_PPO/2026-02-25-21-00-28_NOISE_1.0_ACT_0.0/monitor.csv","PPO_1",),
    # (r"logs_std_train/cogen_PPO/2026-02-25-23-21-59_NOISE_2.0_ACT_0.0/monitor.csv","PPO_2",),
    # (r"logs_std_train/cogen_PPO/2026-02-26-00-24-39_NOISE_3.0_ACT_0.0/monitor.csv","PPO_3",),
    (r"logs_std_train/cogen_PPO/2026-02-26-00-24-39_NOISE_4.0_ACT_0.0/monitor.csv","PPO_4",),
    # (r"logs_std_train/cogen_PPO/2026-02-26-00-58-13_NOISE_5.0_ACT_0.0/monitor.csv","PPO_5",),

    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/cogen_SAC/2026-02-06-11-37-08_DS_0.0_DA_0.0/monitor.csv", "SAC_DS_0.0"),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/cogen_SAC/2026-02-09-11-59-29_DS_0.0_DA_0.0/monitor.csv", "SAC_DS_0.0"),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/cogen_SAC/2026-02-09-18-03-16_DS_0.0_DA_0.0/monitor.csv", "SAC_DS_0.0"),
    
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/cogen_TD3/2026-02-07-18-29-28_DS_0.0_DA_0.0/monitor.csv", "TD3_DS_0.0"),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/cogen_TD3/2026-02-09-11-59-14_DS_0.0_DA_0.0/monitor.csv", "TD3_DS_0.0"),
    # (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/cogen_TD3/2026-02-09-18-03-50_DS_0.0_DA_0.0/monitor.csv", "TD3_DS_0.0"),

]
PPO_DS_YLIM = (-0.5, 0)

# PPO + DA
PPO_DA_PATHS = [
    ("C:/Users/mkoru/OneDrive/Desktop/programs/sustaingym-main/sustaingym-main/sustaingym/logs/evcharging_PPO_2025-02-25-19-47-06_DS_0.0/monitor.csv", "PPO_0.0"),
    ("C:/Users/mkoru/OneDrive/Desktop/programs/sustaingym-main/sustaingym-main/sustaingym/logs/evcharging_PPO_2025-02-15-17-26-30_DA_0.2_0.0/monitor.csv", "PPO_DA_0.2"),
    ("C:/Users/mkoru/OneDrive/Desktop/programs/sustaingym-main/sustaingym-main/sustaingym/logs/evcharging_PPO_2025-02-15-17-27-12_DA_0.4_0.0/monitor.csv", "PPO_DA_0.4"),
    ("C:/Users/mkoru/OneDrive/Desktop/programs/sustaingym-main/sustaingym-main/sustaingym/logs/evcharging_PPO_2025-02-15-17-28-12_DA_0.6_0.0/monitor.csv", "PPO_DA_0.6"),
]
PPO_DA_YLIM = (4.2, 6.25)

# SAC + DS
SAC_DS_PATHS = [
    (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/cogen_SAC/2026-02-06-11-37-08_DS_0.0_DA_0.0/monitor.csv", "SAC_DS_0.0"),
    ("C:/Users/mkoru/OneDrive/Desktop/programs/sustaingym-main/sustaingym-main/sustaingym/logs/evcharging_SAC_2025-02-16-10-21-28_DS_0.4/monitor.csv", "SAC_DS_0.4"),
    ("C:/Users/mkoru/OneDrive/Desktop/programs/sustaingym-main/sustaingym-main/sustaingym/logs/evcharging_SAC_2025-02-16-10-21-35_DS_0.6/monitor.csv", "SAC_DS_0.6"),
]
SAC_DS_YLIM = (3, 5)

# SAC + DA
SAC_DA_PATHS = [
    ("C:/Users/mkoru/OneDrive/Desktop/programs/sustaingym-main/sustaingym-main/sustaingym/logs/evcharging_TD3/2025-03-07-08-53-50_DS_0.0_DA_0.0/monitor.csv", "SAC_DA_0.0"),
    ("C:/Users/mkoru/OneDrive/Desktop/programs/sustaingym-main/sustaingym-main/sustaingym/logs/evcharging_TD3/2025-03-07-08-58-15_DS_0.0_DA_0.2/monitor.csv", "SAC_DA_0.2"),
    ("C:/Users/mkoru/OneDrive/Desktop/programs/sustaingym-main/sustaingym-main/sustaingym/logs/evcharging_TD3/2025-03-07-08-59-36_DS_0.0_DA_0.4/monitor.csv", "SAC_DA_0.4"),
]
SAC_DA_YLIM = (3.5, 7.5)

# TD3 + DS
TD3_DS_PATHS = [
    (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/cogen_TD3/2026-02-05-22-00-01_DS_0.0_DA_0.0/monitor.csv", "TD3_DS_0.0"),
    (r"/home/mkoruturk/programs/sustaingym-main/sustaingym-main/sustaingym/logs_std_train/cogen_TD3/2026-02-06-11-29-47_DS_0.0_DA_0.0/monitor.csv", "TD3_DS_0.0"),
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
    # (r"logs_std_train/cogen_PPO/TODO_NOISE_0.0_ACT_0.0_ENV_0.5/monitor.csv", "PPO_ENV_0.5"),
    # (r"logs_std_train/cogen_PPO/TODO_NOISE_0.0_ACT_0.0_ENV_1.0/monitor.csv", "PPO_ENV_1.0"),
    # (r"logs_std_train/cogen_PPO/TODO_NOISE_0.0_ACT_0.0_ENV_2.0/monitor.csv", "PPO_ENV_2.0"),
    # (r"logs_std_train/cogen_PPO/TODO_NOISE_0.0_ACT_0.0_ENV_3.0/monitor.csv", "PPO_ENV_3.0"),
    # (r"logs_std_train/cogen_PPO/TODO_NOISE_0.0_ACT_0.0_ENV_4.0/monitor.csv", "PPO_ENV_4.0"),
    # (r"logs_std_train/cogen_PPO/TODO_NOISE_0.0_ACT_0.0_ENV_5.0/monitor.csv", "PPO_ENV_5.0"),
]
PPO_DE_YLIM = (-0.5, 0)

# SAC + DE (Environment Noise)
SAC_DE_PATHS = [
    # (r"logs_std_train/cogen_SAC/TODO_NOISE_0.0_ACT_0.0_ENV_0.5/monitor.csv", "SAC_ENV_0.5"),
    # (r"logs_std_train/cogen_SAC/TODO_NOISE_0.0_ACT_0.0_ENV_1.0/monitor.csv", "SAC_ENV_1.0"),
    # (r"logs_std_train/cogen_SAC/TODO_NOISE_0.0_ACT_0.0_ENV_2.0/monitor.csv", "SAC_ENV_2.0"),
    # (r"logs_std_train/cogen_SAC/TODO_NOISE_0.0_ACT_0.0_ENV_3.0/monitor.csv", "SAC_ENV_3.0"),
    # (r"logs_std_train/cogen_SAC/TODO_NOISE_0.0_ACT_0.0_ENV_4.0/monitor.csv", "SAC_ENV_4.0"),
    # (r"logs_std_train/cogen_SAC/TODO_NOISE_0.0_ACT_0.0_ENV_5.0/monitor.csv", "SAC_ENV_5.0"),
]
SAC_DE_YLIM = (3, 5)

# TD3 + DE (Environment Noise)
TD3_DE_PATHS = [
    # (r"logs_std_train/cogen_TD3/TODO_NOISE_0.0_ACT_0.0_ENV_0.5/monitor.csv", "TD3_ENV_0.5"),
    # (r"logs_std_train/cogen_TD3/TODO_NOISE_0.0_ACT_0.0_ENV_1.0/monitor.csv", "TD3_ENV_1.0"),
    # (r"logs_std_train/cogen_TD3/TODO_NOISE_0.0_ACT_0.0_ENV_2.0/monitor.csv", "TD3_ENV_2.0"),
    # (r"logs_std_train/cogen_TD3/TODO_NOISE_0.0_ACT_0.0_ENV_3.0/monitor.csv", "TD3_ENV_3.0"),
    # (r"logs_std_train/cogen_TD3/TODO_NOISE_0.0_ACT_0.0_ENV_4.0/monitor.csv", "TD3_ENV_4.0"),
    # (r"logs_std_train/cogen_TD3/TODO_NOISE_0.0_ACT_0.0_ENV_5.0/monitor.csv", "TD3_ENV_5.0"),
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
        ax.fill_between(x, mean - 0.5 * std, mean + 0.5 * std,
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
    parser.add_argument('--t_steps', type=int, default=96000, help='Training steps')
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
    title = "Cogen-v0"
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
