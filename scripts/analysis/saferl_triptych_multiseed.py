"""
Generate Safe RL constraint satisfaction triptych heatmap using multi-seed data.
Mirrors make_poster_figures.py fig2 but reads from saferl_multiseed_stats.json.

Usage:
  python scripts/analysis/saferl_triptych_multiseed.py
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
KAPPA_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'kappa')
OUT_DIR = os.path.join(PROJECT_ROOT, 'docs', 'myposter')


def load_stats():
    with open(os.path.join(KAPPA_DIR, 'saferl_multiseed_stats.json')) as f:
        return json.load(f)


def extract_env_data(stats, env_name, algos, cost_limits):
    """Extract ratio and cost matrices from multi-seed stats."""
    ratios = np.zeros((len(algos), len(cost_limits)))
    costs = np.zeros((len(algos), len(cost_limits)))
    stds = np.zeros((len(algos), len(cost_limits)))
    n_seeds = np.zeros((len(algos), len(cost_limits)), dtype=int)

    for i, algo in enumerate(algos):
        for j, cl in enumerate(cost_limits):
            key = f"{algo}_CL_{cl}"
            if key in stats.get(env_name, {}):
                entry = stats[env_name][key]
                cost_mean = entry['cost_mean']
                ratios[i, j] = cost_mean / cl
                costs[i, j] = cost_mean
                stds[i, j] = entry['cost_std']
                n_seeds[i, j] = entry['n_seeds']

    return ratios, costs, stds, n_seeds


def make_triptych(stats):
    algos = ['PPOLag', 'CPO', 'OnCRPO', 'FOCOPS']

    # EV Charging
    ev_cls = [3.0, 5.0, 25.0, 1000.0]
    ev_labels = ['CL=3', 'CL=5', 'CL=25', 'CL=1000']
    ev_ratios, ev_costs, ev_stds, ev_n = extract_env_data(stats, 'evcharging', algos, ev_cls)

    # Building
    bu_cls = [25.0, 50.0, 100.0, 200.0]
    bu_labels = ['CL=25', 'CL=50', 'CL=100', 'CL=200']
    bu_ratios, bu_costs, bu_stds, bu_n = extract_env_data(stats, 'building', algos, bu_cls)

    # Cogen (still single-seed)
    co_cls = [10.0, 25.0, 50.0, 200.0]
    co_labels = ['CL=10', 'CL=25', 'CL=50', 'CL=200']
    co_ratios, co_costs, co_stds, co_n = extract_env_data(stats, 'cogen', algos, co_cls)

    # Colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('pastel_constraint', [
        '#4caf50', '#81c784', '#a5d6a7', '#c8e6c9',
        '#f0f4c3', '#fff9c4',
        '#ffcdd2', '#ef9a9a', '#c62828',
    ], N=256)
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=2.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8), gridspec_kw={'wspace': 0.15})

    datasets = [
        ('EV Charging', ev_labels, ev_ratios, ev_costs, ev_stds, ev_n),
        ('Building HVAC', bu_labels, bu_ratios, bu_costs, bu_stds, bu_n),
        ('Cogeneration', co_labels, co_ratios, co_costs, co_stds, co_n),
    ]

    for idx, (title, labels, ratios, costs, stds, ns) in enumerate(datasets):
        ax = axes[idx]
        im = ax.imshow(ratios, cmap=cmap, norm=norm, aspect='auto')

        for i in range(4):
            for j in range(4):
                r = ratios[i, j]
                c = costs[i, j]
                s = stds[i, j]
                n = ns[i, j]

                fg = 'white' if r > 1.3 else '#2c3e50'
                fw = 'bold' if r > 1.0 else 'normal'
                pct = int(round(r * 100))

                if n > 1:
                    # Multi-seed: show mean +/- std
                    ax.text(j, i, f'{pct}%\n({c:.1f}\u00b1{s:.1f})\nn={n}',
                            ha='center', va='center', fontsize=9,
                            color=fg, fontweight=fw)
                else:
                    ax.text(j, i, f'{pct}%\n({c:.0f})',
                            ha='center', va='center', fontsize=10.5,
                            color=fg, fontweight=fw)

        # Grid lines
        for x in (0.5, 1.5, 2.5):
            ax.axvline(x, color='white', linewidth=1.5)
        for y in (0.5, 1.5, 2.5):
            ax.axhline(y, color='white', linewidth=1.5)

        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=11, fontweight='semibold', color='#2c3e50')
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False,
                       left=False, length=0, pad=6)

        ax.set_yticks(range(4))
        if idx == 0:
            ax.set_yticklabels(algos, fontsize=12, fontweight='semibold', color='#34495e')
            ax.tick_params(left=False, length=0, pad=8)
        else:
            ax.set_yticklabels([])

        ax.set_title(title, fontsize=14, fontweight='bold', color='#1a1a2e', pad=12)

        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle('Safe RL: Constraint Satisfaction Across Environments (Multi-Seed)',
                 fontsize=17, fontweight='bold', color='#1a1a2e', y=1.03)

    fig.subplots_adjust(top=0.88, bottom=0.22)

    pos_left = axes[0].get_position()
    pos_right = axes[-1].get_position()
    cbar_width = 0.40
    cbar_center = (pos_left.x0 + pos_right.x1) / 2
    cbar_left = cbar_center - cbar_width / 2
    cbar_ax = fig.add_axes([cbar_left, 0.11, cbar_width, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=10, length=3, color='#bdc3c7')
    cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
    cbar.set_ticklabels(['0%', '50%', '100% (limit)', '150%', '200%'])
    cbar.set_label('Cost / Cost Limit', fontsize=11, color='#4a4a4a', labelpad=6)

    cbar_center = cbar_left + cbar_width / 2
    fig.text(cbar_center, 0.02,
             'cell = mean % of cost limit used (cost mean\u00b1std)  |  green = constraint satisfied  |  multi-seed where available',
             ha='center', fontsize=10.5, color='#4a4a4a', style='italic')

    out_path = os.path.join(OUT_DIR, 'fig2_saferl_triptych_multiseed.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved to {out_path}')


def main():
    stats = load_stats()
    make_triptych(stats)


if __name__ == '__main__':
    main()
