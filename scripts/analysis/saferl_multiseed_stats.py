"""
Compute multi-seed statistics for safe RL experiments.
Parses OmniSafe progress.csv files across all seeds, algos, and cost limits.

Usage:
  python scripts/analysis/saferl_multiseed_stats.py
"""

import os
import sys
import json
import re
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOGS_DIR = os.path.join(PROJECT_ROOT, 'docs', 'thesis', 'logs', 'logs_saferl_train')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'kappa')


def parse_saferl_logs(env_name):
    """Parse all safe RL training logs for a given environment.

    Returns dict: {(algo, cost_limit): [(seed, reward, cost), ...]}
    """
    env_dir = os.path.join(LOGS_DIR, f'omnisafe_{env_name}')
    if not os.path.isdir(env_dir):
        return {}

    results = {}
    for dirname in sorted(os.listdir(env_dir)):
        dirpath = os.path.join(env_dir, dirname)
        if not os.path.isdir(dirpath):
            continue
        # Skip legacy, SACLag, seed dirs
        if dirname.startswith('v') or 'SACLag' in dirname or dirname.startswith('seed-'):
            continue

        # Extract algo and cost_limit from dirname
        algo_match = re.search(r'(PPOLag|CPO|OnCRPO|FOCOPS)', dirname)
        cl_match = re.search(r'CL_([\d.]+)', dirname)
        if not algo_match or not cl_match:
            continue

        algo = algo_match.group(1)
        cl = float(cl_match.group(1))

        # Skip CL_10000 (not in thesis standard set)
        if cl > 1000:
            continue

        # Find seed directory
        seed_dirs = []
        for root, dirs, files in os.walk(dirpath):
            for d in dirs:
                if d.startswith('seed-'):
                    seed_dirs.append(os.path.join(root, d))
        if not seed_dirs:
            continue

        for seed_dir in seed_dirs:
            progress = os.path.join(seed_dir, 'progress.csv')
            if not os.path.isfile(progress):
                continue

            seed_match = re.search(r'seed-(\d+)', os.path.basename(seed_dir))
            seed = int(seed_match.group(1)) if seed_match else -1

            # Last 10 epochs average
            try:
                with open(progress) as f:
                    lines = f.readlines()
                last_lines = lines[-10:]
                rewards, costs = [], []
                for line in last_lines:
                    parts = line.strip().split(',')
                    rewards.append(float(parts[0]))
                    costs.append(float(parts[1]))
                avg_reward = np.mean(rewards)
                avg_cost = np.mean(costs)
            except (ValueError, IndexError):
                continue

            key = (algo, cl)
            if key not in results:
                results[key] = []
            results[key].append({
                'seed': seed,
                'reward': avg_reward,
                'cost': avg_cost,
            })

    return results


def compute_stats(data):
    """Compute mean, std, CI95 for each (algo, cost_limit) group."""
    stats = {}
    for (algo, cl), runs in sorted(data.items()):
        rewards = [r['reward'] for r in runs]
        costs = [r['cost'] for r in runs]
        n = len(runs)
        stats[(algo, cl)] = {
            'n_seeds': n,
            'seeds': [r['seed'] for r in runs],
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards, ddof=1) if n > 1 else 0,
            'reward_ci95': 1.96 * np.std(rewards, ddof=1) / np.sqrt(n) if n > 1 else 0,
            'cost_mean': np.mean(costs),
            'cost_std': np.std(costs, ddof=1) if n > 1 else 0,
            'cost_ci95': 1.96 * np.std(costs, ddof=1) / np.sqrt(n) if n > 1 else 0,
            'rewards': rewards,
            'costs': costs,
        }
    return stats


def print_env_table(env_name, stats):
    """Print formatted multi-seed results table."""
    print(f"\n{'='*80}")
    print(f"SAFE RL MULTI-SEED: {env_name.upper()}")
    print(f"{'='*80}")
    print(f"{'Algo':<10} {'CL':>7} {'N':>3} {'Reward':>20} {'Cost':>20} {'Cost<=CL?':>10}")
    print(f"{'-'*75}")

    for (algo, cl), s in sorted(stats.items()):
        r_str = f"{s['reward_mean']:.2f} +/- {s['reward_std']:.2f}"
        c_str = f"{s['cost_mean']:.2f} +/- {s['cost_std']:.2f}"
        satisfied = "YES" if s['cost_mean'] <= cl else "NO"
        print(f"{algo:<10} {cl:>7.0f} {s['n_seeds']:>3} {r_str:>20} {c_str:>20} {satisfied:>10}")


def analyze_constraint_satisfaction(stats, env_name):
    """Analyze how well each algo satisfies constraints across seeds."""
    print(f"\n{'='*80}")
    print(f"CONSTRAINT SATISFACTION ANALYSIS: {env_name.upper()}")
    print(f"{'='*80}")

    algos = sorted(set(algo for algo, _ in stats.keys()))
    cls = sorted(set(cl for _, cl in stats.keys()))

    for algo in algos:
        print(f"\n--- {algo} ---")
        for cl in cls:
            key = (algo, cl)
            if key not in stats:
                continue
            s = stats[key]
            n_satisfied = sum(1 for c in s['costs'] if c <= cl)
            pct = n_satisfied / s['n_seeds'] * 100
            print(f"  CL={cl:>6.0f}: cost={s['cost_mean']:.2f}+/-{s['cost_std']:.2f}, "
                  f"satisfied={n_satisfied}/{s['n_seeds']} ({pct:.0f}%), "
                  f"reward={s['reward_mean']:.2f}+/-{s['reward_std']:.2f}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}
    for env_name in ['evcharging', 'building', 'cogen']:
        data = parse_saferl_logs(env_name)
        if not data:
            print(f"\n[{env_name}] No multi-seed data found.")
            continue

        stats = compute_stats(data)
        print_env_table(env_name, stats)
        analyze_constraint_satisfaction(stats, env_name)

        # Store for JSON output
        all_results[env_name] = {}
        for (algo, cl), s in stats.items():
            key_str = f"{algo}_CL_{cl}"
            all_results[env_name][key_str] = {
                'n_seeds': s['n_seeds'],
                'reward_mean': s['reward_mean'],
                'reward_std': s['reward_std'],
                'reward_ci95': s['reward_ci95'],
                'cost_mean': s['cost_mean'],
                'cost_std': s['cost_std'],
                'cost_ci95': s['cost_ci95'],
            }

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'saferl_multiseed_stats.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
