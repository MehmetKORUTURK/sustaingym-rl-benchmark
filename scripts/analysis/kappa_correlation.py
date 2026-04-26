"""
Correlate coupling score kappa with three thesis axes:
  1. Perturbation fragility  (from multiseed_stats.json)
  2. MARL performance gap     (from MARL test logs vs stdrl baseline)
  3. CMDP effectiveness       (from safe RL training logs)

Produces a correlation table and a decision rule.

Usage:
  python scripts/analysis/kappa_correlation.py
"""

import json
import os
import sys
import numpy as np
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

LOGS_DIR = os.path.join(PROJECT_ROOT, 'docs', 'thesis', 'logs')
KAPPA_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'kappa')


def load_kappa():
    """Load kappa values from compute_coupling_scores.py output."""
    with open(os.path.join(KAPPA_DIR, 'kappa_summary.json')) as f:
        data = json.load(f)
    return {
        'evcharging': data['evcharging']['kappa'],
        'building': data['building']['kappa'],
        'cogen': data['cogen']['kappa'],
    }


# ============================================================================
# AXIS 1: Perturbation Fragility
# ============================================================================
def compute_perturbation_fragility():
    """Compute % reward degradation at representative noise levels.

    Uses best algo per env from multiseed_stats.json (5-seed averages).
    Fragility = mean degradation across PS, PA, PD at moderate noise.
    """
    with open(os.path.join(LOGS_DIR, 'multiseed_stats.json')) as f:
        data = json.load(f)

    results = {}

    # EV Charging: best algo = PPO (highest baseline reward)
    ev = data['evcharging_PPO']
    baseline = ev['baseline']['mean']
    ps_deg = (baseline - ev['PS=0.15']['mean']) / abs(baseline) * 100
    pa_deg = (baseline - ev['PA=0.15']['mean']) / abs(baseline) * 100
    pd_deg = (baseline - ev['PD=0.15']['mean']) / abs(baseline) * 100
    results['evcharging'] = {
        'algo': 'PPO', 'baseline': baseline,
        'PS_degradation': ps_deg, 'PA_degradation': pa_deg, 'PD_degradation': pd_deg,
        'mean_degradation': (ps_deg + pa_deg + pd_deg) / 3,
    }

    # Building: best algo = SAC (closer to 0 = better, less negative)
    bu = data['building_SAC']
    baseline = bu['baseline']['mean']
    # Note: reward is negative, more negative = worse
    # Degradation = (noisy - baseline) / |baseline| * 100  (positive = worse)
    ps_deg = (bu['PS=0.15']['mean'] - baseline) / abs(baseline) * 100
    pa_deg = (bu['PA=0.15']['mean'] - baseline) / abs(baseline) * 100
    pd_deg = (bu['PD=0.15']['mean'] - baseline) / abs(baseline) * 100
    results['building'] = {
        'algo': 'SAC', 'baseline': baseline,
        'PS_degradation': abs(ps_deg), 'PA_degradation': abs(pa_deg), 'PD_degradation': abs(pd_deg),
        'mean_degradation': (abs(ps_deg) + abs(pa_deg) + abs(pd_deg)) / 3,
    }

    # Cogen: best algo = SAC (closest to 0)
    co = data['cogen_SAC']
    baseline = co['baseline']['mean']
    ps_deg = (co['PS=10']['mean'] - baseline) / abs(baseline) * 100
    pa_deg = (co['PA=0.1']['mean'] - baseline) / abs(baseline) * 100
    pd_deg = (co['PD=0.3']['mean'] - baseline) / abs(baseline) * 100
    results['cogen'] = {
        'algo': 'SAC', 'baseline': baseline,
        'PS_degradation': abs(ps_deg), 'PA_degradation': abs(pa_deg), 'PD_degradation': abs(pd_deg),
        'mean_degradation': (abs(ps_deg) + abs(pa_deg) + abs(pd_deg)) / 3,
    }

    return results


# ============================================================================
# AXIS 2: MARL Performance Gap
# ============================================================================
def compute_marl_gap():
    """Compute MARL performance relative to single-agent baseline.

    gap = (best_MARL_reward - stdrl_baseline) / |stdrl_baseline| * 100
    Positive = MARL better, Negative = MARL worse.
    """
    # StdRL baselines (from multiseed_test_stats.json, best algo per env)
    with open(os.path.join(LOGS_DIR, 'multiseed_test_stats.json')) as f:
        stdrl = json.load(f)

    stdrl_baselines = {
        'evcharging': stdrl['evcharging_PPO']['mean'],  # 6.24
        'building': stdrl['building_SAC']['mean'],       # -31.63
        'cogen': stdrl['cogen_SAC']['mean'],             # -0.041
    }

    # MARL results (best algo per env from evaluation_summary.txt)
    marl_results = {
        # EV: best = PPO_SHARED (7.20)
        'evcharging': {'algo': 'PPO_SHARED', 'mean': 7.198955},
        # Building: best = IMPALA_SHARED (-65.02)
        'building': {'algo': 'IMPALA_SHARED', 'mean': -65.017144},
        # Cogen: best = IMPALA (-0.137)
        'cogen': {'algo': 'IMPALA', 'mean': -0.136595},
    }

    results = {}
    for env in ['evcharging', 'building', 'cogen']:
        bl = stdrl_baselines[env]
        marl = marl_results[env]['mean']
        # For negative rewards (building, cogen): less negative = better
        # gap > 0 means MARL is better
        if bl < 0:
            gap = (marl - bl) / abs(bl) * 100  # positive if marl less negative
        else:
            gap = (marl - bl) / abs(bl) * 100  # positive if marl more positive

        results[env] = {
            'stdrl_baseline': bl,
            'marl_best': marl,
            'marl_algo': marl_results[env]['algo'],
            'gap_pct': gap,
        }

    return results


# ============================================================================
# AXIS 3: CMDP Effectiveness
# ============================================================================
def compute_cmdp_effectiveness():
    """Compute CMDP cost reduction effectiveness.

    For each environment, compare:
    - Unconstrained cost (loosest cost_limit, where CMDP has no effect)
    - Tightest cost_limit's achieved cost

    effectiveness = (unconstrained_cost - constrained_cost) / unconstrained_cost * 100
    Higher = CMDP is more effective at reducing cost.
    """
    # From the safe RL logs (last 10 epoch averages):
    # Using OnCRPO as representative algo (best overall per thesis)
    results = {
        'evcharging': {
            'algo': 'OnCRPO',
            'loose_limit': 1000, 'loose_cost': 182.47,    # CL=1000, cost=182.47
            'tight_limit': 3,    'tight_cost': 4.43,      # CL=3, cost=4.43
            'tight_reward': 4.45, 'loose_reward': 4.46,
        },
        'building': {
            'algo': 'OnCRPO',
            'loose_limit': 200,  'loose_cost': 199.39,    # CL=200, cost≈200 (unconstrained)
            'tight_limit': 25,   'tight_cost': 9.89,      # CL=25, cost=9.89
            'tight_reward': -30.30, 'loose_reward': -157.78,
        },
        'cogen': {
            'algo': 'OnCRPO',
            'loose_limit': 200,  'loose_cost': 21.74,     # CL=200, cost=21.74
            'tight_limit': 10,   'tight_cost': 11.72,     # CL=10, cost=11.72
            'tight_reward': -0.06, 'loose_reward': -0.06,
        },
    }

    for env in results:
        r = results[env]
        if r['loose_cost'] > 0:
            r['cost_reduction_pct'] = (r['loose_cost'] - r['tight_cost']) / r['loose_cost'] * 100
        else:
            r['cost_reduction_pct'] = 0.0

        # Reward sacrifice: how much reward is lost when constraint is tight
        if abs(r['loose_reward']) > 1e-6:
            r['reward_sacrifice_pct'] = abs(r['tight_reward'] - r['loose_reward']) / abs(r['loose_reward']) * 100
        else:
            r['reward_sacrifice_pct'] = 0.0

    return results


# ============================================================================
# CORRELATION & DECISION RULE
# ============================================================================
def compute_correlations(kappa, perturbation, marl, cmdp):
    """Compute Pearson/Spearman correlations between kappa and each axis."""
    envs = ['evcharging', 'building', 'cogen']
    k = np.array([kappa[e] for e in envs])

    metrics = {
        'perturbation_fragility': np.array([perturbation[e]['mean_degradation'] for e in envs]),
        'marl_gap': np.array([marl[e]['gap_pct'] for e in envs]),
        'cmdp_cost_reduction': np.array([cmdp[e]['cost_reduction_pct'] for e in envs]),
    }

    print(f"\n{'='*70}")
    print(f"CORRELATION ANALYSIS: kappa vs Three Axes")
    print(f"{'='*70}")
    print(f"\nkappa values: EV={k[0]:.4f}, Building={k[1]:.4f}, Cogen={k[2]:.4f}")

    correlations = {}
    for name, values in metrics.items():
        # With n=3, use Spearman (rank-based, no normality assumption)
        if len(set(np.argsort(k))) == len(k):
            rho, p_val = stats.spearmanr(k, values)
        else:
            rho, p_val = 0, 1.0
        pearson_r, pearson_p = stats.pearsonr(k, values)

        correlations[name] = {
            'spearman_rho': rho, 'spearman_p': p_val,
            'pearson_r': pearson_r, 'pearson_p': pearson_p,
        }

        print(f"\n--- {name} ---")
        print(f"  Values: {', '.join(f'{e}={v:.2f}' for e, v in zip(envs, values))}")
        print(f"  Spearman rho = {rho:+.4f} (p={p_val:.4f})")
        print(f"  Pearson r    = {pearson_r:+.4f} (p={pearson_p:.4f})")

        # Expected direction
        if name == 'perturbation_fragility':
            expected = "positive (higher kappa -> more fragile)"
            consistent = rho > 0
        elif name == 'marl_gap':
            expected = "negative (higher kappa -> MARL performs worse)"
            consistent = rho < 0
        else:
            # CMDP: high coupling makes constraint satisfaction harder
            # so cost reduction % is LOWER for high-kappa envs
            expected = "negative (higher kappa -> harder to satisfy constraints)"
            consistent = rho < 0
        print(f"  Expected: {expected}")
        print(f"  Observed: {'consistent' if consistent else 'INCONSISTENT'}")

    return correlations


def print_summary_table(kappa, perturbation, marl, cmdp):
    """Print the thesis-ready summary table."""
    envs = ['evcharging', 'building', 'cogen']

    print(f"\n{'='*70}")
    print(f"SUMMARY TABLE (for thesis)")
    print(f"{'='*70}")
    print(f"{'Env':<14} {'kappa':>6} {'Pert.Deg%':>10} {'MARL Gap%':>10} {'CMDP Red%':>10}")
    print(f"{'-'*54}")
    for e in envs:
        print(f"{e:<14} {kappa[e]:>6.3f} {perturbation[e]['mean_degradation']:>10.1f} "
              f"{marl[e]['gap_pct']:>10.1f} {cmdp[e]['cost_reduction_pct']:>10.1f}")

    print(f"\n{'='*70}")
    print(f"DECISION RULE TABLE")
    print(f"{'='*70}")
    print(f"{'kappa Range':<16} {'Perturbation':<20} {'CMDP':<22} {'MARL':<20}")
    print(f"{'-'*78}")
    print(f"{'kappa < 0.1':<16} {'Low fragility':<20} {'Verify need first':<22} {'MARL feasible':<20}")
    print(f"{'0.1 <= kappa < 0.5':<16} {'Moderate, algo-dep.':<20} {'Effective if orthog.':<22} {'Single-agent pref.':<20}")
    print(f"{'kappa >= 0.5':<16} {'High fragility':<20} {'Effective':<22} {'Avoid decomposition':<20}")

    print(f"\n{'='*70}")
    print(f"DETAILED BREAKDOWN")
    print(f"{'='*70}")
    for e in envs:
        print(f"\n--- {e.upper()} (kappa = {kappa[e]:.4f}) ---")
        p = perturbation[e]
        print(f"  Perturbation ({p['algo']}): PS={p['PS_degradation']:.1f}%, "
              f"PA={p['PA_degradation']:.1f}%, PD={p['PD_degradation']:.1f}% "
              f"=> mean={p['mean_degradation']:.1f}%")
        m = marl[e]
        print(f"  MARL ({m['marl_algo']}): stdrl={m['stdrl_baseline']:.3f}, "
              f"marl={m['marl_best']:.3f} => gap={m['gap_pct']:+.1f}%")
        c = cmdp[e]
        print(f"  CMDP ({c['algo']}): cost {c['loose_cost']:.1f} -> {c['tight_cost']:.1f} "
              f"=> reduction={c['cost_reduction_pct']:.1f}%, "
              f"reward sacrifice={c['reward_sacrifice_pct']:.1f}%")


def main():
    kappa = load_kappa()
    perturbation = compute_perturbation_fragility()
    marl = compute_marl_gap()
    cmdp = compute_cmdp_effectiveness()

    correlations = compute_correlations(kappa, perturbation, marl, cmdp)
    print_summary_table(kappa, perturbation, marl, cmdp)

    # Save results
    output = {
        'kappa': kappa,
        'perturbation': {k: {kk: vv for kk, vv in v.items()} for k, v in perturbation.items()},
        'marl': marl,
        'cmdp': {k: {kk: vv for kk, vv in v.items()} for k, v in cmdp.items()},
        'correlations': correlations,
    }
    output_path = os.path.join(KAPPA_DIR, 'kappa_correlation.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
